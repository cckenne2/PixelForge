# scripts/make_lr.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import argparse
import numpy as np
from PIL import Image, ImageFilter
import torch
import torch.nn.functional as F

# Mirror the training dataset’s extension list
from PixelForge.data.paired import IMG_EXTS  # {.png,.jpg,.jpeg,.bmp,.webp}

def center_crop_multiple(img: Image.Image, multiple: int) -> Image.Image:
    w, h = img.size
    W = (w // multiple) * multiple
    H = (h // multiple) * multiple
    left = (w - W) // 2
    top  = (h - H) // 2
    return img.crop((left, top, left + W, top + H))

def _pil_to_tensor(img: Image.Image, device: torch.device) -> torch.Tensor:
    t = torch.from_numpy(np.asarray(img).astype(np.float32) / 255.0)
    if t.ndim == 2:
        t = t[..., None]
    t = t.permute(2, 0, 1).unsqueeze(0).to(device)
    return t

def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    t = t.detach().clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((t * 255.0 + 0.5).astype(np.uint8), mode="RGB")

def _srgb_to_linear(t: torch.Tensor) -> torch.Tensor:
    a = (t <= 0.04045).to(t.dtype)
    return a * (t / 12.92) + (1 - a) * torch.pow((t + 0.055) / 1.055, 2.4)

def _linear_to_srgb(t: torch.Tensor) -> torch.Tensor:
    a = (t <= 0.0031308).to(t.dtype)
    return a * (t * 12.92) + (1 - a) * (1.055 * torch.pow(t.clamp(min=0.0), 1.0 / 2.4) - 0.055)

def _gaussian_kernel1d(sigma: float, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if sigma <= 0:
        return torch.tensor([1.0], dtype=dtype, device=device)
    radius = int(max(1, round(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(x * x) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    return kernel

def _gaussian_blur(img_t: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return img_t
    b, c, h, w = img_t.shape
    k1d = _gaussian_kernel1d(sigma, img_t.dtype, img_t.device)
    kx = k1d.view(1, 1, 1, -1)
    ky = k1d.view(1, 1, -1, 1)
    img_t = F.conv2d(img_t, kx.repeat(c, 1, 1, 1), padding=(0, k1d.numel() // 2), groups=c)
    img_t = F.conv2d(img_t, ky.repeat(c, 1, 1, 1), padding=(k1d.numel() // 2, 0), groups=c)
    return img_t

def downscale_once(hr: Image.Image, scale: int, method: str, gamma_aware: bool, preblur_sigma: float) -> Image.Image:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        if preblur_sigma > 0:
            hr = hr.filter(ImageFilter.GaussianBlur(radius=preblur_sigma))
        method_map_pil = {
            "BICUBIC": Image.BICUBIC,
            "LANCZOS": Image.LANCZOS,
            "BOX": Image.BOX,
            "BILINEAR": Image.BILINEAR,
        }
        resample = method_map_pil.get(method.upper(), Image.BICUBIC)
        if gamma_aware:
            arr = (np.asarray(hr).astype("float32") / 255.0)
            a = (arr <= 0.04045).astype(np.float32)
            lin = a * (arr / 12.92) + (1 - a) * (((arr + 0.055) / 1.055) ** 2.4)
            lin_img = Image.fromarray((lin * 255.0 + 0.5).astype("uint8"), mode="RGB")
            lr_lin = lin_img.resize((hr.width // scale, hr.height // scale), resample=resample)
            arr = (np.asarray(lr_lin).astype("float32") / 255.0)
            a = (arr <= 0.0031308).astype(np.float32)
            srgb = a * (arr * 12.92) + (1 - a) * (1.055 * (arr ** (1.0 / 2.4)) - 0.055)
            return Image.fromarray((np.clip(srgb, 0, 1) * 255.0 + 0.5).astype("uint8"), mode="RGB")
        else:
            return hr.resize((hr.width // scale, hr.height // scale), resample=resample)

    t = _pil_to_tensor(hr, device)
    if preblur_sigma > 0:
        t = _gaussian_blur(t, preblur_sigma)

    mode_map = {
        "BICUBIC": "bicubic",
        "LANCZOS": "bicubic",
        "BOX": "area",
        "BILINEAR": "bilinear",
    }
    mode = mode_map.get(method.upper(), "bicubic")
    size = (hr.height // scale, hr.width // scale)

    if gamma_aware:
        t_lin = _srgb_to_linear(t)
        lr_lin = F.interpolate(t_lin, size=size, mode=mode, align_corners=False if mode != "area" else None, antialias=True)
        t_srgb = _linear_to_srgb(lr_lin)
        return _tensor_to_pil(t_srgb)
    else:
        lr = F.interpolate(t, size=size, mode=mode, align_corners=False if mode != "area" else None, antialias=True)
        return _tensor_to_pil(lr)
def main():
    ap = argparse.ArgumentParser("Regenerate LR images from HR")
    ap.add_argument("--hr", type=str, required=True, help="Input HR root (recurses)")
    ap.add_argument("--out", type=str, required=True, help="Output LR root")
    ap.add_argument("--scale", type=int, default=4, choices=[2,4,8])
    ap.add_argument("--method", type=str, default="BICUBIC",
                    choices=["BICUBIC","LANCZOS","BOX","BILINEAR"])
    ap.add_argument("--gamma_aware", action="store_true", help="Resize in linear light to reduce halos/speckles")
    ap.add_argument("--preblur", type=float, default=0.0, help="Pre-blur sigma before downscaling")
    ap.add_argument("--ext", type=str, default=".png", help="Output file extension (e.g., .png, .jpg)")
    ap.add_argument("--keep_structure", action="store_true", help="Center crop HR to multiple of scale")
    ap.add_argument("--verbose", action="store_true", help="Print progress for every image")
    args = ap.parse_args()

    in_root = Path(args.hr)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    exts = {e.lower() for e in IMG_EXTS}
    paths = [p for p in in_root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    if not paths:
        print(f"No images under {in_root}")
        return

    for i, p in enumerate(paths, 1):
        try:
            hr = Image.open(p).convert("RGB")
        except Exception as e:
            print(f"Skipping {p}: {e}")
            continue

        if args.keep_structure:
            hr = center_crop_multiple(hr, args.scale)

        lr = downscale_once(hr, args.scale, args.method, args.gamma_aware, args.preblur)

        rel = p.relative_to(in_root)
        out_path = (out_root / rel).with_suffix(args.ext)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if args.ext.lower() in [".jpg", ".jpeg"]:
            # Save *safely* if you must use JPEG: high quality, disable subsampling
            lr.save(out_path, quality=95, subsampling=0, optimize=True)
        else:
            lr.save(out_path, optimize=True)

    if args.verbose or (i % 100 == 0 or i == len(paths)):
            print(f"[{i}/{len(paths)}] {p.name} -> {out_path}")

    print(f"\nWrote {len(paths)} LR images to: {out_root.resolve()}")

if __name__ == "__main__":
    main()
