# src/PixelForge/data/paired.py
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

# Define supported image file extensions
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

class PairedOnTheFlyDataset(Dataset):
    """
    Super-resolution training dataset with two modes:
    - Precomputed mode: If an LR dataset root is provided and contains files, pairs LR files with matching HR files by relative path.
    - On-the-fly mode: If no LR dataset is provided, creates LR by bicubic downscale from HR.

    Returns dict with 'lr', 'hr', 'path'.
    """
    def __init__(self, root_hr: Path, fallback: Path | None, root_lr: Path | None = None, *, scale: int = 4, patch_hr: int | None = None):
        """
        Initialize the dataset.

        Args:
            root_hr: Primary directory path to high-resolution images
            fallback: Fallback directory path if primary directory is empty
            root_lr: Optional directory path to precomputed low-resolution images
            scale: Downscaling factor for creating low-resolution images
            patch_hr: Size of high-resolution patches to extract (if None, use whole image)
        """
        self.scale, self.patch_hr = scale, patch_hr
        self.mode_precomputed = False
        self.hr_lr_pairs: list[tuple[Path, Path]] = []
        self.paths: list[Path] = []
        self.hr_root: Path | None = None
        self.lr_root: Path | None = None

        # Normalize HR root (allow nested train/valid folders)
        cand_hr = Path(root_hr)
        root_hr_norm = cand_hr / "train" if (cand_hr / "train").exists() else cand_hr
        if not (root_hr_norm.exists() and root_hr_norm.is_dir()):
            raise FileNotFoundError(f"Directory not found: {root_hr_norm}")

        hr_paths = [p for p in root_hr_norm.rglob("*") if p.suffix.lower() in IMG_EXTS]
        self.hr_root = root_hr_norm

        # If empty, try fallback HR
        if not hr_paths and fallback and Path(fallback).exists():
            fb = Path(fallback)
            fb = fb / "train" if (fb / "train").exists() else fb
            hr_paths = [p for p in fb.rglob("*") if p.suffix.lower() in IMG_EXTS]
            self.hr_root = fb

        if not hr_paths:
            raise FileNotFoundError("No images found in data/HR/ or data/samples/.")

        # Attempt to pair with LR dataset if provided
        if root_lr is not None and self.hr_root is not None:
            lr_root_norm = Path(root_lr)
            lr_root_norm = lr_root_norm / "train" if (lr_root_norm / "train").exists() else lr_root_norm
            if lr_root_norm.exists() and lr_root_norm.is_dir():
                self.lr_root = lr_root_norm

                def find_lr_for(hr_path: Path) -> Path | None:
                    # Map HR relative path under hr_root to LR under lr_root
                    try:
                        rel = hr_path.relative_to(self.hr_root)  # may be just filename or nested path
                    except Exception:
                        return None
                    candidate = (self.lr_root / rel)
                    if candidate.exists() and candidate.suffix.lower() in IMG_EXTS:
                        return candidate
                    # Else try any extension with same stem in the same directory
                    stem = candidate.stem
                    lr_dir = candidate.parent
                    for ext in IMG_EXTS:
                        alt = lr_dir / f"{stem}{ext}"
                        if alt.exists():
                            return alt
                    return None

                pairs: list[tuple[Path, Path]] = []
                for hp in hr_paths:
                    lp = find_lr_for(hp)
                    if lp is not None:
                        pairs.append((hp, lp))
                if pairs:
                    self.mode_precomputed = True
                    self.hr_lr_pairs = pairs
                # If no pairs found, silently fall back to on-the-fly

        # If not using precomputed, keep only HR paths for on-the-fly mode
        if not self.mode_precomputed:
            self.paths = hr_paths

    def _center_crop_multiple(self, img: Image.Image, multiple: int) -> Image.Image:
        """
        Center crop an image to dimensions that are multiples of a given number.
        
        Args:
            img: Input PIL image
            multiple: Number that dimensions should be multiples of
            
        Returns:
            Cropped PIL image
        """
        w, h = img.size
        W = (w // multiple) * multiple; H = (h // multiple) * multiple
        left = (w - W) // 2; top  = (h - H) // 2
        return img.crop((left, top, left + W, top + H))

    def __len__(self): 
        """Return the number of samples available"""
        if self.mode_precomputed:
            return len(self.hr_lr_pairs)
        return len(self.paths)

    def __getitem__(self, idx):
        """
        Get a paired sample of LR and HR images.

        Returns a dict with tensors 'lr' and 'hr' in [0,1], and 'path' to the HR source.
        """
        if self.mode_precomputed:
            hr_path, lr_path = self.hr_lr_pairs[idx]
            hr = Image.open(hr_path).convert("RGB")
            lr = Image.open(lr_path).convert("RGB")

            # Center-crop HR to multiple of scale; crop LR accordingly to maintain alignment
            hr = self._center_crop_multiple(hr, self.scale)
            w_hr, h_hr = hr.size
            target_lr_size = (w_hr // self.scale, h_hr // self.scale)

            # Center-crop LR to expected dimensions if differs
            w_lr, h_lr = lr.size
            if (w_lr, h_lr) != target_lr_size:
                # Compute centered crop for LR to match target size
                left = max(0, (w_lr - target_lr_size[0]) // 2)
                top  = max(0, (h_lr - target_lr_size[1]) // 2)
                lr = lr.crop((left, top, left + target_lr_size[0], top + target_lr_size[1]))

            # Optional center patching
            if self.patch_hr is not None:
                w_hr, h_hr = hr.size
                if w_hr >= self.patch_hr and h_hr >= self.patch_hr:
                    left_hr = (w_hr - self.patch_hr) // 2; top_hr = (h_hr - self.patch_hr) // 2
                    hr = hr.crop((left_hr, top_hr, left_hr + self.patch_hr, top_hr + self.patch_hr))
                    # Corresponding LR patch
                    patch_lr = self.patch_hr // self.scale
                    w_lr, h_lr = lr.size
                    if w_lr >= patch_lr and h_lr >= patch_lr:
                        left_lr = (w_lr - patch_lr) // 2; top_lr = (h_lr - patch_lr) // 2
                        lr = lr.crop((left_lr, top_lr, left_lr + patch_lr, top_lr + patch_lr))

            hr_t = TF.to_tensor(hr); lr_t = TF.to_tensor(lr)
            return {"lr": lr_t, "hr": hr_t, "path": str(hr_path)}
        else:
            path = self.paths[idx]
            hr = Image.open(path).convert("RGB")
            hr = self._center_crop_multiple(hr, self.scale)
            if self.patch_hr is not None:
                w, h = hr.size
                if w >= self.patch_hr and h >= self.patch_hr:
                    left = (w - self.patch_hr) // 2; top = (h - self.patch_hr) // 2
                    hr = hr.crop((left, top, left + self.patch_hr, top + self.patch_hr))
            w, h = hr.size
            lr = hr.resize((w // self.scale, h // self.scale), Image.BICUBIC)
            hr_t = TF.to_tensor(hr); lr_t = TF.to_tensor(lr)
            return {"lr": lr_t, "hr": hr_t, "path": str(path)}