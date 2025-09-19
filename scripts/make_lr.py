# scripts/make_lr.py
"""
Low-Resolution Image Generation Script

This script generates low-resolution (LR) images from high-resolution (HR) images
using various downscaling methods. It supports gamma-aware processing, pre-blur
filtering, and multiple resampling algorithms to create training datasets for
super-resolution models.

Key features:
- Multiple downscaling methods (bicubic, Lanczos, box, bilinear)
- Gamma-aware processing for perceptually accurate downscaling
- GPU acceleration when available
- Preserves directory structure
- Pre-blur filtering to reduce aliasing
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import argparse
import numpy as np
from PIL import Image, ImageFilter
import torch
import torch.nn.functional as F

# Mirror the training dataset's extension list
from PixelForge.data.paired import IMG_EXTS  # {.png,.jpg,.jpeg,.bmp,.webp}

def center_crop_multiple(img: Image.Image, multiple: int) -> Image.Image:
    """
    Center crop an image to dimensions that are multiples of a given value.
    
    This ensures that the image dimensions are evenly divisible by the scale factor,
    which is important for clean downscaling without fractional pixel issues.
    
    Args:
        img (Image.Image): Input PIL image
        multiple (int): The value that width and height should be multiples of
        
    Returns:
        Image.Image: Center-cropped image with dimensions as multiples of 'multiple'
        
    Example:
        If img is 1023x767 and multiple=4, returns 1020x764 image centered on original
    """
    w, h = img.size
    # Calculate largest dimensions that are multiples of the scale factor
    W = (w // multiple) * multiple
    H = (h // multiple) * multiple
    
    # Calculate crop box for center cropping
    left = (w - W) // 2
    top  = (h - H) // 2
    return img.crop((left, top, left + W, top + H))

def _pil_to_tensor(img: Image.Image, device: torch.device) -> torch.Tensor:
    """
    Convert PIL image to PyTorch tensor with proper normalization and format.
    
    Converts from PIL's HWC uint8 format to PyTorch's BCHW float32 format
    with values normalized to [0,1] range.
    
    Args:
        img (Image.Image): Input PIL image (RGB or grayscale)
        device (torch.device): Target device for tensor
        
    Returns:
        torch.Tensor: Normalized tensor in BCHW format, shape [1, C, H, W]
    """
    # Convert to numpy array and normalize to [0,1]
    t = torch.from_numpy(np.asarray(img).astype(np.float32) / 255.0)
    
    # Handle grayscale images by adding channel dimension
    if t.ndim == 2:
        t = t[..., None]
    
    # Convert from HWC to BCHW format: (H,W,C) -> (C,H,W) -> (1,C,H,W)
    t = t.permute(2, 0, 1).unsqueeze(0).to(device)
    return t

def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """
    Convert PyTorch tensor back to PIL image format.
    
    Converts from PyTorch's BCHW float format to PIL's HWC uint8 format.
    
    Args:
        t (torch.Tensor): Input tensor in BCHW format with values in [0,1]
        
    Returns:
        Image.Image: PIL RGB image
    """
    # Convert from BCHW to HWC format and clamp values
    t = t.detach().clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0).cpu().numpy()
    # Convert to uint8 with proper rounding
    return Image.fromarray((t * 255.0 + 0.5).astype(np.uint8), mode="RGB")

def _srgb_to_linear(t: torch.Tensor) -> torch.Tensor:
    """
    Convert sRGB color space to linear RGB color space.
    
    The sRGB color space uses a gamma curve to better match human perception.
    Converting to linear space is important for mathematically correct image
    processing operations like filtering and resampling.
    
    Formula:
    - If sRGB <= 0.04045: linear = sRGB / 12.92
    - If sRGB > 0.04045: linear = ((sRGB + 0.055) / 1.055) ^ 2.4
    
    Args:
        t (torch.Tensor): Input tensor with sRGB values in [0,1]
        
    Returns:
        torch.Tensor: Tensor with linear RGB values
    """
    a = (t <= 0.04045).to(t.dtype)
    return a * (t / 12.92) + (1 - a) * torch.pow((t + 0.055) / 1.055, 2.4)

def _linear_to_srgb(t: torch.Tensor) -> torch.Tensor:
    """
    Convert linear RGB color space back to sRGB color space.
    
    This is the inverse of sRGB to linear conversion, applying the sRGB gamma curve.
    
    Formula:
    - If linear <= 0.0031308: sRGB = linear * 12.92
    - If linear > 0.0031308: sRGB = 1.055 * linear^(1/2.4) - 0.055
    
    Args:
        t (torch.Tensor): Input tensor with linear RGB values
        
    Returns:
        torch.Tensor: Tensor with sRGB values in [0,1]
    """
    a = (t <= 0.0031308).to(t.dtype)
    return a * (t * 12.92) + (1 - a) * (1.055 * torch.pow(t.clamp(min=0.0), 1.0 / 2.4) - 0.055)

def _gaussian_kernel1d(sigma: float, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """
    Generate 1D Gaussian kernel for image blurring.
    
    Creates a discrete approximation of the Gaussian function for convolution-based
    blurring. The kernel size is automatically determined based on sigma.
    
    Args:
        sigma (float): Standard deviation of Gaussian. If <= 0, returns identity kernel
        dtype (torch.dtype): Data type for the kernel
        device (torch.device): Device to create kernel on
        
    Returns:
        torch.Tensor: Normalized 1D Gaussian kernel
    """
    if sigma <= 0:
        return torch.tensor([1.0], dtype=dtype, device=device)
    
    # Rule of thumb: kernel radius = 3*sigma covers ~99.7% of distribution
    radius = int(max(1, round(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    
    # Compute Gaussian values: exp(-(x^2)/(2*sigma^2))
    kernel = torch.exp(-(x * x) / (2 * sigma * sigma))
    
    # Normalize so sum equals 1 (preserves image brightness)
    kernel = kernel / kernel.sum()
    return kernel

def _gaussian_blur(img_t: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Apply Gaussian blur to image tensor using separable convolution.
    
    Uses two 1D convolutions (horizontal then vertical) instead of one 2D convolution
    for efficiency. This is mathematically equivalent but computationally faster.
    Uses reflection padding to avoid edge darkening artifacts.
    
    Args:
        img_t (torch.Tensor): Input image tensor in BCHW format
        sigma (float): Blur strength (standard deviation)
        
    Returns:
        torch.Tensor: Blurred image tensor
    """
    if sigma <= 0:
        return img_t
    
    b, c, h, w = img_t.shape
    k1d = _gaussian_kernel1d(sigma, img_t.dtype, img_t.device)
    
    # Reshape kernels for horizontal and vertical convolution
    kx = k1d.view(1, 1, 1, -1)  # Horizontal kernel: [1,1,1,K]
    ky = k1d.view(1, 1, -1, 1)  # Vertical kernel: [1,1,K,1]
    
    # Calculate padding needed (half kernel size on each side)
    pad_x = k1d.numel() // 2
    pad_y = k1d.numel() // 2
    
    # Apply horizontal convolution with reflection padding
    x = F.pad(img_t, (pad_x, pad_x, 0, 0), mode="reflect")
    x = F.conv2d(x, kx.repeat(c, 1, 1, 1), padding=0, groups=c)
    
    # Apply vertical convolution with reflection padding
    x = F.pad(x, (0, 0, pad_y, pad_y), mode="reflect")
    x = F.conv2d(x, ky.repeat(c, 1, 1, 1), padding=0, groups=c)
    
    return x

def downscale_once(hr: Image.Image, scale: int, method: str, gamma_aware: bool, preblur_sigma: float) -> Image.Image:
    """
    Downscale a high-resolution image by an integer factor.
    
    This function provides multiple downscaling approaches optimized for different
    hardware and quality requirements. It can process in linear color space for
    better perceptual quality and apply pre-blur to reduce aliasing.
    
    Args:
        hr (Image.Image): High-resolution input image
        scale (int): Downscaling factor (e.g., 4 for 4x smaller output)
        method (str): Resampling method - "BICUBIC", "LANCZOS", "BOX", "BILINEAR"
        gamma_aware (bool): Whether to process in linear color space
        preblur_sigma (float): Gaussian blur sigma to apply before downscaling
        
    Returns:
        Image.Image: Downscaled low-resolution image
        
    Notes:
        - Uses GPU when available for faster processing
        - BOX method uses exact averaging for integer scales
        - Gamma-aware processing reduces color fringing and improves quality
    """
    # Determine processing device (prefer GPU for speed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
    
    if device.type == "cpu":
        # CPU processing path: use PIL for simplicity and avoid GPU overhead
        method_map_pil = {
            "BICUBIC": Image.BICUBIC,
            "LANCZOS": Image.LANCZOS,
            "BOX": Image.BOX,
            "BILINEAR": Image.BILINEAR,
        }
        resample = method_map_pil.get(method.upper(), Image.BICUBIC)
        
        if gamma_aware:
            # Process in linear color space for better quality
            arr = (np.asarray(hr).astype("float32") / 255.0)
            
            # Convert sRGB to linear (manual implementation for CPU)
            a = (arr <= 0.04045).astype(np.float32)
            lin = a * (arr / 12.92) + (1 - a) * (((arr + 0.055) / 1.055) ** 2.4)
            
            # Apply pre-blur in linear space if requested
            if preblur_sigma > 0:
                # Approximate Gaussian blur using PIL (not perfect but fast)
                lin_img = Image.fromarray((np.clip(lin, 0, 1) * 255.0 + 0.5).astype("uint8"), mode="RGB")
                lin_img = lin_img.filter(ImageFilter.GaussianBlur(radius=preblur_sigma))
                lin = np.asarray(lin_img).astype("float32") / 255.0
            
            # Perform downscaling in linear space
            lin_img = Image.fromarray((np.clip(lin, 0, 1) * 255.0 + 0.5).astype("uint8"), mode="RGB")
            lr_lin = lin_img.resize((hr.width // scale, hr.height // scale), 
                                   resample=resample if method.upper() != "BOX" else Image.BOX)
            
            # Convert back to sRGB
            arr = (np.asarray(lr_lin).astype("float32") / 255.0)
            a = (arr <= 0.0031308).astype(np.float32)
            srgb = a * (arr * 12.92) + (1 - a) * (1.055 * (arr ** (1.0 / 2.4)) - 0.055)
            return Image.fromarray((np.clip(srgb, 0, 1) * 255.0 + 0.5).astype("uint8"), mode="RGB")
        else:
            # Simple sRGB processing
            if preblur_sigma > 0:
                hr = hr.filter(ImageFilter.GaussianBlur(radius=preblur_sigma))
            return hr.resize((hr.width // scale, hr.height // scale), 
                           resample=resample if method.upper() != "BOX" else Image.BOX)

    # GPU processing path: use PyTorch for better performance and quality
    t = _pil_to_tensor(hr, device)

    # Map method names to PyTorch interpolation modes
    mode_map = {
        "BICUBIC": "bicubic",
        "LANCZOS": "bicubic",  # PyTorch doesn't have Lanczos, use bicubic as approximation
        "BOX": "area",         # area mode is similar to box filtering
        "BILINEAR": "bilinear",
    }
    mode = mode_map.get(method.upper(), "bicubic")
    size = (hr.height // scale, hr.width // scale)

    def _box_downsample(x: torch.Tensor) -> torch.Tensor:
        """Exact box filtering using average pooling for integer scales."""
        return F.avg_pool2d(x, kernel_size=scale, stride=scale)

    if gamma_aware:
        # Process in linear color space for perceptually accurate results
        t_lin = _srgb_to_linear(t)
        
        # Apply pre-blur in linear space to reduce aliasing
        if preblur_sigma > 0:
            t_lin = _gaussian_blur(t_lin, preblur_sigma)
        
        # Perform downscaling in linear space
        if method.upper() == "BOX":
            lr_lin = _box_downsample(t_lin)  # Use exact averaging
        elif mode == "area":
            lr_lin = F.interpolate(t_lin, size=size, mode="area")
        else:
            lr_lin = F.interpolate(t_lin, size=size, mode=mode, 
                                 align_corners=False, antialias=True)
        
        # Convert back to sRGB for display
        t_srgb = _linear_to_srgb(lr_lin)
        return _tensor_to_pil(t_srgb)
    else:
        # Standard sRGB processing (faster but potentially lower quality)
        x = t
        if preblur_sigma > 0:
            x = _gaussian_blur(x, preblur_sigma)
        
        # Perform downscaling
        if method.upper() == "BOX":
            lr = _box_downsample(x)
        elif mode == "area":
            lr = F.interpolate(x, size=size, mode="area")
        else:
            lr = F.interpolate(x, size=size, mode=mode, 
                             align_corners=False, antialias=True)
        return _tensor_to_pil(lr)

def main():
    """
    Main function to process command line arguments and batch process images.
    
    Recursively finds all images in the input directory, downscales them according
    to specified parameters, and saves the results while preserving directory structure.
    """
    ap = argparse.ArgumentParser("Regenerate LR images from HR", 
                                description="Generate low-resolution images from high-resolution inputs for super-resolution training datasets")
    
    # Required arguments
    ap.add_argument("--hr", type=str, required=True, 
                    help="Input HR root directory (will recurse through subdirectories)")
    ap.add_argument("--out", type=str, required=True, 
                    help="Output LR root directory")
    
    # Downscaling parameters
    ap.add_argument("--scale", type=int, default=4, choices=[2,4,8],
                    help="Downscaling factor (2x, 4x, or 8x)")
    ap.add_argument("--method", type=str, default="BICUBIC",
                    choices=["BICUBIC","LANCZOS","BOX","BILINEAR"],
                    help="Resampling method for downscaling")
    
    # Quality options
    ap.add_argument("--gamma_aware", action="store_true", 
                    help="Resize in linear light to reduce halos/speckles (higher quality)")
    ap.add_argument("--preblur", type=float, default=0.0, 
                    help="Pre-blur sigma before downscaling (reduces aliasing)")
    
    # Output options
    ap.add_argument("--ext", type=str, default=".png", 
                    help="Output file extension (e.g., .png, .jpg)")
    ap.add_argument("--keep_structure", action="store_true", 
                    help="Center crop HR images to be multiples of scale factor")
    
    # Utility options
    ap.add_argument("--verbose", action="store_true", 
                    help="Print progress for every image processed")
    
    args = ap.parse_args()

    # Setup input and output paths
    in_root = Path(args.hr)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    # Find all supported image files recursively
    exts = {e.lower() for e in IMG_EXTS}  # Convert to lowercase set for comparison
    paths = [p for p in in_root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    
    if not paths:
        print(f"No images found under {in_root}")
        return

    # Print device information when verbose
    if args.verbose:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            print(f"Using CUDA device: {name}")
        else:
            print("Using CPU for processing")

    # Process each image
    for i, p in enumerate(paths, 1):
        try:
            # Load and convert to RGB (handles various input formats)
            hr = Image.open(p).convert("RGB")
        except Exception as e:
            print(f"Error loading {p}: {e}")
            continue

        # Optionally crop to ensure dimensions are multiples of scale factor
        if args.keep_structure:
            hr = center_crop_multiple(hr, args.scale)

        # Perform the downscaling
        lr = downscale_once(hr, args.scale, args.method, args.gamma_aware, args.preblur)

        # Calculate output path preserving directory structure
        rel = p.relative_to(in_root)
        out_path = (out_root / rel).with_suffix(args.ext)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Save with format-appropriate settings
        if args.ext.lower() in [".jpg", ".jpeg"]:
            # JPEG settings optimized for quality while avoiding common artifacts
            lr.save(out_path, quality=95, subsampling=0, optimize=True, progressive=True)
        else:
            # PNG or other lossless formats
            lr.save(out_path, optimize=True, compress_level=6)

        # Progress reporting
        if args.verbose or (i % 100 == 0 or i == len(paths)):
            print(f"[{i}/{len(paths)}] {p.name} -> {out_path}")

    print(f"\nSuccessfully processed {len(paths)} images to: {out_root.resolve()}")

if __name__ == "__main__":
    main()
