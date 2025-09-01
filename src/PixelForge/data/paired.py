# src/PixelForge/data/paired.py
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

# Define supported image file extensions
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

class PairedOnTheFlyDataset(Dataset):
    """
    Loads HR images from root_hr (falls back to fallback) and creates LR by bicubic downscale.
    Returns dict with 'lr', 'hr', 'path'.
    """
    def __init__(self, root_hr: Path, fallback: Path | None, scale: int = 4, patch_hr: int | None = None):
        """
        Initialize the dataset.
        
        Args:
            root_hr: Primary directory path to high-resolution images
            fallback: Fallback directory path if primary directory is empty
            scale: Downscaling factor for creating low-resolution images
            patch_hr: Size of high-resolution patches to extract (if None, use whole image)
        """
        self.scale, self.patch_hr = scale, patch_hr
        self.paths = []
        
        # Try to load images from primary directory
        root_hr = Path(root_hr)
        if root_hr.exists():
            self.paths = [p for p in root_hr.rglob("*") if p.suffix.lower() in IMG_EXTS]
        
        # If primary directory is empty, try fallback directory
        if not self.paths and fallback and Path(fallback).exists():
            self.paths = [p for p in Path(fallback).rglob("*") if p.suffix.lower() in IMG_EXTS]
        
        # Raise error if no images found in either directory
        if not self.paths:
            raise FileNotFoundError("No images found in data/HR/ or data/samples/.")

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
        """Return the number of images in the dataset"""
        return len(self.paths)

    def __getitem__(self, idx):
        """
        Get a paired sample of LR and HR images.
        
        Args:
            idx: Index of the sample to fetch
            
        Returns:
            Dictionary containing:
            - lr: Low-resolution tensor
            - hr: High-resolution tensor
            - path: Path to the original image
        """
        path = self.paths[idx]
        # Load and convert image to RGB
        hr = Image.open(path).convert("RGB")
        
        # Crop to ensure dimensions are multiples of scale factor
        hr = self._center_crop_multiple(hr, self.scale)
        
        # Extract a center patch if patch_hr is specified and image is large enough
        if self.patch_hr is not None:
            w, h = hr.size
            if w >= self.patch_hr and h >= self.patch_hr:
                left = (w - self.patch_hr) // 2; top = (h - self.patch_hr) // 2
                hr = hr.crop((left, top, left + self.patch_hr, top + self.patch_hr))
        
        # Create low-resolution version by downscaling
        w, h = hr.size
        lr = hr.resize((w // self.scale, h // self.scale), Image.BICUBIC)
        
        # Convert images to tensors
        hr_t = TF.to_tensor(hr); lr_t = TF.to_tensor(lr)
        
        return {"lr": lr_t, "hr": hr_t, "path": str(path)}