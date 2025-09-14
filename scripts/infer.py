# scripts/infer.py
"""
PixelForge Inference Script

This script performs image super-resolution inference using a trained PixelForge model.
It processes all images in an input directory and outputs upscaled versions to a specified
output directory, maintaining the folder structure.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import argparse, yaml
import torch
from torchvision.transforms.functional import to_tensor, to_pil_image

from PixelForge.models.generator import UpscaleGeneratorV1
from PixelForge.data.paired import IMG_EXTS  # reuse same extensions

def list_images(root: Path):
    """
    Recursively find all image files in a directory tree.
    
    Args:
        root (Path): Root directory to search for images
        
    Returns:
        list[Path]: List of paths to all found image files with supported extensions
        
    Note:
        Uses the same image extensions as defined in the training data module
        to ensure consistency between training and inference.
    """
    # Convert all extensions to lowercase for case-insensitive matching
    exts = {e.lower() for e in IMG_EXTS}
    
    # Recursively search for files with matching extensions
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]

@torch.no_grad()  # Disable gradient computation for inference to save memory
def main():
    """
    Main inference function that handles command-line arguments, model loading,
    and batch processing of images for super-resolution.
    
    The function performs the following steps:
    1. Parse command-line arguments
    2. Load configuration from YAML file
    3. Initialize device and model
    4. Process all images in the input directory
    5. Save upscaled results to output directory
    """
    
    # === Command-line Argument Parsing ===
    ap = argparse.ArgumentParser(description="PixelForge inference: upscale a folder")
    ap.add_argument("--config", type=str, required=True, 
                   help="path to YAML config (same one used for training)")
    ap.add_argument("--ckpt", type=str, default="", 
                   help="optional explicit checkpoint path (.pt). If omitted, uses cfg['checkpoint'].")
    ap.add_argument("--input", type=str, required=True, 
                   help="input folder of LR images")
    ap.add_argument("--output", type=str, required=True, 
                   help="output folder for upscaled images")
    ap.add_argument("--alpha", type=float, default=0.0, 
                   help="detail knob in [0,1] - controls balance between smoothness and detail")
    ap.add_argument("--preserve_graphics", action="store_true", 
                   help="enable graphics preservation hook (experimental) - better for synthetic images")
    ap.add_argument("--ext", type=str, default=".png", 
                   help="output file extension, e.g., .png or .jpg")
    args = ap.parse_args()

    # === Configuration Loading ===
    # Load YAML config and select platform-specific settings
    # This ensures consistency with training configuration
    with open(args.config, "r") as f:
        cfg_all = yaml.safe_load(f)
    
    # Select configuration based on current platform (Windows vs Linux)
    cfg = cfg_all["windows"] if sys.platform.startswith("win") else cfg_all["linux"]

    # === Device Setup ===
    # Configure GPU if available, otherwise fall back to CPU
    if torch.cuda.is_available():
        torch.cuda.set_device(0)  # Use first GPU
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
        device = torch.device("cuda:0")
    else:
        print("WARNING: CUDA not available; running on CPU.")
        device = torch.device("cpu")

    # === Checkpoint Path Resolution ===
    # Use explicit checkpoint if provided, otherwise use config settings
    if args.ckpt:
        ckpt_path = Path(args.ckpt)
    else:
        # Extract checkpoint info from configuration
        ck_cfg = cfg["checkpoint"]
        ckpt_path = Path(ck_cfg["dir"]) / ck_cfg["name"]

    # Verify checkpoint exists
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # === Model Loading and Setup ===
    # Load the saved model package which contains:
    # - "model": the state dictionary
    # - "cfg": the configuration used during training
    # - "stats": training statistics (optional)
    pkg = torch.load(ckpt_path, map_location="cpu")
    
    # Prefer the saved configuration over runtime config to ensure compatibility
    saved_cfg = pkg.get("cfg", cfg)  # fall back to runtime cfg if absent
    mcfg = saved_cfg["model"]  # Model-specific configuration
    scale = saved_cfg["scale"]  # Upscaling factor (e.g., 2x, 4x)

    # Initialize generator with saved configuration
    gen = UpscaleGeneratorV1(
        scale=scale, 
        n_feats=mcfg["n_feats"],  # Number of feature channels
        n_res=mcfg["n_res"]       # Number of residual blocks
    ).to(device)
    
    # Load the trained weights
    gen.load_state_dict(pkg["model"], strict=True)
    gen.eval()  # Set to evaluation mode (disables dropout, batch norm updates)

    # === Input/Output Directory Setup ===
    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)  # Create output directory if needed
    
    # Find all images in input directory
    images = list_images(in_dir)
    if not images:
        raise FileNotFoundError(f"No images found under: {in_dir}")

    # === Processing Information Display ===
    print(f"Device: {device} | Scale: x{scale} | Alpha: {args.alpha} | Graphics: {args.preserve_graphics}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Input images: {len(images)}")

    # === Inference Setup ===
    # Use automatic mixed precision (AMP) on GPU for faster inference
    use_amp = (device.type == "cuda")
    from torch.amp import autocast

    # === Main Processing Loop ===
    for i, path in enumerate(images, 1):
        # Load and preprocess image
        # Convert PIL Image -> Tensor -> add batch dimension -> move to device
        img = to_tensor(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        
        # Perform inference with optional mixed precision
        with autocast(device_type=device.type, enabled=use_amp):
            # Generate super-resolution image
            # alpha: controls detail enhancement (0=smooth, 1=detailed)
            # preserve_graphics: experimental feature for synthetic images
            sr = gen(img, alpha=float(args.alpha), preserve_graphics=bool(args.preserve_graphics)).clamp(0, 1)
        
        # Convert back to PIL Image for saving
        sr_img = to_pil_image(sr.squeeze(0).cpu())

        # === Output Path Construction ===
        # Maintain directory structure from input to output
        rel = path.relative_to(in_dir)  # Get relative path from input root
        out_path = (out_dir / rel).with_suffix(args.ext)  # Change extension if needed
        
        # Create subdirectories as needed
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the upscaled image
        sr_img.save(out_path)
        print(f"[{i}/{len(images)}] wrote {out_path}")

    # === Completion Summary ===
    print(f"\nDone. Saved {len(images)} images to: {out_dir.resolve()}")

if __name__ == "__main__":
    # Lazy import PIL to speed up CLI startup when just checking help
    from PIL import Image
    main()
