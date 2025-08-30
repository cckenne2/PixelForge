# scripts/train.py
import argparse, yaml
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.PixelForge.models.generator import UpscaleGeneratorV1
from src.PixelForge.data.paired import PairedOnTheFlyDataset
from src.PixelForge.train.train_l1 import train_l1

def main(cfg):
    # Set up GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Initialize dataset
    # PairedOnTheFlyDataset generates high-res and low-res image pairs on the fly
    ds = PairedOnTheFlyDataset(
        Path(cfg["data"]["root_hr"]),     # Path to high-resolution images
        Path(cfg["data"]["fallback"]),    # Fallback path for missing images
        scale=cfg["scale"],               # Upscaling factor
        patch_hr=cfg["img_size_hr"],      # Size of high-res patches
    )

    # Create data loader for batch processing
    dl = DataLoader(
        ds, 
        batch_size=cfg["batch_size"],     # Number of samples per batch
        shuffle=True,                     # Randomly shuffle data
        num_workers=cfg["num_workers"],   # Number of parallel workers
        pin_memory=cfg["pin_memory"],     # Pin memory for faster GPU transfer
        drop_last=True                    # Drop incomplete final batch
    )

    # Initialize the generator model
    mcfg = cfg["model"]
    gen = UpscaleGeneratorV1(
        scale=cfg["scale"],               # Upscaling factor
        n_feats=mcfg["n_feats"],         # Number of features
        n_res=mcfg["n_res"]              # Number of residual blocks
    ).to(device)
    
    # Print model size in millions of parameters
    print("Params (M):", round(sum(p.numel() for p in gen.parameters() if p.requires_grad)/1e6, 3))

    # Train the model using L1 loss
    stats = train_l1(gen, dl, device,
                     steps=cfg["steps"],   # Number of training steps
                     lr=cfg["lr"],         # Learning rate
                     alpha=cfg["alpha"])   # Loss function parameter

    # Save the trained model
    ck = cfg["checkpoint"]
    out_dir = Path(ck["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)    # Create output directory if needed
    out_path = out_dir / ck["name"]
    
    # Save model state, configuration, and training statistics
    torch.save({"model": gen.state_dict(), "cfg": cfg, "stats": stats}, out_path)
    print("Saved:", out_path.resolve())

if __name__ == "__main__":
    # Parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="path to YAML config")
    args = ap.parse_args()
    
    # Load configuration from YAML file
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    main(cfg)