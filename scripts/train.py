# scripts/train.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import argparse, yaml
import torch
from torch.utils.data import DataLoader
import psutil
import threading
import time

from PixelForge.models.generator import UpscaleGeneratorV1
from PixelForge.data.paired import PairedOnTheFlyDataset
from PixelForge.train.train_l1 import train_l1

def gpu_monitor(stop_event):
    """Monitor and display GPU usage in a separate thread"""
    while not stop_event.is_set():
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            gpu_allocated = torch.cuda.memory_allocated(0) / 1e9
            gpu_cached = torch.cuda.memory_reserved(0) / 1e9
            gpu_util = (gpu_allocated / gpu_memory) * 100
            
            print(f"\rGPU Memory: {gpu_allocated:.2f}GB/{gpu_memory:.2f}GB ({gpu_util:.1f}%) | "
                  f"Cached: {gpu_cached:.2f}GB", end="", flush=True)
        
        time.sleep(2)  # Update every 2 seconds

def main(cfg):
    # Set data directory path
    DATA_ROOT = Path(cfg["data"]["root_hr"])

    try:
        assert DATA_ROOT.exists()
    except AssertionError:
        raise FileNotFoundError(f"Data root directory {DATA_ROOT} does not exist.")
    
    # Prioritize CUDA GPU setup
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available! Training will be significantly slower on CPU.")
        device = torch.device("cpu")
    else:
        # Set CUDA device and optimize settings
        torch.cuda.set_device(0)  # Use first GPU
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic algorithms for speed
        device = torch.device("cuda:0")
        
        # Display GPU information
        gpu_props = torch.cuda.get_device_properties(0)
        print(f"GPU: {gpu_props.name}")
        print(f"GPU Memory: {gpu_props.total_memory / 1e9:.2f}GB")
        print(f"CUDA Version: {torch.version.cuda}")
    
    print("Device:", device)

    # Start GPU monitoring thread
    stop_monitor = threading.Event()
    if torch.cuda.is_available():
        monitor_thread = threading.Thread(target=gpu_monitor, args=(stop_monitor,))
        monitor_thread.daemon = True
        monitor_thread.start()

    try:
        # Initialize dataset
        ds = PairedOnTheFlyDataset(
            DATA_ROOT,
            Path(cfg["data"]["fallback"]),
            scale=cfg["scale"],
            patch_hr=cfg["img_size_hr"],
        )

        # Optimize DataLoader for GPU
        dl = DataLoader(
            ds, 
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=cfg["num_workers"] if device.type == "cuda" else 0,  # Disable workers on CPU
            pin_memory=device.type == "cuda",  # Only pin memory for GPU
            drop_last=True,
            persistent_workers=cfg["num_workers"] > 0 and device.type == "cuda"
        )

        # Initialize model with proper device placement
        mcfg = cfg["model"]
        gen = UpscaleGeneratorV1(
            scale=cfg["scale"],
            n_feats=mcfg["n_feats"],
            n_res=mcfg["n_res"]
        ).to(device)
        
        print("Params (M):", round(sum(p.numel() for p in gen.parameters() if p.requires_grad)/1e6, 3))

        # Clear GPU cache before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("\nStarting training...")

        # Train the model
        stats = train_l1(gen, dl, device,
                         steps=cfg["steps"],
                         lr=cfg["lr"],
                         alpha=cfg["alpha"])

        # Save model
        ck = cfg["checkpoint"]
        out_dir = Path(ck["dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / ck["name"]
        
        # Convert model back to FP32 for saving if using FP16
        if device.type == "cuda":
            gen = gen.float()
        
        torch.save({"model": gen.state_dict(), "cfg": cfg, "stats": stats}, out_path)
        print(f"\nSaved: {out_path.resolve()}")

    finally:
        # Stop monitoring thread
        stop_monitor.set()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("\nGPU cache cleared")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="path to YAML config")
    args = ap.parse_args()
    
    with open(args.config, "r") as f:
        cfg_all = yaml.safe_load(f)
    # Select config section based on platform
    if sys.platform.startswith("win"):
        cfg = cfg_all["windows"]
    else:
        cfg = cfg_all["linux"]
    main(cfg)