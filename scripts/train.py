# scripts/train.py
import argparse, yaml
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.PixelForge.models.generator import UpscaleGeneratorV1
from src.PixelForge.data.paired import PairedOnTheFlyDataset
from src.PixelForge.train.train_l1 import train_l1

def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Data
    ds = PairedOnTheFlyDataset(
        Path(cfg["data"]["root_hr"]),
        Path(cfg["data"]["fallback"]),
        scale=cfg["scale"],
        patch_hr=cfg["img_size_hr"],
    )
    dl = DataLoader(
        ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=cfg["pin_memory"],
        drop_last=True
    )

    # Model
    mcfg = cfg["model"]
    gen = UpscaleGeneratorV1(scale=cfg["scale"], n_feats=mcfg["n_feats"], n_res=mcfg["n_res"]).to(device)
    print("Params (M):", round(sum(p.numel() for p in gen.parameters() if p.requires_grad)/1e6, 3))

    # Train (L1 only)
    stats = train_l1(gen, dl, device,
                     steps=cfg["steps"], lr=cfg["lr"], alpha=cfg["alpha"])

    # Save
    ck = cfg["checkpoint"]
    out_dir = Path(ck["dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / ck["name"]
    torch.save({"model": gen.state_dict(), "cfg": cfg, "stats": stats}, out_path)
    print("Saved:", out_path.resolve())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="path to YAML config")
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    main(cfg)