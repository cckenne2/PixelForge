# scripts/download_div2k.py
import os, zipfile, sys
from pathlib import Path
import urllib.request

DIV2K_BASE = "https://data.vision.ee.ethz.ch/cvl/DIV2K"
URLS = {
    "valid_hr": f"{DIV2K_BASE}/DIV2K_valid_HR.zip",
    "train_hr": f"{DIV2K_BASE}/DIV2K_train_HR.zip",  # large; grab later
}

DATA_ROOT = Path("data/DIV2K")
HR_VALID = DATA_ROOT / "HR" / "valid"
HR_TRAIN = DATA_ROOT / "HR" / "train"
TMP_DIR = DATA_ROOT / "_tmp"

def download(url: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        print(f"[skip] {out_path.name} already exists")
        return out_path
    print(f"[downloading] {url}")
    urllib.request.urlretrieve(url, out_path)
    print(f"[saved] {out_path}")
    return out_path

def unzip(zip_path: Path, target_dir: Path):
    target_dir.mkdir(parents=True, exist_ok=True)
    # Extract into a temp folder next to the zip
    extract_root = zip_path.parent / (zip_path.stem)
    with zipfile.ZipFile(zip_path, 'r') as z:
        # Only extract if not already extracted
        if not extract_root.exists():
            z.extractall(zip_path.parent)

    # The zips unpack to DIV2K_*_HR with *.png inside
    src_dir = extract_root
    if src_dir.exists():
        for p in src_dir.glob("*.png"):
            dest = target_dir / p.name
            if dest.exists():
                # Skip if we already moved this file before
                continue
            p.replace(dest)  # replace() works across volumes; acts like move
        # Clean up extracted folder if empty
        try:
            for leftover in src_dir.iterdir():
                # should be empty; if not, leave it
                pass
            src_dir.rmdir()
        except OSError:
            # non-empty; leave it
            pass

def main():
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Start small: validation HR (100 images)
    val_zip = download(URLS["valid_hr"], TMP_DIR / "DIV2K_valid_HR.zip")
    if not any(HR_VALID.glob("*.png")):
        unzip(val_zip, HR_VALID)
        print(f"[done] Valid HR images in {HR_VALID}")
    else:
        print(f"[skip] Valid HR images in {HR_VALID} already exist")

    # 2) Optional: uncomment to fetch the big train HR set now
    train_zip = download(URLS['train_hr'], TMP_DIR / 'DIV2K_train_HR.zip')
    if not any(HR_TRAIN.glob("*.png")):
        unzip(train_zip, HR_TRAIN)
        print(f"[done] Train HR images in {HR_TRAIN}")
    else:
        print(f"[skip] Train HR images in {HR_TRAIN} already exist")

    print("[note] You can delete the data/DIV2K/_tmp folder after verifying files.")

if __name__ == "__main__":
    main()
