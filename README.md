# PixelForge

A PyTorch-based image super-resolution framework featuring alpha-conditioned generators with Feature-wise Linear Modulation (FiLM). PixelForge supports 2x, 4x, and 8x upscaling using deep learning to generate high-quality detail rather than simple pixel interpolation.

## Features

- **Multi-scale upsampling**: 2x, 4x, and 8x super-resolution
- **Alpha conditioning**: Control reconstruction style with alpha parameter (0.0-1.0)
- **FiLM modulation**: Feature-wise Linear Modulation for conditional processing
- **GPU optimized**: CUDA support with automatic mixed precision training
- **Flexible architecture**: Configurable generator with residual blocks and pixel shuffle upsampling

## Architecture

- **UpscaleGeneratorV1**: Alpha-conditioned generator with FiLM modulation
- **Conditional residual blocks**: Feature modulation via gamma/beta parameters
- **Pixel shuffle upsampling**: Efficient sub-pixel convolution for upscaling
- **Structure preservation**: Experimental graphics preservation hooks

## Project Structure

```
PixelForge/
├── src/PixelForge/           # Main package
│   ├── models/               # Generator architecture
│   ├── data/                 # Dataset utilities
│   ├── train/                # Training and validation
│   ├── utils/                # Metrics (PSNR, SSIM)
│   └── config/               # YAML configurations
├── scripts/                  # Training scripts
├── notebooks/                # Development notebooks
├── data/                     # Training data
└── experiments/              # Checkpoints and logs
```

## Quick Start

### 1. Environment Setup

**Option A: Conda (recommended)**
```bash
conda env create -f environment.yml
conda activate PixelForge
```

**Option B: pip**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Data Preparation

Place high-resolution training images in one of these directories:
- `data/HR/train/` (preferred structure)
- `data/HR/` (fallback)
- `data/samples/` (for testing)

### 3. Training

```bash
# Run training with baseline configuration
python scripts/train.py --config src/PixelForge/config/baseline.yaml
```

The training script will:
- Auto-detect GPU and optimize settings
- Display real-time GPU memory usage
- Run validation every 50 steps (if enabled)
- Save checkpoints to `experiments/checkpoints/`

## Configuration

Training is controlled via YAML config files in `src/PixelForge/config/`. The baseline config includes:

- **Model**: 64 features, 4 residual blocks
- **Training**: 200 steps, batch size 4, learning rate 2e-4
- **Data**: Auto-detects `data/HR/train` or `data/HR`
- **Platform**: Separate configs for Windows/Linux

## Notebooks

Development notebooks in `notebooks/`:
- `01_data_prep.ipynb`: Data exploration and preparation
- `02_model_prototype.ipynb`: Model development and testing

## Requirements

- Python 3.8+
- PyTorch 2.0+ with CUDA support (recommended)
- 4GB+ GPU memory for training
- High-resolution images for training data

## Performance

The UpscaleGeneratorV1 model:
- **Parameters**: ~600K (0.6M) for default config
- **Training**: ~100 steps in <30s on modern GPU
- **Memory**: Supports batch size 4-8 on 6GB GPU

## License

This project is for educational and research purposes.
