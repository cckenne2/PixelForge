import torch
from PixelForge.utils.metrics import psnr, ssim

@torch.no_grad()
def run_validation(gen, dataloader, device, *, alpha: float = 0.0, max_batches: int = 2):
    """
    Runs validation on a generator model using a dataloader.

    Args:
        gen: The generator model to evaluate.
        dataloader: DataLoader providing batches of low-res and high-res images.
        device: Device to run inference on (e.g., 'cuda' or 'cpu').
        alpha (float, optional): Alpha blending parameter for the generator. Default is 0.0.
        max_batches (int, optional): Maximum number of batches to validate. Default is 2.

    Returns:
        dict: Dictionary containing mean PSNR and SSIM scores.
    """
    gen.eval()  # Set generator to evaluation mode
    psnrs, ssims = [], []  # Lists to store PSNR and SSIM scores
    it = 0  # Batch counter
    for batch in dataloader:
        if it >= max_batches:  # Stop if max_batches reached
            break
        lr = batch["lr"].to(device, non_blocking=True)  # Move low-res images to device
        hr = batch["hr"].to(device, non_blocking=True)  # Move high-res images to device
        # Generate super-resolved images and clamp values to [0, 1]
        sr = gen(lr, alpha=alpha, preserve_graphics=False).clamp(0, 1)
        psnrs.append(psnr(sr, hr))  # Compute and store PSNR
        ssims.append(ssim(sr, hr))  # Compute and store SSIM
        it += 1  # Increment batch counter
    if not psnrs:  # If no batches processed, return zeros
        return {"psnr": 0.0, "ssim": 0.0}
    psnr_all = torch.cat(psnrs, dim=0)  # Concatenate PSNR scores
    ssim_all = torch.cat(ssims, dim=0)  # Concatenate SSIM scores
    # Return mean PSNR and SSIM as floats
    return {"psnr": float(psnr_all.mean().item()), "ssim": float(ssim_all.mean().item())}
