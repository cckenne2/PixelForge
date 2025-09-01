# src/PixelForge/train/train_l1.py
from time import perf_counter
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

def train_l1(gen, dataloader, device, *, steps=100, lr=2e-4, alpha=0.0, log_every=10):
    """
    Train a generator model using L1 loss.
    
    Args:
        gen: Generator model to train
        dataloader: DataLoader providing training batches
        device: Device to run training on (CPU or GPU)
        steps: Maximum number of training steps
        lr: Learning rate for Adam optimizer
        alpha: Alpha parameter for generator (if applicable)
        log_every: Frequency of logging progress
    
    Returns:
        Dictionary containing loss history and training time
    """
    # Set model to training mode
    gen.train()
    
    # Initialize optimizer
    opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.9, 0.99))
    
    # Setup gradient scaler for mixed precision training (if on CUDA)
    scaler = GradScaler(device_type=device.type if device.type == "cuda" else None)
    
    # Track loss history and time
    loss_hist = []
    t0 = perf_counter()
    
    step = 0
    for batch in dataloader:
        if step >= steps: break
        
        # Move batch data to device
        lr_img = batch["lr"].to(device, non_blocking=True)  # Low resolution image
        hr_img = batch["hr"].to(device, non_blocking=True)  # High resolution image (target)
        
        # Clear gradients
        opt.zero_grad(set_to_none=True)
        
        # Use mixed precision for forward pass if on CUDA
        with autocast(device_type=device.type if device.type == "cuda" else "cpu"):
            # Generate super-resolution image
            sr = gen(lr_img, alpha=alpha, preserve_graphics=False)
            # Calculate L1 loss between generated and target images
            loss = F.l1_loss(sr, hr_img)
        
        # Scale gradients and perform backward pass
        scaler.scale(loss).backward()
        
        # Update weights and scaler
        scaler.step(opt); scaler.update()
        
        # Record loss
        loss_hist.append(loss.item())
        
        # Log progress
        if step % log_every == 0:
            print(f"step {step:04d} L1: {loss.item():.4f}")
            
        step += 1
    
    # Calculate and print training summary
    t1 = perf_counter()
    print(f"done {step} steps in {t1 - t0:.2f}s ; last L1={loss_hist[-1]:.4f}")
    
    return {"loss_hist": loss_hist, "seconds": t1 - t0}