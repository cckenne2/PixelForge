import torch
import torch.nn.functional as F

def psnr(sr: torch.Tensor, hr: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """
    Computes the Peak Signal-to-Noise Ratio (PSNR) between super-resolved (sr) and high-resolution (hr) images.

    Args:
        sr (torch.Tensor): Super-resolved images of shape (N, 3, H, W), with pixel values in [0, 1].
        hr (torch.Tensor): Ground-truth high-resolution images of shape (N, 3, H, W), with pixel values in [0, 1].
        max_val (float, optional): Maximum possible pixel value. Default is 1.0.

    Returns:
        torch.Tensor: PSNR values for each image in the batch, shape (N,).

    Notes:
        - PSNR is a common metric for measuring the quality of reconstructed images.
        - Higher PSNR indicates better reconstruction quality.
        - The function computes the mean squared error (MSE) per image and applies the PSNR formula:
            PSNR = 10 * log10(max_val^2 / MSE)
        - A small epsilon is added to the denominator to avoid division by zero.
    """
    # Compute mean squared error (MSE) for each image in the batch
    mse = torch.mean((sr - hr) ** 2, dim=(1, 2, 3))
    # Small epsilon to avoid division by zero in PSNR calculation
    eps = 1e-10
    # Compute PSNR for each image: 10 * log10(max_val^2 / MSE)
    return 10.0 * torch.log10((max_val ** 2) / torch.clamp(mse, min=eps))

def ssim(sr: torch.Tensor, hr: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """
    Computes the Structural Similarity Index (SSIM) between super-resolved (sr) and high-resolution (hr) images,
    using a fast approximation on the luminance (Y) channel.

    Args:
        sr (torch.Tensor): Super-resolved images of shape (N, 3, H, W), with pixel values in [0, 1].
        hr (torch.Tensor): Ground-truth high-resolution images of shape (N, 3, H, W), with pixel values in [0, 1].
        window_size (int, optional): Size of the box filter window for local statistics. Default is 11.

    Returns:
        torch.Tensor: SSIM values for each image in the batch, shape (N,).

    Notes:
        - SSIM is a perceptual metric that quantifies image similarity based on luminance, contrast, and structure.
        - This implementation converts RGB images to luminance using standard weights, then computes local means,
          variances, and covariances using a box filter.
        - The SSIM formula is applied per-pixel, and the result is averaged over spatial dimensions to yield a
          single SSIM score per image.
        - This version is lightweight and suitable for quick sanity checks, but may not be as accurate as full SSIM.
    """
    # Split the super-resolved and high-resolution images into R, G, B channels
    r_s, g_s, b_s = sr[:, 0:1], sr[:, 1:2], sr[:, 2:3]
    r_h, g_h, b_h = hr[:, 0:1], hr[:, 1:2], hr[:, 2:3]
    # Convert RGB images to luminance (Y) channel using standard weights
    y_s = 0.299 * r_s + 0.587 * g_s + 0.114 * b_s
    y_h = 0.299 * r_h + 0.587 * g_h + 0.114 * b_h

    # Calculate padding for the box filter
    pad = window_size // 2
    # Create a normalized box filter kernel
    weight = torch.ones(1, 1, window_size, window_size, device=sr.device) / (window_size ** 2)

    # Define a helper function for box filtering
    def filt(x):  # box filter
        return F.conv2d(x, weight, padding=pad, groups=1)

    # Compute local means for super-resolved and high-resolution images
    mu_s = filt(y_s)
    mu_h = filt(y_h)
    # Compute local variances for super-resolved and high-resolution images
    sigma_s2 = filt(y_s * y_s) - mu_s * mu_s
    sigma_h2 = filt(y_h * y_h) - mu_h * mu_h
    # Compute local covariance between super-resolved and high-resolution images
    sigma_sh = filt(y_s * y_h) - mu_s * mu_h

    # Constants for numerical stability (from SSIM paper)
    c1 = (0.01 ** 2)
    c2 = (0.03 ** 2)

    # Compute SSIM numerator and denominator
    num = (2 * mu_s * mu_h + c1) * (2 * sigma_sh + c2)
    den = (mu_s * mu_s + mu_h * mu_h + c1) * (sigma_s2 + sigma_h2 + c2)
    # Compute SSIM map for each pixel
    ssim_map = num / (den + 1e-10)
    # Average SSIM map over spatial dimensions to get a single score per image
    return ssim_map.mean(dim=(1, 2, 3))
