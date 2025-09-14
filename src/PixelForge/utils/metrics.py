import torch
import torch.nn.functional as F

def psnr(sr: torch.Tensor, hr: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """
    sr, hr: (N,3,H,W) in [0, 1]
    returns: PSNR per-image (N,)
    """
    mse = torch.mean((sr - hr) ** 2, dim=(1, 2, 3))
    eps = 1e-10
    return 10.0 * torch.log10((max_val ** 2) / torch.clamp(mse, min=eps))

def ssim(sr: torch.Tensor, hr: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """
    Lightweight SSIM on luminance channel. sr/hr in [0,1].
    Returns per-image SSIM (N,). Simple & fast for sanity checks.
    """
    # Convert to luma
    r_s, g_s, b_s = sr[:, 0:1], sr[:, 1:2], sr[:, 2:3]
    r_h, g_h, b_h = hr[:, 0:1], hr[:, 1:2], hr[:, 2:3]
    y_s = 0.299 * r_s + 0.587 * g_s + 0.114 * b_s
    y_h = 0.299 * r_h + 0.587 * g_h + 0.114 * b_h

    pad = window_size // 2
    weight = torch.ones(1, 1, window_size, window_size, device=sr.device) / (window_size ** 2)

    def filt(x):  # box filter
        return F.conv2d(x, weight, padding=pad, groups=1)

    mu_s = filt(y_s)
    mu_h = filt(y_h)
    sigma_s2 = filt(y_s * y_s) - mu_s * mu_s
    sigma_h2 = filt(y_h * y_h) - mu_h * mu_h
    sigma_sh = filt(y_s * y_h) - mu_s * mu_h

    c1 = (0.01 ** 2)
    c2 = (0.03 ** 2)

    num = (2 * mu_s * mu_h + c1) * (2 * sigma_sh + c2)
    den = (mu_s * mu_s + mu_h * mu_h + c1) * (sigma_s2 + sigma_h2 + c2)
    ssim_map = num / (den + 1e-10)
    return ssim_map.mean(dim=(1, 2, 3))
