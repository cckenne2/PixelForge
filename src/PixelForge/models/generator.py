# src/PixelForge/models/generator.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["UpscaleGeneratorV1", "structure_mask_v1"]

class AlphaMLP(nn.Module):
    """
    Multi-layer perceptron that generates conditioning parameters (gamma, beta) 
    from a scalar alpha value for feature modulation.
    """
    def __init__(self, n_feats: int, hidden: int = 32, gain: float = 0.1):
        """
        Args:
            n_feats: Number of feature channels to generate parameters for
            hidden: Hidden layer size
            gain: Scaling factor for output parameters
        """
        super().__init__()
        # Two-layer MLP: 1 -> hidden -> 2*n_feats
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 2 * n_feats)  # Output gamma and beta parameters
        )
        self.gain = gain

    def forward(self, alpha: torch.Tensor):
        """
        Convert scalar alpha to gamma/beta conditioning parameters.
        
        Args:
            alpha: Scalar or tensor controlling the conditioning strength
            
        Returns:
            gamma, beta: Multiplicative and additive conditioning parameters
        """
        if alpha.dim() == 0:
            alpha = alpha.view(1)
        out = self.net(alpha.unsqueeze(-1))
        out = torch.tanh(out) * self.gain  # Bound output and scale
        gamma, beta = out.chunk(2, dim=-1)  # Split into gamma and beta
        return gamma, beta

class ResidualBlockCond(nn.Module):
    """
    Residual block with conditional feature modulation using gamma/beta parameters.
    Applies: x + modulated_conv_features where modulation = gamma * features + beta
    """
    def __init__(self, n_feats: int):
        """
        Args:
            n_feats: Number of input/output feature channels
        """
        super().__init__()
        self.conv1 = nn.Conv2d(n_feats, n_feats, 3, padding=1)
        self.act   = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_feats, n_feats, 3, padding=1)

    def forward(self, x, gamma: torch.Tensor, beta: torch.Tensor):
        """
        Args:
            x: Input features [N, C, H, W]
            gamma: Multiplicative modulation parameters [N, C]
            beta: Additive modulation parameters [N, C]
            
        Returns:
            Residual connection: x + modulated_features
        """
        # Apply two conv layers with activation
        y = self.conv1(x); y = self.act(y); y = self.conv2(y)
        N, C, _, _ = y.shape
        # Apply conditional modulation: y = y * (1 + gamma) + beta
        y = y * (1.0 + gamma.view(N, C, 1, 1)) + beta.view(N, C, 1, 1)
        return x + y  # Residual connection

class UpsampleBlock(nn.Module):
    """
    Upsampling block using sub-pixel convolution (pixel shuffle) for 2x upscaling.
    """
    def __init__(self, n_feats: int):
        """
        Args:
            n_feats: Number of input feature channels
        """
        super().__init__()
        # Conv to 4x channels for 2x2 pixel shuffle
        self.conv = nn.Conv2d(n_feats, 4 * n_feats, 3, padding=1)
        self.ps = nn.PixelShuffle(2)  # Rearrange 4x channels to 2x spatial
        self.act = nn.ReLU(inplace=True)
        
    def forward(self, x): 
        """2x spatial upsampling via pixel shuffle"""
        return self.act(self.ps(self.conv(x)))

class UpscaleGeneratorV1(nn.Module):
    """
    Generator network for image upscaling with conditional feature modulation.
    Architecture: head -> residual_blocks -> upsampler -> tail
    """
    def __init__(self, scale: int = 4, n_feats: int = 64, n_res: int = 4):
        """
        Args:
            scale: Upscaling factor (2, 4, or 8)
            n_feats: Number of feature channels
            n_res: Number of residual blocks
        """
        super().__init__()
        assert scale in (2,4,8), "Scale must be 2, 4, or 8"
        self.scale = scale
        
        # Network components
        self.head = nn.Conv2d(3, n_feats, 3, padding=1)  # RGB -> features
        self.body = nn.ModuleList([ResidualBlockCond(n_feats) for _ in range(n_res)])
        # Cascade of upsample blocks: log2(scale) blocks for total scale factor
        self.upsampler = nn.Sequential(*[UpsampleBlock(n_feats) for _ in range(int(math.log2(scale)))])
        self.tail = nn.Conv2d(n_feats, 3, 3, padding=1)  # features -> RGB
        
        # MLP for generating conditional parameters from alpha
        self.alpha_mlp = AlphaMLP(n_feats, hidden=32, gain=0.1)

    @torch.no_grad()
    def count_params(self): 
        """Count trainable parameters in the model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x, alpha: float = 0.0, preserve_graphics: bool = False):
        """
        Forward pass for image upscaling.
        
        Args:
            x: Input image [N, 3, H, W]
            alpha: Conditioning parameter [0, 1] controlling feature modulation
            preserve_graphics: If True, computes structure mask (experimental)
            
        Returns:
            Upscaled image [N, 3, H*scale, W*scale]
        """
        # Ensure alpha is a tensor and clamp to valid range
        if not torch.is_tensor(alpha):
            alpha = torch.tensor(alpha, dtype=x.dtype, device=x.device)
        alpha = torch.clamp(alpha, 0.0, 1.0)
        
        # Extract initial features
        feat = self.head(x)
        N = x.shape[0]
        
        # Generate conditioning parameters from alpha
        gamma, beta = self.alpha_mlp(alpha.expand(N))
        
        # Apply residual blocks with conditioning
        for block in self.body: 
            feat = block(feat, gamma, beta)
            
        # Upsample features
        feat = self.upsampler(feat)
        
        # Convert back to RGB
        out = self.tail(feat)
        
        # Experimental graphics preservation mode
        if preserve_graphics:
            _ = structure_mask_v1(out)  # Computed but not applied yet
            
        return out

def structure_mask_v1(x_rgb: torch.Tensor, thresh: float = 0.2) -> torch.Tensor:
    """
    Compute edge/structure mask from RGB image using Sobel edge detection.
    Used for potential graphics preservation in upscaling.
    
    Args:
        x_rgb: RGB image tensor [N, 3, H, W]
        thresh: Threshold for edge detection [0, 1]
        
    Returns:
        Edge mask [N, 1, H, W] with values in [0, 1]
    """
    # Normalize each image to [0, 1] range
    x = (x_rgb - x_rgb.amin(dim=(2,3), keepdim=True)) / (x_rgb.amax(dim=(2,3), keepdim=True) - x_rgb.amin(dim=(2,3), keepdim=True) + 1e-8)
    
    # Convert RGB to grayscale using standard luminance weights
    r, g, b = x[:,0:1], x[:,1:2], x[:,2:3]
    y = 0.299*r + 0.587*g + 0.114*b
    
    # Sobel edge detection kernels
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=x.dtype, device=x.device).view(1,1,3,3)  # Horizontal edges
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=x.dtype, device=x.device).view(1,1,3,3)  # Vertical edges
    
    # Compute gradients in x and y directions
    gx = F.conv2d(y, kx, padding=1)
    gy = F.conv2d(y, ky, padding=1)
    
    # Compute gradient magnitude
    mag = torch.sqrt(gx*gx + gy*gy)
    
    # Normalize magnitude to [0, 1]
    mag = mag / (mag.amax(dim=(2,3), keepdim=True) + 1e-8)
    
    # Apply threshold and create soft mask
    mask = torch.clamp((mag - thresh) / max(1e-6, 1.0 - thresh), 0.0, 1.0)
    
    return mask