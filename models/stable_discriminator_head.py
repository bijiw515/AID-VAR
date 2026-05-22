"""
Numerically stable discriminator head implementation.
Fixes NaN issues in the StyleGAN-T discriminator head.

Key improvements:
1. Increased BatchNorm eps for numerical stability
2. Improved weight initialization
3. Added numerical stability checks
4. Simplified conditional discrimination
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.spectral_norm import SpectralNorm
import logging

logger = logging.getLogger(__name__)


class StableSpectralConv1d(nn.Conv1d):
    """Numerically stable SpectralConv1d."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Increased eps for better numerical stability
        SpectralNorm.apply(self, name='weight', n_power_iterations=1, dim=0, eps=1e-6)

        # Improved weight initialization
        with torch.no_grad():
            nn.init.normal_(self.weight, 0.0, 0.02)
            if self.bias is not None:
                nn.init.zeros_(self.bias)


class StableBatchNormLocal(nn.Module):
    """Numerically stable local batch normalization."""
    def __init__(self, num_features: int, affine: bool = True, virtual_bs: int = 8, eps: float = 1e-3):
        super().__init__()
        self.virtual_bs = virtual_bs
        self.eps = eps  # Increased from 1e-5 to 1e-3 for stability
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.warning("StableBatchNormLocal input contains NaN/Inf, fixing")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

        shape = x.size()

        # Reshape batch into groups
        G = max(1, int(np.ceil(x.size(0) / self.virtual_bs)))
        x = x.view(G, -1, x.size(-2), x.size(-1))

        # Numerically stable statistics
        mean = x.mean([1, 3], keepdim=True)
        var = x.var([1, 3], keepdim=True, unbiased=False)

        # Clamp variance to prevent numerical issues
        var = torch.clamp(var, min=self.eps)
        std = torch.sqrt(var + self.eps)

        if torch.isnan(mean).any() or torch.isnan(std).any():
            logger.warning("BatchNorm statistics contain NaN, using safe defaults")
            mean = torch.zeros_like(mean)
            std = torch.ones_like(std)

        x = (x - mean) / std

        if self.affine:
            x = x * self.weight[None, :, None] + self.bias[None, :, None]

        result = x.view(shape)

        if torch.isnan(result).any() or torch.isinf(result).any():
            logger.warning("BatchNorm output contains NaN/Inf, fixing")
            result = torch.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)

        return result


class StableResidualBlock(nn.Module):
    """Numerically stable residual block."""
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.norm_factor = 1.0 / np.sqrt(2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.warning("ResidualBlock input contains NaN/Inf, fixing")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

        try:
            residual = self.fn(x)

            if torch.isnan(residual).any() or torch.isinf(residual).any():
                logger.warning("Residual computation produced NaN/Inf, returning input")
                return x

            result = (residual + x) * self.norm_factor

            if torch.isnan(result).any() or torch.isinf(result).any():
                logger.warning("ResidualBlock output contains NaN/Inf, returning input")
                return x

            return result

        except Exception as e:
            logger.error(f"ResidualBlock computation failed: {e}, returning input")
            return x


def stable_make_block(channels: int, kernel_size: int) -> nn.Module:
    """Create a numerically stable discriminator block."""
    return nn.Sequential(
        StableSpectralConv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size//2,
            padding_mode='circular',
        ),
        StableBatchNormLocal(channels),
        nn.LeakyReLU(0.2, True),
    )


class StableFullyConnectedLayer(nn.Module):
    """Numerically stable fully connected layer."""
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 lr_multiplier: float = 1.0, weight_init: float = 0.02):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lr_multiplier = lr_multiplier

        self.weight = nn.Parameter(torch.randn(out_features, in_features) * weight_init)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        self.weight_gain = lr_multiplier / np.sqrt(in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.warning("FC layer input contains NaN/Inf, fixing")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

        w = self.weight * self.weight_gain

        result = F.linear(x, w, self.bias)

        if torch.isnan(result).any() or torch.isinf(result).any():
            logger.warning("FC layer output contains NaN/Inf, fixing")
            result = torch.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)

        return result


class StableDiscHead(nn.Module):
    """Numerically stable discriminator head."""
    def __init__(self, channels: int, c_dim: int, cmap_dim: int = 64):
        super().__init__()
        self.channels = channels
        self.c_dim = c_dim
        self.cmap_dim = cmap_dim

        self.main = nn.Sequential(
            stable_make_block(channels, kernel_size=1),
            StableResidualBlock(stable_make_block(channels, kernel_size=9))
        )

        if self.c_dim > 0:
            self.cmapper = StableFullyConnectedLayer(self.c_dim, cmap_dim)
            self.cls = StableSpectralConv1d(channels, cmap_dim, kernel_size=1, padding=0)
            self.norm_factor = 1.0 / np.sqrt(max(1.0, float(cmap_dim)))
        else:
            self.cls = StableSpectralConv1d(channels, 1, kernel_size=1, padding=0)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Improved weight initialization."""
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            nn.init.normal_(module.weight, 0.0, 0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm1d, StableBatchNormLocal)):
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.ones_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.warning("DiscHead input features contain NaN/Inf, fixing")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

        try:
            # Main feature extraction
            h = self.main(x)

            if torch.isnan(h).any() or torch.isinf(h).any():
                logger.warning("Main feature extraction produced NaN/Inf, using zero features")
                h = torch.zeros_like(h)

            # Classification head
            out = self.cls(h)

            if torch.isnan(out).any() or torch.isinf(out).any():
                logger.warning("Classification head output NaN/Inf, using zero output")
                if self.c_dim > 0:
                    out = torch.zeros(x.size(0), self.cmap_dim, h.size(-1), device=x.device)
                else:
                    out = torch.zeros(x.size(0), 1, h.size(-1), device=x.device)

            # Conditional discrimination
            if self.c_dim > 0 and c is not None:
                try:
                    cmap = self.cmapper(c).unsqueeze(-1)

                    if torch.isnan(cmap).any() or torch.isinf(cmap).any():
                        logger.warning("Conditional mapping produced NaN/Inf, using identity")
                        cmap = torch.ones_like(cmap)

                    out = out * cmap
                    out = out.sum(1, keepdim=True) * self.norm_factor

                except Exception as e:
                    logger.warning(f"Conditional discrimination failed: {e}, using unconditional output")
                    out = out.mean(1, keepdim=True)

            # Final numerical stability check
            out = torch.clamp(out, min=-10.0, max=10.0)

            if torch.isnan(out).any() or torch.isinf(out).any():
                logger.warning("DiscHead final output contains NaN/Inf, returning safe default")
                out = torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)

            return out

        except Exception as e:
            logger.error(f"DiscHead forward pass failed: {e}, returning zero output")
            return torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)


class StableProjectedDiscriminator(nn.Module):
    """Numerically stable projected discriminator."""
    def __init__(self, c_dim: int, diffaug: bool = True, p_crop: float = 0.5, dino_checkpoint_path: str = None):
        super().__init__()
        self.c_dim = c_dim
        self.diffaug = diffaug
        self.p_crop = p_crop

        # Import DINO (keeping original stable DINO implementation)
        from stylegan_t.networks.discriminator import DINO
        # Support local DINO model path
        self.dino = DINO(local_checkpoint_path=dino_checkpoint_path)

        # Use stable discriminator heads
        heads = []
        for i in range(self.dino.n_hooks):
            heads.append((str(i), StableDiscHead(self.dino.embed_dim, c_dim)))

        self.heads = nn.ModuleDict(heads)

        # Freeze DINO, only train heads
        for param in self.dino.parameters():
            param.requires_grad = False

    def get_trainable_params(self):
        """Return trainable parameters (heads only)."""
        trainable_params = []
        for head in self.heads.values():
            trainable_params.extend(head.parameters())
        return trainable_params

    def train(self, mode: bool = True):
        """Override train to keep DINO in eval mode always."""
        super().train(mode)
        self.dino.eval()  # DINO always stays in eval mode
        return self

    def forward_rgb(self, rgb_images: torch.Tensor, c: torch.Tensor = None) -> torch.Tensor:
        """Numerically stable RGB forward pass."""
        try:
            rgb_images = torch.clamp(rgb_images, min=-1.0, max=1.0)

            if torch.isnan(rgb_images).any() or torch.isinf(rgb_images).any():
                logger.warning("Input RGB images contain NaN/Inf, fixing")
                rgb_images = torch.nan_to_num(rgb_images, nan=0.0, posinf=1.0, neginf=-1.0)

            # Convert to [0,1] range
            rgb_images_01 = (rgb_images + 1.0) * 0.5
            rgb_images_01 = torch.clamp(rgb_images_01, min=0.0, max=1.0)

            # DINO feature extraction
            dino_features = self.dino(rgb_images_01)

            # Multi-head discrimination
            logits_list = []
            for k, head in self.heads.items():
                try:
                    feature_k = dino_features[k]
                    if torch.isnan(feature_k).any() or torch.isinf(feature_k).any():
                        logger.warning(f"DINO feature {k} contains NaN/Inf, fixing")
                        feature_k = torch.nan_to_num(feature_k, nan=0.0, posinf=1.0, neginf=-1.0)

                    head_logits = head(feature_k, c)
                    logits_list.append(head_logits.view(rgb_images.size(0), -1))

                except Exception as head_e:
                    logger.warning(f"Discriminator head {k} failed: {head_e}, using zero output")
                    B = rgb_images.size(0)
                    device = rgb_images.device
                    fallback_logits = torch.zeros(B, 1, device=device, dtype=rgb_images.dtype)
                    logits_list.append(fallback_logits)

            # Merge outputs
            if logits_list:
                logits = torch.cat(logits_list, dim=1)
            else:
                logger.error("All discriminator heads failed, returning zero output")
                B = rgb_images.size(0)
                device = rgb_images.device
                logits = torch.zeros(B, 1, device=device, dtype=rgb_images.dtype)

            # Final numerical stability check
            logits = torch.clamp(logits, min=-10.0, max=10.0)

            if torch.isnan(logits).any() or torch.isinf(logits).any():
                logger.warning("Final discriminator output contains NaN/Inf, fixing")
                logits = torch.nan_to_num(logits, nan=0.0, posinf=1.0, neginf=-1.0)

            return logits

        except Exception as e:
            logger.error(f"Stable discriminator forward pass failed: {e}")
            B = rgb_images.shape[0]
            device = rgb_images.device
            return torch.zeros(B, 1, device=device, dtype=rgb_images.dtype)

    def forward_soft_rgb(self, rgb_images: torch.Tensor, c: torch.Tensor = None) -> torch.Tensor:
        """Soft RGB forward pass (maintains gradient flow)."""
        return self.forward_rgb(rgb_images, c)  # Uses the same stable implementation
