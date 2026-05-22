"""
AID-VAR planning module: lightweight GuidanceInjector
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

def safe_layer_norm(x, weight, bias, eps=1e-5):
    """Numerically stable LayerNorm implementation."""
    # Clamp input to prevent overflow
    x = torch.clamp(x, min=-10.0, max=10.0)

    # Compute mean and variance with numerical protection
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)

    # Prevent division by zero
    var = torch.clamp(var, min=eps, max=100.0)
    std = torch.sqrt(var + eps)

    # Normalize
    normalized = (x - mean) / std

    # Clamp normalized output
    normalized = torch.clamp(normalized, min=-5.0, max=5.0)

    # Apply weight and bias
    if weight is not None:
        # Clamp weight to prevent extreme scaling
        safe_weight = torch.clamp(weight, min=0.1, max=2.0)
        normalized = normalized * safe_weight

    if bias is not None:
        safe_bias = torch.clamp(bias, min=-1.0, max=1.0)
        normalized = normalized + safe_bias

    return torch.clamp(normalized, min=-3.0, max=3.0)

class SafeLayerNorm(nn.Module):
    """Numerically safe LayerNorm layer."""
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps

        # Conservative initialization
        self.weight = nn.Parameter(torch.ones(normalized_shape) * 0.5)
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        return safe_layer_norm(x, self.weight, self.bias, self.eps)

class UltraSafeTransformerBlock(nn.Module):
    """Numerically safe Transformer block."""

    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.norm1 = SafeLayerNorm(embed_dim)
        self.norm2 = SafeLayerNorm(embed_dim)

        # Multi-head attention with conservative settings
        self.self_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )

        # FFN with smaller hidden dimension to prevent overflow
        hidden_dim = min(embed_dim * 2, 512)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Very small init std for stability
                nn.init.normal_(module.weight, mean=0.0, std=0.001)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        x = torch.clamp(x, min=-5.0, max=5.0)

        # First residual: self-attention
        x_norm = self.norm1(x)

        try:
            attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
            attn_out = torch.clamp(attn_out, min=-3.0, max=3.0)

            # Scale down residual contribution
            x = x + 0.1 * attn_out
            x = torch.clamp(x, min=-5.0, max=5.0)

        except Exception:
            return torch.clamp(x, min=-3.0, max=3.0)

        # Second residual: FFN
        x_norm2 = self.norm2(x)

        try:
            ffn_out = self.ffn(x_norm2)
            ffn_out = torch.clamp(ffn_out, min=-3.0, max=3.0)

            x = x + 0.1 * ffn_out
            x = torch.clamp(x, min=-5.0, max=5.0)

        except Exception:
            pass

        return torch.clamp(x, min=-3.0, max=3.0)

class UltraSafeTransformerEncoder(nn.Module):
    """Numerically safe Transformer encoder."""

    def __init__(self, num_layers, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            UltraSafeTransformerBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers

    def forward(self, x):
        x = torch.clamp(x, min=-5.0, max=5.0)

        for layer in self.layers:
            if torch.isnan(x).any() or torch.isinf(x).any():
                return torch.clamp(x, min=-3.0, max=3.0)

            try:
                x_new = layer(x)

                if torch.isnan(x_new).any() or torch.isinf(x_new).any():
                    return torch.clamp(x, min=-3.0, max=3.0)

                x = x_new

            except Exception:
                return torch.clamp(x, min=-3.0, max=3.0)

        return torch.clamp(x, min=-3.0, max=3.0)

class GuidanceInjector(nn.Module):
    """
    Implicit planner that generates spatially-aware planning token maps.

    Key properties:
    1. Outputs a Planning Token Map rather than a single token
    2. Size-matched: output matches the target generation scale (L_k = pn_k^2)
    3. Spatial refinement: each position carries independent planning information
    4. Element-wise addition: added to S_k position-by-position, not broadcast
    """

    def __init__(self, input_dim=1024, embed_dim=1024, num_layers=2, num_heads=8):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        # Input projection with small initialization
        self.input_proj = nn.Linear(input_dim, embed_dim)
        nn.init.normal_(self.input_proj.weight, mean=0.0, std=0.001)
        nn.init.constant_(self.input_proj.bias, 0.0)

        self.encoder = UltraSafeTransformerEncoder(
            num_layers=num_layers,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.1
        )

        # Positional encoding cache
        self.pos_encodings = {}

        # Output projection preserving per-position independence
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=0.001)
        nn.init.constant_(self.output_proj.bias, 0.0)

        self.final_norm = SafeLayerNorm(embed_dim)

    def _get_position_encoding(self, size: int, device: torch.device) -> torch.Tensor:
        """Get or create positional encoding for the given sequence length."""
        key = f"pos_{size}"

        if key not in self.pos_encodings:
            # Sinusoidal positional encoding for flattened spatial positions
            pos_embed = torch.zeros(1, size, self.embed_dim)

            for i in range(size):
                for j in range(self.embed_dim):
                    if j % 2 == 0:
                        pos_embed[0, i, j] = math.sin(i / (10000 ** (j / self.embed_dim)))
                    else:
                        pos_embed[0, i, j] = math.cos(i / (10000 ** ((j-1) / self.embed_dim)))

            # Register as buffer (not parameter) to avoid training
            self.register_buffer(key, pos_embed * 0.01)
            self.pos_encodings[key] = self.__dict__['_buffers'][key]

        return self.pos_encodings[key].to(device)

    def forward(self, prev_features, target_patch_num=None):
        """Generate spatially-aware planning token map.

        Args:
            prev_features: features from previous scale (B, L_prev, C)
            target_patch_num: patch count for target scale (pn_k); inferred if None

        Returns:
            planning_token_map: (B, L_k, C) where L_k = pn_k^2
        """
        B, L_prev, C = prev_features.shape
        device = prev_features.device

        if target_patch_num is None:
            import math
            pn_prev = int(math.sqrt(L_prev))
            target_patch_num = pn_prev

        target_L = target_patch_num * target_patch_num

        prev_features = torch.clamp(prev_features, min=-10.0, max=10.0)

        if prev_features.numel() == 0:
            return torch.zeros(B, target_L, self.embed_dim, device=device)

        x_proj = self.input_proj(prev_features)
        x_proj = torch.clamp(x_proj, min=-3.0, max=3.0)

        # Spatial resize to target scale
        if L_prev != target_L:
            if target_L > L_prev:
                # Upsample via interpolation
                x_proj = x_proj.transpose(1, 2)
                x_proj = F.interpolate(x_proj, size=target_L, mode='linear', align_corners=False)
                x_proj = x_proj.transpose(1, 2)
            else:
                # Downsample via pooling
                x_proj = x_proj.transpose(1, 2)
                x_proj = F.adaptive_avg_pool1d(x_proj, target_L)
                x_proj = x_proj.transpose(1, 2)

        pos_encoding = self._get_position_encoding(target_L, device)
        x_proj = x_proj + pos_encoding.expand(B, -1, -1)
        x_proj = torch.clamp(x_proj, min=-3.0, max=3.0)

        encoded = self.encoder(x_proj)

        output = self.output_proj(encoded)
        output = torch.clamp(output, min=-3.0, max=3.0)

        planning_token_map = self.final_norm(output)
        planning_token_map = torch.clamp(planning_token_map, min=-2.0, max=2.0)

        if torch.isnan(planning_token_map).any() or torch.isinf(planning_token_map).any():
            return torch.zeros(B, target_L, self.embed_dim, device=device)

        return planning_token_map

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

# Backward-compatible alias
def create_iplanner(embed_dim=1024, hidden_dim=512, num_layers=2, num_heads=8):
    """Create a GuidanceInjector instance."""
    model = GuidanceInjector(embed_dim, hidden_dim, num_layers, num_heads)
    return model


class IPlannerTiny(nn.Module):
    """Ultra-lightweight planner (<5M parameters) using MLP instead of Transformer."""

    def __init__(
        self,
        embed_dim: int = 1024,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        in_dim = embed_dim

        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else embed_dim
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU() if i < num_layers - 1 else nn.Identity(),
                nn.Dropout(dropout) if i < num_layers - 1 else nn.Identity(),
            ])
            in_dim = out_dim

        self.mlp = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, C)
        Returns:
            I_k: (B, C)
        """
        x = x.transpose(1, 2)  # (B, C, L)
        x = self.pool(x)       # (B, C, 1)
        x = x.squeeze(-1)      # (B, C)
        I_k = self.mlp(x)
        return I_k


def _test():
    B, L, C = 2, 128, 512
    m = GuidanceInjector(embed_dim=C)
    x = torch.randn(B, L, C)
    out = m(x)
    assert out.shape == (B, C)


if __name__ == "__main__":
    _test()
