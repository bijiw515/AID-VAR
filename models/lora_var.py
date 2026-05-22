"""
LoRA (Low-Rank Adaptation) implementation for VAR model.

This module provides LoRA adapters for efficient fine-tuning of the VAR model
as an ablation study to compare with the guidance injector approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple


class LoRALayer(nn.Module):
    """
    LoRA adapter layer implementing low-rank decomposition.

    Decomposes weight update as: ΔW = B @ A, where A ∈ R^(r×in), B ∈ R^(out×r)
    Output: base_output + (B @ A @ x) * (alpha / r)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 32,
        alpha: float = 16.0,
        dropout: float = 0.05
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Dropout for regularization
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # Initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize LoRA parameters conservatively."""
        # Initialize A with Kaiming uniform (like nn.Linear)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # Initialize B to zeros (no adaptation at start)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LoRA adapter.

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            LoRA output of shape (..., out_features)
        """
        # x: (..., in_features)
        # lora_A: (rank, in_features)
        # lora_B: (out_features, rank)

        # Apply dropout to input
        x_dropped = self.dropout(x)

        # Compute LoRA output: (B @ A) @ x
        # First: x @ A^T -> (..., rank)
        lora_out = F.linear(x_dropped, self.lora_A)
        # Second: lora_out @ B^T -> (..., out_features)
        lora_out = F.linear(lora_out, self.lora_B)

        # Scale by alpha / rank
        return lora_out * self.scaling


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adapter.

    Combines a frozen base linear layer with a trainable LoRA adapter.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int = 32,
        alpha: float = 16.0,
        dropout: float = 0.05
    ):
        super().__init__()

        # Store frozen base layer
        self.base_linear = base_linear
        # Freeze base parameters
        for param in self.base_linear.parameters():
            param.requires_grad = False

        # Create LoRA adapter
        self.lora = LoRALayer(
            in_features=base_linear.in_features,
            out_features=base_linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: base output + LoRA adaptation.

        Args:
            x: Input tensor

        Returns:
            Combined output
        """
        # Base output (frozen)
        base_out = self.base_linear(x)

        # LoRA adaptation (trainable)
        lora_out = self.lora(x)

        # Combine
        return base_out + lora_out

    @property
    def weight(self):
        """Property for compatibility with code that accesses .weight"""
        return self.base_linear.weight

    @property
    def bias(self):
        """Property for compatibility with code that accesses .bias"""
        return self.base_linear.bias


def apply_lora_to_linear(
    module: nn.Module,
    target_name: str,
    rank: int,
    alpha: float,
    dropout: float
) -> None:
    """
    Replace a linear layer in a module with LoRALinear.

    Args:
        module: Parent module containing the target linear layer
        target_name: Name of the linear layer attribute
        rank: LoRA rank
        alpha: LoRA alpha scaling factor
        dropout: LoRA dropout rate
    """
    # Get the target linear layer
    target_linear = getattr(module, target_name)

    if not isinstance(target_linear, nn.Linear):
        raise ValueError(f"{target_name} is not a nn.Linear layer")

    # Create LoRA-wrapped version
    lora_linear = LoRALinear(
        base_linear=target_linear,
        rank=rank,
        alpha=alpha,
        dropout=dropout
    )

    # Replace the original layer
    setattr(module, target_name, lora_linear)


def apply_lora_to_var(
    var_model: nn.Module,
    lora_config: Optional[Dict] = None
) -> nn.Module:
    """
    Apply LoRA adapters to VAR model's attention and FFN layers.

    Args:
        var_model: VAR model instance
        lora_config: Configuration dict with keys:
            - 'attention': {'rank': int, 'alpha': float, 'dropout': float}
            - 'ffn': {'rank': int, 'alpha': float, 'dropout': float}

    Returns:
        VAR model with LoRA adapters applied
    """
    # Default configuration
    if lora_config is None:
        lora_config = {
            'attention': {'rank': 64, 'alpha': 16.0, 'dropout': 0.05},
            'ffn': {'rank': 32, 'alpha': 8.0, 'dropout': 0.05}
        }

    attn_config = lora_config['attention']
    ffn_config = lora_config['ffn']

    # Apply LoRA to each transformer block
    num_blocks_modified = 0
    for block_idx, block in enumerate(var_model.blocks):
        # Apply LoRA to attention layers
        # mat_qkv: (C, 3*C) - projects to Q, K, V
        apply_lora_to_linear(
            module=block.attn,
            target_name='mat_qkv',
            rank=attn_config['rank'],
            alpha=attn_config['alpha'],
            dropout=attn_config['dropout']
        )

        # proj: (C, C) - output projection
        apply_lora_to_linear(
            module=block.attn,
            target_name='proj',
            rank=attn_config['rank'],
            alpha=attn_config['alpha'],
            dropout=attn_config['dropout']
        )

        # Apply LoRA to FFN layers
        # fc1: (C, 4*C) - expansion
        apply_lora_to_linear(
            module=block.ffn,
            target_name='fc1',
            rank=ffn_config['rank'],
            alpha=ffn_config['alpha'],
            dropout=ffn_config['dropout']
        )

        # fc2: (4*C, C) - projection
        apply_lora_to_linear(
            module=block.ffn,
            target_name='fc2',
            rank=ffn_config['rank'],
            alpha=ffn_config['alpha'],
            dropout=ffn_config['dropout']
        )

        num_blocks_modified += 1

    print(f"[LoRA] Applied LoRA adapters to {num_blocks_modified} transformer blocks")
    print(f"[LoRA] Attention config: rank={attn_config['rank']}, alpha={attn_config['alpha']}")
    print(f"[LoRA] FFN config: rank={ffn_config['rank']}, alpha={ffn_config['alpha']}")

    # Freeze all base parameters
    for name, param in var_model.named_parameters():
        if 'lora' not in name:
            param.requires_grad = False

    # Count trainable parameters
    total_params = sum(p.numel() for p in var_model.parameters())
    trainable_params = sum(p.numel() for p in var_model.parameters() if p.requires_grad)

    print(f"[LoRA] Total parameters: {total_params:,}")
    print(f"[LoRA] Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

    return var_model


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Extract only LoRA parameters from model state dict.

    Args:
        model: Model with LoRA adapters

    Returns:
        State dict containing only LoRA parameters
    """
    lora_state = {}
    for name, param in model.named_parameters():
        if 'lora' in name and param.requires_grad:
            lora_state[name] = param.data.clone()

    return lora_state


def load_lora_state_dict(
    model: nn.Module,
    lora_state_dict: Dict[str, torch.Tensor],
    strict: bool = True
) -> None:
    """
    Load LoRA parameters into model.

    Args:
        model: Model with LoRA adapters
        lora_state_dict: State dict containing LoRA parameters
        strict: Whether to strictly enforce parameter name matching
    """
    model_state = model.state_dict()

    # Filter to only LoRA parameters
    lora_params = {k: v for k, v in lora_state_dict.items() if 'lora' in k}

    # Update model state
    missing_keys = []
    unexpected_keys = []

    for key, value in lora_params.items():
        if key in model_state:
            model_state[key].copy_(value)
        else:
            unexpected_keys.append(key)

    for key in model_state.keys():
        if 'lora' in key and key not in lora_params:
            missing_keys.append(key)

    if strict and (missing_keys or unexpected_keys):
        error_msg = []
        if missing_keys:
            error_msg.append(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            error_msg.append(f"Unexpected keys: {unexpected_keys}")
        raise RuntimeError("\n".join(error_msg))

    if missing_keys:
        print(f"[LoRA] Warning: Missing keys in checkpoint: {missing_keys}")
    if unexpected_keys:
        print(f"[LoRA] Warning: Unexpected keys in checkpoint: {unexpected_keys}")

    print(f"[LoRA] Loaded {len(lora_params)} LoRA parameters")


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """
    Merge LoRA weights into base weights for inference.

    This creates a single weight matrix by adding LoRA adaptation to base weights.
    After merging, LoRA adapters can be removed for faster inference.

    Args:
        model: Model with LoRA adapters

    Returns:
        Model with merged weights
    """
    for module in model.modules():
        if isinstance(module, LoRALinear):
            # Compute merged weight: W_base + (B @ A) * scaling
            with torch.no_grad():
                lora_weight = (module.lora.lora_B @ module.lora.lora_A) * module.lora.scaling
                module.base_linear.weight.data += lora_weight

                # Zero out LoRA weights after merging
                module.lora.lora_A.zero_()
                module.lora.lora_B.zero_()

    print("[LoRA] Merged LoRA weights into base weights")
    return model


if __name__ == "__main__":
    # Test LoRA implementation
    print("Testing LoRA implementation...")

    # Create a simple linear layer
    base_linear = nn.Linear(512, 1024)
    print(f"Base linear: {base_linear.in_features} -> {base_linear.out_features}")

    # Wrap with LoRA
    lora_linear = LoRALinear(base_linear, rank=32, alpha=16.0, dropout=0.1)
    print(f"LoRA rank: {lora_linear.lora.rank}")
    print(f"LoRA scaling: {lora_linear.lora.scaling}")

    # Test forward pass
    x = torch.randn(4, 16, 512)  # (batch, seq, features)
    output = lora_linear(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Check trainable parameters
    trainable = sum(p.numel() for p in lora_linear.parameters() if p.requires_grad)
    total = sum(p.numel() for p in lora_linear.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    print("\nLoRA implementation test passed!")
