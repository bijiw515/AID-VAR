"""
Empirical experiment: Add random noise at the second scale during VAR generation
and visualize all scales progressively.

This script:
1. Generates images using VAR autoregressively
2. Injects random noise at the second scale (si=1)
3. Continues generation for remaining scales
4. Visualizes intermediate results at each scale
"""

import os
import os.path as osp
import torch
import torchvision
import random
import numpy as np
import PIL.Image as PImage
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
from typing import Optional, Union, List
import argparse

# Disable default parameter init for faster speed
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)

from models import VQVAE, build_vae_var
from models.helpers import sample_with_top_k_top_p_
from models.basic_var import AdaLNSelfAttn


def autoregressive_infer_with_scale_noise(
    var_model,
    B: int,
    label_B: Optional[Union[int, torch.LongTensor]],
    noise_scale_idx: int = 2,  # Which scale to add noise (default: 2nd scale, idx=1)
    noise_magnitude: float = 1.0,  # Magnitude of random noise
    noise_type: str = 'local',  # 'global' or 'local'
    noise_region: tuple = None,  # (h_start, h_end, w_start, w_end) for local noise, None means center region
    g_seed: Optional[int] = None,
    cfg: float = 1.5,
    top_k: int = 900,
    top_p: float = 0.95,
    more_smooth: bool = False,
) -> tuple:
    """
    Modified autoregressive inference that adds random noise at a specific scale.

    Args:
        var_model: VAR model
        B: batch size
        label_B: class labels
        noise_scale_idx: which scale to inject noise (0-indexed)
        noise_magnitude: standard deviation of Gaussian noise
        noise_type: 'global' for full feature map, 'local' for specific region
        noise_region: (h_start, h_end, w_start, w_end) for local noise region
                     If None with local noise, uses center quarter of the feature map
        g_seed: random seed
        cfg: classifier-free guidance strength
        top_k: top-k sampling
        top_p: top-p sampling
        more_smooth: use gumbel softmax for smoothing

    Returns:
        final_image: final generated image (B, 3, H, W)
        scale_images: list of intermediate images at each scale
        scale_features: list of features at each scale
        noise_mask: binary mask showing where noise was applied (for visualization)
    """
    if g_seed is None:
        rng = None
    else:
        var_model.rng.manual_seed(g_seed)
        rng = var_model.rng

    device = var_model.lvl_1L.device

    # Prepare labels
    if label_B is None:
        label_B = torch.multinomial(
            var_model.uniform_prob, num_samples=B, replacement=True, generator=rng
        ).reshape(B)
    elif isinstance(label_B, int):
        label_B = torch.full(
            (B,),
            fill_value=var_model.num_classes if label_B < 0 else label_B,
            device=device
        )

    # Prepare conditioning for CFG
    sos = cond_BD = var_model.class_emb(
        torch.cat((label_B, torch.full_like(label_B, fill_value=var_model.num_classes)), dim=0)
    )

    # Position and level embeddings
    lvl_pos = var_model.lvl_embed(var_model.lvl_1L) + var_model.pos_1LC
    next_token_map = (
        sos.unsqueeze(1).expand(2 * B, var_model.first_l, -1)
        + var_model.pos_start.expand(2 * B, var_model.first_l, -1)
        + lvl_pos[:, :var_model.first_l]
    )

    cur_L = 0
    f_hat = sos.new_zeros(B, var_model.Cvae, var_model.patch_nums[-1], var_model.patch_nums[-1])

    # Storage for intermediate results
    scale_images = []
    scale_features = []
    noise_mask = None  # Will store the noise mask for visualization

    # Enable KV caching
    for b in var_model.blocks:
        b.attn.kv_caching(True)

    try:
        for si, pn in enumerate(var_model.patch_nums):
            print(f"Generating scale {si}/{len(var_model.patch_nums)-1} (patch_num={pn})")

            ratio = si / var_model.num_stages_minus_1
            cur_L += pn * pn

            # Forward through transformer blocks
            cond_BD_or_gss = var_model.shared_ada_lin(cond_BD)
            x = next_token_map

            for b in var_model.blocks:
                x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)

            # Get logits and apply CFG
            logits_BlV = var_model.get_logits(x, cond_BD)
            t = cfg * ratio
            logits_BlV = (1 + t) * logits_BlV[:B] - t * logits_BlV[B:]

            # Sample tokens
            idx_Bl = sample_with_top_k_top_p_(
                logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1
            )[:, :, 0]

            # Get embeddings
            if not more_smooth:
                h_BChw = var_model.vae_quant_proxy[0].embedding(idx_Bl)  # B, l, Cvae
            else:
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)
                h_BChw = (
                    torch.nn.functional.gumbel_softmax(
                        logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1
                    )
                    @ var_model.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
                )

            h_BChw = h_BChw.transpose_(1, 2).reshape(B, var_model.Cvae, pn, pn)

            # ===== ADD NOISE AT SPECIFIED SCALE =====
            if si == noise_scale_idx:
                # Create noise mask based on noise type
                if noise_type == 'global':
                    # Apply noise to entire feature map
                    mask = torch.ones_like(h_BChw)
                    noise = torch.randn_like(h_BChw) * noise_magnitude
                    h_BChw = h_BChw + noise
                    print(f"  → Added GLOBAL Gaussian noise (std={noise_magnitude}) at scale {si}")

                elif noise_type == 'local':
                    # Apply noise to local region
                    mask = torch.zeros_like(h_BChw)

                    # Determine noise region
                    if noise_region is None:
                        # Default: center quarter of the feature map
                        h_start = pn // 4
                        h_end = 3 * pn // 4
                        w_start = pn // 4
                        w_end = 3 * pn // 4
                    else:
                        h_start, h_end, w_start, w_end = noise_region
                        # Clamp to valid range
                        h_start = max(0, min(h_start, pn))
                        h_end = max(h_start, min(h_end, pn))
                        w_start = max(0, min(w_start, pn))
                        w_end = max(w_start, min(w_end, pn))

                    # Generate and apply noise only in the specified region
                    noise = torch.randn_like(h_BChw[:, :, h_start:h_end, w_start:w_end]) * noise_magnitude
                    h_BChw[:, :, h_start:h_end, w_start:w_end] = h_BChw[:, :, h_start:h_end, w_start:w_end] + noise
                    mask[:, :, h_start:h_end, w_start:w_end] = 1.0

                    print(f"  → Added LOCAL Gaussian noise (std={noise_magnitude}) at scale {si}")
                    print(f"     Region: h[{h_start}:{h_end}], w[{w_start}:{w_end}] (size: {h_end-h_start}x{w_end-w_start})")

                # Store noise mask for visualization (use first channel)
                noise_mask = mask[0, 0].detach().cpu()  # Shape: (pn, pn)

            # Update f_hat and get next input
            f_hat, next_token_map = var_model.vae_quant_proxy[0].get_next_autoregressive_input(
                si, len(var_model.patch_nums), f_hat, h_BChw
            )

            # Save intermediate results
            scale_features.append(h_BChw.detach().cpu())

            # Decode current f_hat to image
            with torch.no_grad():
                img_si = var_model.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)
                scale_images.append(img_si.detach().cpu())

            # Prepare for next scale
            if si != var_model.num_stages_minus_1:
                next_token_map = next_token_map.view(B, var_model.Cvae, -1).transpose(1, 2)
                next_token_map = var_model.word_embed(next_token_map) + lvl_pos[
                    :, cur_L:cur_L + var_model.patch_nums[si + 1] ** 2
                ]
                next_token_map = next_token_map.repeat(2, 1, 1)  # CFG doubling

    finally:
        # Disable KV caching
        for b in var_model.blocks:
            b.attn.kv_caching(False)

    # Final image
    final_image = var_model.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)

    return final_image, scale_images, scale_features, noise_mask


def compute_scale_losses(
    scale_features_with_noise: List[torch.Tensor],
    scale_features_no_noise: List[torch.Tensor],
    loss_type: str = 'l2'
) -> dict:
    """
    Compute losses between scale features with and without noise.

    Args:
        scale_features_with_noise: list of feature tensors from noisy generation (B, C, H, W)
        scale_features_no_noise: list of feature tensors from clean generation (B, C, H, W)
        loss_type: 'l1', 'l2', or 'cosine'

    Returns:
        dict containing:
            - 'losses_per_scale': list of scalar loss values for each scale (averaged over batch)
            - 'losses_per_sample': list of loss tensors for each scale (B,)
            - 'spatial_losses': list of spatial loss maps for each scale (B, H, W)
    """
    assert len(scale_features_with_noise) == len(scale_features_no_noise), \
        "Number of scales must match"

    num_scales = len(scale_features_with_noise)
    losses_per_scale = []
    losses_per_sample = []
    spatial_losses = []

    for si in range(num_scales):
        feat_noisy = scale_features_with_noise[si]  # (B, C, H, W)
        feat_clean = scale_features_no_noise[si]    # (B, C, H, W)

        if loss_type == 'l1':
            # L1 loss
            diff = torch.abs(feat_noisy - feat_clean)
        elif loss_type == 'l2':
            # L2 (MSE) loss
            diff = (feat_noisy - feat_clean) ** 2
        elif loss_type == 'cosine':
            # Cosine similarity loss (1 - cosine_similarity)
            # Flatten spatial dimensions for cosine similarity
            B, C, H, W = feat_noisy.shape
            feat_noisy_flat = feat_noisy.reshape(B, C, -1)  # (B, C, H*W)
            feat_clean_flat = feat_clean.reshape(B, C, -1)  # (B, C, H*W)

            # Compute cosine similarity per spatial location
            cos_sim = torch.nn.functional.cosine_similarity(
                feat_noisy_flat, feat_clean_flat, dim=1
            )  # (B, H*W)
            diff = (1 - cos_sim).reshape(B, H, W).unsqueeze(1)  # (B, 1, H, W)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        # Spatial loss: average over channels
        if loss_type in ['l1', 'l2']:
            spatial_loss = diff.mean(dim=1)  # (B, H, W)
        else:
            spatial_loss = diff.squeeze(1)   # (B, H, W)

        # Per-sample loss: average over all spatial dimensions and channels
        per_sample_loss = diff.reshape(diff.shape[0], -1).mean(dim=1)  # (B,)

        # Per-scale loss: average over batch
        per_scale_loss = per_sample_loss.mean().item()

        losses_per_scale.append(per_scale_loss)
        losses_per_sample.append(per_sample_loss)
        spatial_losses.append(spatial_loss)

    return {
        'losses_per_scale': losses_per_scale,
        'losses_per_sample': losses_per_sample,
        'spatial_losses': spatial_losses,
    }


def visualize_scales(
    scale_images: List[torch.Tensor],
    patch_nums: tuple,
    noise_mask: Optional[torch.Tensor] = None,
    noise_scale_idx: Optional[int] = None,
    save_path: str = None
):
    """
    Visualize all scales in a grid, optionally showing noise mask.

    Args:
        scale_images: list of images at each scale (each is B, 3, H, W)
        patch_nums: tuple of patch numbers for each scale
        noise_mask: optional binary mask showing where noise was applied
        noise_scale_idx: which scale had noise applied
        save_path: optional path to save the figure
    """
    num_scales = len(scale_images)
    B = scale_images[0].shape[0]

    # Create figure with subplots for each scale
    fig, axes = plt.subplots(B, num_scales, figsize=(3 * num_scales, 3 * B))

    if B == 1:
        axes = axes.reshape(1, -1)

    for b_idx in range(B):
        for si, (img_tensor, pn) in enumerate(zip(scale_images, patch_nums)):
            ax = axes[b_idx, si]

            # Convert to numpy and transpose to HWC
            # Ensure it's on CPU and convert to float32 for matplotlib compatibility
            img_np = img_tensor[b_idx].float().permute(1, 2, 0).numpy()
            img_np = np.clip(img_np, 0, 1).astype(np.float32)

            ax.imshow(img_np)
            ax.axis('off')

            # Highlight title if this scale had noise
            title_text = f'Scale {si}\n(patch={pn})'
            if noise_mask is not None and si == noise_scale_idx:
                title_text = f'Scale {si} [NOISE]\n(patch={pn})'
                ax.set_title(title_text, fontsize=10, fontweight='bold', color='red')
            else:
                ax.set_title(title_text, fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    plt.close(fig)  # Close figure to free memory


def visualize_loss_curves(
    losses_per_scale: List[float],
    patch_nums: tuple,
    noise_scale_idx: int,
    loss_type: str = 'l2',
    save_path: str = None
):
    """
    Visualize loss curves across scales.

    Args:
        losses_per_scale: list of scalar loss values for each scale
        patch_nums: tuple of patch numbers for each scale
        noise_scale_idx: which scale had noise injected
        loss_type: type of loss used
        save_path: optional path to save the figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    scales = list(range(len(losses_per_scale)))
    ax.plot(scales, losses_per_scale, marker='o', linewidth=2, markersize=8, label=f'{loss_type.upper()} Loss')

    # Highlight the scale where noise was injected
    ax.axvline(x=noise_scale_idx, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Noise Injection Scale')

    # Mark the noise injection point
    ax.scatter([noise_scale_idx], [losses_per_scale[noise_scale_idx]],
               color='red', s=200, zorder=5, marker='*', label='Noise Injection Point')

    ax.set_xlabel('Scale Index', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{loss_type.upper()} Loss', fontsize=12, fontweight='bold')
    ax.set_title(f'Feature Loss Across Scales (Noise vs. Clean)', fontsize=14, fontweight='bold')
    ax.set_xticks(scales)
    ax.set_xticklabels([f'{i}\n({pn}×{pn})' for i, pn in enumerate(patch_nums)], fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved loss curve to {save_path}")

    plt.close(fig)


def visualize_spatial_losses(
    spatial_losses: List[torch.Tensor],
    patch_nums: tuple,
    noise_scale_idx: int,
    batch_idx: int = 0,
    save_path: str = None
):
    """
    Visualize spatial loss maps across all scales for a specific sample.

    Args:
        spatial_losses: list of spatial loss tensors (B, H, W) for each scale
        patch_nums: tuple of patch numbers for each scale
        noise_scale_idx: which scale had noise injected
        batch_idx: which sample from the batch to visualize
        save_path: optional path to save the figure
    """
    num_scales = len(spatial_losses)

    # Create subplots: 2 rows x num_scales columns
    # Row 1: spatial loss heatmaps, Row 2: normalized heatmaps
    fig, axes = plt.subplots(2, num_scales, figsize=(3 * num_scales, 6))

    if num_scales == 1:
        axes = axes.reshape(2, 1)

    # Find global min/max for consistent colorbar
    vmin_global = min([sl[batch_idx].min().item() for sl in spatial_losses])
    vmax_global = max([sl[batch_idx].max().item() for sl in spatial_losses])

    for si, (spatial_loss, pn) in enumerate(zip(spatial_losses, patch_nums)):
        loss_map = spatial_loss[batch_idx].cpu().numpy()  # (H, W)

        # Row 0: Absolute loss values (global scale)
        ax0 = axes[0, si]
        im0 = ax0.imshow(loss_map, cmap='hot', vmin=vmin_global, vmax=vmax_global, interpolation='nearest')
        title_text = f'Scale {si}\n({pn}×{pn})'
        if si == noise_scale_idx:
            title_text = f'Scale {si} [NOISE]\n({pn}×{pn})'
            ax0.set_title(title_text, fontsize=10, fontweight='bold', color='red')
        else:
            ax0.set_title(title_text, fontsize=10)
        ax0.axis('off')
        plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

        # Row 1: Normalized loss values (per-scale normalization)
        ax1 = axes[1, si]
        vmin_local = loss_map.min()
        vmax_local = loss_map.max()
        im1 = ax1.imshow(loss_map, cmap='hot', vmin=vmin_local, vmax=vmax_local, interpolation='nearest')
        ax1.set_title('Normalized', fontsize=9)
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # Add row labels
    axes[0, 0].text(-0.1, 0.5, 'Global Scale', transform=axes[0, 0].transAxes,
                    fontsize=12, fontweight='bold', va='center', rotation=90)
    axes[1, 0].text(-0.1, 0.5, 'Local Scale', transform=axes[1, 0].transAxes,
                    fontsize=12, fontweight='bold', va='center', rotation=90)

    fig.suptitle(f'Spatial Loss Distribution Across Scales (Sample {batch_idx})',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved spatial loss visualization to {save_path}")

    plt.close(fig)


def visualize_noise_mask(noise_mask: torch.Tensor, patch_num: int, save_path: str = None):
    """
    Visualize the noise mask.

    Args:
        noise_mask: binary mask tensor (H, W)
        patch_num: patch number for the scale
        save_path: optional path to save the figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # Convert to numpy
    mask_np = noise_mask.numpy()

    # Visualize mask
    im = ax.imshow(mask_np, cmap='RdYlGn', vmin=0, vmax=1, interpolation='nearest')
    ax.set_title(f'Noise Mask (patch_num={patch_num})', fontsize=14, fontweight='bold')
    ax.axis('off')

    # Add colorbar
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Add grid to show patch boundaries
    for i in range(patch_num + 1):
        ax.axhline(y=i - 0.5, color='gray', linewidth=0.5, alpha=0.3)
        ax.axvline(x=i - 0.5, color='gray', linewidth=0.5, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved noise mask visualization to {save_path}")

    plt.close(fig)


def main():
    # ===== Parse arguments =====
    parser = argparse.ArgumentParser(description='VAR scale noise injection experiment')
    parser.add_argument('--model_depth', type=int, default=16, choices=[16, 20, 24, 30],
                        help='VAR model depth')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_samples', type=int, default=4,
                        help='Number of samples to generate')
    parser.add_argument('--class_labels', type=int, nargs='+', default=[207, 283, 22, 284],
                        help='Class labels for generation (goldfish=980, baseball=437, boar=22, french_horn=562)')
    parser.add_argument('--noise_scale_idx', type=int, default=4,
                        help='Which scale to inject noise (0-9, where 0=1st scale, 1=2nd scale, etc.)')
    parser.add_argument('--noise_magnitude', type=float, default=0.2,
                        help='Noise standard deviation')
    parser.add_argument('--noise_type', type=str, default='global', choices=['global', 'local'],
                        help='Type of noise: global (entire feature map) or local (specific region)')
    parser.add_argument('--noise_region', type=int, nargs=4, default=None, metavar=('H_START', 'H_END', 'W_START', 'W_END'),
                        help='Noise region for local noise (h_start h_end w_start w_end). If not specified, uses center quarter.')
    parser.add_argument('--cfg', type=float, default=4.0,
                        help='Classifier-free guidance strength')
    parser.add_argument('--top_k', type=int, default=900,
                        help='Top-k sampling parameter')
    parser.add_argument('--top_p', type=float, default=0.95,
                        help='Top-p (nucleus) sampling parameter')
    parser.add_argument('--output_dir', type=str, default='experiments/scale_noise_visualization',
                        help='Output directory for visualizations')
    parser.add_argument('--loss_type', type=str, default='l2', choices=['l1', 'l2', 'cosine'],
                        help='Type of loss to compute: l1, l2 (MSE), or cosine')

    args = parser.parse_args()

    # ===== Configuration from args =====
    MODEL_DEPTH = args.model_depth
    SEED = args.seed
    NUM_SAMPLES = args.num_samples
    CLASS_LABELS = args.class_labels

    NOISE_SCALE_IDX = args.noise_scale_idx
    NOISE_MAGNITUDE = args.noise_magnitude
    NOISE_TYPE = args.noise_type
    NOISE_REGION = tuple(args.noise_region) if args.noise_region else None

    CFG = args.cfg
    TOP_K = args.top_k
    TOP_P = args.top_p

    # ===== Setup =====
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Set random seeds
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Enable TF32 for faster computation
    tf32 = True
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.set_float32_matmul_precision('high' if tf32 else 'highest')

    # ===== Load models =====
    print("Loading models...")

    # Download checkpoints if needed
    hf_home = 'https://huggingface.co/FoundationVision/var/resolve/main'
    vae_ckpt = 'vae_ch160v4096z32.pth'
    var_ckpt = f'checkpoints/var_d{MODEL_DEPTH}.pth'

    if not osp.exists(vae_ckpt):
        os.system(f'wget {hf_home}/{vae_ckpt}')
    if not osp.exists(var_ckpt):
        print(f"Warning: {var_ckpt} not found. You may need to download it manually.")

    # Build models
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        device=device, patch_nums=patch_nums,
        num_classes=1000, depth=MODEL_DEPTH, shared_aln=False,
    )

    # Load checkpoints
    vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
    var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=True)

    vae.eval()
    var.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    for p in var.parameters():
        p.requires_grad_(False)

    print("Models loaded successfully!")

    # ===== Generate samples WITH noise =====
    print(f"\n{'='*70}")
    print(f"STEP 1: Generating {NUM_SAMPLES} samples WITH noise")
    print(f"{'='*70}")
    print(f"  - Noise type: {NOISE_TYPE.upper()}")
    print(f"  - Adding noise at scale {NOISE_SCALE_IDX} with magnitude {NOISE_MAGNITUDE}")
    if NOISE_TYPE == 'local' and NOISE_REGION:
        print(f"  - Noise region: h[{NOISE_REGION[0]}:{NOISE_REGION[1]}], w[{NOISE_REGION[2]}:{NOISE_REGION[3]}]")
    elif NOISE_TYPE == 'local':
        print(f"  - Noise region: center quarter (auto-determined)")
    print(f"  - Class labels: {CLASS_LABELS[:NUM_SAMPLES]}")

    B = min(NUM_SAMPLES, len(CLASS_LABELS))
    label_B = torch.tensor(CLASS_LABELS[:B], device=device)

    with torch.inference_mode():
        with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):
            final_image_noisy, scale_images_noisy, scale_features_noisy, noise_mask = autoregressive_infer_with_scale_noise(
                var_model=var,
                B=B,
                label_B=label_B,
                noise_scale_idx=NOISE_SCALE_IDX,
                noise_magnitude=NOISE_MAGNITUDE,
                noise_type=NOISE_TYPE,
                noise_region=NOISE_REGION,
                g_seed=SEED,
                cfg=CFG,
                top_k=TOP_K,
                top_p=TOP_P,
                more_smooth=False,
            )

    print(f"Generation complete! Generated {len(scale_images_noisy)} scales")

    # ===== Generate samples WITHOUT noise (clean baseline) =====
    print(f"\n{'='*70}")
    print(f"STEP 2: Generating {NUM_SAMPLES} samples WITHOUT noise (clean baseline)")
    print(f"{'='*70}")

    with torch.inference_mode():
        with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):
            # Use a different noise_scale_idx that is out of range to effectively disable noise
            final_image_clean, scale_images_clean, scale_features_clean, _ = autoregressive_infer_with_scale_noise(
                var_model=var,
                B=B,
                label_B=label_B,
                noise_scale_idx=-1,  # Out of range, effectively no noise
                noise_magnitude=0.0,  # No noise
                noise_type=NOISE_TYPE,
                noise_region=NOISE_REGION,
                g_seed=SEED,  # Same seed for fair comparison
                cfg=CFG,
                top_k=TOP_K,
                top_p=TOP_P,
                more_smooth=False,
            )

    print(f"Clean generation complete! Generated {len(scale_images_clean)} scales")

    # ===== Compute losses between noisy and clean features =====
    print(f"\n{'='*70}")
    print(f"STEP 3: Computing losses between noisy and clean features")
    print(f"{'='*70}")
    print(f"  - Loss type: {args.loss_type.upper()}")

    loss_results = compute_scale_losses(
        scale_features_with_noise=scale_features_noisy,
        scale_features_no_noise=scale_features_clean,
        loss_type=args.loss_type
    )

    losses_per_scale = loss_results['losses_per_scale']
    losses_per_sample = loss_results['losses_per_sample']
    spatial_losses = loss_results['spatial_losses']

    print(f"\nLoss statistics:")
    for si, loss_val in enumerate(losses_per_scale):
        marker = " ← NOISE INJECTED" if si == NOISE_SCALE_IDX else ""
        print(f"  Scale {si} (patch={patch_nums[si]:2d}): {loss_val:.6f}{marker}")

    # ===== Visualize results =====
    print(f"\n{'='*70}")
    print(f"STEP 4: Visualizing results")
    print(f"{'='*70}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # File prefix based on noise type
    file_prefix = f'noise_{NOISE_TYPE}_scale{NOISE_SCALE_IDX}_mag{NOISE_MAGNITUDE}'

    # 1. Visualize loss curves
    print("\n1. Visualizing loss curves...")
    loss_curve_path = f'{args.output_dir}/{file_prefix}_loss_curve.png'
    visualize_loss_curves(
        losses_per_scale=losses_per_scale,
        patch_nums=patch_nums,
        noise_scale_idx=NOISE_SCALE_IDX,
        loss_type=args.loss_type,
        save_path=loss_curve_path
    )

    # 2. Visualize spatial loss heatmaps for each sample
    print("\n2. Visualizing spatial loss heatmaps...")
    for batch_idx in range(B):
        spatial_loss_path = f'{args.output_dir}/{file_prefix}_spatial_loss_sample{batch_idx}.png'
        visualize_spatial_losses(
            spatial_losses=spatial_losses,
            patch_nums=patch_nums,
            noise_scale_idx=NOISE_SCALE_IDX,
            batch_idx=batch_idx,
            save_path=spatial_loss_path
        )

    # 3. Visualize all scales for NOISY version
    print("\n3. Visualizing all scales (noisy version)...")
    noisy_scales_path = f'{args.output_dir}/{file_prefix}_noisy_all_scales.png'
    visualize_scales(scale_images_noisy, patch_nums, noise_mask=noise_mask,
                    noise_scale_idx=NOISE_SCALE_IDX, save_path=noisy_scales_path)

    # 4. Visualize all scales for CLEAN version
    print("\n4. Visualizing all scales (clean version)...")
    clean_scales_path = f'{args.output_dir}/{file_prefix}_clean_all_scales.png'
    visualize_scales(scale_images_clean, patch_nums, noise_mask=None,
                    noise_scale_idx=None, save_path=clean_scales_path)

    # 5. Visualize noise mask if available
    if noise_mask is not None:
        print("\n5. Visualizing noise mask...")
        mask_save_path = f'{args.output_dir}/{file_prefix}_mask.png'
        visualize_noise_mask(noise_mask, patch_nums[NOISE_SCALE_IDX], save_path=mask_save_path)

    # 6. Save final images (both noisy and clean) side by side
    print("\n6. Saving final images...")
    # Noisy version
    final_grid_noisy = torchvision.utils.make_grid(final_image_noisy.cpu(), nrow=4, padding=2, pad_value=1.0)
    final_grid_noisy_np = final_grid_noisy.permute(1, 2, 0).mul_(255).numpy().astype(np.uint8)
    final_img_noisy = PImage.fromarray(final_grid_noisy_np)
    final_noisy_path = f'{args.output_dir}/{file_prefix}_final_noisy.png'
    final_img_noisy.save(final_noisy_path)
    print(f"   - Saved noisy final images to {final_noisy_path}")

    # Clean version
    final_grid_clean = torchvision.utils.make_grid(final_image_clean.cpu(), nrow=4, padding=2, pad_value=1.0)
    final_grid_clean_np = final_grid_clean.permute(1, 2, 0).mul_(255).numpy().astype(np.uint8)
    final_img_clean = PImage.fromarray(final_grid_clean_np)
    final_clean_path = f'{args.output_dir}/{file_prefix}_final_clean.png'
    final_img_clean.save(final_clean_path)
    print(f"   - Saved clean final images to {final_clean_path}")

    # 7. Create a comparison figure with noisy and clean side by side
    print("\n7. Creating comparison visualization...")
    fig, axes = plt.subplots(B, 2, figsize=(8, 4*B))
    if B == 1:
        axes = axes.reshape(1, -1)

    for b_idx in range(B):
        # Noisy image
        img_noisy = final_image_noisy[b_idx].cpu().float().permute(1, 2, 0).numpy()
        img_noisy = np.clip(img_noisy, 0, 1)
        axes[b_idx, 0].imshow(img_noisy)
        axes[b_idx, 0].set_title(f'Sample {b_idx}: WITH Noise', fontweight='bold')
        axes[b_idx, 0].axis('off')

        # Clean image
        img_clean = final_image_clean[b_idx].cpu().float().permute(1, 2, 0).numpy()
        img_clean = np.clip(img_clean, 0, 1)
        axes[b_idx, 1].imshow(img_clean)
        axes[b_idx, 1].set_title(f'Sample {b_idx}: WITHOUT Noise', fontweight='bold')
        axes[b_idx, 1].axis('off')

    plt.tight_layout()
    comparison_path = f'{args.output_dir}/{file_prefix}_comparison.png'
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   - Saved comparison to {comparison_path}")

    print(f"\n{'='*70}")
    print("Experiment complete!")
    print(f"{'='*70}")
    print(f"All results saved to: {args.output_dir}")
    print(f"\nKey outputs:")
    print(f"  - Loss curve: {loss_curve_path}")
    print(f"  - Spatial loss heatmaps: {args.output_dir}/{file_prefix}_spatial_loss_sample*.png")
    print(f"  - Noisy final images: {final_noisy_path}")
    print(f"  - Clean final images: {final_clean_path}")
    print(f"  - Side-by-side comparison: {comparison_path}")


if __name__ == "__main__":
    main()
