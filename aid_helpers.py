#!/usr/bin/env python3
"""
AID-VAR training helper functions.

Auxiliary functions used during AID-VAR training,
extracted from trainer_planner.py for code organization.

Includes:
- Visual debugging and sample saving
- MSE loss calculation
- Visual debug summary generation
- Other training utilities
"""
import os
import time
import json
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Tuple
from PIL import Image
import numpy as np

# Get logger
import logging
logger = logging.getLogger(__name__)

# Type definitions
Ten = torch.Tensor
ITen = torch.LongTensor


def calculate_mse_loss(pred_tokens: ITen, gt_tokens: ITen) -> float:
    """Calculate MSE loss

    Args:
        pred_tokens: predicted tokens (B, L) or (L,)
        gt_tokens: ground truth tokens (B, L) or (L,)

    Returns:
        MSE loss value
    """
    try:
        # Key fix: ensure input and target tensor shapes match
        pred_tokens = pred_tokens.float()
        gt_tokens = gt_tokens.float()

        # Record original shapes
        original_pred_shape = pred_tokens.shape
        original_gt_shape = gt_tokens.shape

        # Ensure both tensors are 2D
        if pred_tokens.dim() == 1:
            pred_tokens = pred_tokens.unsqueeze(0)  # (L,) -> (1, L)
        if gt_tokens.dim() == 1:
            gt_tokens = gt_tokens.unsqueeze(0)  # (L,) -> (1, L)

        # Ensure shapes match
        if pred_tokens.shape != gt_tokens.shape:
            # If shapes don't match, try broadcasting or adjusting
            if pred_tokens.shape[0] != gt_tokens.shape[0]:
                # Batch size mismatch, take the smaller one
                min_batch = min(pred_tokens.shape[0], gt_tokens.shape[0])
                pred_tokens = pred_tokens[:min_batch]
                gt_tokens = gt_tokens[:min_batch]

            if pred_tokens.shape[1] != gt_tokens.shape[1]:
                # Sequence length mismatch, take the smaller one
                min_length = min(pred_tokens.shape[1], gt_tokens.shape[1])
                pred_tokens = pred_tokens[:, :min_length]
                gt_tokens = gt_tokens[:, :min_length]

        mse_value = F.mse_loss(pred_tokens, gt_tokens).item()
        return mse_value

    except Exception as e:
        logger.warning(f"MSE loss calculation failed: {e}")
        logger.warning(f"   Original shapes: pred={original_pred_shape if 'original_pred_shape' in locals() else 'N/A'}, gt={original_gt_shape if 'original_gt_shape' in locals() else 'N/A'}")
        logger.warning(f"   Current shapes: pred={pred_tokens.shape if 'pred_tokens' in locals() else 'N/A'}, gt={gt_tokens.shape if 'gt_tokens' in locals() else 'N/A'}")
        return 0.0


def save_visual_debugging_samples(
    vae: nn.Module,
    patch_nums: Tuple[int],
    Cvae: int,
    device: torch.device,
    epoch: int,
    batch_idx: int,
    scale_idx: int,
    real_tokens: torch.Tensor,  # real tokens at current scale
    fake_tokens: torch.Tensor,  # fake tokens at current scale
    std_tokens: List[torch.Tensor],  # multi-scale tokens predicted by standard VAR
    save_dir: str,
    real_tokens_history: List[torch.Tensor],  # multi-scale history of real tokens (required)
    fake_tokens_history: List[torch.Tensor],  # multi-scale history of fake tokens (required)
    max_samples: int = 4,
    guidance_tokens_list: Optional[List[torch.Tensor]] = None  # guidance tokens from GuidanceInjector (attention feature vectors)
):
    """Save visual debugging samples

    Args:
        vae: VQ-VAE model for token-to-image conversion
        patch_nums: multi-scale patch counts
        Cvae: VAE channel count
        device: device
        epoch: current epoch
        batch_idx: current batch index
        scale_idx: current scale index
        real_tokens: real tokens at current scale (B, pn^2)
        fake_tokens: fake tokens at current scale (B, pn^2)
        std_tokens: list of multi-scale tokens from standard VAR
        save_dir: save directory
        real_tokens_history: multi-scale history of real tokens (required, for building full sequence)
        fake_tokens_history: multi-scale history of fake tokens (required, for building full sequence)
        max_samples: max number of samples to save
        guidance_tokens_list: multi-scale guidance tokens from GuidanceInjector (attention feature vector form)

    Note:
        - Real samples: use real_tokens_history as history + current real_tokens
        - Fake samples: use fake_tokens_history as history + current fake_tokens
        - Standard VAR samples: use std_tokens as history + current std_tokens
        - Guidance visualization: visualize guidance_tokens_list as attention heatmaps overlaid on std images
    """
    if std_tokens is not None:
        pass

    # Create save directory
    debug_dir = os.path.join(save_dir, "visual_debug", f"epoch_{epoch+1}", f"batch_{batch_idx+1}", f"scale_{scale_idx+1}")
    os.makedirs(debug_dir, exist_ok=True)

    B = min(real_tokens.shape[0], max_samples)
    save_stats = {"real": 0, "fake": 0, "std": 0, "guidance": 0}

    # Key fix: check data integrity
    # Check if std_tokens has enough data
    if len(std_tokens) <= scale_idx:
        logger.warning(f"   std_tokens length insufficient: {len(std_tokens)} <= {scale_idx}")
        return

    # Check each required std_tokens entry exists and has enough samples
    for si in range(scale_idx + 1):
        if si >= len(std_tokens) or std_tokens[si] is None:
            logger.warning(f"   std_tokens[{si}] missing or None")
            return
        if std_tokens[si].shape[0] < B:
            logger.warning(f"   std_tokens[{si}] insufficient samples: {std_tokens[si].shape[0]} < {B}")
            B = std_tokens[si].shape[0]  # Adjust to minimum available samples

    # Key fix: use correct VQ-VAE multi-scale reconstruction workflow
    for i in range(B):
        try:

            # Save reconstruction image for real samples
            try:
                # Key fix: build complete 10-scale token list
                # Use real token history; current scale uses real tokens; future scales zero-padded
                ms_real_tokens = []
                for si in range(len(patch_nums)):  # Full 10 scales
                    if si < scale_idx:
                        # Previous scales: use real token history
                        if real_tokens_history is None or si >= len(real_tokens_history):
                            raise ValueError(f"Insufficient real token history: need scale {si}, only have {len(real_tokens_history) if real_tokens_history else 0} scales")
                        token = real_tokens_history[si][i:i+1]  # (1, L_si)
                        # Ensure 2D tensor
                        if token.dim() == 1:
                            token = token.unsqueeze(0)  # (1, L_si)
                        ms_real_tokens.append(token)
                    elif si == scale_idx:
                        # Current scale: use real tokens
                        token = real_tokens[i:i+1]  # (1, L_current)
                        # Ensure 2D tensor
                        if token.dim() == 1:
                            token = token.unsqueeze(0)  # (1, L_current)
                        ms_real_tokens.append(token)
                    else:
                        # Future scales: zero-pad
                        pn = patch_nums[si]
                        L_future = pn * pn
                        zero_tokens = torch.zeros(1, L_future, dtype=real_tokens.dtype, device=real_tokens.device)
                        ms_real_tokens.append(zero_tokens)

                # Key fix: use correct VQ-VAE reconstruction method
                real_img = vae.idxBl_to_img(ms_real_tokens, same_shape=True, last_one=True)

                # Convert to PIL image and save
                if real_img is not None and real_img.numel() > 0:
                    # Ensure image is in [0,1] range (converted from [-1,1])
                    real_img_np = real_img[0].detach().cpu().add(1).mul(0.5).clamp(0, 1).permute(1, 2, 0).numpy()
                    real_img_pil = Image.fromarray((real_img_np * 255).astype(np.uint8))
                    real_img_pil.save(os.path.join(debug_dir, f"real_sample_{i+1}_scale_{scale_idx+1}.png"))
                    save_stats["real"] += 1

            except Exception as e:
                logger.warning(f"       Failed to save real sample {i+1}: {e}")
                import traceback
                logger.warning(f"       Traceback: {traceback.format_exc()}")

            # Save reconstruction image for fake samples
            try:
                # Key fix: build complete 10-scale token list
                # Use fake token history; current scale uses fake tokens; future scales zero-padded
                ms_fake_tokens = []
                for si in range(len(patch_nums)):  # Full 10 scales
                    if si < scale_idx:
                        # Previous scales: use fake token history
                        if fake_tokens_history is None or si >= len(fake_tokens_history):
                            raise ValueError(f"Insufficient fake token history: need scale {si}, only have {len(fake_tokens_history) if fake_tokens_history else 0} scales")
                        token = fake_tokens_history[si][i:i+1]  # (1, L_si)
                        # Ensure 2D tensor
                        if token.dim() == 1:
                            token = token.unsqueeze(0)  # (1, L_si)
                        ms_fake_tokens.append(token)
                    elif si == scale_idx:
                        # Current scale: use generated fake tokens
                        token = fake_tokens[i:i+1]  # (1, L_current)
                        # Ensure 2D tensor
                        if token.dim() == 1:
                            token = token.unsqueeze(0)  # (1, L_current)
                        ms_fake_tokens.append(token)
                    else:
                        # Future scales: zero-pad
                        pn = patch_nums[si]
                        L_future = pn * pn
                        zero_tokens = torch.zeros(1, L_future, dtype=fake_tokens.dtype, device=fake_tokens.device)
                        ms_fake_tokens.append(zero_tokens)

                # Key fix: use correct VQ-VAE reconstruction method
                fake_img = vae.idxBl_to_img(ms_fake_tokens, same_shape=True, last_one=True)

                # Convert to PIL image and save
                if fake_img is not None and fake_img.numel() > 0:
                    # Ensure image is in [0,1] range (converted from [-1,1])
                    fake_img_np = fake_img[0].detach().cpu().add(1).mul(0.5).clamp(0, 1).permute(1, 2, 0).numpy()
                    fake_img_pil = Image.fromarray((fake_img_np * 255).astype(np.uint8))
                    fake_img_pil.save(os.path.join(debug_dir, f"fake_sample_{i+1}_scale_{scale_idx+1}.png"))
                    save_stats["fake"] += 1

            except Exception as e:
                logger.warning(f"       Failed to save fake sample {i+1}: {e}")
                import traceback
                logger.warning(f"       Traceback: {traceback.format_exc()}")

            # Save reconstruction image for standard VAR samples
            try:
                # Key fix: build complete 10-scale token list
                ms_std_tokens = []
                for si in range(len(patch_nums)):  # Full 10 scales
                    if si <= scale_idx:
                        # Available scales: use standard VAR predictions
                        token = std_tokens[si][i:i+1]  # (1, L_si)
                        # Ensure 2D tensor
                        if token.dim() == 1:
                            token = token.unsqueeze(0)  # (1, L_si)
                        ms_std_tokens.append(token)
                    else:
                        # Future scales: zero-pad
                        pn = patch_nums[si]
                        L_future = pn * pn
                        zero_tokens = torch.zeros(1, L_future, dtype=std_tokens[0].dtype, device=std_tokens[0].device)
                        ms_std_tokens.append(zero_tokens)

                # Key fix: use correct VQ-VAE reconstruction method
                std_img = vae.idxBl_to_img(ms_std_tokens, same_shape=True, last_one=True)

                # Convert to PIL image and save
                if std_img is not None and std_img.numel() > 0:
                    # Ensure image is in [0,1] range (converted from [-1,1])
                    std_img_np = std_img[0].detach().cpu().add(1).mul(0.5).clamp(0, 1).permute(1, 2, 0).numpy()
                    std_img_pil = Image.fromarray((std_img_np * 255).astype(np.uint8))
                    std_img_pil.save(os.path.join(debug_dir, f"std_sample_{i+1}_scale_{scale_idx+1}.png"))
                    save_stats["std"] += 1

            except Exception as e:
                logger.warning(f"       Failed to save std sample {i+1}: {e}")
                import traceback
                logger.warning(f"       Traceback: {traceback.format_exc()}")

            # Save guidance attention visualization
            if guidance_tokens_list is not None:
                try:
                    # Redesign: visualize guidance tokens as attention heatmaps
                    # Design:
                    # 1. Background: full fake tokens reconstruction (no zero-padding)
                    # 2. Heatmap: per-scale attention heatmap from guidance tokens

                    # Use complete fake tokens image as background (no zero-padding)
                    ms_complete_fake_tokens = []
                    for si in range(len(patch_nums)):
                        if si < scale_idx:
                            # Previous scales: use fake_tokens_history
                            if fake_tokens_history is None or si >= len(fake_tokens_history):
                                # Fall back to std_tokens if no history
                                token = std_tokens[si][i:i+1]
                            else:
                                token = fake_tokens_history[si][i:i+1]
                        elif si == scale_idx:
                            # Current scale: use current fake_tokens
                            token = fake_tokens[i:i+1]
                        else:
                            # Future scales: use fake_tokens (avoid zero-padding)
                            token = fake_tokens[si][i:i+1]

                        if token.dim() == 1:
                            token = token.unsqueeze(0)
                        ms_complete_fake_tokens.append(token)

                    # Generate complete fake background image (no zero-padding)
                    fake_base_img = vae.idxBl_to_img(ms_complete_fake_tokens, same_shape=True, last_one=True)

                    if fake_base_img is not None and scale_idx < len(guidance_tokens_list) and guidance_tokens_list[scale_idx] is not None:
                        # Get guidance features for current scale
                        guidance_features = guidance_tokens_list[scale_idx][i:i+1]  # (1, patch_size^2, C)

                        with torch.no_grad():
                            # Compute attention weights: use L2 norm or mean of features as attention intensity
                            B, L_patch, C = guidance_features.shape
                            patch_size = int(L_patch ** 0.5)  # patch_nums[scale_idx]

                            # Method 1: use L2 norm as attention intensity
                            attention_weights = torch.norm(guidance_features, dim=-1)  # (1, L_patch)

                            # Normalize to [0,1]
                            attention_weights = attention_weights.squeeze(0)  # (L_patch,)
                            attention_min = attention_weights.min()
                            attention_max = attention_weights.max()
                            if attention_max > attention_min:
                                attention_weights = (attention_weights - attention_min) / (attention_max - attention_min)
                            else:
                                attention_weights = torch.zeros_like(attention_weights)

                            # Reshape to 2D heatmap
                            attention_heatmap = attention_weights.view(patch_size, patch_size)  # (patch_size, patch_size)

                            # Convert background image to numpy
                            fake_img_np = fake_base_img[0].detach().cpu().add(1).mul(0.5).clamp(0, 1).permute(1, 2, 0).numpy()

                            # Upsample attention heatmap to image size
                            import torch.nn.functional as F
                            import matplotlib.pyplot as plt
                            import matplotlib.cm as cm

                            # Upsample heatmap to image size
                            img_size = fake_img_np.shape[0]  # Assume square image
                            attention_heatmap_upsampled = F.interpolate(
                                attention_heatmap.unsqueeze(0).unsqueeze(0).float(),  # (1, 1, patch_size, patch_size)
                                size=(img_size, img_size),
                                mode='bilinear',
                                align_corners=False
                            ).squeeze(0).squeeze(0)  # (img_size, img_size)

                            # Convert to numpy
                            attention_np = attention_heatmap_upsampled.cpu().numpy()

                            # Use matplotlib heatmap colormap
                            colormap = cm.get_cmap('jet')  # jet colormap: blue (low) -> red (high)
                            attention_colored = colormap(attention_np)[:, :, :3]  # RGB, drop alpha channel

                            # Create overlay image: background + transparency-modulated heatmap
                            # High attention areas: opaque; low attention areas: transparent
                            alpha = attention_np * 0.6  # Max transparency 60%

                            # Blend images
                            overlay_img = fake_img_np * (1 - alpha[:, :, np.newaxis]) + attention_colored * alpha[:, :, np.newaxis]
                            overlay_img = np.clip(overlay_img, 0, 1)

                            # Convert to PIL image and save
                            overlay_img_pil = Image.fromarray((overlay_img * 255).astype(np.uint8))
                            overlay_img_pil.save(os.path.join(debug_dir, f"guidance_overlay_sample_{i+1}_scale_{scale_idx+1}.png"))

                            # Also save pure heatmap for debugging
                            heatmap_img_pil = Image.fromarray((attention_colored * 255).astype(np.uint8))
                            heatmap_img_pil.save(os.path.join(debug_dir, f"guidance_heatmap_sample_{i+1}_scale_{scale_idx+1}.png"))

                            save_stats["guidance"] += 1
                    else:
                        pass

                except Exception as e:
                    logger.warning(f"       Failed to save guidance attention visualization sample {i+1}: {e}")
                    import traceback
                    logger.warning(f"       Traceback: {traceback.format_exc()}")

        except Exception as e:
            logger.warning(f"       Error processing sample {i+1}: {e}")
            import traceback
            logger.warning(f"       Traceback: {traceback.format_exc()}")

    logger.info(f"🔬 Visual debug save complete: Real={save_stats['real']}, Fake={save_stats['fake']}, Std={save_stats['std']}, Guidance={save_stats['guidance']}")

    # Save metadata
    metadata = {
        "epoch": epoch + 1,
        "batch_idx": batch_idx + 1,
        "scale_idx": scale_idx + 1,
        "patch_size": f"{patch_nums[scale_idx]}x{patch_nums[scale_idx]}",
        "save_stats": save_stats,
        "save_time": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(os.path.join(debug_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)


def create_multi_scale_composite_images(
    save_dir: str,
    epoch: int,
    batch_idx: int,
    max_samples: int = 2
):
    """
    Create composite comparison images from saved per-scale images.
    Layout: 10 columns (one per scale), each with 3 rows (fake, real, std).

    Args:
        save_dir: base save directory path
        epoch: epoch number
        batch_idx: batch number
        max_samples: max number of samples
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
        import os

        # Base debug directory path
        base_debug_dir = os.path.join(save_dir, "visual_debug", f"epoch_{epoch+1}", f"batch_{batch_idx+1}")

        if not os.path.exists(base_debug_dir):
            logger.warning(f"Debug directory does not exist: {base_debug_dir}")
            return

        # Get all scale directories
        scale_dirs = []
        for scale_idx in range(1, 11):  # scale_1 to scale_10
            scale_dir = os.path.join(base_debug_dir, f"scale_{scale_idx}")
            if os.path.exists(scale_dir):
                scale_dirs.append((scale_idx, scale_dir))

        if not scale_dirs:
            logger.warning(f"No scale directories found in: {base_debug_dir}")
            return

        # Create composite comparison image for each sample
        for sample_idx in range(1, max_samples + 1):

            # Collect images for all scales
            scale_images = {}  # {scale_idx: {'fake': img, 'real': img, 'std': img}}

            for scale_idx, scale_dir in scale_dirs:
                images = {}

                # Try to load each image type
                for img_type in ['fake', 'real', 'std', 'guidance']:
                    if img_type == 'guidance':
                        # Guidance type uses overlay image
                        img_path = os.path.join(scale_dir, f"guidance_overlay_sample_{sample_idx}_scale_{scale_idx}.png")
                    else:
                        img_path = os.path.join(scale_dir, f"{img_type}_sample_{sample_idx}_scale_{scale_idx}.png")

                    if os.path.exists(img_path):
                        try:
                            img = Image.open(img_path)
                            # Resize for consistency
                            img = img.resize((96, 96), Image.LANCZOS)
                            images[img_type] = img
                        except Exception as e:
                            logger.warning(f"Failed to load image {img_path}: {e}")
                            # Create placeholder image
                            images[img_type] = Image.new('RGB', (96, 96), color='gray')
                    else:
                        # Create placeholder image
                        images[img_type] = Image.new('RGB', (96, 96), color='lightgray')

                scale_images[scale_idx] = images

            if not scale_images:
                logger.warning(f"No images found for sample {sample_idx}")
                continue

            # Create composite comparison image
            try:
                # Calculate canvas size
                img_size = 96
                header_height = 50
                label_height = 20
                num_scales = len(scale_images)

                # Layout: 10 columns (scales), 4 rows (fake, real, std, guidance)
                canvas_width = num_scales * img_size + 50  # extra space for labels
                canvas_height = header_height + label_height + 4 * img_size + 30  # 4 image rows + label space

                # Create canvas
                composite_img = Image.new('RGB', (canvas_width, canvas_height), color='white')
                draw = ImageDraw.Draw(composite_img)

                # Try to load fonts
                try:
                    font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 10)
                    title_font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 14)
                except:
                    font = ImageFont.load_default()
                    title_font = ImageFont.load_default()

                # Draw title
                title = f"Sample {sample_idx} - Multi-Scale Comparison (Epoch {epoch+1}, Batch {batch_idx+1})"
                draw.text((10, 10), title, fill='black', font=title_font)

                # Draw row labels
                row_labels = ['Fake', 'Real', 'Std', 'Guidance']
                for row_idx, label in enumerate(row_labels):
                    y_pos = header_height + label_height + row_idx * img_size + img_size // 2
                    draw.text((5, y_pos), label, fill='black', font=font)

                # Draw column labels and images
                sorted_scales = sorted(scale_images.keys())
                for col_idx, scale_idx in enumerate(sorted_scales):
                    # Draw scale label
                    x_center = 25 + col_idx * img_size + img_size // 2
                    draw.text((x_center - 15, header_height), f"Scale {scale_idx}", fill='black', font=font)

                    # Draw four images for this scale
                    images = scale_images[scale_idx]
                    for row_idx, img_type in enumerate(['fake', 'real', 'std', 'guidance']):
                        if img_type in images:
                            x_pos = 25 + col_idx * img_size
                            y_pos = header_height + label_height + row_idx * img_size
                            composite_img.paste(images[img_type], (x_pos, y_pos))

                # Save composite comparison image
                composite_dir = os.path.join(base_debug_dir, "composite")
                os.makedirs(composite_dir, exist_ok=True)
                composite_path = os.path.join(composite_dir, f"sample_{sample_idx}_multi_scale_comparison.jpg")
                composite_img.save(composite_path, quality=90)

            except Exception as e:
                logger.error(f"Failed to create composite image for sample {sample_idx}: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")

    except Exception as e:
        logger.error(f"Error creating multi-scale composite comparison: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")


def generate_visual_debug_summary(save_dir: str, epoch: int):
    """Generate visual debug summary report.

    Analyzes saved visual debug samples and generates a summary to help understand discriminator behavior.
    """
    try:
        debug_dir = os.path.join(save_dir, "validation", f"epoch_{epoch+1}", "visual_debug")

        if not os.path.exists(debug_dir):
            return

        # Collect debug info files
        debug_info_files = glob.glob(os.path.join(debug_dir, "debug_info_*.json"))

        if not debug_info_files:
            return

        summary = {
            "epoch": epoch + 1,
            "total_debug_sessions": len(debug_info_files),
            "scales_analyzed": [],
            "discriminator_analysis": {
                "samples_saved": {"real": 0, "fake": 0, "std": 0},
                "notes": []
            },
            "image_files": []
        }

        for info_file in debug_info_files:
            try:
                with open(info_file, 'r') as f:
                    info = json.load(f)

                scale_info = f"Scale_{info['scale_idx']}_{info['patch_size']}"
                if scale_info not in summary["scales_analyzed"]:
                    summary["scales_analyzed"].append(scale_info)

                # Accumulate sample statistics
                for sample_type, count in info["samples_saved"].items():
                    summary["discriminator_analysis"]["samples_saved"][sample_type] += count

            except Exception as e:
                logger.warning(f"Failed to process debug info file {info_file}: {e}")

        # Collect image file list
        image_files = glob.glob(os.path.join(debug_dir, "*.png"))
        summary["image_files"] = [os.path.basename(f) for f in image_files]

        # Generate analysis notes
        total_samples = sum(summary["discriminator_analysis"]["samples_saved"].values())
        if total_samples > 0:
            summary["discriminator_analysis"]["notes"].append(
                f"Saved {total_samples} decoded samples for visual debugging"
            )
            summary["discriminator_analysis"]["notes"].append(
                "Check the following files to analyze discriminator collapse:"
            )
            summary["discriminator_analysis"]["notes"].append(
                "- 'real' samples: real images (discriminator positive)"
            )
            summary["discriminator_analysis"]["notes"].append(
                "- 'fake' samples: AID-VAR generated images (discriminator negative)"
            )
            summary["discriminator_analysis"]["notes"].append(
                "- 'std' samples: standard VAR generated images (reference baseline)"
            )
            summary["discriminator_analysis"]["notes"].append(
                "- 'guidance' samples: GuidanceInjector guidance token visualization"
            )
            summary["discriminator_analysis"]["notes"].append(
                "Analysis points: 1) Are fake samples clearly worse than real? 2) Are fake samples too different from std? 3) What planning intent do guidance samples show?"
            )

        # Save summary
        summary_file = os.path.join(debug_dir, f"visual_debug_summary_epoch_{epoch+1}.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    except Exception as e:
        logger.warning(f"🔬 Failed to generate visual debug summary: {e}")


def should_save_visual_debug(
    epoch: int,
    batch_idx: int,
    scale_idx: int,
    acc_real: float,
    acc_fake: float
) -> bool:
    """Determine whether to save visual debug samples.

    Args:
        epoch: current epoch
        batch_idx: current batch index
        scale_idx: current scale index
        acc_real: real sample accuracy
        acc_fake: fake sample accuracy

    Returns:
        Whether to save visual debug samples
    """
    # Save real/fake token images on every validation pass
    return True


# Training state helper functions
def is_warmup_phase(enable_staged_training: bool, step: int, warmup_steps: int) -> bool:
    """Determine whether currently in warmup phase."""
    return enable_staged_training and step < warmup_steps


def format_training_metrics(metrics: Dict[str, float], phase: str = "") -> str:
    """Format training metrics for output."""
    phase_prefix = f"[{phase}] " if phase else ""

    return (f"{phase_prefix}D_loss: {metrics.get('loss_discriminator', 0):.4f}, "
            f"P_loss: {metrics.get('loss_planner', 0):.4f}, "
            f"D_acc: {metrics.get('discriminator_accuracy', 0):.3f}")


def log_discriminator_collapse_warning(epoch: int, acc_real: float, acc_fake: float):
    """Log discriminator collapse warning."""
    if acc_real > 0.95 and acc_fake < 0.05:
        logger.warning(f"🚨 Epoch {epoch+1}: Discriminator collapse detected - always predicts real (Real: {acc_real:.3f}, Fake: {acc_fake:.3f})")
    elif acc_real < 0.05 and acc_fake > 0.95:
        logger.warning(f"🚨 Epoch {epoch+1}: Discriminator collapse detected - always predicts fake (Real: {acc_real:.3f}, Fake: {acc_fake:.3f})")
    elif abs(acc_real - 0.5) < 0.1 and abs(acc_fake - 0.5) < 0.1:
        logger.warning(f"⚠️ Epoch {epoch+1}: Discriminator near random level (Real: {acc_real:.3f}, Fake: {acc_fake:.3f})")


def create_debug_info_dict(
    epoch: int,
    batch_idx: int,
    scale_idx: int,
    patch_size: str,
    success_count: Dict[str, int],
    token_shapes: Dict[str, List[int]]
) -> Dict:
    """Create debug info dictionary."""
    return {
        "epoch": epoch + 1,
        "batch_idx": batch_idx + 1,
        "scale_idx": scale_idx + 1,
        "patch_size": patch_size,
        "samples_saved": success_count,
        "token_shapes": token_shapes,
        "notes": "Visual debugging samples for discriminator collapse analysis"
    }


def ensure_debug_dir_exists(save_dir: str, epoch: int) -> str:
    """Ensure debug directory exists."""
    debug_dir = os.path.join(save_dir, "visual_debug", f"epoch_{epoch+1}")
    os.makedirs(debug_dir, exist_ok=True)
    return debug_dir


# ============================================================================
# Alternating training helper functions
# ============================================================================

class AlternatingTrainingManager:
    """Alternating training manager - manages update strategy for discriminator and GuidanceInjector."""

    def __init__(
        self,
        strategy: str = "adaptive",      # alternating strategy
        warmup_steps: int = 150,         # warmup steps
        guidance_weight_max: float = 0.01, # max guidance weight
        progressive_steps: int = 50,     # progressive training steps
    ):
        self.strategy = strategy
        self.warmup_steps = warmup_steps
        self.guidance_weight_max = guidance_weight_max
        self.progressive_steps = progressive_steps

        # State tracking
        self.step_count = 0
        self.disc_updates = 0
        self.planner_updates = 0
        self.last_metrics = {}

    def should_update_discriminator(self, current_step: int, metrics: Optional[Dict] = None) -> bool:
        """Determine whether to update discriminator at current step."""

        # Warmup phase: only train discriminator
        if current_step < self.warmup_steps:
            return True

        # Joint training phase strategies
        if self.strategy == "simple_1_1":
            # Simple 1:1 alternation
            return (current_step - self.warmup_steps) % 2 == 0

        elif self.strategy == "adaptive":
            # Adaptive strategy: adjust based on performance
            if metrics:
                acc_real = metrics.get('acc_real', 0.5)
                acc_fake = metrics.get('acc_fake', 0.5)

                # If discriminator is too weak, train discriminator more
                if acc_real < 0.6 or acc_fake < 0.6:
                    return (current_step - self.warmup_steps) % 3 != 2  # 2/3 time on discriminator

                # If discriminator is too strong, train GuidanceInjector more
                elif acc_real > 0.9 and acc_fake > 0.9:
                    return (current_step - self.warmup_steps) % 4 == 0  # 1/4 time on discriminator

            # Default 1:1 alternation
            return (current_step - self.warmup_steps) % 2 == 0

        elif self.strategy == "discriminator_priority":
            # 2 out of 3 steps train discriminator
            return (current_step - self.warmup_steps) % 3 != 2

        else:
            # Default simple alternation
            return (current_step - self.warmup_steps) % 2 == 0

    def should_update_planner(self, current_step: int, metrics: Optional[Dict] = None) -> bool:
        """Determine whether to update GuidanceInjector at current step."""

        # Warmup phase: do not train GuidanceInjector
        if current_step < self.warmup_steps:
            return False

        # Joint training phase: mutually exclusive with discriminator
        return not self.should_update_discriminator(current_step, metrics)

    def get_progressive_guidance_weight(self, current_step: int) -> float:
        """Compute guidance weight - modified to use fixed weight."""
        if current_step < self.warmup_steps:
            return 0.0

        # Use fixed guidance weight to avoid discriminator collapse from progressive growth
        return self.guidance_weight_max

        # # Original progressive growth logic (commented out)
        # # Joint training phase: progressive growth
        # steps_since_joint = current_step - self.warmup_steps
        # if steps_since_joint < self.progressive_steps:
        #     # Linear growth to max value
        #     progress = steps_since_joint / self.progressive_steps
        #     return self.guidance_weight_max * progress
        # else:
        #     return self.guidance_weight_max

    def update_step_count(self, updated_discriminator: bool, updated_planner: bool, metrics: Optional[Dict] = None):
        """Update step count statistics."""
        self.step_count += 1
        if updated_discriminator:
            self.disc_updates += 1
        if updated_planner:
            self.planner_updates += 1

        # Save metrics for next decision
        if metrics:
            self.last_metrics = {
                'acc_real': metrics.get('acc_real', 0.5),
                'acc_fake': metrics.get('acc_fake', 0.5),
                'loss_D': metrics.get('loss_D', 0.0),
                'loss_P': metrics.get('loss_P', 0.0)
            }

    def get_training_stats(self) -> Dict:
        """Get training statistics."""
        total_updates = self.disc_updates + self.planner_updates
        return {
            'total_steps': self.step_count,
            'discriminator_updates': self.disc_updates,
            'planner_updates': self.planner_updates,
            'disc_update_ratio': self.disc_updates / max(total_updates, 1),
            'planner_update_ratio': self.planner_updates / max(total_updates, 1),
            'current_strategy': self.strategy,
            'warmup_completed': self.step_count >= self.warmup_steps
        }


def detect_discriminator_collapse(
    acc_real: float,
    acc_fake: float,
    current_step: int,
    collapse_history: List[Dict],
    threshold_real: float = 0.95,
    threshold_fake: float = 0.05
) -> Tuple[bool, bool]:
    """
    Detect discriminator collapse.

    Args:
        acc_real: real sample accuracy
        acc_fake: fake sample accuracy
        current_step: current step count
        collapse_history: collapse history records
        threshold_real: real sample accuracy threshold
        threshold_fake: fake sample accuracy threshold

    Returns:
        (is_collapsed, should_recover): whether collapsed, whether to recover
    """
    # Discriminator collapse: overfits on real samples, fails to identify fake samples
    is_collapsed = acc_real > threshold_real and acc_fake < threshold_fake

    # Record current state
    collapse_history.append({
        'step': current_step,
        'acc_real': acc_real,
        'acc_fake': acc_fake,
        'collapsed': is_collapsed
    })

    # Keep history length bounded
    if len(collapse_history) > 10:
        collapse_history.pop(0)

    # Detect persistent collapse
    recent_collapses = sum(1 for h in collapse_history[-3:] if h['collapsed'])
    is_persistent_collapse = recent_collapses >= 2

    # Detect recovery
    should_recover = acc_real < 0.8 and acc_fake > 0.3

    if is_persistent_collapse:
        logger.warning(f"🚨 Persistent discriminator collapse detected! step={current_step}, consecutive collapses={recent_collapses}")

    return is_collapsed, should_recover


def apply_learning_rate_fix(
    opt_planner,
    opt_disc,
    base_lr: float = 5e-6,
    recovery_mode: bool = False
):
    """
    Apply learning rate fix.

    Args:
        opt_planner: GuidanceInjector optimizer
        opt_disc: discriminator optimizer
        base_lr: base learning rate
        recovery_mode: whether in recovery mode (reduce learning rate)
    """
    target_lr_planner = base_lr
    target_lr_disc = base_lr

    # Recovery mode: reduce learning rate
    if recovery_mode:
        target_lr_planner *= 0.8
        target_lr_disc *= 0.5
        logger.info(f"Recovery mode: reducing lr planner={target_lr_planner:.2e}, disc={target_lr_disc:.2e}")

    # Update learning rates
    for param_group in opt_planner.optimizer.param_groups:
        param_group['lr'] = target_lr_planner
    for param_group in opt_disc.optimizer.param_groups:
        param_group['lr'] = target_lr_disc


class NoOpOptimizer:
    """No-op optimizer wrapper - temporarily disables an optimizer during alternating training."""

    def __init__(self, real_opt):
        self.real_opt = real_opt
        self.amp_ctx = real_opt.amp_ctx
        self.optimizer = real_opt.optimizer  # Maintain interface compatibility

    def backward_clip_step(self, stepping=True, loss=None, retain_graph=False):
        """Run backward pass but do not update parameters."""
        # Allow loss computation to proceed normally, but skip parameter update
        # This preserves gradient computation correctness without modifying parameters
        return


def log_alternating_training_step(
    current_step: int,
    phase: str,
    updated_discriminator: bool,
    updated_planner: bool,
    guidance_weight: float,
    acc_real: float,
    acc_fake: float,
    strategy: str
):
    """Log alternating training step information."""
    pass


def format_alternating_training_metrics(metrics: Dict, alt_manager: AlternatingTrainingManager) -> str:
    """Format alternating training metrics for display."""

    stats = alt_manager.get_training_stats()

    lines = [
        f"Alternating training stats:",
        f"   Total steps: {stats['total_steps']}",
        f"   Discriminator updates: {stats['discriminator_updates']} ({stats['disc_update_ratio']*100:.1f}%)",
        f"   Planner updates: {stats['planner_updates']} ({stats['planner_update_ratio']*100:.1f}%)",
        f"   Strategy: {stats['current_strategy']}",
        f"   Warmup completed: {'yes' if stats['warmup_completed'] else 'no'}",
        f"   Current accuracy: real={metrics.get('acc_real', 0):.3f}, fake={metrics.get('acc_fake', 0):.3f}",
        f"   Guidance weight: {metrics.get('guidance_weight', 0):.4f}"
    ]

    return "\n".join(lines)
