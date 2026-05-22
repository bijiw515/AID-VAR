#!/usr/bin/env python3

import os
import os.path as osp
import torch
import torch.nn.functional as F
import random
import numpy as np
import PIL.Image as PImage
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import argparse
from datetime import datetime
from typing import Optional, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2

setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)

from models import build_vae_var
from models.guidance_injector import GuidanceInjector
from utils.misc import create_npz_from_sample_folder

def parse_args():
    parser = argparse.ArgumentParser(description='Generate 50K AID-VAR guided images with guidance visualization')
    parser.add_argument('--model_depth', type=int, default=16, choices=[16, 20, 24, 30],
                        help='VAR model depth (default: 16)')
    parser.add_argument('--cfg', type=float, default=1.5,
                        help='Classifier-free guidance scale (default: 1.5)')
    parser.add_argument('--top_p', type=float, default=0.96,
                        help='Top-p sampling parameter (default: 0.96)')
    parser.add_argument('--top_k', type=int, default=900,
                        help='Top-k sampling parameter (default: 900)')
    parser.add_argument('--more_smooth', action='store_true', default=False,
                        help='Enable more_smooth for better visual quality')
    parser.add_argument('--output_dir', type=str, default='guidance_visualization',
                        help='Output directory for generated samples and visualizations')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    parser.add_argument('--vae_ckpt', type=str, default='vae_ch160v4096z32.pth',
                        help='VAE checkpoint path')
    parser.add_argument('--var_ckpt', type=str, default=None,
                        help='VAR checkpoint path (auto-determined if None)')
    parser.add_argument('--planner_ckpt', type=str, required=True,
                        help='GuidanceInjector checkpoint path')
    parser.add_argument('--create_npz', action='store_true',
                        help='Create .npz file after generation')
    parser.add_argument('--save_format', type=str, default='viz_only', choices=['png', 'npz', 'both', 'viz_only'],
                        help='Save format: viz_only (guidance visualizations only), png (PNG files), npz (direct NPZ), both (PNG+NPZ) (default: viz_only)')
    parser.add_argument('--save_png_for_npz', action='store_true',
                        help='When using npz format, also save PNG files for visualization')
    parser.add_argument('--dtype', type=str, default='float16', choices=['float16', 'bfloat16'],
                        help='Data type for inference (default: float16)')
    parser.add_argument('--save_guidance_viz', action='store_true', default=True,
                        help='Save guidance visualization images (default: True)')
    parser.add_argument('--guidance_alpha', type=float, default=0.4,
                        help='Transparency of guidance heatmap overlay (default: 0.4)')
    parser.add_argument('--save_original_images', action='store_true',
                        help='Save original generated images (only needed for FID evaluation)')
    return parser.parse_args()

def download_checkpoints(vae_ckpt, var_ckpt):
    hf_home = 'https://huggingface.co/FoundationVision/var/resolve/main'
    
    if not osp.exists(vae_ckpt):
        print(f"Downloading {vae_ckpt}...")
        os.system(f'wget {hf_home}/{vae_ckpt}')
    
    if not osp.exists(var_ckpt):
        print(f"Downloading {var_ckpt}...")
        os.system(f'wget {hf_home}/{var_ckpt}')

def create_guidance_heatmap(guidance_features: torch.Tensor, patch_size: int, target_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    with torch.no_grad():
        B, L_patch, C = guidance_features.shape
        attention_weights = torch.norm(guidance_features, dim=-1).squeeze(0)  # (L_patch,)

        attention_min = attention_weights.min()
        attention_max = attention_weights.max()
        if attention_max > attention_min:
            attention_weights = (attention_weights - attention_min) / (attention_max - attention_min)
        else:
            attention_weights = torch.zeros_like(attention_weights)

        attention_heatmap = attention_weights.view(patch_size, patch_size)  # (patch_size, patch_size)

        heatmap_np = attention_heatmap.cpu().numpy()

        colormap = cm.get_cmap('jet')
        heatmap_colored = colormap(heatmap_np)[:, :, :3]

        heatmap_resized = cv2.resize(heatmap_colored, target_size, interpolation=cv2.INTER_LINEAR)

        heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)

        return heatmap_uint8

def overlay_heatmap_on_image(image_np: np.ndarray, heatmap_np: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    overlay = cv2.addWeighted(image_np, 1-alpha, heatmap_np, alpha, 0)
    return overlay

def create_guidance_visualization_grid(
    original_image: np.ndarray,
    guidance_features_list: List[torch.Tensor],
    scale_decoded_images_list: List[np.ndarray],
    baseline_scale_images_list: List[np.ndarray],
    patch_nums: List[int],
    alpha: float = 0.4
) -> np.ndarray:
    H, W, C = original_image.shape
    num_scales = len(guidance_features_list)

    grid_width = W * num_scales
    grid_height = H * 3
    grid_image = np.zeros((grid_height, grid_width, C), dtype=np.uint8)

    for i, (guidance_features, patch_size) in enumerate(zip(guidance_features_list, patch_nums)):
        start_col = W * i
        end_col = W * (i + 1)

        if i < len(scale_decoded_images_list) and scale_decoded_images_list[i] is not None:
            aid_scale_image = scale_decoded_images_list[i]
        else:
            aid_scale_image = original_image

        if i < len(baseline_scale_images_list) and baseline_scale_images_list[i] is not None:
            baseline_scale_image = baseline_scale_images_list[i]
        else:
            baseline_scale_image = np.zeros_like(aid_scale_image)

        grid_image[H:2*H, start_col:end_col, :] = aid_scale_image

        grid_image[2*H:3*H, start_col:end_col, :] = baseline_scale_image

        if guidance_features is not None:
            heatmap = create_guidance_heatmap(guidance_features, patch_size, (H, W))
            overlay = overlay_heatmap_on_image(aid_scale_image, heatmap, alpha)
            grid_image[:H, start_col:end_col, :] = overlay
        else:
            grid_image[:H, start_col:end_col, :] = aid_scale_image

    return grid_image

def add_scale_labels_to_grid(
    grid_image: np.ndarray,
    patch_nums: List[int],
    image_width: int
) -> np.ndarray:
    grid_height, W_total, C = grid_image.shape
    H = grid_height // 3

    label_height = 35
    row_separator_height = 15
    total_height = label_height + H + row_separator_height + H + row_separator_height + H
    labeled_image = np.ones((total_height, W_total, C), dtype=np.uint8) * 248

    labeled_image[label_height:label_height + H, :, :] = grid_image[:H, :, :]

    row2_start = label_height + H + row_separator_height
    labeled_image[row2_start:row2_start + H, :, :] = grid_image[H:2*H, :, :]

    row3_start = row2_start + H + row_separator_height
    labeled_image[row3_start:row3_start + H, :, :] = grid_image[2*H:3*H, :, :]

    labeled_pil = Image.fromarray(labeled_image)
    draw = ImageDraw.Draw(labeled_pil)

    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        scale_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        try:
            title_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
            scale_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 12)
        except:
            title_font = ImageFont.load_default()
            scale_font = ImageFont.load_default()

    colors = {
        'original': '#2E86C1',
        'scale': '#E74C3C',
        'text': '#2C3E50',
        'bg': '#ECF0F1',
        'border': '#BDC3C7'
    }

    labels_config = [(f"Scale {i+1}", f"Guidance {pn}×{pn}", colors['scale']) for i, pn in enumerate(patch_nums)]

    for i, (title, subtitle, color) in enumerate(labels_config):
        x_center = i * image_width + image_width // 2
        y_start = 8

        scale_text = f"Scale {i+1}"
        title_bbox = draw.textbbox((0, 0), scale_text, font=title_font)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = x_center - title_width // 2

        text_bg_padding = 8
        draw.rectangle([title_x - text_bg_padding, y_start - 3,
                       title_x + title_width + text_bg_padding, y_start + 18],
                      fill=(255, 255, 255, 200), outline=color, width=1)

        draw.text((title_x, y_start), scale_text, fill=color, font=title_font)

        if i < len(labels_config) - 1:
            sep_x = (i + 1) * image_width
            draw.line([sep_x, 0, sep_x, label_height], fill=colors['border'], width=1)

    draw.line([0, label_height - 1, W_total, label_height - 1], fill=colors['border'], width=2)

    row_labels = [
        ("", colors['scale']),
        ("", colors['original']),
        ("", (128, 128, 128))
    ]

    for row_idx, (row_title, row_color) in enumerate(row_labels):
        if row_idx == 0:
            row_y_center = label_height + H // 2
        elif row_idx == 1:
            row_y_center = label_height + H + row_separator_height + H // 2
        else:
            row_y_center = label_height + H + row_separator_height + H + row_separator_height + H // 2

        text_img = Image.new('RGBA', (100, 20), (255, 255, 255, 0))
        text_draw = ImageDraw.Draw(text_img)
        text_draw.text((0, 0), row_title, fill=row_color, font=scale_font)

        text_rotated = text_img.rotate(90, expand=True)

        text_y = row_y_center - text_rotated.height // 2
        labeled_pil.paste(text_rotated, (5, text_y), text_rotated)

    sep1_y = label_height + H + row_separator_height // 2
    draw.line([0, sep1_y - 1, W_total, sep1_y - 1], fill=colors['border'], width=1)
    draw.line([0, sep1_y + 1, W_total, sep1_y + 1], fill=colors['border'], width=1)

    sep2_y = label_height + H + row_separator_height + H + row_separator_height // 2
    draw.line([0, sep2_y - 1, W_total, sep2_y - 1], fill=colors['border'], width=1)
    draw.line([0, sep2_y + 1, W_total, sep2_y + 1], fill=colors['border'], width=1)

    legend_width = 120
    legend_height = 20
    legend_x = W_total - legend_width - 10
    legend_y = 5

    draw.rectangle([legend_x - 5, legend_y - 3, legend_x + legend_width + 5, legend_y + legend_height + 8],
                  fill='white', outline=colors['border'], width=1)

    for j in range(legend_width):
        color_value = j / legend_width
        if color_value < 0.5:
            r = int(255 * (2 * color_value))
            g = int(255 * (2 * color_value))
            b = 255
        else:
            r = 255
            g = int(255 * (2 * (1 - color_value)))
            b = int(255 * (2 * (1 - color_value)))

        gradient_color = (r, g, b)
        draw.line([legend_x + j, legend_y, legend_x + j, legend_y + legend_height],
                 fill=gradient_color)

    draw.text((legend_x, legend_y + legend_height + 2), "Low", fill=colors['text'], font=scale_font)
    draw.text((legend_x + legend_width - 25, legend_y + legend_height + 2), "High", fill=colors['text'], font=scale_font)
    draw.text((legend_x + 25, legend_y - 15), "Guidance Intensity", fill=colors['text'], font=scale_font)

    return np.array(labeled_pil)

def setup_models(args):
    print(f"Setting up AID-VAR models...")

    if args.var_ckpt is None:
        args.var_ckpt = f'checkpoints/var_d{args.model_depth}.pth'

    download_checkpoints(args.vae_ckpt, args.var_ckpt)

    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    device = torch.device(args.device)

    print(f"Building VAE and VAR models (depth={args.model_depth})...")
    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        device=device, patch_nums=patch_nums,
        num_classes=1000, depth=args.model_depth, shared_aln=False,
    )
    
    print(f"Loading VAE checkpoint: {args.vae_ckpt}")
    vae.load_state_dict(torch.load(args.vae_ckpt, map_location='cpu'), strict=True)

    print(f"Loading VAR checkpoint: {args.var_ckpt}")
    var.load_state_dict(torch.load(args.var_ckpt, map_location='cpu'), strict=True)

    vae.eval()
    var.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    for p in var.parameters():
        p.requires_grad_(False)

    print(f"Building GuidanceInjector...")

    var_embed_dim = args.model_depth * 64
    print(f"GuidanceInjector configuration: VAR depth={args.model_depth}, embed_dim={var_embed_dim}")
    
    planner = GuidanceInjector(
        input_dim=var_embed_dim,
        embed_dim=var_embed_dim,
        num_layers=2,
        num_heads=8
    ).to(device)
    
    print(f"Loading GuidanceInjector checkpoint: {args.planner_ckpt}")
    if not osp.exists(args.planner_ckpt):
        raise FileNotFoundError(f"GuidanceInjector checkpoint not found: {args.planner_ckpt}")

    planner_state = torch.load(args.planner_ckpt, map_location='cpu')

    if 'planner_state_dict' in planner_state:
        missing_keys, unexpected_keys = planner.load_state_dict(planner_state['planner_state_dict'], strict=False)
        if unexpected_keys:
            pos_keys = [k for k in unexpected_keys if k.startswith('pos_')]
            other_keys = [k for k in unexpected_keys if not k.startswith('pos_')]
            if pos_keys:
                print(f"Ignoring positional encoding buffers: {pos_keys}")
            if other_keys:
                print(f"Warning: unexpected keys: {other_keys}")
        if missing_keys:
            print(f"Warning: missing keys: {missing_keys}")
        print("GuidanceInjector loaded from training checkpoint.")
    elif 'planner' in planner_state:
        planner.load_state_dict(planner_state['planner'], strict=True)
        print("GuidanceInjector loaded from legacy checkpoint.")
    else:
        planner.load_state_dict(planner_state, strict=True)
        print("GuidanceInjector weights loaded directly.")
    
    planner.eval()
    for p in planner.parameters():
        p.requires_grad_(False)

    print(f"AID-VAR models loaded successfully!")
    print(f"  VAE parameters: {sum(p.numel() for p in vae.parameters()):,}")
    print(f"  VAR parameters: {sum(p.numel() for p in var.parameters()):,}")
    print(f"  GuidanceInjector parameters: {sum(p.numel() for p in planner.parameters()):,}")
    
    return vae, var, planner, device

def baseline_var_inference_with_scale_extraction(
    var_model,
    B: int,
    label_B,
    cfg=1.5,
    top_k=0,
    top_p=0.0,
    g_seed: Optional[int] = None,
    more_smooth=False
) -> Tuple[torch.Tensor, List[np.ndarray]]:
    if g_seed is None:
        rng = None
    else:
        var_model.rng.manual_seed(g_seed)
        rng = var_model.rng

    if label_B is None:
        label_B = torch.multinomial(var_model.uniform_prob, num_samples=B, replacement=True, generator=rng).reshape(B)
    elif isinstance(label_B, int):
        label_B = torch.full((B,), fill_value=var_model.num_classes if label_B < 0 else label_B, device=var_model.lvl_1L.device)

    sos = cond_BD = var_model.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=var_model.num_classes)), dim=0))

    lvl_pos = var_model.lvl_embed(var_model.lvl_1L) + var_model.pos_1LC
    next_token_map = sos.unsqueeze(1).expand(2 * B, var_model.first_l, -1) + \
                    var_model.pos_start.expand(2 * B, var_model.first_l, -1) + \
                    lvl_pos[:, :var_model.first_l]

    cur_L = 0
    f_hat = sos.new_zeros(B, var_model.Cvae, var_model.patch_nums[-1], var_model.patch_nums[-1])

    for b in var_model.blocks:
        b.attn.kv_caching(True)

    baseline_scale_decoded_images_list = []

    try:
        for si, pn in enumerate(var_model.patch_nums):
            ratio = si / var_model.num_stages_minus_1
            cur_L += pn * pn

            cond_BD_or_gss = var_model.shared_ada_lin(cond_BD)
            x = next_token_map

            for b in var_model.blocks:
                x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)

            logits_BlV = var_model.get_logits(x, cond_BD)

            t = cfg * ratio
            logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]

            from models.helpers import sample_with_top_k_top_p_
            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]

            if not more_smooth:
                h_BChw = var_model.vae_quant_proxy[0].embedding(idx_Bl)   # B, l, Cvae
            else:
                from models.helpers import gumbel_softmax_with_rng
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)
                h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ var_model.vae_quant_proxy[0].embedding.weight.unsqueeze(0)

            h_BChw = h_BChw.transpose_(1, 2).reshape(B, var_model.Cvae, pn, pn)
            f_hat, next_token_map = var_model.vae_quant_proxy[0].get_next_autoregressive_input(si, len(var_model.patch_nums), f_hat, h_BChw)

            try:
                current_scale_image = var_model.vae_proxy[0].fhat_to_img(f_hat.clone())  # (B, 3, H, W) in [-1, 1]
                current_scale_image = current_scale_image.add_(1).mul_(0.5)
                current_scale_image = torch.clamp(current_scale_image, 0, 1)

                scale_imgs_batch = []
                for sample_idx in range(current_scale_image.shape[0]):
                    scale_img_np = (current_scale_image[sample_idx].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    scale_imgs_batch.append(scale_img_np)
                baseline_scale_decoded_images_list.append(scale_imgs_batch)

            except Exception as e:
                print(f"Warning: Failed to decode baseline image at scale {si}: {e}")
                baseline_scale_decoded_images_list.append(None)

            if si != var_model.num_stages_minus_1:
                next_token_map = next_token_map.view(B, var_model.Cvae, -1).transpose(1, 2)
                next_token_map = var_model.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + var_model.patch_nums[si+1] ** 2]
                next_token_map = next_token_map.repeat(2, 1, 1)

        for b in var_model.blocks:
            b.attn.kv_caching(False)

        final_images = var_model.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)

        return final_images, baseline_scale_decoded_images_list

    except Exception as e:
        print(f"Baseline VAR inference failed: {e}")
        for b in var_model.blocks:
            b.attn.kv_caching(False)

        final_images = var_model.autoregressive_infer_cfg(B=B, label_B=label_B, cfg=cfg, top_k=top_k, top_p=top_p, g_seed=g_seed, more_smooth=more_smooth)
        empty_baseline_images = [None] * len(var_model.patch_nums)
        return final_images, empty_baseline_images

def aid_guided_inference_with_guidance_extraction(
    var_model,
    planner,
    B: int,
    label_B,
    cfg=1.5,
    top_k=0,
    top_p=0.0,
    g_seed: Optional[int] = None,
    more_smooth=False
) -> Tuple[torch.Tensor, List[torch.Tensor], List[np.ndarray], List[np.ndarray]]:
    if g_seed is None:
        rng = None
    else:
        var_model.rng.manual_seed(g_seed)
        rng = var_model.rng

    if label_B is None:
        label_B = torch.multinomial(var_model.uniform_prob, num_samples=B, replacement=True, generator=rng).reshape(B)
    elif isinstance(label_B, int):
        label_B = torch.full((B,), fill_value=var_model.num_classes if label_B < 0 else label_B, device=var_model.lvl_1L.device)

    sos = cond_BD = var_model.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=var_model.num_classes)), dim=0))

    lvl_pos = var_model.lvl_embed(var_model.lvl_1L) + var_model.pos_1LC
    next_token_map = sos.unsqueeze(1).expand(2 * B, var_model.first_l, -1) + \
                    var_model.pos_start.expand(2 * B, var_model.first_l, -1) + \
                    lvl_pos[:, :var_model.first_l]

    cur_L = 0
    f_hat = sos.new_zeros(B, var_model.Cvae, var_model.patch_nums[-1], var_model.patch_nums[-1])

    for b in var_model.blocks:
        b.attn.kv_caching(True)

    generated_tokens_per_scale = []

    guidance_features_list = []

    scale_decoded_images_list = []

    try:
        for si, pn in enumerate(var_model.patch_nums):
            ratio = si / var_model.num_stages_minus_1
            cur_L += pn * pn

            guidance_tokens = None
            guidance_features = None

            if si > 0 and len(generated_tokens_per_scale) > 0:
                prev_tokens = generated_tokens_per_scale[-1]

                try:
                    prev_embed = F.embedding(prev_tokens, var_model.vae_quant_proxy[0].embedding.weight.detach())
                    prev_features = var_model.word_embed(prev_embed)  # (B, L_prev, C)

                    prev_features = torch.clamp(prev_features, min=-10.0, max=10.0)
                    prev_features = prev_features.detach().clone().requires_grad_(False)

                    guidance_tokens = planner(prev_features, target_patch_num=pn)  # (B, pn*pn, C)

                    guidance_features = guidance_tokens.clone().detach()  # (B, pn*pn, C)

                    if guidance_tokens is not None:
                        guidance_tokens = torch.clamp(guidance_tokens, min=-5.0, max=5.0)

                        guidance_tokens = torch.cat([guidance_tokens, guidance_tokens], dim=0)  # (2*B, pn*pn, C)

                except Exception as e:
                    print(f"Warning: Failed to generate guidance for scale {si}: {e}")
                    guidance_tokens = None
                    guidance_features = None

            guidance_features_list.append(guidance_features)

            cond_BD_or_gss = var_model.shared_ada_lin(cond_BD)
            x = next_token_map

            if guidance_tokens is not None:
                if guidance_tokens.shape[0] == x.shape[0] and guidance_tokens.shape[1] == x.shape[1]:
                    x = x + 0.001 * guidance_tokens
                else:
                    print(f"Warning: Guidance token shape mismatch at scale {si}: {guidance_tokens.shape} vs {x.shape}")

            for b in var_model.blocks:
                x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)

            logits_BlV = var_model.get_logits(x, cond_BD)

            t = cfg * ratio
            logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]

            from models.helpers import sample_with_top_k_top_p_
            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
            generated_tokens_per_scale.append(idx_Bl)

            if not more_smooth:  # default case
                h_BChw = var_model.vae_quant_proxy[0].embedding(idx_Bl)   # B, l, Cvae
            else:   # smooth sampling for visualization
                from models.helpers import gumbel_softmax_with_rng
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)   # reference mask-git
                h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ var_model.vae_quant_proxy[0].embedding.weight.unsqueeze(0)

            h_BChw = h_BChw.transpose_(1, 2).reshape(B, var_model.Cvae, pn, pn)
            f_hat, next_token_map = var_model.vae_quant_proxy[0].get_next_autoregressive_input(si, len(var_model.patch_nums), f_hat, h_BChw)

            try:
                current_scale_image = var_model.vae_proxy[0].fhat_to_img(f_hat.clone())  # (B, 3, H, W) in [-1, 1]
                current_scale_image = current_scale_image.add_(1).mul_(0.5)
                current_scale_image = torch.clamp(current_scale_image, 0, 1)

                scale_imgs_batch = []
                for sample_idx in range(current_scale_image.shape[0]):
                    scale_img_np = (current_scale_image[sample_idx].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    scale_imgs_batch.append(scale_img_np)
                scale_decoded_images_list.append(scale_imgs_batch)

            except Exception as e:
                print(f"Warning: Failed to decode image at scale {si}: {e}")
                scale_decoded_images_list.append(None)

            if si != var_model.num_stages_minus_1:
                next_token_map = next_token_map.view(B, var_model.Cvae, -1).transpose(1, 2)
                next_token_map = var_model.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + var_model.patch_nums[si+1] ** 2]
                next_token_map = next_token_map.repeat(2, 1, 1)   # CFG: double batch size

        for b in var_model.blocks:
            b.attn.kv_caching(False)

        final_images = var_model.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)   # de-normalize, from [-1, 1] to [0, 1]

        _, baseline_scale_decoded_images_list = baseline_var_inference_with_scale_extraction(
            var_model=var_model,
            B=B,
            label_B=label_B,
            cfg=cfg,
            top_k=top_k,
            top_p=top_p,
            g_seed=g_seed,
            more_smooth=more_smooth
        )

        return final_images, guidance_features_list, scale_decoded_images_list, baseline_scale_decoded_images_list

    except Exception as e:
        # Fallback on error: use standard VAR inference
        print(f"AID-VAR inference failed: {e}, falling back to standard VAR")
        for b in var_model.blocks:
            b.attn.kv_caching(False)

        final_images = var_model.autoregressive_infer_cfg(B=B, label_B=label_B, cfg=cfg, top_k=top_k, top_p=top_p, g_seed=g_seed, more_smooth=more_smooth)
        empty_guidance = [None] * len(var_model.patch_nums)
        empty_scale_images = [None] * len(var_model.patch_nums)
        empty_baseline_images = [None] * len(var_model.patch_nums)
        return final_images, empty_guidance, empty_scale_images, empty_baseline_images

def generate_guidance_visualization_samples(var, planner, device, args):
    print(f"\nGenerating 50,000 AID-VAR guidance visualizations...")
    print(f"Parameters: cfg={args.cfg}, top_p={args.top_p}, top_k={args.top_k}, more_smooth={args.more_smooth}")
    print(f"Save format: {args.save_format}")
    if args.save_format == 'viz_only':
        print(f"Focus: saving guidance visualizations only, no original images")
    else:
        print(f"Save original images: {args.save_original_images}")
    print(f"Guidance visualization: {args.save_guidance_viz}")
    print(f"Guidance alpha: {args.guidance_alpha}")

    batch_size = 10
    samples_per_class = 10

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sample_folder = f"{args.output_dir}_d{args.model_depth}_{timestamp}"
    os.makedirs(sample_folder, exist_ok=True)

    if args.save_guidance_viz:
        guidance_viz_folder = os.path.join(sample_folder, "guidance_visualizations")
        os.makedirs(guidance_viz_folder, exist_ok=True)
        print(f"Saving guidance visualizations to: {guidance_viz_folder}")

    print(f"Saving AID-VAR samples to: {sample_folder}")

    if args.save_format in ['npz', 'both']:
        all_samples = []
    else:
        all_samples = None

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    tf32 = True
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.set_float32_matmul_precision('high' if tf32 else 'highest')

    if args.dtype == 'float16':
        dtype = torch.float16
    else:
        dtype = torch.bfloat16

    total_samples = 0
    num_classes = 1000
    patch_nums = var.patch_nums

    for class_id in tqdm(range(num_classes), desc="Generating AID-VAR classes with guidance viz"):
        torch.manual_seed(class_id)
        random.seed(class_id)
        np.random.seed(class_id)

        class_labels = [class_id] * samples_per_class
        current_batch_size = batch_size
        current_labels = class_labels

        label_B = torch.tensor(current_labels, device=device)

        with torch.inference_mode():
            with torch.autocast(device.type, enabled=True, dtype=dtype, cache_enabled=True):
                recon_B3HW, guidance_features_list, scale_decoded_images_list, baseline_scale_decoded_images_list = aid_guided_inference_with_guidance_extraction(
                    var_model=var,
                    planner=planner,
                    B=current_batch_size,
                    label_B=label_B,
                    cfg=args.cfg,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    g_seed=class_id,
                    more_smooth=args.more_smooth
                )

                for i in range(current_batch_size):
                    img_tensor = recon_B3HW[i].cpu()
                    img_tensor = torch.clamp(img_tensor, 0, 1)

                    if args.save_guidance_viz:
                        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

                        sample_guidance_features = []
                        for scale_idx, guidance_features in enumerate(guidance_features_list):
                            if guidance_features is not None:
                                sample_guidance = guidance_features[i:i+1]  # (1, pn*pn, C)
                                sample_guidance_features.append(sample_guidance)
                            else:
                                sample_guidance_features.append(None)

                        valid_patch_nums = [pn for pn, gf in zip(patch_nums[1:], sample_guidance_features[1:]) if gf is not None]
                        valid_guidance = [gf for gf in sample_guidance_features[1:] if gf is not None]

                        if valid_guidance:
                            valid_aid_scale_images = []
                            valid_baseline_scale_images = []

                            for scale_idx in range(len(valid_guidance)):
                                actual_scale_idx = scale_idx + 1

                                if (actual_scale_idx < len(scale_decoded_images_list) and
                                    scale_decoded_images_list[actual_scale_idx] is not None and
                                    i < len(scale_decoded_images_list[actual_scale_idx])):
                                    valid_aid_scale_images.append(scale_decoded_images_list[actual_scale_idx][i])
                                else:
                                    valid_aid_scale_images.append(None)

                                if (actual_scale_idx < len(baseline_scale_decoded_images_list) and
                                    baseline_scale_decoded_images_list[actual_scale_idx] is not None and
                                    i < len(baseline_scale_decoded_images_list[actual_scale_idx])):
                                    valid_baseline_scale_images.append(baseline_scale_decoded_images_list[actual_scale_idx][i])
                                else:
                                    valid_baseline_scale_images.append(None)

                            viz_grid = create_guidance_visualization_grid(
                                original_image=img_np,
                                guidance_features_list=valid_guidance,
                                scale_decoded_images_list=valid_aid_scale_images,
                                baseline_scale_images_list=valid_baseline_scale_images,
                                patch_nums=valid_patch_nums,
                                alpha=args.guidance_alpha
                            )

                            viz_grid_labeled = add_scale_labels_to_grid(
                                viz_grid,
                                valid_patch_nums,
                                img_np.shape[1]
                            )

                            viz_path = os.path.join(guidance_viz_folder, f"guidance_viz_{total_samples:05d}.png")
                            viz_pil = Image.fromarray(viz_grid_labeled)
                            viz_pil.save(viz_path, "PNG")

                    if args.save_format in ['npz', 'both'] and all_samples is not None:
                        img_np_float = img_tensor.permute(1, 2, 0).numpy().astype(np.float32)
                        all_samples.append(img_np_float)

                    if (args.save_format in ['png', 'both'] or args.save_png_for_npz or args.save_original_images):
                        if not args.save_guidance_viz:
                            img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                        img_pil = PImage.fromarray(img_np)

                        img_path = os.path.join(sample_folder, f"aid_sample_{total_samples:05d}.png")
                        img_pil.save(img_path, "PNG")

                    total_samples += 1

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if (class_id + 1) % 100 == 0:
            print(f"Completed {class_id + 1}/{num_classes} classes, {total_samples} total AID-VAR samples")
            if args.save_guidance_viz:
                print(f"   Guidance visualizations: {total_samples} saved")

    print(f"\nAID-VAR generation with guidance visualization completed! Total samples: {total_samples}")

    if args.save_format in ['npz', 'both'] and all_samples is not None:
        print(f"Saving direct NPZ file (no PNG compression loss)...")
        all_samples_array = np.stack(all_samples, axis=0)  # Shape: (50000, H, W, C)
        npz_path = f"{sample_folder}_aid_guidance.npz"

        # Save in standard FID evaluation format
        np.savez_compressed(npz_path,
                          arr_0=all_samples_array,
                          samples=all_samples_array)  # Compatibility alias

        print(f"Direct AID-VAR NPZ file saved: {npz_path}")
        print(f"   Shape: {all_samples_array.shape}")
        print(f"   Dtype: {all_samples_array.dtype}")
        print(f"   Value range: [{all_samples_array.min():.3f}, {all_samples_array.max():.3f}]")

        # Free memory
        del all_samples_array
        del all_samples

    print(f"AID-VAR samples folder: {sample_folder}")
    if args.save_guidance_viz:
        print(f"Guidance visualizations folder: {guidance_viz_folder}")

    return sample_folder

def main():
    args = parse_args()

    print("="*80)
    print("AID-VAR Guidance Visualization Sample Generation")
    print("="*80)
    print(f"Model depth: {args.model_depth}")
    print(f"Batch size: 50 (hardcoded)")
    print(f"Samples per class: 50 (hardcoded)")
    print(f"Sampling parameters: cfg={args.cfg}, top_p={args.top_p}, top_k={args.top_k}")
    print(f"More smooth: {args.more_smooth}")
    print(f"Random seed: class_id for each class (reproducible)")
    print(f"Save format: {args.save_format}")
    print(f"Output directory: {args.output_dir}")
    print(f"GuidanceInjector checkpoint: {args.planner_ckpt}")
    print(f"Device: {args.device}")
    print(f"Guidance visualization: {args.save_guidance_viz}")
    print(f"Guidance alpha: {args.guidance_alpha}")
    print("="*80)

    try:
        vae, var, planner, device = setup_models(args)

        sample_folder = generate_guidance_visualization_samples(var, planner, device, args)

        if args.create_npz:
            print(f"\nCreating .npz file...")
            create_npz_from_sample_folder(sample_folder)
            print(f"NPZ file created: {sample_folder}.npz")
        else:
            print(f"\nTo create .npz file later, run:")
            print(f"python -c \"from utils.misc import create_npz_from_sample_folder; create_npz_from_sample_folder('{sample_folder}')\"")

        print(f"\nAID-VAR guidance visualization generation complete!")
        print(f"Sample folder: {sample_folder}")
        if args.create_npz:
            print(f"NPZ file: {sample_folder}.npz")
        if args.save_guidance_viz:
            print(f"Guidance visualizations: {sample_folder}/guidance_visualizations/")

        print(f"\nVisualization layout (3-row comparison):")
        print(f"  Row 1: AID-VAR decoded image + guidance heatmap overlay")
        print(f"  Row 2: AID-VAR per-scale decoded images (clean)")
        print(f"  Row 3: Baseline VAR per-scale decoded images (same seed)")
        print(f"Heatmap: red = high attention, blue = low attention")

    except Exception as e:
        print(f"Error during AID-VAR guidance visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 