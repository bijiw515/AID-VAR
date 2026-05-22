#!/usr/bin/env python3
"""
AID-VAR Guidance可视化脚本
基于generate_aid_fid_samples.py，生成50K张图片的同时
提取和可视化每个尺度的GuidanceInjector guidance feature map
将guidance feature map以热力图形式叠加在生成的图片上
"""

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

# 禁用默认参数初始化以加速
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
                        help='Save format: viz_only (只保存guidance可视化), png (PNG files), npz (direct NPZ), both (PNG+NPZ) (default: viz_only)')
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
    """下载模型检查点如果不存在"""
    hf_home = 'https://huggingface.co/FoundationVision/var/resolve/main'
    
    if not osp.exists(vae_ckpt):
        print(f"Downloading {vae_ckpt}...")
        os.system(f'wget {hf_home}/{vae_ckpt}')
    
    if not osp.exists(var_ckpt):
        print(f"Downloading {var_ckpt}...")
        os.system(f'wget {hf_home}/{var_ckpt}')

def create_guidance_heatmap(guidance_features: torch.Tensor, patch_size: int, target_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """
    将guidance features转换为热力图
    
    Args:
        guidance_features: (1, patch_size^2, C) guidance特征
        patch_size: patch的尺度大小
        target_size: 目标图像尺寸 (H, W)
    
    Returns:
        热力图 numpy数组 (H, W, 3) in [0, 255]
    """
    with torch.no_grad():
        # 计算attention权重：使用L2范数作为强度指标
        B, L_patch, C = guidance_features.shape
        attention_weights = torch.norm(guidance_features, dim=-1).squeeze(0)  # (L_patch,)
        
        # 归一化到[0,1]
        attention_min = attention_weights.min()
        attention_max = attention_weights.max()
        if attention_max > attention_min:
            attention_weights = (attention_weights - attention_min) / (attention_max - attention_min)
        else:
            attention_weights = torch.zeros_like(attention_weights)
        
        # 重塑为2D
        attention_heatmap = attention_weights.view(patch_size, patch_size)  # (patch_size, patch_size)
        
        # 转换为numpy
        heatmap_np = attention_heatmap.cpu().numpy()
        
        # 使用matplotlib的热力图colormap
        colormap = cm.get_cmap('jet')  # 红色表示高attention，蓝色表示低attention
        heatmap_colored = colormap(heatmap_np)[:, :, :3]  # 去掉alpha通道
        
        # 缩放到目标尺寸
        heatmap_resized = cv2.resize(heatmap_colored, target_size, interpolation=cv2.INTER_LINEAR)
        
        # 转换为[0, 255]
        heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
        
        return heatmap_uint8

def overlay_heatmap_on_image(image_np: np.ndarray, heatmap_np: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    将热力图叠加到原图像上
    
    Args:
        image_np: 原图像 (H, W, 3) in [0, 255]
        heatmap_np: 热力图 (H, W, 3) in [0, 255]
        alpha: 热力图透明度
    
    Returns:
        叠加后的图像 (H, W, 3) in [0, 255]
    """
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
    """
    创建guidance可视化网格：三行显示
    第一行：热力图叠加到AID-VAR解码图像上
    第二行：AID-VAR各个尺度的纯净解码图像
    第三行：普通VAR各个尺度的解码图像（对比）
    
    Args:
        original_image: 原始生成的完整图像 (H, W, 3) in [0, 255]
        guidance_features_list: 每个尺度的guidance features
        scale_decoded_images_list: AID-VAR每个尺度解码出的图像列表
        baseline_scale_images_list: 普通VAR每个尺度解码出的图像列表
        patch_nums: 每个尺度的patch数量
        alpha: 热力图透明度
        
    Returns:
        可视化网格图像 (3*H, W_total, 3) in [0, 255] - 三行显示
    """
    H, W, C = original_image.shape
    num_scales = len(guidance_features_list)
    
    # 创建网格：三行，每行有 num_scales 列
    grid_width = W * num_scales
    grid_height = H * 3  # 三行
    grid_image = np.zeros((grid_height, grid_width, C), dtype=np.uint8)
    
    # 第一行：热力图叠加到AID-VAR解码图像上
    # 第二行：AID-VAR纯净的每个尺度解码图像
    # 第三行：普通VAR每个尺度解码图像
    for i, (guidance_features, patch_size) in enumerate(zip(guidance_features_list, patch_nums)):
        # 列的位置
        start_col = W * i
        end_col = W * (i + 1)
        
        # 获取当前尺度的AID-VAR解码图像
        if i < len(scale_decoded_images_list) and scale_decoded_images_list[i] is not None:
            aid_scale_image = scale_decoded_images_list[i]
        else:
            # 如果没有解码图像，使用原图作为fallback
            aid_scale_image = original_image
        
        # 获取当前尺度的普通VAR解码图像
        if i < len(baseline_scale_images_list) and baseline_scale_images_list[i] is not None:
            baseline_scale_image = baseline_scale_images_list[i]
        else:
            # 如果没有基线图像，使用黑色背景
            baseline_scale_image = np.zeros_like(aid_scale_image)
        
        # 第二行：AID-VAR纯净的尺度解码图像
        grid_image[H:2*H, start_col:end_col, :] = aid_scale_image
        
        # 第三行：普通VAR尺度解码图像
        grid_image[2*H:3*H, start_col:end_col, :] = baseline_scale_image
        
        # 第一行：热力图叠加到AID-VAR解码图像上
        if guidance_features is not None:
            # 创建热力图
            heatmap = create_guidance_heatmap(guidance_features, patch_size, (H, W))
            
            # 叠加到AID-VAR解码图像上
            overlay = overlay_heatmap_on_image(aid_scale_image, heatmap, alpha)
            
            # 第一行放置叠加后的图像
            grid_image[:H, start_col:end_col, :] = overlay
        else:
            # 如果没有guidance，第一行也使用AID-VAR解码图像
            grid_image[:H, start_col:end_col, :] = aid_scale_image
    
    return grid_image

def add_scale_labels_to_grid(
    grid_image: np.ndarray, 
    patch_nums: List[int], 
    image_width: int
) -> np.ndarray:
    """
    在可视化网格上添加美观的尺度标签（适配三行布局）
    
    Args:
        grid_image: 网格图像 (3*H, W_total, 3) - 三行布局
        patch_nums: 每个尺度的patch数量  
        image_width: 单个图像的宽度
        
    Returns:
        添加标签后的图像
    """
    grid_height, W_total, C = grid_image.shape
    H = grid_height // 3  # 单行图像高度
    
    # 创建一个带标签栏的新图像 (顶部增加标签区域，中间增加行分隔)
    label_height = 35  # 简化后的标签区域高度
    row_separator_height = 15  # 行间分隔区域高度
    total_height = label_height + H + row_separator_height + H + row_separator_height + H
    labeled_image = np.ones((total_height, W_total, C), dtype=np.uint8) * 248  # 浅灰色背景
    
    # 放置第一行图像（热力图叠加）
    labeled_image[label_height:label_height + H, :, :] = grid_image[:H, :, :]
    
    # 放置第二行图像（AID-VAR纯净解码图像）
    row2_start = label_height + H + row_separator_height
    labeled_image[row2_start:row2_start + H, :, :] = grid_image[H:2*H, :, :]
    
    # 放置第三行图像（普通VAR解码图像）
    row3_start = row2_start + H + row_separator_height
    labeled_image[row3_start:row3_start + H, :, :] = grid_image[2*H:3*H, :, :]
    
    # 转换为PIL图像进行绘制
    labeled_pil = Image.fromarray(labeled_image)
    draw = ImageDraw.Draw(labeled_pil)
    
    # 尝试加载更好的字体
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
    
    # 定义颜色方案
    colors = {
        'original': '#2E86C1',      # 蓝色
        'scale': '#E74C3C',         # 红色
        'text': '#2C3E50',          # 深灰色
        'bg': '#ECF0F1',            # 浅灰色
        'border': '#BDC3C7'         # 边框灰色
    }
    
    # 标签配置 - 只为尺度配置，不包含原图
    labels_config = [(f"Scale {i+1}", f"Guidance {pn}×{pn}", colors['scale']) for i, pn in enumerate(patch_nums)]
    
    # 绘制简化的列标签（尺度标签）- 只在顶部显示尺度信息
    for i, (title, subtitle, color) in enumerate(labels_config):
        # 计算位置
        x_center = i * image_width + image_width // 2
        y_start = 8
        
        # 绘制简化的标题（只显示尺度编号）
        scale_text = f"Scale {i+1}"
        title_bbox = draw.textbbox((0, 0), scale_text, font=title_font)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = x_center - title_width // 2
        
        # 绘制文字背景（半透明）
        text_bg_padding = 8
        draw.rectangle([title_x - text_bg_padding, y_start - 3, 
                       title_x + title_width + text_bg_padding, y_start + 18], 
                      fill=(255, 255, 255, 200), outline=color, width=1)
        
        # 绘制标题文字
        draw.text((title_x, y_start), scale_text, fill=color, font=title_font)
        
        # 添加简单的分隔线（除了最后一个）
        if i < len(labels_config) - 1:
            sep_x = (i + 1) * image_width
            draw.line([sep_x, 0, sep_x, label_height], fill=colors['border'], width=1)
    
    # 添加底部分隔线
    draw.line([0, label_height - 1, W_total, label_height - 1], fill=colors['border'], width=2)
    
    # 添加简化的行标签（在左侧边缘）
    row_labels = [
        ("", colors['scale']),    # 第一行：AID-VAR + guidance热力图
        ("", colors['original']),   # 第二行：AID-VAR纯净
        ("", (128, 128, 128))       # 第三行：普通VAR
    ]
    
    # 绘制简化的行标签（最小侵入）
    for row_idx, (row_title, row_color) in enumerate(row_labels):
        # 计算行的Y位置
        if row_idx == 0:
            # 第一行：热力图叠加
            row_y_center = label_height + H // 2
        elif row_idx == 1:
            # 第二行：AID-VAR解码图像
            row_y_center = label_height + H + row_separator_height + H // 2
        else:
            # 第三行：普通VAR解码图像
            row_y_center = label_height + H + row_separator_height + H + row_separator_height + H // 2
        
        # 绘制简单的行标签（竖直文字）
        # 创建竖直文字
        text_img = Image.new('RGBA', (100, 20), (255, 255, 255, 0))
        text_draw = ImageDraw.Draw(text_img)
        text_draw.text((0, 0), row_title, fill=row_color, font=scale_font)
        
        # 旋转90度
        text_rotated = text_img.rotate(90, expand=True)
        
        # 计算位置并粘贴
        text_y = row_y_center - text_rotated.height // 2
        labeled_pil.paste(text_rotated, (5, text_y), text_rotated)
    
    # 添加中间分隔线
    # 第一条分隔线（第一行和第二行之间）
    sep1_y = label_height + H + row_separator_height // 2
    draw.line([0, sep1_y - 1, W_total, sep1_y - 1], fill=colors['border'], width=1)
    draw.line([0, sep1_y + 1, W_total, sep1_y + 1], fill=colors['border'], width=1)
    
    # 第二条分隔线（第二行和第三行之间）
    sep2_y = label_height + H + row_separator_height + H + row_separator_height // 2
    draw.line([0, sep2_y - 1, W_total, sep2_y - 1], fill=colors['border'], width=1)
    draw.line([0, sep2_y + 1, W_total, sep2_y + 1], fill=colors['border'], width=1)
    
    # 在右上角添加热力图图例
    legend_width = 120
    legend_height = 20
    legend_x = W_total - legend_width - 10
    legend_y = 5
    
    # 绘制图例背景
    draw.rectangle([legend_x - 5, legend_y - 3, legend_x + legend_width + 5, legend_y + legend_height + 8], 
                  fill='white', outline=colors['border'], width=1)
    
    # 绘制渐变条
    for j in range(legend_width):
        color_value = j / legend_width
        # 从蓝色渐变到红色（jet colormap近似）
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
    
    # 图例标签
    draw.text((legend_x, legend_y + legend_height + 2), "Low", fill=colors['text'], font=scale_font)
    draw.text((legend_x + legend_width - 25, legend_y + legend_height + 2), "High", fill=colors['text'], font=scale_font)
    draw.text((legend_x + 25, legend_y - 15), "Guidance Intensity", fill=colors['text'], font=scale_font)
    
    return np.array(labeled_pil)

def setup_models(args):
    """设置VAE、VAR和GuidanceInjector模型"""
    print(f"Setting up AID-VAR models...")
    
    # 确定VAR检查点路径
    if args.var_ckpt is None:
        args.var_ckpt = f'checkpoints/var_d{args.model_depth}.pth'
    
    # 下载基础模型检查点
    download_checkpoints(args.vae_ckpt, args.var_ckpt)
    
    # 构建模型
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    device = torch.device(args.device)
    
    print(f"Building VAE and VAR models (depth={args.model_depth})...")
    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,    # VQVAE超参数
        device=device, patch_nums=patch_nums,
        num_classes=1000, depth=args.model_depth, shared_aln=False,
    )
    
    # 加载VAE和VAR检查点
    print(f"Loading VAE checkpoint: {args.vae_ckpt}")
    vae.load_state_dict(torch.load(args.vae_ckpt, map_location='cpu'), strict=True)
    
    print(f"Loading VAR checkpoint: {args.var_ckpt}")
    var.load_state_dict(torch.load(args.var_ckpt, map_location='cpu'), strict=True)
    
    # 设置为评估模式并冻结参数
    vae.eval()
    var.eval()
    for p in vae.parameters(): 
        p.requires_grad_(False)
    for p in var.parameters(): 
        p.requires_grad_(False)
    
    # 构建GuidanceInjector
    print(f"Building GuidanceInjector...")
    
    # 根据VAR模型深度计算正确的嵌入维度（与build_vae_var保持一致）
    var_embed_dim = args.model_depth * 64  # width = depth * 64
    print(f"GuidanceInjector configuration: VAR depth={args.model_depth}, embed_dim={var_embed_dim}")
    
    planner = GuidanceInjector(
        input_dim=var_embed_dim,    # VAR的嵌入维度（根据深度动态计算）
        embed_dim=var_embed_dim,    # 保持一致
        num_layers=2,
        num_heads=8
    ).to(device)
    
    # 加载GuidanceInjector检查点
    print(f"Loading GuidanceInjector checkpoint: {args.planner_ckpt}")
    if not osp.exists(args.planner_ckpt):
        raise FileNotFoundError(f"GuidanceInjector checkpoint not found: {args.planner_ckpt}")
    
    planner_state = torch.load(args.planner_ckpt, map_location='cpu')
    
    # 🔥 修复：正确的GuidanceInjector权重加载方式，与train_planner.py保持一致
    if 'planner_state_dict' in planner_state:
        # 从完整的训练检查点加载
        missing_keys, unexpected_keys = planner.load_state_dict(planner_state['planner_state_dict'], strict=False)
        if unexpected_keys:
            pos_keys = [k for k in unexpected_keys if k.startswith('pos_')]
            other_keys = [k for k in unexpected_keys if not k.startswith('pos_')]
            if pos_keys:
                print(f"🔄 忽略位置编码buffer: {pos_keys}")
            if other_keys:
                print(f"⚠️ 其他未预期的键: {other_keys}")
        if missing_keys:
            print(f"⚠️ 缺失的键: {missing_keys}")
        print("✅ GuidanceInjector模型状态从训练检查点加载成功")
    elif 'planner' in planner_state:
        # 从旧格式的检查点加载
        planner.load_state_dict(planner_state['planner'], strict=True)
        print("✅ GuidanceInjector模型状态从旧格式检查点加载成功")
    else:
        # 直接加载模型权重
        planner.load_state_dict(planner_state, strict=True)
        print("✅ GuidanceInjector模型状态直接加载成功")
    
    planner.eval()
    for p in planner.parameters():
        p.requires_grad_(False)
    
    print(f"AID-VAR models loaded successfully!")
    print(f"  📦 VAE parameters: {sum(p.numel() for p in vae.parameters()):,}")
    print(f"  🧠 VAR parameters: {sum(p.numel() for p in var.parameters()):,}")
    print(f"  🎯 GuidanceInjector parameters: {sum(p.numel() for p in planner.parameters()):,}")
    
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
    """
    普通VAR推理并提取每个尺度的解码图像（用于对比）
    
    Returns:
        Tuple[生成的图像 (B, 3, H, W), baseline_scale_decoded_images_list]
    """
    if g_seed is None: 
        rng = None
    else: 
        var_model.rng.manual_seed(g_seed)
        rng = var_model.rng
    
    # 准备标签 - 与原始实现保持一致
    if label_B is None:
        label_B = torch.multinomial(var_model.uniform_prob, num_samples=B, replacement=True, generator=rng).reshape(B)
    elif isinstance(label_B, int):
        label_B = torch.full((B,), fill_value=var_model.num_classes if label_B < 0 else label_B, device=var_model.lvl_1L.device)
    
    # CFG准备
    sos = cond_BD = var_model.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=var_model.num_classes)), dim=0))
    
    # 位置和层级嵌入
    lvl_pos = var_model.lvl_embed(var_model.lvl_1L) + var_model.pos_1LC
    next_token_map = sos.unsqueeze(1).expand(2 * B, var_model.first_l, -1) + \
                    var_model.pos_start.expand(2 * B, var_model.first_l, -1) + \
                    lvl_pos[:, :var_model.first_l]
    
    cur_L = 0
    f_hat = sos.new_zeros(B, var_model.Cvae, var_model.patch_nums[-1], var_model.patch_nums[-1])
    
    # 启用KV缓存
    for b in var_model.blocks: 
        b.attn.kv_caching(True)
    
    # 🔥 存储每个尺度解码出的图像用于可视化
    baseline_scale_decoded_images_list = []
    
    try:
        for si, pn in enumerate(var_model.patch_nums):
            ratio = si / var_model.num_stages_minus_1
            cur_L += pn * pn
            
            # VAR前向传播（标准流程，没有guidance）
            cond_BD_or_gss = var_model.shared_ada_lin(cond_BD)
            x = next_token_map
            
            # Transformer blocks处理
            for b in var_model.blocks:
                x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
            
            # 获取logits
            logits_BlV = var_model.get_logits(x, cond_BD)
            
            # CFG应用
            t = cfg * ratio
            logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]
            
            # 采样
            from models.helpers import sample_with_top_k_top_p_
            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
            
            # 更新f_hat和next_token_map
            if not more_smooth:
                h_BChw = var_model.vae_quant_proxy[0].embedding(idx_Bl)   # B, l, Cvae
            else:
                from models.helpers import gumbel_softmax_with_rng
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)
                h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ var_model.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
            
            h_BChw = h_BChw.transpose_(1, 2).reshape(B, var_model.Cvae, pn, pn)
            f_hat, next_token_map = var_model.vae_quant_proxy[0].get_next_autoregressive_input(si, len(var_model.patch_nums), f_hat, h_BChw)
            
            # 🔥 解码当前尺度的图像用于可视化
            try:
                # 使用当前accumulated f_hat解码图像
                current_scale_image = var_model.vae_proxy[0].fhat_to_img(f_hat.clone())  # (B, 3, H, W) in [-1, 1]
                current_scale_image = current_scale_image.add_(1).mul_(0.5)  # 转换到[0, 1]
                current_scale_image = torch.clamp(current_scale_image, 0, 1)
                
                # 转换为numpy格式用于可视化 (保存所有样本)
                scale_imgs_batch = []
                for sample_idx in range(current_scale_image.shape[0]):
                    scale_img_np = (current_scale_image[sample_idx].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    scale_imgs_batch.append(scale_img_np)
                baseline_scale_decoded_images_list.append(scale_imgs_batch)
                
            except Exception as e:
                print(f"Warning: Failed to decode baseline image at scale {si}: {e}")
                # 添加None作为占位符
                baseline_scale_decoded_images_list.append(None)
            
            # 准备下一阶段
            if si != var_model.num_stages_minus_1:
                next_token_map = next_token_map.view(B, var_model.Cvae, -1).transpose(1, 2)
                next_token_map = var_model.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + var_model.patch_nums[si+1] ** 2]
                next_token_map = next_token_map.repeat(2, 1, 1)
        
        # 清理KV缓存
        for b in var_model.blocks: 
            b.attn.kv_caching(False)
        
        # 生成最终图像
        final_images = var_model.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)
        
        return final_images, baseline_scale_decoded_images_list
        
    except Exception as e:
        print(f"Baseline VAR inference failed: {e}")
        for b in var_model.blocks: 
            b.attn.kv_caching(False)
        
        # 简化的fallback
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
    """
    修改版的AID引导推理，同时返回生成的图像、每个尺度的guidance features和尺度解码图像
    同时生成普通VAR的对比图像
    
    Returns:
        Tuple[生成的图像 (B, 3, H, W), guidance_features_list, aid_scale_decoded_images_list, baseline_scale_decoded_images_list]
    """
    if g_seed is None: 
        rng = None
    else: 
        var_model.rng.manual_seed(g_seed)
        rng = var_model.rng
    
    # 准备标签 - 与原始实现保持一致
    if label_B is None:
        label_B = torch.multinomial(var_model.uniform_prob, num_samples=B, replacement=True, generator=rng).reshape(B)
    elif isinstance(label_B, int):
        label_B = torch.full((B,), fill_value=var_model.num_classes if label_B < 0 else label_B, device=var_model.lvl_1L.device)
    
    # CFG准备 - 与原始实现保持一致，无条件类别为num_classes
    sos = cond_BD = var_model.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=var_model.num_classes)), dim=0))
    
    # 位置和层级嵌入
    lvl_pos = var_model.lvl_embed(var_model.lvl_1L) + var_model.pos_1LC
    next_token_map = sos.unsqueeze(1).expand(2 * B, var_model.first_l, -1) + \
                    var_model.pos_start.expand(2 * B, var_model.first_l, -1) + \
                    lvl_pos[:, :var_model.first_l]
    
    cur_L = 0
    f_hat = sos.new_zeros(B, var_model.Cvae, var_model.patch_nums[-1], var_model.patch_nums[-1])
    
    # 启用KV缓存
    for b in var_model.blocks: 
        b.attn.kv_caching(True)
    
    # 存储每个尺度生成的tokens，用于下一尺度的引导生成
    generated_tokens_per_scale = []
    
    # 🔥 存储每个尺度的guidance features用于可视化
    guidance_features_list = []
    
    # 🔥 存储每个尺度解码出的图像用于可视化
    scale_decoded_images_list = []
    
    try:
        for si, pn in enumerate(var_model.patch_nums):
            ratio = si / var_model.num_stages_minus_1  # 使用与原始实现相同的ratio计算
            cur_L += pn * pn
            
            # 生成引导tokens（从第二个尺度开始）
            guidance_tokens = None
            guidance_features = None
            
            if si > 0 and len(generated_tokens_per_scale) > 0:
                # 获取前一尺度的tokens
                prev_tokens = generated_tokens_per_scale[-1]  # 前一尺度生成的tokens
                
                try:
                    # 转换为embedding features（与训练时保持一致）
                    prev_embed = F.embedding(prev_tokens, var_model.vae_quant_proxy[0].embedding.weight.detach())
                    prev_features = var_model.word_embed(prev_embed)  # (B, L_prev, C)
                    
                    # 数值稳定性保护
                    prev_features = torch.clamp(prev_features, min=-10.0, max=10.0)
                    prev_features = prev_features.detach().clone().requires_grad_(False)
                    
                    # 使用GuidanceInjector生成引导tokens - 与训练时完全一致
                    guidance_tokens = planner(prev_features, target_patch_num=pn)  # (B, pn*pn, C)
                    
                    # 🔥 保存guidance features用于可视化（在CFG之前）
                    guidance_features = guidance_tokens.clone().detach()  # (B, pn*pn, C)
                    
                    if guidance_tokens is not None:
                        # 数值稳定性保护
                        guidance_tokens = torch.clamp(guidance_tokens, min=-5.0, max=5.0)
                        
                        # CFG处理：为引导tokens也复制一份用于无条件生成
                        guidance_tokens = torch.cat([guidance_tokens, guidance_tokens], dim=0)  # (2*B, pn*pn, C)
                    
                except Exception as e:
                    print(f"Warning: Failed to generate guidance for scale {si}: {e}")
                    guidance_tokens = None
                    guidance_features = None
            
            # 存储guidance features（包括第一尺度的None）
            guidance_features_list.append(guidance_features)
            
            # VAR前向传播
            cond_BD_or_gss = var_model.shared_ada_lin(cond_BD)
            x = next_token_map
            
            # 注入引导tokens（与训练时autoregressive_train_step保持一致）
            if guidance_tokens is not None:
                # 确保维度匹配
                if guidance_tokens.shape[0] == x.shape[0] and guidance_tokens.shape[1] == x.shape[1]:
                    # 🔥 应用训练时的引导权重 0.001，与训练时保持完全一致
                    x = x + 0.001 * guidance_tokens  # 与训练时权重一致
                else:
                    print(f"Warning: Guidance token shape mismatch at scale {si}: {guidance_tokens.shape} vs {x.shape}")
            
            # Transformer blocks处理
            for b in var_model.blocks:
                x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
            
            # 获取logits
            logits_BlV = var_model.get_logits(x, cond_BD)
            
            # CFG应用 - 与原始实现保持一致
            t = cfg * ratio
            logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]
            
            # 采样
            from models.helpers import sample_with_top_k_top_p_
            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
            generated_tokens_per_scale.append(idx_Bl)
            
            # 更新f_hat和next_token_map（与原始实现保持一致）
            if not more_smooth:  # 默认情况
                h_BChw = var_model.vae_quant_proxy[0].embedding(idx_Bl)   # B, l, Cvae
            else:   # 用于可视化的平滑采样
                from models.helpers import gumbel_softmax_with_rng
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)   # 参考mask-git
                h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ var_model.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
            
            h_BChw = h_BChw.transpose_(1, 2).reshape(B, var_model.Cvae, pn, pn)
            f_hat, next_token_map = var_model.vae_quant_proxy[0].get_next_autoregressive_input(si, len(var_model.patch_nums), f_hat, h_BChw)
            
            # 🔥 解码当前尺度的图像用于可视化
            try:
                # 使用当前accumulated f_hat解码图像
                current_scale_image = var_model.vae_proxy[0].fhat_to_img(f_hat.clone())  # (B, 3, H, W) in [-1, 1]
                current_scale_image = current_scale_image.add_(1).mul_(0.5)  # 转换到[0, 1]
                current_scale_image = torch.clamp(current_scale_image, 0, 1)
                
                # 转换为numpy格式用于可视化 (保存所有样本)
                scale_imgs_batch = []
                for sample_idx in range(current_scale_image.shape[0]):
                    scale_img_np = (current_scale_image[sample_idx].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    scale_imgs_batch.append(scale_img_np)
                scale_decoded_images_list.append(scale_imgs_batch)
                
            except Exception as e:
                print(f"Warning: Failed to decode image at scale {si}: {e}")
                # 添加None作为占位符
                scale_decoded_images_list.append(None)
            
            # 准备下一阶段（与原始实现保持一致）
            if si != var_model.num_stages_minus_1:
                next_token_map = next_token_map.view(B, var_model.Cvae, -1).transpose(1, 2)
                next_token_map = var_model.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + var_model.patch_nums[si+1] ** 2]
                next_token_map = next_token_map.repeat(2, 1, 1)   # CFG: 双倍批次大小
        
        # 清理KV缓存
        for b in var_model.blocks: 
            b.attn.kv_caching(False)
        
        # 生成最终图像（与原始实现保持一致）
        final_images = var_model.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)   # de-normalize, from [-1, 1] to [0, 1]
        
        # 🔥 生成普通VAR的对比图像（使用完全相同的种子进行公平对比）
        _, baseline_scale_decoded_images_list = baseline_var_inference_with_scale_extraction(
            var_model=var_model,
            B=B,
            label_B=label_B,
            cfg=cfg,
            top_k=top_k,
            top_p=top_p,
            g_seed=g_seed,  # 使用相同的种子（class_id）
            more_smooth=more_smooth
        )
        
        return final_images, guidance_features_list, scale_decoded_images_list, baseline_scale_decoded_images_list
        
    except Exception as e:
        # 出错时的fallback：使用标准VAR推理
        print(f"AID-VAR inference failed: {e}, falling back to standard VAR")
        for b in var_model.blocks: 
            b.attn.kv_caching(False)
        
        final_images = var_model.autoregressive_infer_cfg(B=B, label_B=label_B, cfg=cfg, top_k=top_k, top_p=top_p, g_seed=g_seed, more_smooth=more_smooth)
        empty_guidance = [None] * len(var_model.patch_nums)
        empty_scale_images = [None] * len(var_model.patch_nums)  # 每个尺度都是None
        empty_baseline_images = [None] * len(var_model.patch_nums)  # 基线图像也是None
        return final_images, empty_guidance, empty_scale_images, empty_baseline_images

def generate_guidance_visualization_samples(var, planner, device, args):
    """使用AID-VAR生成50,000个引导样本并创建guidance可视化"""
    print(f"\n🎯 Generating 50,000 AID-VAR guidance visualizations...")
    print(f"Parameters: cfg={args.cfg}, top_p={args.top_p}, top_k={args.top_k}, more_smooth={args.more_smooth}")
    print(f"Save format: {args.save_format}")
    if args.save_format == 'viz_only':
        print(f"📊 Focus: 只保存guidance可视化结果，不保存原图")
    else:
        print(f"📊 Save original images: {args.save_original_images}")
    print(f"Guidance visualization: {args.save_guidance_viz}")
    print(f"Guidance alpha: {args.guidance_alpha}")
    
    # 硬编码配置
    batch_size = 10  # 固定批次大小
    samples_per_class = 10  # 每类固定样本数
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sample_folder = f"{args.output_dir}_d{args.model_depth}_{timestamp}"
    os.makedirs(sample_folder, exist_ok=True)
    
    # 创建guidance可视化子目录
    if args.save_guidance_viz:
        guidance_viz_folder = os.path.join(sample_folder, "guidance_visualizations")
        os.makedirs(guidance_viz_folder, exist_ok=True)
        print(f"Saving guidance visualizations to: {guidance_viz_folder}")
    
    print(f"Saving AID-VAR samples to: {sample_folder}")
    
    # 初始化直接NPZ保存的数组（仅在需要时）
    if args.save_format in ['npz', 'both']:
        all_samples = []  # 存储所有样本的numpy数组
    else:
        all_samples = None
    
    # 随机种子设置
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 启用优化
    tf32 = True
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.set_float32_matmul_precision('high' if tf32 else 'highest')
    
    # 确定数据类型
    if args.dtype == 'float16':
        dtype = torch.float16
    else:
        dtype = torch.bfloat16
    
    total_samples = 0
    num_classes = 1000
    patch_nums = var.patch_nums  # 获取patch数量序列
    
    # 逐类生成样本
    for class_id in tqdm(range(num_classes), desc="Generating AID-VAR classes with guidance viz"):
        # 为每个类设置种子以确保可重现性
        torch.manual_seed(class_id)
        random.seed(class_id)
        np.random.seed(class_id)
        
        # 为当前类生成确切50个样本
        class_labels = [class_id] * samples_per_class
        current_batch_size = batch_size  # 始终50
        current_labels = class_labels
            
        label_B = torch.tensor(current_labels, device=device)
        
        with torch.inference_mode():
            with torch.autocast(device.type, enabled=True, dtype=dtype, cache_enabled=True):
                # 使用修改版的AID-VAR引导生成，同时提取guidance features和尺度解码图像
                recon_B3HW, guidance_features_list, scale_decoded_images_list, baseline_scale_decoded_images_list = aid_guided_inference_with_guidance_extraction(
                    var_model=var,
                    planner=planner,
                    B=current_batch_size, 
                    label_B=label_B, 
                    cfg=args.cfg, 
                    top_k=args.top_k, 
                    top_p=args.top_p, 
                    g_seed=class_id,  # 使用class_id作为种子
                    more_smooth=args.more_smooth
                )
                
                # 保存图像为所需格式
                for i in range(current_batch_size):
                    img_tensor = recon_B3HW[i].cpu()
                    img_tensor = torch.clamp(img_tensor, 0, 1)
                    
                    # 🔥 创建guidance可视化（只保存可视化结果，不保存原图）
                    if args.save_guidance_viz:
                        # 转换为numpy图像用于可视化
                        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                        
                        # 提取当前样本的guidance features
                        sample_guidance_features = []
                        for scale_idx, guidance_features in enumerate(guidance_features_list):
                            if guidance_features is not None:
                                # 提取第i个样本的guidance features
                                sample_guidance = guidance_features[i:i+1]  # (1, pn*pn, C)
                                sample_guidance_features.append(sample_guidance)
                            else:
                                sample_guidance_features.append(None)
                        
                        # 创建可视化网格（原图 + 各尺度guidance叠加图）
                        valid_patch_nums = [pn for pn, gf in zip(patch_nums[1:], sample_guidance_features[1:]) if gf is not None]
                        valid_guidance = [gf for gf in sample_guidance_features[1:] if gf is not None]
                        
                        if valid_guidance:  # 如果有有效的guidance
                            # 提取对应的AID-VAR尺度解码图像（当前样本的）
                            valid_aid_scale_images = []
                            valid_baseline_scale_images = []
                            
                            for scale_idx in range(len(valid_guidance)):
                                actual_scale_idx = scale_idx + 1  # 跳过第0个尺度
                                
                                # AID-VAR图像
                                if (actual_scale_idx < len(scale_decoded_images_list) and 
                                    scale_decoded_images_list[actual_scale_idx] is not None and
                                    i < len(scale_decoded_images_list[actual_scale_idx])):
                                    valid_aid_scale_images.append(scale_decoded_images_list[actual_scale_idx][i])  # 取第i个样本
                                else:
                                    valid_aid_scale_images.append(None)
                                
                                # 普通VAR图像
                                if (actual_scale_idx < len(baseline_scale_decoded_images_list) and 
                                    baseline_scale_decoded_images_list[actual_scale_idx] is not None and
                                    i < len(baseline_scale_decoded_images_list[actual_scale_idx])):
                                    valid_baseline_scale_images.append(baseline_scale_decoded_images_list[actual_scale_idx][i])  # 取第i个样本
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
                            
                            # 添加尺度标签
                            viz_grid_labeled = add_scale_labels_to_grid(
                                viz_grid, 
                                valid_patch_nums, 
                                img_np.shape[1]
                            )
                            
                            # 保存可视化结果
                            viz_path = os.path.join(guidance_viz_folder, f"guidance_viz_{total_samples:05d}.png")
                            viz_pil = Image.fromarray(viz_grid_labeled)
                            viz_pil.save(viz_path, "PNG")
                    
                    # 可选：保存到NPZ数组（如果需要FID评估）
                    if args.save_format in ['npz', 'both'] and all_samples is not None:
                        # 直接保存为float32数组，避免PNG压缩损失
                        img_np_float = img_tensor.permute(1, 2, 0).numpy().astype(np.float32)
                        all_samples.append(img_np_float)
                    
                    # 可选：保存原图为PNG（仅在明确需要或特定格式时）
                    if (args.save_format in ['png', 'both'] or args.save_png_for_npz or args.save_original_images):
                        # 转换为PIL图像以保存PNG
                        if not args.save_guidance_viz:  # 如果已经在guidance可视化中转换过，就不重复
                            img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                        img_pil = PImage.fromarray(img_np)
                        
                        # 保存为PNG
                        img_path = os.path.join(sample_folder, f"aid_sample_{total_samples:05d}.png")
                        img_pil.save(img_path, "PNG")
                    
                    total_samples += 1
        
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 每100个类更新进度
        if (class_id + 1) % 100 == 0:
            print(f"✅ Completed {class_id + 1}/{num_classes} classes, {total_samples} total AID-VAR samples")
            if args.save_guidance_viz:
                print(f"   📊 Guidance visualizations: {total_samples} saved")
    
    print(f"\n🎉 AID-VAR generation with guidance visualization completed! Total samples: {total_samples}")
    
    # 保存直接NPZ文件（如果需要）
    if args.save_format in ['npz', 'both'] and all_samples is not None:
        print(f"💾 Saving direct NPZ file (no PNG compression loss)...")
        all_samples_array = np.stack(all_samples, axis=0)  # Shape: (50000, H, W, C)
        npz_path = f"{sample_folder}_aid_guidance.npz"
        
        # 保存为标准FID评估格式
        np.savez_compressed(npz_path, 
                          arr_0=all_samples_array,  # 主要数组
                          samples=all_samples_array)  # 兼容性别名
        
        print(f"✅ Direct AID-VAR NPZ file saved: {npz_path}")
        print(f"   Shape: {all_samples_array.shape}")
        print(f"   Dtype: {all_samples_array.dtype}")
        print(f"   Value range: [{all_samples_array.min():.3f}, {all_samples_array.max():.3f}]")
        
        # 清理内存
        del all_samples_array
        del all_samples
    
    print(f"📁 AID-VAR samples folder: {sample_folder}")
    if args.save_guidance_viz:
        print(f"📊 Guidance visualizations folder: {guidance_viz_folder}")
    
    return sample_folder

def main():
    args = parse_args()
    
    print("="*80)
    print("🎯 AID-VAR Guidance可视化样本生成")
    print("🚀 使用GuidanceInjector引导的高质量图像生成 + Guidance Feature Map可视化")
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
        # 设置模型
        vae, var, planner, device = setup_models(args)
        
        # 生成AID-VAR引导样本并创建guidance可视化
        sample_folder = generate_guidance_visualization_samples(var, planner, device, args)
        
        # 创建NPZ文件（如果需要）
        if args.create_npz:
            print(f"\n📦 Creating .npz file...")
            create_npz_from_sample_folder(sample_folder)
            print(f"✅ NPZ file created: {sample_folder}.npz")
        else:
            print(f"\n💡 To create .npz file later, run:")
            print(f"python -c \"from utils.misc import create_npz_from_sample_folder; create_npz_from_sample_folder('{sample_folder}')\"")
        
        print(f"\n🎉 AID-VAR Guidance可视化样本生成完成!")
        print(f"📁 Sample folder: {sample_folder}")
        if args.create_npz:
            print(f"📦 NPZ file: {sample_folder}.npz")
        if args.save_guidance_viz:
            print(f"📊 Guidance visualizations: {sample_folder}/guidance_visualizations/")
        
        print(f"\n📊 Guidance可视化说明:")
        print(f"• 三行对比显示：")
        print(f"  - 第一行：AID-VAR解码图像 + Guidance热力图叠加")
        print(f"  - 第二行：AID-VAR各尺度纯净解码图像")
        print(f"  - 第三行：普通VAR各尺度解码图像（使用相同class_id种子）")
        print(f"• 热力图：红色表示高attention区域，蓝色表示低attention区域")
        print(f"• 🔥 公平对比：AID-VAR和普通VAR使用完全相同的种子")
        print(f"• 可以清楚看到AID-VAR相比普通VAR的真实改进效果")
        print(f"• 最小化标签设计，不遮挡图像内容")
        
        print(f"\n📊 Next steps for AID-VAR analysis:")
        print(f"1. 分析guidance feature map的空间分布模式")
        print(f"2. 比较不同尺度guidance的重点关注区域")
        print(f"3. 评估guidance对最终生成质量的影响")
        print(f"4. 计算FID分数与标准VAR对比")
        
    except Exception as e:
        print(f"❌ Error during AID-VAR guidance visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 