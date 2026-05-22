#!/usr/bin/env python3
"""
AID-VAR FID评估样本生成脚本
基于原始generate_fid_samples.py，但使用AID-VAR的引导推理方法
生成带有GuidanceInjector引导的高质量样本用于FID评估
"""

import os
import os.path as osp
import torch
import random
import numpy as np
import PIL.Image as PImage
from tqdm import tqdm
import argparse
from datetime import datetime
from typing import Optional

# 禁用默认参数初始化以加速
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)

from models import build_vae_var
from models.guidance_injector import GuidanceInjector
from utils.misc import create_npz_from_sample_folder

def parse_args():
    parser = argparse.ArgumentParser(description='Generate 50K AID-VAR guided images for FID evaluation')
    parser.add_argument('--model_depth', type=int, default=16, choices=[16, 20, 24, 30],
                        help='VAR model depth (default: 16)')
    parser.add_argument('--cfg', type=float, default=1.5,
                        help='Classifier-free guidance scale (default: 1.5)')
    parser.add_argument('--top_p', type=float, default=0.96,
                        help='Top-p sampling parameter (default: 0.96)')
    parser.add_argument('--top_k', type=int, default=900,
                        help='Top-k sampling parameter (default: 900)')
    parser.add_argument('--more_smooth', action='store_true',default=True,
                        help='Enable more_smooth for better visual quality')
    parser.add_argument('--output_dir', type=str, default='aid_fid_samples',
                        help='Output directory for generated samples')
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
    parser.add_argument('--save_format', type=str, default='png', choices=['png', 'npz', 'both'],
                        help='Save format: png (PNG files), npz (direct NPZ), both (PNG+NPZ) (default: png)')
    parser.add_argument('--save_png_for_npz', action='store_true',
                        help='When using npz format, also save PNG files for visualization')
    parser.add_argument('--dtype', type=str, default='float16', choices=['float16', 'bfloat16'],
                        help='Data type for inference (default: float16)')
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

def generate_aid_samples(var, planner, device, args):
    """使用AID-VAR生成50,000个引导样本"""
    print(f"\n🎯 Generating 50,000 AID-VAR guided samples for FID evaluation...")
    print(f"Parameters: cfg={args.cfg}, top_p={args.top_p}, top_k={args.top_k}, more_smooth={args.more_smooth}")
    print(f"Save format: {args.save_format}")
    
    # 硬编码配置
    batch_size = 10  # 固定批次大小
    samples_per_class = 10  # 每类固定样本数
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sample_folder = f"{args.output_dir}_d{args.model_depth}_{timestamp}"
    os.makedirs(sample_folder, exist_ok=True)
    print(f"Saving AID-VAR samples to: {sample_folder}")
    
    # 初始化直接NPZ保存的数组
    if args.save_format in ['npz', 'both']:
        all_samples = []  # 存储所有样本的numpy数组
    
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
    
    # 逐类生成样本
    for class_id in tqdm(range(num_classes), desc="Generating AID-VAR classes"):
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
                # 使用AID-VAR引导生成
                recon_B3HW = var.aid_guided_inference(
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
                    
                    # 保存到NPZ数组（如果需要）
                    if args.save_format in ['npz', 'both']:
                        # 直接保存为float32数组，避免PNG压缩损失
                        img_np_float = img_tensor.permute(1, 2, 0).numpy().astype(np.float32)
                        all_samples.append(img_np_float)
                    
                    # 保存为PNG（如果需要）
                    if args.save_format in ['png', 'both'] or args.save_png_for_npz:
                        # 转换为PIL图像以保存PNG
                        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                        img_pil = PImage.fromarray(img_np)
                        
                        # 保存为PNG（按官方指导）
                        img_path = os.path.join(sample_folder, f"aid_sample_{total_samples:05d}.png")
                        img_pil.save(img_path, "PNG")
                    
                    total_samples += 1
        
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 每100个类更新进度
        if (class_id + 1) % 100 == 0:
            print(f"✅ Completed {class_id + 1}/{num_classes} classes, {total_samples} total AID-VAR samples")
    
    print(f"\n🎉 AID-VAR generation completed! Total samples: {total_samples}")
    
    # 保存直接NPZ文件（如果需要）
    if args.save_format in ['npz', 'both']:
        print(f"💾 Saving direct NPZ file (no PNG compression loss)...")
        all_samples_array = np.stack(all_samples, axis=0)  # Shape: (50000, H, W, C)
        npz_path = f"{sample_folder}_aid_direct.npz"
        
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
    return sample_folder

def main():
    args = parse_args()
    
    print("="*80)
    print("🎯 AID-VAR FID评估样本生成")
    print("🚀 使用GuidanceInjector引导的高质量图像生成")
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
    print("="*80)
    
    try:
        # 设置模型
        vae, var, planner, device = setup_models(args)
        
        # 生成AID-VAR引导样本
        sample_folder = generate_aid_samples(var, planner, device, args)
        
        # 创建NPZ文件（如果需要）
        if args.create_npz:
            print(f"\n📦 Creating .npz file...")
            create_npz_from_sample_folder(sample_folder)
            print(f"✅ NPZ file created: {sample_folder}.npz")
        else:
            print(f"\n💡 To create .npz file later, run:")
            print(f"python -c \"from utils.misc import create_npz_from_sample_folder; create_npz_from_sample_folder('{sample_folder}')\"")
        
        print(f"\n🎉 AID-VAR FID sample generation completed successfully!")
        print(f"📁 Sample folder: {sample_folder}")
        if args.create_npz:
            print(f"📦 NPZ file: {sample_folder}.npz")
        
        print(f"\n📊 Next steps for AID-VAR FID evaluation:")
        print(f"1. Compare with baseline VAR FID samples")
        print(f"2. Use OpenAI's FID evaluation toolkit")
        print(f"3. Evaluate FID, IS, precision, and recall improvements")
        print(f"4. Analyze quality enhancement from GuidanceInjector guidance")
        
    except Exception as e:
        print(f"❌ Error during AID-VAR generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 