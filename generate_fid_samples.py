#!/usr/bin/env python3
"""
Official FID Evaluation Script for VAR
Generate 50,000 images (50 per class) for FID evaluation as per official guidance:
- Use var.autoregressive_infer_cfg(..., cfg=1.5, top_p=0.96, top_k=900, more_smooth=False)
- Save as PNG files
- Pack into .npz file via create_npz_from_sample_folder
"""

import os
import os.path as osp
import torch
import torchvision
import random
import numpy as np
import PIL.Image as PImage
from tqdm import tqdm
import argparse
from datetime import datetime

# Disable default parameter init for faster speed
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)

from models import build_vae_var
from utils.misc import create_npz_from_sample_folder

def parse_args():
    parser = argparse.ArgumentParser(description='Generate 50K images for FID evaluation')
    parser.add_argument('--model_depth', type=int, default=16, choices=[16, 20, 24, 30],
                        help='VAR model depth (default: 16)')
    # Batch size is hardcoded to 50
    # parser.add_argument('--batch_size', type=int, default=50, 
    #                     help='Batch size for generation (default: 50, recommended: 50+ for consistency)')
    parser.add_argument('--cfg', type=float, default=1.5,
                        help='Classifier-free guidance scale (default: 1.5)')
    parser.add_argument('--top_p', type=float, default=0.96,
                        help='Top-p sampling parameter (default: 0.96)')
    parser.add_argument('--top_k', type=int, default=900,
                        help='Top-k sampling parameter (default: 900)')
    parser.add_argument('--more_smooth', action='store_true',
                        help='Enable more_smooth for better visual quality')
    # Seed is automatically set to class_id, no need for manual configuration
    # parser.add_argument('--seed', type=int, default=0,
    #                     help='Random seed (default: 0)')
    parser.add_argument('--output_dir', type=str, default='fid_samples',
                        help='Output directory for generated samples')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    parser.add_argument('--vae_ckpt', type=str, default='vae_ch160v4096z32.pth',
                        help='VAE checkpoint path')
    parser.add_argument('--var_ckpt', type=str, default=None,
                        help='VAR checkpoint path (auto-determined if None)')
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
    """Download checkpoints if they don't exist"""
    hf_home = 'https://huggingface.co/FoundationVision/var/resolve/main'
    
    if not osp.exists(vae_ckpt):
        print(f"Downloading {vae_ckpt}...")
        os.system(f'wget {hf_home}/{vae_ckpt}')
    
    if not osp.exists(var_ckpt):
        print(f"Downloading {var_ckpt}...")
        os.system(f'wget {hf_home}/{var_ckpt}')

def setup_models(args):
    """Setup VAE and VAR models"""
    print(f"Setting up models...")
    
    # Determine VAR checkpoint path if not provided
    if args.var_ckpt is None:
        args.var_ckpt = f'checkpoints/var_d{args.model_depth}.pth'
    
    # Download checkpoints if needed
    download_checkpoints(args.vae_ckpt, args.var_ckpt)
    
    # Build models
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    device = torch.device(args.device)
    
    print(f"Building VAE and VAR models (depth={args.model_depth})...")
    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,    # VQVAE hyperparameters
        device=device, patch_nums=patch_nums,
        num_classes=1000, depth=args.model_depth, shared_aln=False,
    )
    
    # Load checkpoints
    print(f"Loading VAE checkpoint: {args.vae_ckpt}")
    vae.load_state_dict(torch.load(args.vae_ckpt, map_location='cpu'), strict=True)
    
    print(f"Loading VAR checkpoint: {args.var_ckpt}")
    var.load_state_dict(torch.load(args.var_ckpt, map_location='cpu'), strict=True)
    
    # Set to eval mode
    vae.eval()
    var.eval()
    for p in vae.parameters(): 
        p.requires_grad_(False)
    for p in var.parameters(): 
        p.requires_grad_(False)
    
    print(f"Models loaded successfully!")
    return vae, var, device

def generate_samples(var, device, args):
    """Generate 50,000 samples (50 per class)"""
    print(f"\nGenerating 50,000 samples for FID evaluation...")
    print(f"Parameters: cfg={args.cfg}, top_p={args.top_p}, top_k={args.top_k}, more_smooth={args.more_smooth}")
    print(f"Save format: {args.save_format}")
    
    # Hardcoded configurations
    batch_size = 50  # Fixed batch size
    samples_per_class = 50  # Fixed samples per class
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sample_folder = f"{args.output_dir}_d{args.model_depth}_{timestamp}"
    os.makedirs(sample_folder, exist_ok=True)
    print(f"Saving samples to: {sample_folder}")
    
    # Initialize arrays for direct NPZ saving
    if args.save_format in ['npz', 'both']:
        all_samples = []  # 存储所有样本的numpy数组
    
    # Random seeds will be set dynamically for each class based on class_id
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Enable optimizations
    tf32 = True
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.set_float32_matmul_precision('high' if tf32 else 'highest')
    
    # Determine dtype
    if args.dtype == 'float16':
        dtype = torch.float16
    else:
        dtype = torch.bfloat16
    
    total_samples = 0
    num_classes = 1000
    
    # Generate samples class by class
    for class_id in tqdm(range(num_classes), desc="Generating classes"):
        # Set seed to class_id for reproducible per-class generation
        torch.manual_seed(class_id)
        random.seed(class_id)
        np.random.seed(class_id)
        
        # Generate exactly 50 samples for this class in one batch
        class_labels = [class_id] * samples_per_class
        current_batch_size = batch_size  # Always 50
        current_labels = class_labels
            
        label_B = torch.tensor(current_labels, device=device)
        
        with torch.inference_mode():
            with torch.autocast(device.type, enabled=True, dtype=dtype, cache_enabled=True):
                # Generate samples using the official parameters
                # Seed is already set to class_id above
                recon_B3HW = var.autoregressive_infer_cfg(
                    B=current_batch_size, 
                    label_B=label_B, 
                    cfg=args.cfg, 
                    top_k=args.top_k, 
                    top_p=args.top_p, 
                    g_seed=class_id,  # Simple: use class_id as seed
                    more_smooth=args.more_smooth
                )
                
                # Save images in the requested format(s)
                for i in range(current_batch_size):
                    img_tensor = recon_B3HW[i].cpu()
                    img_tensor = torch.clamp(img_tensor, 0, 1)
                    
                    # Save to NPZ array if requested
                    if args.save_format in ['npz', 'both']:
                        # 直接保存为float32数组，避免PNG压缩损失
                        img_np_float = img_tensor.permute(1, 2, 0).numpy().astype(np.float32)
                        all_samples.append(img_np_float)
                    
                    # Save as PNG if requested
                    if args.save_format in ['png', 'both'] or args.save_png_for_npz:
                        # Convert to PIL Image for PNG saving
                        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                        img_pil = PImage.fromarray(img_np)
                        
                        # Save as PNG (not JPEG as per official guidance)
                        img_path = os.path.join(sample_folder, f"sample_{total_samples:05d}.png")
                        img_pil.save(img_path, "PNG")
                    
                    total_samples += 1
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Progress update every 100 classes
        if (class_id + 1) % 100 == 0:
            print(f"Completed {class_id + 1}/{num_classes} classes, {total_samples} total samples")
    
    print(f"\nGeneration completed! Total samples: {total_samples}")
    
    # Save direct NPZ file if requested
    if args.save_format in ['npz', 'both']:
        print(f"Saving direct NPZ file (no PNG compression loss)...")
        all_samples_array = np.stack(all_samples, axis=0)  # Shape: (50000, H, W, C)
        npz_path = f"{sample_folder}_direct.npz"
        
        # 保存为标准FID评估格式
        np.savez_compressed(npz_path, 
                          arr_0=all_samples_array,  # 主要数组
                          samples=all_samples_array)  # 兼容性别名
        
        print(f"✅ Direct NPZ file saved: {npz_path}")
        print(f"   Shape: {all_samples_array.shape}")
        print(f"   Dtype: {all_samples_array.dtype}")
        print(f"   Value range: [{all_samples_array.min():.3f}, {all_samples_array.max():.3f}]")
        
        # 清理内存
        del all_samples_array
        del all_samples
    
    print(f"Samples folder: {sample_folder}")
    return sample_folder

def main():
    args = parse_args()
    
    print("="*80)
    print("Enhanced VAR FID Evaluation Sample Generation")
    print("🔥 Improvements: Dynamic seeds per class + Direct NPZ saving")
    print("="*80)
    print(f"Model depth: {args.model_depth}")
    print(f"Batch size: 50 (hardcoded)")
    print(f"Samples per class: 50 (hardcoded)")
    print(f"Sampling parameters: cfg={args.cfg}, top_p={args.top_p}, top_k={args.top_k}")
    print(f"More smooth: {args.more_smooth}")
    print(f"Random seed: class_id for each class (simplified seeding)")
    print(f"Save format: {args.save_format}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print("="*80)
    
    try:
        # Setup models
        vae, var, device = setup_models(args)
        
        # Generate samples
        sample_folder = generate_samples(var, device, args)
        
        # Create NPZ file if requested
        if args.create_npz:
            print(f"\nCreating .npz file...")
            create_npz_from_sample_folder(sample_folder)
            print(f"✅ NPZ file created: {sample_folder}.npz")
        else:
            print(f"\n💡 To create .npz file later, run:")
            print(f"python -c \"from utils.misc import create_npz_from_sample_folder; create_npz_from_sample_folder('{sample_folder}')\"")
        
        print(f"\n✅ FID sample generation completed successfully!")
        print(f"📁 Sample folder: {sample_folder}")
        if args.create_npz:
            print(f"📦 NPZ file: {sample_folder}.npz")
        
        print(f"\n💡 Next steps for FID evaluation:")
        print(f"1. Use OpenAI's FID evaluation toolkit")
        print(f"2. Compare with ImageNet-1K reference ground truth")
        print(f"3. Evaluate FID, IS, precision, and recall")
        
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()