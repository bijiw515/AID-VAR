"""
LoRA-based Training Script for AID-VAR Ablation Study

This script replaces the guidance injector (GuidanceInjector) with LoRA fine-tuning
while preserving all discriminator training and adversarial loss components.

Usage:
    torchrun --nproc_per_node=8 train_lora.py \
        --data_path /path/to/imagenet \
        --depth 16 \
        --lr_var 1e-6 \
        --lr_discriminator 1e-6 \
        --ep 100
"""

import os
import sys
import time
import math
import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# Import models
from models.vqvae import VQVAE
from models.var import VAR
from models.lora_var import apply_lora_to_var, get_lora_state_dict, load_lora_state_dict
from models.feature_discriminator_adapter import StyleGANDiscriminatorAdapter

# Import trainer
from trainer_lora import LoRATrainer

# Import utilities
from utils.data_sampler import DistInfiniteBatchSampler, worker_init_fn
from utils.amp_opt import AmpOptimizer
from utils.arg_util import arg_util
from utils.misc import auto_resume

import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_args_parser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('LoRA-VAR Training', add_help=False)

    # Data parameters
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to ImageNet dataset')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of data loading workers')

    # Model parameters
    parser.add_argument('--depth', type=int, default=16, choices=[16, 20, 24],
                        help='VAR model depth (16/20/24)')
    parser.add_argument('--var_ckpt', type=str, default=None,
                        help='Path to pretrained VAR checkpoint')

    # LoRA parameters
    parser.add_argument('--lora_rank_attention', type=int, default=64,
                        help='LoRA rank for attention layers')
    parser.add_argument('--lora_rank_ffn', type=int, default=32,
                        help='LoRA rank for FFN layers')
    parser.add_argument('--lora_alpha_attention', type=float, default=16.0,
                        help='LoRA alpha for attention layers')
    parser.add_argument('--lora_alpha_ffn', type=float, default=8.0,
                        help='LoRA alpha for FFN layers')
    parser.add_argument('--lora_dropout', type=float, default=0.05,
                        help='LoRA dropout rate')

    # Training parameters
    parser.add_argument('--ep', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--bs', type=int, default=32,
                        help='Batch size per GPU')
    parser.add_argument('--lr_var', type=float, default=1e-6,
                        help='Learning rate for LoRA parameters')
    parser.add_argument('--lr_discriminator', type=float, default=1e-6,
                        help='Learning rate for discriminator')
    parser.add_argument('--wd', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--clip_var', type=float, default=0.5,
                        help='Gradient clipping for VAR')
    parser.add_argument('--clip_discriminator', type=float, default=1.0,
                        help='Gradient clipping for discriminator')

    # Loss weights
    parser.add_argument('--lambda_rec', type=float, default=0.1,
                        help='Reconstruction loss weight')
    parser.add_argument('--lambda_adv', type=float, default=1.0,
                        help='Adversarial loss weight')

    # Training strategy
    parser.add_argument('--enable_staged_training', type=arg_util.str2bool, default=True,
                        help='Enable staged training (warmup + joint)')
    parser.add_argument('--warmup_steps', type=int, default=150,
                        help='Number of warmup steps')
    parser.add_argument('--alternating_strategy', type=str, default='adaptive',
                        choices=['adaptive', 'fixed', 'balanced'],
                        help='Alternating training strategy')
    parser.add_argument('--guidance_weight_max', type=float, default=0.01,
                        help='Maximum guidance weight (repurposed as LoRA scaling)')

    # Discriminator parameters
    parser.add_argument('--disc_backbone', type=str, default='dino',
                        help='Discriminator backbone')
    parser.add_argument('--disc_freeze_backbone', type=arg_util.str2bool, default=True,
                        help='Freeze discriminator backbone')

    # Checkpoint and logging
    parser.add_argument('--output_dir', type=str, default='experiments/lora_var',
                        help='Output directory')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--load_optimizer', type=arg_util.str2bool, default=True,
                        help='Load optimizer state when resuming')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--val_interval', type=int, default=5,
                        help='Validate every N epochs')

    # Distributed training
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Local rank for distributed training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Visual debugging
    parser.add_argument('--enable_visual_debug', type=arg_util.str2bool, default=True,
                        help='Enable visual debugging')
    parser.add_argument('--visual_debug_interval', type=int, default=50,
                        help='Visual debugging interval')

    return parser


def setup_distributed():
    """Setup distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    dist.barrier()

    return rank, world_size, local_rank


def load_vqvae(device: torch.device) -> VQVAE:
    """Load frozen VQ-VAE model."""
    logger.info("Loading VQ-VAE...")

    vqvae = VQVAE(
        vocab_size=4096,
        z_channels=32,
        ch=160,
        test_mode=True,
        share_quant_resi=4,
        v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    ).to(device)

    # Load checkpoint
    vae_ckpt_path = 'checkpoints/vae_ch160v4096z32.pth'
    if not os.path.exists(vae_ckpt_path):
        logger.info(f"VQ-VAE checkpoint not found at {vae_ckpt_path}, downloading...")
        os.makedirs('checkpoints', exist_ok=True)
        os.system(f'wget -O {vae_ckpt_path} https://huggingface.co/FoundationVision/var/resolve/main/vae_ch160v4096z32.pth')

    vqvae.load_state_dict(torch.load(vae_ckpt_path, map_location=device), strict=True)
    vqvae.eval()

    # Freeze all parameters
    for param in vqvae.parameters():
        param.requires_grad = False

    logger.info("✅ VQ-VAE loaded and frozen")
    return vqvae


def load_var(args, device: torch.device) -> VAR:
    """Load VAR model and apply LoRA."""
    logger.info(f"Loading VAR model (depth={args.depth})...")

    # Determine embedding dimension
    embed_dim = args.depth * 64  # d16: 1024, d20: 1280, d24: 1536

    # Create VAR model
    var = VAR(
        vae_local=None,  # We'll use separate VQ-VAE
        depth=args.depth,
        embed_dim=embed_dim,
        num_heads=16,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_eps=1e-6,
        shared_aln=False,
        cond_drop_rate=0.1,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        vocab_size=4096,
        num_classes=1000
    ).to(device)

    # Load pretrained VAR weights
    if args.var_ckpt is not None:
        var_ckpt_path = args.var_ckpt
    else:
        var_ckpt_path = f'checkpoints/var_d{args.depth}.pth'

    if not os.path.exists(var_ckpt_path):
        raise FileNotFoundError(
            f"VAR checkpoint not found at {var_ckpt_path}. "
            f"Please download from https://huggingface.co/FoundationVision/var"
        )

    var_state = torch.load(var_ckpt_path, map_location=device)
    if 'trainer_state' in var_state:
        var_state = var_state['model_state_dict']
    var.load_state_dict(var_state, strict=True)

    logger.info("✅ VAR model loaded")

    # Apply LoRA adapters
    logger.info("Applying LoRA adapters...")
    lora_config = {
        'attention': {
            'rank': args.lora_rank_attention,
            'alpha': args.lora_alpha_attention,
            'dropout': args.lora_dropout
        },
        'ffn': {
            'rank': args.lora_rank_ffn,
            'alpha': args.lora_alpha_ffn,
            'dropout': args.lora_dropout
        }
    }

    var = apply_lora_to_var(var, lora_config)
    var.train()

    logger.info("✅ LoRA adapters applied")
    return var


def load_discriminator(args, device: torch.device) -> StyleGANDiscriminatorAdapter:
    """Load discriminator."""
    logger.info("Loading discriminator...")

    discriminator = StyleGANDiscriminatorAdapter(
        backbone_type=args.disc_backbone,
        freeze_backbone=args.disc_freeze_backbone,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        img_size=256
    ).to(device)

    discriminator.train()

    logger.info("✅ Discriminator loaded")
    return discriminator


def create_optimizers(args, lora_var: VAR, discriminator: nn.Module):
    """Create optimizers for LoRA and discriminator."""
    # LoRA optimizer (only LoRA parameters)
    lora_params = [p for n, p in lora_var.named_parameters() if 'lora' in n and p.requires_grad]
    logger.info(f"LoRA trainable parameters: {sum(p.numel() for p in lora_params):,}")

    opt_var = AmpOptimizer(
        optimizer=torch.optim.AdamW(
            lora_params,
            lr=args.lr_var,
            betas=(0.9, 0.999),
            weight_decay=args.wd
        ),
        grad_clip=args.clip_var,
        foreach=True
    )

    # Discriminator optimizer
    disc_params = discriminator.get_trainable_params()
    logger.info(f"Discriminator trainable parameters: {sum(p.numel() for p in disc_params):,}")

    opt_discriminator = AmpOptimizer(
        optimizer=torch.optim.AdamW(
            disc_params,
            lr=args.lr_discriminator,
            betas=(0.9, 0.999),
            weight_decay=args.wd
        ),
        grad_clip=args.clip_discriminator,
        foreach=True
    )

    return opt_var, opt_discriminator


def create_dataloaders(args, rank: int, world_size: int):
    """Create training and validation dataloaders."""
    from torchvision import datasets, transforms

    # Training transforms
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Validation transforms
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Training dataset
    train_dataset = datasets.ImageFolder(
        root=os.path.join(args.data_path, 'train'),
        transform=train_transform
    )

    # Validation dataset
    val_dataset = datasets.ImageFolder(
        root=os.path.join(args.data_path, 'val'),
        transform=val_transform
    )

    # Training sampler and loader
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.bs,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )

    # Validation sampler and loader
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.bs,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")

    return train_loader, val_loader, train_sampler


def save_checkpoint(
    args,
    epoch: int,
    lora_var: VAR,
    discriminator: nn.Module,
    opt_var: AmpOptimizer,
    opt_discriminator: AmpOptimizer,
    trainer: LoRATrainer,
    train_metrics: Dict,
    val_metrics: Dict,
    is_best: bool = False
):
    """Save checkpoint."""
    if dist.get_rank() != 0:
        return

    checkpoint = {
        'epoch': epoch,
        'lora_state_dict': get_lora_state_dict(lora_var),
        'discriminator_state_dict': discriminator.state_dict(),
        'var_optimizer_state_dict': opt_var.state_dict(),
        'discriminator_optimizer_state_dict': opt_discriminator.state_dict(),
        'trainer_state': trainer.state_dict(),
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'args': vars(args)
    }

    # Save regular checkpoint
    ckpt_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch:04d}.pth')
    torch.save(checkpoint, ckpt_path)
    logger.info(f"💾 Saved checkpoint: {ckpt_path}")

    # Save best checkpoint
    if is_best:
        best_path = os.path.join(args.output_dir, 'checkpoint_best.pth')
        torch.save(checkpoint, best_path)
        logger.info(f"🏆 Saved best checkpoint: {best_path}")

    # Save latest checkpoint
    latest_path = os.path.join(args.output_dir, 'checkpoint_latest.pth')
    torch.save(checkpoint, latest_path)


def load_checkpoint(
    args,
    lora_var: VAR,
    discriminator: nn.Module,
    opt_var: AmpOptimizer,
    opt_discriminator: AmpOptimizer,
    trainer: LoRATrainer
) -> Tuple[int, Dict, Dict]:
    """Load checkpoint."""
    if args.resume is None:
        return 0, {}, {}

    logger.info(f"Loading checkpoint from {args.resume}")
    checkpoint = torch.load(args.resume, map_location='cpu')

    # Load LoRA state
    load_lora_state_dict(lora_var, checkpoint['lora_state_dict'], strict=True)

    # Load discriminator state
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'], strict=True)

    # Load optimizer states
    if args.load_optimizer:
        opt_var.load_state_dict(checkpoint['var_optimizer_state_dict'])
        opt_discriminator.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])
        logger.info("✅ Loaded optimizer states")
    else:
        logger.info("⚠️ Skipped loading optimizer states")

    # Load trainer state
    trainer.load_state_dict(checkpoint['trainer_state'])

    start_epoch = checkpoint['epoch'] + 1
    train_metrics = checkpoint.get('train_metrics', {})
    val_metrics = checkpoint.get('val_metrics', {})

    logger.info(f"✅ Resumed from epoch {checkpoint['epoch']}")
    return start_epoch, train_metrics, val_metrics


def train_one_epoch(
    args,
    epoch: int,
    trainer: LoRATrainer,
    train_loader: DataLoader,
    train_sampler: DistributedSampler
) -> Dict[str, float]:
    """Train for one epoch."""
    train_sampler.set_epoch(epoch)

    epoch_metrics = {
        'loss_discriminator': 0.0,
        'loss_var': 0.0,
        'loss_rec': 0.0,
        'loss_adv': 0.0,
        'acc_real': 0.0,
        'acc_fake': 0.0,
        'discriminator_accuracy': 0.0
    }

    num_batches = 0
    start_time = time.time()

    for batch_idx, (img_B3HW, label_B) in enumerate(train_loader):
        img_B3HW = img_B3HW.cuda(non_blocking=True)
        label_B = label_B.cuda(non_blocking=True)

        # Training step
        batch_metrics = trainer.train_step(
            img_B3HW=img_B3HW,
            label_B=label_B,
            epoch=epoch,
            batch_idx=batch_idx,
            exp_dir=args.output_dir
        )

        # Accumulate metrics
        for key in epoch_metrics:
            if key in batch_metrics:
                epoch_metrics[key] += batch_metrics[key]

        num_batches += 1

        # Log progress
        if batch_idx % 10 == 0 and dist.get_rank() == 0:
            elapsed = time.time() - start_time
            samples_per_sec = (batch_idx + 1) * args.bs * dist.get_world_size() / elapsed
            logger.info(
                f"Epoch [{epoch+1}/{args.ep}] Batch [{batch_idx}/{len(train_loader)}] "
                f"loss_D={batch_metrics['loss_discriminator']:.4f} "
                f"loss_VAR={batch_metrics['loss_var']:.4f} "
                f"acc_D={batch_metrics['discriminator_accuracy']:.3f} "
                f"({samples_per_sec:.1f} samples/s)"
            )

    # Average metrics
    for key in epoch_metrics:
        epoch_metrics[key] /= max(num_batches, 1)

    return epoch_metrics


def main():
    """Main training function."""
    # Parse arguments
    parser = get_args_parser()
    args = parser.parse_args()

    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')

    # Set random seed
    torch.manual_seed(args.seed + rank)

    # Create output directory
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"Output directory: {args.output_dir}")

    # Load models
    vqvae = load_vqvae(device)
    lora_var = load_var(args, device)
    discriminator = load_discriminator(args, device)

    # Wrap with DDP
    lora_var = DDP(lora_var, device_ids=[local_rank], find_unused_parameters=False)
    discriminator = DDP(discriminator, device_ids=[local_rank], find_unused_parameters=False)

    # Create optimizers
    opt_var, opt_discriminator = create_optimizers(args, lora_var.module, discriminator.module)

    # Create dataloaders
    train_loader, val_loader, train_sampler = create_dataloaders(args, rank, world_size)

    # Create trainer
    trainer = LoRATrainer(
        device=device,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        vae=vqvae,
        lora_var=lora_var.module,
        discriminator=discriminator.module,
        opt_var=opt_var,
        opt_discriminator=opt_discriminator,
        lambda_rec=args.lambda_rec,
        lambda_adv=args.lambda_adv,
        enable_staged_training=args.enable_staged_training,
        warmup_steps=args.warmup_steps,
        alternating_strategy=args.alternating_strategy,
        guidance_weight_max=args.guidance_weight_max,
        enable_visual_debug=args.enable_visual_debug,
        visual_debug_interval=args.visual_debug_interval
    )

    # Load checkpoint if resuming
    start_epoch, train_metrics, val_metrics = load_checkpoint(
        args, lora_var.module, discriminator.module,
        opt_var, opt_discriminator, trainer
    )

    # Training loop
    best_fid = float('inf')

    for epoch in range(start_epoch, args.ep):
        logger.info(f"\n{'='*80}")
        logger.info(f"Epoch {epoch+1}/{args.ep}")
        logger.info(f"{'='*80}")

        # Train one epoch
        train_metrics = train_one_epoch(args, epoch, trainer, train_loader, train_sampler)

        if rank == 0:
            logger.info(f"📊 Training metrics:")
            for key, value in train_metrics.items():
                logger.info(f"   {key}: {value:.4f}")

        # Validation
        if (epoch + 1) % args.val_interval == 0:
            val_metrics = trainer.validate_one_epoch(val_loader, epoch, args.output_dir)

            if rank == 0:
                logger.info(f"📊 Validation metrics:")
                for key, value in val_metrics.items():
                    logger.info(f"   {key}: {value:.4f}")

        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            is_best = False  # TODO: Implement FID-based best model selection
            save_checkpoint(
                args, epoch, lora_var.module, discriminator.module,
                opt_var, opt_discriminator, trainer,
                train_metrics, val_metrics, is_best
            )

    logger.info("🎉 Training completed!")


if __name__ == '__main__':
    main()
