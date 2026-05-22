#!/usr/bin/env python3
"""
AID-VAR full ImageNet multi-GPU distributed training script.
Based on the successful architecture of train_single_class.py, extended to full ImageNet and multi-GPU training.
Includes: staged training, spatial-aware planning token maps, discriminator accuracy monitoring, etc.

Checkpoint resume functionality:
- Supports resuming training from a specified checkpoint
- Automatically restores model state, optimizer state, and training progress
- Supports checkpoint synchronization for distributed training

Usage examples:
  # Train from scratch (8 GPUs)
  torchrun --nproc_per_node=8 train_planner.py --data_path /path/to/imagenet --num_epochs 100

  # Resume from checkpoint
  torchrun --nproc_per_node=8 train_planner.py --data_path /path/to/imagenet --num_epochs 100 \
    --resume_checkpoint experiments/aid_var_full_imagenet_staged_20241219_143021/checkpoints/checkpoint_epoch_10.pth
"""

import os
import sys
import argparse
import logging
import json
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import time
from datetime import datetime
from dataclasses import dataclass

# Add stylegan_t path
stylegan_path = os.path.join(os.path.dirname(__file__), 'stylegan_t')
if stylegan_path not in sys.path:
    sys.path.append(stylegan_path)

# Import VAR project modules
import dist
from utils import arg_util, misc
from utils.data import build_dataset
from utils.data_sampler import DistInfiniteBatchSampler, EvalDistributedSampler
from models import build_vae_var
from models.guidance_injector import GuidanceInjector
from models.discriminator_adapter import StyleGANDiscriminatorAdapter
from trainer_planner import PlannerTrainer
from utils.amp_sc import AmpOptimizer
from utils.misc import MetricLogger, TensorboardLogger, DistLogger

# Configure logging - will be reconfigured to correct path after setup_experiment_dir
logger = logging.getLogger(__name__)

@dataclass
class AIDVARTrainingConfig:
    """AID-VAR full ImageNet training configuration"""

    # Basic configuration
    DEVICE: str = 'cuda'
    OUTPUT_DIR: str = "experiments"

    # Model configuration
    VAR_CKPT: str = "checkpoints/var_d24.pth"  # VAR weights path
    VQVAE_CKPT: str = "vae_ch160v4096z32.pth"  # VQVAE weights path
    VAR_DEPTH: int = 24  # VAR model depth (16, 20, 24, etc.)

    # Training configuration
    EPOCHS: int = 2
    GLOBAL_BATCH_SIZE: int = 16 * 32  # Global batch size (sum across all GPUs)
    VAL_BATCH_SIZE: int = 32  # Validation batch size (per GPU)
    NUM_WORKERS: int = 8

    # Unified learning rates to avoid training dynamics imbalance
    LEARNING_RATE_PLANNER: float = 1e-06       # GuidanceInjector learning rate
    LEARNING_RATE_DISCRIMINATOR: float = 1e-06  # Discriminator learning rate (unified with planner)

    # Gradient clipping
    CLIP_PLANNER: float = 0.5   # Conservative gradient clipping
    CLIP_DISCRIMINATOR: float = 0.5  # Conservative gradient clipping

    # AID-VAR hyperparameters
    LAMBDA_REC: float = 0  # Reconstruction loss weight
    GUIDANCE_WEIGHT: float = 0.005  # Conservative initial guidance weight
    GUIDANCE_TARGET_WEIGHT: float = 0.001  # Fixed guidance weight
    GUIDANCE_RAMP_EPOCHS: int = 15  # Progressive increase period

    # Discriminator stability parameters
    R1_GAMMA: float = 0.2  # R1 gradient penalty weight
    PROGRESSIVE_GUIDANCE: bool = True  # Progressive guidance weight
    COLLAPSE_DETECTION: bool = True  # Collapse detection

    # Staged training configuration
    ENABLE_STAGED_TRAINING: bool = True
    WARMUP_STEPS: int = 0  # Discriminator warmup period

    # Sampling parameters
    TOP_K: int = 0
    TOP_P: float = 0
    CFG: float = 1.5

    # VAR related
    PATCH_NUMS: tuple = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)  # VAR multi-scale

    # GuidanceInjector architecture - dynamically adjusted based on VAR depth
    PLANNER_LAYERS: int = 2
    PLANNER_DIM: int = VAR_DEPTH * 64  # Automatically computed: depth * 64
    PLANNER_HEADS: int = 8

    # Logging and saving
    LOG_INTERVAL: int = 10  # Print every 10 batches
    SAVE_INTERVAL: int = 1  # Save checkpoint every epoch
    VALIDATION_INTERVAL: int = 1  # Validate every epoch

    # Checkpoint resume
    RESUME_CHECKPOINT: Optional[str] = None  # Checkpoint path to resume training
    LOAD_OPTIMIZER_STATE: bool = True  # Whether to load optimizer state (including learning rate)

    @classmethod
    def to_dict(cls) -> Dict:
        """Convert to dictionary format"""
        return {
            key: value for key, value in cls.__dict__.items()
            if not key.startswith('_') and not callable(value) and not isinstance(value, classmethod)
        }

    def update_for_var_depth(self, depth: int):
        """Update configuration parameters based on VAR depth

        Args:
            depth: VAR model depth (16, 20, 24, etc.)
        """
        self.VAR_DEPTH = depth
        self.PLANNER_DIM = depth * 64  # VAR embedding dimension: depth * 64

        # Auto-detect VAR checkpoint path
        var_ckpt_path = f"checkpoints/var_d{depth}.pth"
        if os.path.exists(var_ckpt_path):
            self.VAR_CKPT = var_ckpt_path
        else:
            logger.warning(f"VAR weights file not found: {var_ckpt_path}")

class NullDDP(torch.nn.Module):
    """DDP replacement class for single-GPU training"""
    def __init__(self, module, *args, **kwargs):
        super(NullDDP, self).__init__()
        self.module = module
        self.require_backward_grad_sync = False

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

def setup_experiment_dir(config: AIDVARTrainingConfig, args) -> str:
    """Set up experiment directory and configure logging (master process only)"""
    if not dist.is_master():
        return ""  # Non-master processes return empty string

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    staged_suffix = "staged" if config.ENABLE_STAGED_TRAINING else "joint"
    exp_name = f"aid_var_full_imagenet_{staged_suffix}_{timestamp}"
    exp_dir = os.path.join(config.OUTPUT_DIR, exp_name)

    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "validation"), exist_ok=True)

    # Configure logging to experiment directory
    log_file = os.path.join(exp_dir, "logs", "full_imagenet_training.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ],
        force=True  # Force reconfiguration
    )

    # Save configuration
    config_dict = config.to_dict()
    config_dict.update({
        'world_size': dist.get_world_size(),
        'global_batch_size': args.glb_batch_size,
        'batch_size_per_gpu': args.batch_size,
        'data_path': args.data_path,
        'patch_nums': args.patch_nums,
        'depth': args.depth
    })

    with open(os.path.join(exp_dir, "config.json"), 'w') as f:
        json.dump(config_dict, f, indent=2)

    logger.info(f"Experiment directory created: {exp_dir}")
    return exp_dir

def load_models(config: AIDVARTrainingConfig, args) -> Tuple[nn.Module, nn.Module, nn.Module, nn.Module]:
    """Load all models - adapted for AID-VAR architecture"""
    if dist.is_master():
        logger.info("Loading AID-VAR model architecture...")

    # Use correct VQ-VAE configuration parameters
    # Based on actual configuration of vae_ch160v4096z32.pth
    vqvae, var = build_vae_var(
        V=4096,      # Vocabulary size, matches 4096 in filename
        Cvae=32,     # VQ-VAE feature dimension, matches 32 in filename
        ch=160,      # Channel count, matches 160 in filename
        share_quant_resi=4,
        device=dist.get_device(),
        patch_nums=args.patch_nums,
        num_classes=1000,  # Full ImageNet 1000 classes
        depth=args.depth,
        shared_aln=args.saln,
        attn_l2_norm=args.anorm,
        flash_if_available=args.fuse,
        fused_if_available=args.fuse,
        init_adaln=args.aln,
        init_adaln_gamma=args.alng,
        init_head=args.hd,
        init_std=args.ini
    )

    # Download and load pretrained weights
    vae_ckpt = config.VQVAE_CKPT
    if dist.is_local_master():
        if not os.path.exists(vae_ckpt):
            os.system(f'wget https://huggingface.co/FoundationVision/var/resolve/main/{vae_ckpt}')
    dist.barrier()
    vqvae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)

    # Load VAR weights
    if os.path.exists(config.VAR_CKPT):
        var_state = torch.load(config.VAR_CKPT, map_location='cpu')
        var.load_state_dict(var_state, strict=True)
        if dist.is_master():
            logger.info(f"VAR model weights loaded: {config.VAR_CKPT}")
    else:
        raise FileNotFoundError(f"VAR weights file not found: {config.VAR_CKPT}")

    # Freeze pretrained models
    for param in var.parameters():
        param.requires_grad = False
    for param in vqvae.parameters():
        param.requires_grad = False

    var.eval()
    vqvae.eval()

    # Count frozen parameters
    frozen_vae_params = sum(p.numel() for p in vqvae.parameters())
    frozen_var_params = sum(p.numel() for p in var.parameters())
    if dist.is_master():
        logger.info(f"Frozen VAE parameters: {frozen_vae_params:,}")
        logger.info(f"Frozen VAR parameters: {frozen_var_params:,}")

    # Verify VAR and VQ-VAE configuration match

    # Create AID-VAR components - use VAR's actual embedding dimension
    planner = GuidanceInjector(
        input_dim=var.C,  # VAR feature dimension (dynamic: 16*64=1024 or 20*64=1280)
        embed_dim=var.C,  # Planner embedding dimension matches VAR
        num_layers=config.PLANNER_LAYERS,
        num_heads=config.PLANNER_HEADS
    ).to(dist.get_device())

    # Use pixel-level discriminator adapter
    discriminator = StyleGANDiscriminatorAdapter(
        vae=vqvae,  # VQ-VAE instance for Token->RGB decoding
        patch_nums=args.patch_nums,  # Multi-scale patch configuration
        img_resolution=256,  # Image resolution
        c_dim=0  # Class condition dimension
    ).to(dist.get_device())

    return vqvae, var, planner, discriminator

def create_optimizers(planner: nn.Module, discriminator: nn.Module, config: AIDVARTrainingConfig) -> Tuple[AmpOptimizer, AmpOptimizer]:
    """Create optimizers"""

    # Get trainable parameters
    planner_params = list(planner.parameters())

    # Use pixel-level discriminator's trainable parameter getter
    try:
        disc_trainable_params = discriminator.get_trainable_params()
    except AttributeError:
        # Fallback: get all trainable parameters
        disc_trainable_params = [p for p in discriminator.parameters() if p.requires_grad]
        if dist.is_master():
            logger.warning("Falling back to all trainable parameters")

    if not disc_trainable_params:
        # Create dummy parameter to avoid optimizer error
        dummy_param = torch.nn.Parameter(torch.tensor(0.0, device=dist.get_device()))
        disc_trainable_params = [dummy_param]
        if dist.is_master():
            logger.warning("Discriminator head parameters are empty, using dummy parameter")

    # Count parameters
    planner_param_count = sum(p.numel() for p in planner_params) / 1e6
    disc_param_count = sum(p.numel() for p in disc_trainable_params) / 1e6
    if dist.is_master():
        logger.info(f"Trainable parameter statistics:")
        logger.info(f"   GuidanceInjector: {planner_param_count:.2f}M parameters")
        logger.info(f"   StyleGAN-T Discriminator (heads only): {disc_param_count:.2f}M parameters")

    # Create AmpOptimizer - GuidanceInjector
    amp_planner_opt = AmpOptimizer(
        mixed_precision=0,  # Use float32
        optimizer=torch.optim.AdamW(
            planner_params,
            lr=config.LEARNING_RATE_PLANNER,
            weight_decay=1e-5,
            betas=(0.9, 0.999)
        ),
        names=['planner'],
        paras=planner_params,
        grad_clip=config.CLIP_PLANNER,
        n_gradient_accumulation=1
    )

    # Create AmpOptimizer - Discriminator
    amp_disc_opt = AmpOptimizer(
        mixed_precision=0,  # Use float32
        optimizer=torch.optim.AdamW(
            disc_trainable_params,
            lr=config.LEARNING_RATE_DISCRIMINATOR,
            weight_decay=1e-5,
            betas=(0.9, 0.999)
        ),
        names=['discriminator'],
        paras=disc_trainable_params,
        grad_clip=config.CLIP_DISCRIMINATOR,
        n_gradient_accumulation=1
    )

    return amp_planner_opt, amp_disc_opt

def create_data_loaders(args) -> Tuple[DataLoader, DataLoader, int]:
    """Create distributed data loaders"""

    # Build ImageNet dataset
    num_classes, dataset_train, dataset_val = build_dataset(
        args.data_path,
        final_reso=args.data_load_reso,
        hflip=args.hflip,
        mid_reso=args.mid_reso
    )

    # Create distributed validation data loader
    val_loader = DataLoader(
        dataset_val,
        num_workers=0,
        pin_memory=True,
        batch_size=round(args.batch_size * 1.5),  # Slightly larger batch for validation
        sampler=EvalDistributedSampler(
            dataset_val,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank()
        ),
        shuffle=False,
        drop_last=False,
    )

    # Create distributed training data loader
    train_loader = DataLoader(
        dataset=dataset_train,
        num_workers=args.workers,
        pin_memory=True,
        generator=args.get_different_generator_for_each_rank(),
        batch_sampler=DistInfiniteBatchSampler(
            dataset_len=len(dataset_train),
            glb_batch_size=args.glb_batch_size,
            same_seed_for_all_ranks=args.same_seed_for_all_ranks,
            shuffle=True,
            fill_last=True,
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
        ),
    )

    iters_train = len(train_loader)

    if dist.is_master():
        logger.info(f"   Training samples total: {len(dataset_train):,}")
        logger.info(f"   Validation samples total: {len(dataset_val):,}")
        logger.info(f"   World size: {dist.get_world_size()}")
        logger.info(f"   Global batch size: {args.glb_batch_size}")
        logger.info(f"   Per-GPU batch size: {args.batch_size}")
        logger.info(f"   Training iterations: {iters_train}")

    return train_loader, val_loader, iters_train

def train_full_imagenet(config: AIDVARTrainingConfig, args) -> str:
    """
    Execute AID-VAR full ImageNet distributed training

    Returns:
        exp_dir: Experiment directory path (valid only for master process)
    """
    if dist.is_master():
        logger.info(f"Starting AID-VAR training: data={args.data_path}, epochs={config.EPOCHS}, world_size={dist.get_world_size()}, glb_batch={args.glb_batch_size}")
        if config.ENABLE_STAGED_TRAINING:
            logger.info(f"Staged training: warmup_steps={config.WARMUP_STEPS}")

    # Set up experiment directory (master process only)
    exp_dir = setup_experiment_dir(config, args)

    # Synchronize all processes
    dist.barrier()

    # Load models
    vqvae, var, planner, discriminator = load_models(config, args)

    # Wrap with DDP
    planner = (DDP if dist.initialized() else NullDDP)(
        planner,
        device_ids=[dist.get_local_rank()],
        find_unused_parameters=False,
        broadcast_buffers=False
    )
    discriminator = (DDP if dist.initialized() else NullDDP)(
        discriminator,
        device_ids=[dist.get_local_rank()],
        find_unused_parameters=False,
        broadcast_buffers=False
    )

    # Create optimizers
    amp_planner_opt, amp_disc_opt = create_optimizers(
        planner.module if isinstance(planner, DDP) else planner,
        discriminator.module if isinstance(discriminator, DDP) else discriminator,
        config
    )

    # Create data loaders
    train_loader, val_loader, iters_train = create_data_loaders(args)

    # Create AID-VAR trainer
    trainer = PlannerTrainer(
        device=dist.get_device(),
        vae=vqvae,
        var=var,
        planner=planner.module if isinstance(planner, DDP) else planner,
        disc=discriminator.module if isinstance(discriminator, DDP) else discriminator,
        opt_planner=amp_planner_opt,
        opt_disc=amp_disc_opt,
        lambda_rec=config.LAMBDA_REC,
        patch_nums=var.patch_nums,
        top_k=config.TOP_K,
        top_p=config.TOP_P,
        cfg=config.CFG,
        # Staged training parameters
        warmup_steps=config.WARMUP_STEPS,
        enable_staged_training=config.ENABLE_STAGED_TRAINING,
        alternating_strategy="adaptive",
        guidance_weight_max=config.GUIDANCE_TARGET_WEIGHT,
    )

    # Create metric logger
    metric_logger = MetricLogger(delimiter='  ')

    # Set up TensorBoard logging (master process only)
    if dist.is_master() and exp_dir:
        tb_log_dir = os.path.join(exp_dir, "logs", "tensorboard")
        os.makedirs(tb_log_dir, exist_ok=True)
        tb_logger = DistLogger(
            TensorboardLogger(
                log_dir=tb_log_dir,
                filename_suffix='aid_var_full_imagenet'
            ),
            verbose=True
        )
    else:
        tb_logger = DistLogger(None, verbose=False)

    # Check if we need to resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')

    if config.RESUME_CHECKPOINT and os.path.exists(config.RESUME_CHECKPOINT):
        if dist.is_master():
            logger.info(f"Resuming from checkpoint: {config.RESUME_CHECKPOINT}")
        try:
            start_epoch, resume_train_metrics, resume_val_metrics = load_checkpoint(
                config.RESUME_CHECKPOINT,
                planner.module if isinstance(planner, DDP) else planner,
                discriminator.module if isinstance(discriminator, DDP) else discriminator,
                amp_planner_opt.optimizer,
                amp_disc_opt.optimizer,
                dist.get_device(),
                trainer,  # Pass trainer to restore its internal state
                load_optimizer=config.LOAD_OPTIMIZER_STATE  # Control whether to load optimizer state
            )
            if dist.is_master():
                logger.info(f"Training resumed successfully, starting from epoch {start_epoch + 1}")
            # Get best loss from resumed validation metrics
            best_val_loss = resume_val_metrics.get('loss', float('inf'))

            # Reset learning rates (if optimizer state was loaded but different LR is needed)
            if config.LOAD_OPTIMIZER_STATE:
                for param_group in amp_planner_opt.optimizer.param_groups:
                    old_lr = param_group['lr']
                    param_group['lr'] = config.LEARNING_RATE_PLANNER
                    if dist.is_master() and old_lr != config.LEARNING_RATE_PLANNER:
                        logger.info(f"GuidanceInjector LR reset: {old_lr} -> {config.LEARNING_RATE_PLANNER}")

                for param_group in amp_disc_opt.optimizer.param_groups:
                    old_lr = param_group['lr']
                    param_group['lr'] = config.LEARNING_RATE_DISCRIMINATOR
                    if dist.is_master() and old_lr != config.LEARNING_RATE_DISCRIMINATOR:
                        logger.info(f"Discriminator LR reset: {old_lr} -> {config.LEARNING_RATE_DISCRIMINATOR}")
            else:
                if dist.is_master():
                    logger.info(f"Using new learning rates: GuidanceInjector={config.LEARNING_RATE_PLANNER}, Discriminator={config.LEARNING_RATE_DISCRIMINATOR}")
        except Exception as e:
            if dist.is_master():
                logger.error(f"Checkpoint loading failed: {e}")
            start_epoch = 0
    elif config.RESUME_CHECKPOINT:
        if dist.is_master():
            logger.warning(f"Specified checkpoint file not found: {config.RESUME_CHECKPOINT}")

    # Synchronize all processes
    dist.barrier()

    train_loader_iter = iter(train_loader)

    for epoch in range(start_epoch, config.EPOCHS):

        # Train one epoch
        train_metrics = train_one_epoch(
            trainer, train_loader_iter, iters_train, epoch, config, metric_logger, tb_logger
        )

        # Validate (if needed)
        val_metrics = {}
        if (epoch + 1) % config.VALIDATION_INTERVAL == 0:
            val_metrics = trainer.validate_one_epoch(
                val_loader, epoch, exp_dir if dist.is_master() else ""
            )

            # Log validation metrics
            for key, value in val_metrics.items():
                if isinstance(value, (int, float)):  # Only log numeric types
                    tb_logger.update(head='val', step=epoch, **{key: value})

        # Save checkpoint (master process only)
        if dist.is_master() and (epoch + 1) % config.SAVE_INTERVAL == 0:
            is_best = val_metrics.get('loss', float('inf')) < best_val_loss
            if is_best:
                best_val_loss = val_metrics.get('loss', float('inf'))

            save_checkpoint(
                exp_dir, epoch,
                planner.module if isinstance(planner, DDP) else planner,
                discriminator.module if isinstance(discriminator, DDP) else discriminator,
                amp_planner_opt.optimizer, amp_disc_opt.optimizer,
                train_metrics, val_metrics, is_best, trainer
            )

        # Log training metrics
        for key, value in train_metrics.items():
            tb_logger.update(head='train', step=epoch, **{key: value})

        # Synchronize all processes
        dist.barrier()

    if dist.is_master():
        logger.info(f"Training complete. Results saved to: {exp_dir}")

    return exp_dir if dist.is_master() else ""

def train_one_epoch(
    trainer: PlannerTrainer,
    train_loader_iter,
    iters_train: int,
    epoch: int,
    config: AIDVARTrainingConfig,
    metric_logger: MetricLogger,
    tb_logger
) -> Dict[str, float]:
    """Train one epoch"""

    trainer.planner.train()
    trainer.disc.train()

    epoch_metrics = {
        'loss_planner': 0.0,
        'loss_discriminator': 0.0,
        'loss_adversarial': 0.0,
        'loss_reconstruction': 0.0,
        'discriminator_accuracy': 0.0,
        'discriminator_acc_real': 0.0,
        'discriminator_acc_fake': 0.0,
        'valid_scales': 0.0,
        'warmup_phase': False,
        'collapse_count': 0,
        'guidance_weight': 0.0,
        'r1_penalty': 0.0,
    }

    for batch_idx in range(iters_train):
        images, labels = next(train_loader_iter)

        # Execute AID-VAR per-scale training step
        step_metrics = trainer.train_step(images, labels, metric_logger)

        # Use trainer's returned metric names directly
        if 'loss_D' in step_metrics:
            epoch_metrics['loss_discriminator'] += step_metrics['loss_D']
        if 'loss_P' in step_metrics:
            epoch_metrics['loss_planner'] += step_metrics['loss_P']
        if 'loss_adversarial' in step_metrics:
            epoch_metrics['loss_adversarial'] += step_metrics['loss_adversarial']
        if 'loss_reconstruction' in step_metrics:
            epoch_metrics['loss_reconstruction'] += step_metrics['loss_reconstruction']
        if 'acc_D' in step_metrics:
            epoch_metrics['discriminator_accuracy'] += step_metrics['acc_D']
        if 'acc_real' in step_metrics:
            epoch_metrics['discriminator_acc_real'] += step_metrics['acc_real']
        if 'acc_fake' in step_metrics:
            epoch_metrics['discriminator_acc_fake'] += step_metrics['acc_fake']
        if 'valid_scales' in step_metrics:
            epoch_metrics['valid_scales'] += step_metrics['valid_scales']
        if 'warmup_phase' in step_metrics:
            epoch_metrics['warmup_phase'] = step_metrics['warmup_phase']

        # Log collapse detection and stability metrics
        if hasattr(trainer, 'collapse_history') and len(trainer.collapse_history) > 0:
            epoch_metrics['collapse_count'] = len(trainer.collapse_history)

        # Get current guidance weight from trainer's alternating_manager
        current_guidance_weight = step_metrics.get('guidance_weight', 0.0)
        epoch_metrics['guidance_weight'] = current_guidance_weight

        # Log current step metrics (master process only) - simplified version
        if dist.is_master() and batch_idx % config.LOG_INTERVAL == 0:
            warmup_phase = step_metrics.get('warmup_phase', False)
            phase_str = "warmup" if warmup_phase else "joint"

            # Concise log: only key monitoring info, including real/fake accuracy
            logger.info(f"Epoch {epoch+1} Batch {batch_idx}/{iters_train} [{phase_str}] "
                       f"D_loss={step_metrics.get('loss_D', 0):.4f} "
                       f"P_loss={step_metrics.get('loss_P', 0):.4f} "
                       f"D_acc={step_metrics.get('acc_D', 0):.3f} "
                       f"real_acc={step_metrics.get('acc_real', 0):.3f} "
                       f"fake_acc={step_metrics.get('acc_fake', 0):.3f} "
                       f"guidance={current_guidance_weight:.4f}")

            # Only warn when there are collapses
            if hasattr(trainer, 'collapse_history') and len(trainer.collapse_history) > 0:
                recent_collapses = [c for c in trainer.collapse_history if c['collapsed']]
                if recent_collapses:
                    logger.warning(f"Discriminator collapses: {len(recent_collapses)}")

    # Compute averages
    if iters_train > 0:
        for key in epoch_metrics:
            if key not in ['warmup_phase', 'collapse_count']:
                epoch_metrics[key] /= iters_train

    return epoch_metrics

def save_checkpoint(
    exp_dir: str,
    epoch: int,
    planner: nn.Module,
    discriminator: nn.Module,
    planner_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    is_best: bool = False,
    trainer: Optional['PlannerTrainer'] = None
):
    """Save checkpoint (master process only)"""
    if not exp_dir:  # Non-master process
        return

    checkpoint = {
        'epoch': epoch,
        'planner_state_dict': planner.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'planner_optimizer_state_dict': planner_optimizer.state_dict(),
        'discriminator_optimizer_state_dict': discriminator_optimizer.state_dict(),
        'train_metrics': train_metrics,
        'val_metrics': val_metrics
    }

    # Save trainer's internal state (if provided)
    if trainer is not None:
        trainer_state = {
            'current_step': getattr(trainer, 'current_step', 0),
            'is_warmup_phase': getattr(trainer, 'is_warmup_phase', True),
            'collapse_history': getattr(trainer, 'collapse_history', []),
            'recovery_mode': getattr(trainer, 'recovery_mode', False),
        }

        # Save alternating training manager state
        if hasattr(trainer, 'alternating_manager'):
            alternating_state = {
                'step_count': trainer.alternating_manager.step_count,
                'disc_updates': trainer.alternating_manager.disc_updates,
                'planner_updates': trainer.alternating_manager.planner_updates,
                'last_metrics': trainer.alternating_manager.last_metrics,
            }
            trainer_state['alternating_manager'] = alternating_state

        checkpoint['trainer_state'] = trainer_state
        logger.info(f"Saving trainer state: step={trainer_state['current_step']}, warmup={trainer_state['is_warmup_phase']}")

    # Save regular checkpoint
    checkpoint_path = os.path.join(exp_dir, "checkpoints", f"checkpoint_epoch_{epoch+1}.pth")
    torch.save(checkpoint, checkpoint_path)

    # Save best checkpoint
    if is_best:
        best_path = os.path.join(exp_dir, "checkpoints", "best_checkpoint.pth")
        torch.save(checkpoint, best_path)
        logger.info(f"Best checkpoint saved: {best_path}")

    logger.info(f"Checkpoint saved: {checkpoint_path}")

def load_checkpoint(
    checkpoint_path: str,
    planner: nn.Module,
    discriminator: nn.Module,
    planner_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    device: str = 'cuda',
    trainer: Optional['PlannerTrainer'] = None,
    load_optimizer: bool = True  # Whether to load optimizer state
) -> Tuple[int, Dict[str, float], Dict[str, float]]:
    """Load training state from checkpoint"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    if dist.is_master():
        logger.info(f"Loading checkpoint: {checkpoint_path}")

    # Load checkpoint data
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Restore model state
    try:
        missing_keys, unexpected_keys = planner.load_state_dict(checkpoint['planner_state_dict'], strict=False)
        if unexpected_keys:
            pos_keys = [k for k in unexpected_keys if k.startswith('pos_')]
            other_keys = [k for k in unexpected_keys if not k.startswith('pos_')]
            if pos_keys and dist.is_master():
                logger.info(f"Ignoring positional encoding buffers: {pos_keys}")
            if other_keys and dist.is_master():
                logger.warning(f"Other unexpected keys: {other_keys}")
        if missing_keys and dist.is_master():
            logger.warning(f"Missing keys: {missing_keys}")
    except Exception as e:
        if dist.is_master():
            logger.error(f"GuidanceInjector model state loading failed: {e}")
        raise

    try:
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'], strict=True)
    except Exception as e:
        if dist.is_master():
            logger.error(f"Discriminator model state loading failed: {e}")
        raise

    # Restore optimizer state (optional)
    if load_optimizer:
        try:
            planner_optimizer.load_state_dict(checkpoint['planner_optimizer_state_dict'])
        except Exception as e:
            if dist.is_master():
                logger.warning(f"GuidanceInjector optimizer state loading failed, using default state: {e}")

        try:
            discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])
        except Exception as e:
            if dist.is_master():
                logger.warning(f"Discriminator optimizer state loading failed, using default state: {e}")
    else:
        if dist.is_master():
            logger.info("Skipping optimizer state loading, using new learning rate settings")

    # Get training progress and metrics
    # Checkpoint stores the completed epoch; next training starts from the next epoch
    completed_epoch = checkpoint.get('epoch', -1)  # Completed epoch
    start_epoch = completed_epoch + 1  # Next epoch to start
    train_metrics = checkpoint.get('train_metrics', {})
    val_metrics = checkpoint.get('val_metrics', {})

    # Restore trainer's internal state (if provided)
    if trainer is not None and 'trainer_state' in checkpoint:
        trainer_state = checkpoint['trainer_state']

        # Restore basic state
        if hasattr(trainer, 'current_step'):
            trainer.current_step = trainer_state.get('current_step', 0)
        if hasattr(trainer, 'is_warmup_phase'):
            trainer.is_warmup_phase = trainer_state.get('is_warmup_phase', True)
        if hasattr(trainer, 'collapse_history'):
            trainer.collapse_history = trainer_state.get('collapse_history', [])
        if hasattr(trainer, 'recovery_mode'):
            trainer.recovery_mode = trainer_state.get('recovery_mode', False)

        # Restore alternating training manager state
        if hasattr(trainer, 'alternating_manager') and 'alternating_manager' in trainer_state:
            alt_state = trainer_state['alternating_manager']
            trainer.alternating_manager.step_count = alt_state.get('step_count', 0)
            trainer.alternating_manager.disc_updates = alt_state.get('disc_updates', 0)
            trainer.alternating_manager.planner_updates = alt_state.get('planner_updates', 0)
            trainer.alternating_manager.last_metrics = alt_state.get('last_metrics', {})

        if dist.is_master():
            logger.info(f"Trainer state restored: step={trainer.current_step}, warmup={trainer.is_warmup_phase}")
    elif trainer is not None and dist.is_master():
        logger.warning("Trainer state not found in checkpoint, using default state")

    if dist.is_master():
        logger.info(f"Checkpoint info: completed_epoch={completed_epoch + 1}, next_epoch={start_epoch + 1}, train_metrics={train_metrics}, val_metrics={val_metrics}")

    return start_epoch, train_metrics, val_metrics

def main_training():
    """Main training function"""
    # Initialize distributed training and parse arguments
    args: arg_util.Args = arg_util.init_dist_and_get_args()

    # Check data path
    if not args.local_debug and not os.path.exists(args.data_path):
        raise ValueError(f"Data path not found: {args.data_path}")

    # If in local_debug mode, quick test
    if args.local_debug:
        if dist.is_master():
            print("[local debug] AID-VAR debug mode - quick test")
        # Simple functional test can be added here
        return

    # Create configuration object and update from args
    config = AIDVARTrainingConfig()

    # Update configuration from command-line arguments
    if hasattr(args, 'ep'):
        config.EPOCHS = args.ep
    if hasattr(args, 'resume_checkpoint') and args.resume_checkpoint:
        config.RESUME_CHECKPOINT = args.resume_checkpoint

    # Auto-adjust configuration based on depth parameter
    if hasattr(args, 'depth'):
        config.update_for_var_depth(args.depth)

    # Support learning rate command-line arguments
    if hasattr(args, 'lr_planner') and args.lr_planner is not None:
        config.LEARNING_RATE_PLANNER = args.lr_planner
        if dist.is_master():
            logger.info(f"Overriding GuidanceInjector LR from command line: {args.lr_planner}")

    if hasattr(args, 'lr_discriminator') and args.lr_discriminator is not None:
        config.LEARNING_RATE_DISCRIMINATOR = args.lr_discriminator
        if dist.is_master():
            logger.info(f"Overriding discriminator LR from command line: {args.lr_discriminator}")

    if hasattr(args, 'load_optimizer') and args.load_optimizer is not None:
        config.LOAD_OPTIMIZER_STATE = args.load_optimizer
        if dist.is_master():
            logger.info(f"Setting optimizer state loading from command line: {args.load_optimizer}")

    try:
        exp_dir = train_full_imagenet(config, args)
        if dist.is_master():
            print(f"\nDistributed training completed. Experiment results: {exp_dir}")
            if exp_dir:
                print(f"View logs: tail -f {exp_dir}/logs/*.log")

    except Exception as e:
        if dist.is_master():
            logger.error(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
        raise
    finally:
        dist.finalize()

if __name__ == "__main__":
    try:
        main_training()
    finally:
        if dist.initialized():
            dist.finalize()
