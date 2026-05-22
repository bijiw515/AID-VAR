#!/usr/bin/env python3
"""
AID-VAR single-class training script.
Used to validate the AID-VAR framework on a single ImageNet class.
Includes staged training, spatial-aware planning token maps, and discriminator accuracy monitoring.

Checkpoint resume support:
- Resume from a specified checkpoint
- Automatically restores model state, optimizer state, and training progress
- CLI argument: --resume_checkpoint /path/to/checkpoint.pth

Usage:
  # Train from scratch
  python train_single_class.py --class_id 207 --num_epochs 100

  # Resume from checkpoint
  python train_single_class.py --class_id 207 --num_epochs 100 \
    --resume_checkpoint single_class_experiments/aid_var_class_207_staged_20241219_143021/checkpoints/checkpoint_epoch_10.pth

  # Resume from best checkpoint
  python train_single_class.py --class_id 207 --num_epochs 100 \
    --resume_checkpoint single_class_experiments/aid_var_class_207_staged_20241219_143021/checkpoints/best_checkpoint.pth
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
import time
from datetime import datetime
from dataclasses import dataclass

stylegan_path = os.path.join(os.path.dirname(__file__), 'stylegan_t')
if stylegan_path not in sys.path:
    sys.path.append(stylegan_path)

from models import build_vae_var
from models.guidance_injector import GuidanceInjector
from models.discriminator_adapter import StyleGANDiscriminatorAdapter
from trainer_planner import PlannerTrainer
from utils.single_class_data import create_single_class_dataloaders
from utils.amp_sc import AmpOptimizer
from utils.misc import MetricLogger, TensorboardLogger
import aid_helpers

logger = logging.getLogger(__name__)

@dataclass
class SingleClassTrainingConfig:

    DEVICE: str = 'cuda'
    CLASS_ID: int = 207
    DATA_DIR: str = "data"
    DATA_ROOT: str = "/home/intern/zkx/Dataset/imagenet-1k/datasets--ILSVRC--imagenet-1k/snapshots/4603483700ee984ea9debe3ddbfdeae86f6489eb/data"
    OUTPUT_DIR: str = "single_class_experiments"

    VAR_CKPT: str = "checkpoints/var_d16.pth"
    VQVAE_CKPT: str = "vae_ch160v4096z32.pth"
    VAR_DEPTH: int = 20

    EPOCHS: int = 100
    BATCH_SIZE: int = 8
    VAL_BATCH_SIZE: int = 4
    NUM_WORKERS: int = 4

    LEARNING_RATE_PLANNER: float = 5e-06
    LEARNING_RATE_DISCRIMINATOR: float = 1e-06

    CLIP_PLANNER: float = 0.5
    CLIP_DISCRIMINATOR: float = 0.5

    LAMBDA_REC: float = 0.01
    GUIDANCE_WEIGHT: float = 0.005
    GUIDANCE_TARGET_WEIGHT: float = 0.001
    GUIDANCE_RAMP_EPOCHS: int = 15

    R1_GAMMA: float = 0.2
    PROGRESSIVE_GUIDANCE: bool = True
    COLLAPSE_DETECTION: bool = True

    ENABLE_STAGED_TRAINING: bool = True
    WARMUP_STEPS: int = 0

    TOP_K: int = 0
    TOP_P: float = 0
    CFG: float = 1.5

    PATCH_NUMS: tuple = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)

    PLANNER_LAYERS: int = 2
    PLANNER_DIM: int = 1024
    PLANNER_HEADS: int = 8

    LOG_INTERVAL: int = 10
    SAVE_INTERVAL: int = 5
    MAX_SAMPLES_PER_CLASS: int = 1000
    VALIDATION_INTERVAL: int = 1

    RESUME_CHECKPOINT: Optional[str] = None

    @classmethod
    def to_dict(cls) -> Dict:
        return {
            key: value for key, value in cls.__dict__.items()
            if not key.startswith('_') and not callable(value) and not isinstance(value, classmethod)
        }

    def update_for_var_depth(self, depth: int):
        self.VAR_DEPTH = depth
        self.PLANNER_DIM = depth * 64

        var_ckpt_path = f"checkpoints/var_d{depth}.pth"
        if os.path.exists(var_ckpt_path):
            self.VAR_CKPT = var_ckpt_path
        else:
            logger.warning(f"VAR checkpoint not found: {var_ckpt_path}")

def setup_experiment_dir(config: SingleClassTrainingConfig) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    staged_suffix = "staged" if config.ENABLE_STAGED_TRAINING else "joint"
    exp_name = f"aid_var_class_{config.CLASS_ID}_{staged_suffix}_{timestamp}"
    exp_dir = os.path.join(config.OUTPUT_DIR, exp_name)

    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "validation"), exist_ok=True)

    log_file = os.path.join(exp_dir, "logs", "single_class_training.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )

    with open(os.path.join(exp_dir, "config.json"), 'w') as f:
        json.dump(config.to_dict(), f, indent=2)

    return exp_dir

def load_models(config: SingleClassTrainingConfig) -> Tuple[nn.Module, nn.Module, nn.Module, nn.Module]:
    vqvae, var = build_vae_var(
        V=4096,
        Cvae=32,
        ch=160,
        share_quant_resi=4,
        device=config.DEVICE,
        patch_nums=config.PATCH_NUMS,
        num_classes=1000,
        depth=config.VAR_DEPTH,
        shared_aln=False,
        attn_l2_norm=True,
        flash_if_available=False,
        fused_if_available=False,
        init_adaln=0.5,
        init_adaln_gamma=1e-5,
        init_head=0.02,
        init_std=-1
    )

    if os.path.exists(config.VAR_CKPT):
        var_state = torch.load(config.VAR_CKPT, map_location=config.DEVICE)
        var.load_state_dict(var_state, strict=True)
        logger.info(f"Loaded VAR weights: {config.VAR_CKPT}")
    else:
        raise FileNotFoundError(f"VAR checkpoint not found: {config.VAR_CKPT}")

    if os.path.exists(config.VQVAE_CKPT):
        vqvae_state = torch.load(config.VQVAE_CKPT, map_location=config.DEVICE)
        vqvae.load_state_dict(vqvae_state, strict=True)
        logger.info(f"Loaded VQVAE weights: {config.VQVAE_CKPT}")
    else:
        raise FileNotFoundError(f"VQVAE checkpoint not found: {config.VQVAE_CKPT}")

    for param in var.parameters():
        param.requires_grad = False
    for param in vqvae.parameters():
        param.requires_grad = False

    var.eval()
    vqvae.eval()

    planner = GuidanceInjector(
        input_dim=var.C,
        embed_dim=var.C,
        num_layers=config.PLANNER_LAYERS,
        num_heads=config.PLANNER_HEADS
    ).to(config.DEVICE)

    discriminator = StyleGANDiscriminatorAdapter(
        vae=vqvae,
        patch_nums=config.PATCH_NUMS,
        img_resolution=256,
        c_dim=0
    ).to(config.DEVICE)

    planner_params = sum(p.numel() for p in planner.parameters() if p.requires_grad)
    disc_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    total_params = planner_params + disc_params

    logger.info(f"Trainable params - GuidanceInjector: {planner_params/1e6:.2f}M, "
                f"Discriminator: {disc_params/1e6:.2f}M, Total: {total_params/1e6:.2f}M")

    return vqvae, var, planner, discriminator

def create_optimizers(planner: nn.Module, discriminator: nn.Module, config: SingleClassTrainingConfig) -> Tuple[AmpOptimizer, AmpOptimizer]:
    planner_params = list(planner.parameters())

    try:
        disc_trainable_params = discriminator.get_trainable_params()
    except AttributeError:
        disc_trainable_params = [p for p in discriminator.parameters() if p.requires_grad]
        logger.warning("Falling back to all trainable discriminator parameters")

    if not disc_trainable_params:
        dummy_param = torch.nn.Parameter(torch.tensor(0.0, device=config.DEVICE))
        disc_trainable_params = [dummy_param]
        logger.warning("Discriminator head parameters are empty, using dummy parameter")

    amp_planner_opt = AmpOptimizer(
        mixed_precision=0,
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

    amp_disc_opt = AmpOptimizer(
        mixed_precision=0,
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

def create_data_loaders(config: SingleClassTrainingConfig) -> Tuple[DataLoader, DataLoader]:
    try:
        train_loader, val_loader = create_single_class_dataloaders(
            data_root=config.DATA_ROOT,
            class_id=config.CLASS_ID,
            batch_size=config.BATCH_SIZE,
            num_workers=0,
            max_train_samples=config.MAX_SAMPLES_PER_CLASS,
            max_val_samples=config.MAX_SAMPLES_PER_CLASS // 2
        )

        return train_loader, val_loader

    except Exception as e:
        logger.error(f"Failed to create data loaders: {e}")
        raise

def train_single_class(config: SingleClassTrainingConfig) -> str:
    """
    Run AID-VAR single-class training.

    Returns:
        exp_dir: experiment directory path
    """
    exp_dir = setup_experiment_dir(config)

    vqvae, var, planner, discriminator = load_models(config)

    amp_planner_opt, amp_disc_opt = create_optimizers(planner, discriminator, config)

    train_loader, val_loader = create_data_loaders(config)

    trainer = PlannerTrainer(
        device=config.DEVICE,
        vae=vqvae,
        var=var,
        planner=planner,
        disc=discriminator,
        opt_planner=amp_planner_opt,
        opt_disc=amp_disc_opt,
        lambda_rec=config.LAMBDA_REC,
        patch_nums=var.patch_nums,
        top_k=config.TOP_K,
        top_p=config.TOP_P,
        cfg=config.CFG,
        warmup_steps=config.WARMUP_STEPS,
        enable_staged_training=config.ENABLE_STAGED_TRAINING,
        alternating_strategy="adaptive",
        guidance_weight_max=config.GUIDANCE_TARGET_WEIGHT,
    )

    metric_logger = MetricLogger(delimiter='  ')

    tb_log_dir = os.path.join(exp_dir, "logs", "tensorboard")
    os.makedirs(tb_log_dir, exist_ok=True)
    tb_logger = TensorboardLogger(log_dir=tb_log_dir, filename_suffix='aid_var_single_class')

    start_epoch = 0
    if config.RESUME_CHECKPOINT and os.path.exists(config.RESUME_CHECKPOINT):
        try:
            start_epoch, resume_train_metrics, resume_val_metrics = load_checkpoint(
                config.RESUME_CHECKPOINT,
                planner,
                discriminator,
                amp_planner_opt.optimizer,
                amp_disc_opt.optimizer,
                config.DEVICE,
                trainer
            )
            logger.info(f"Resumed training from epoch {start_epoch}")
            best_val_loss = resume_val_metrics.get('loss', float('inf'))
        except Exception as e:
            logger.error(f"Checkpoint loading failed: {e}")
            logger.info("Starting training from scratch")
            start_epoch = 0
    elif config.RESUME_CHECKPOINT:
        logger.warning(f"Specified checkpoint does not exist: {config.RESUME_CHECKPOINT}")

    best_val_loss = float('inf') if start_epoch == 0 else best_val_loss

    for epoch in range(start_epoch, config.EPOCHS):
        train_metrics = train_one_epoch(
            trainer, train_loader, epoch, config, metric_logger, tb_logger
        )

        val_metrics = {}
        if (epoch + 1) % config.VALIDATION_INTERVAL == 0:
            val_metrics = trainer.validate_one_epoch(
                val_loader, epoch, exp_dir
            )

            for key, value in val_metrics.items():
                if isinstance(value, (int, float)):
                    tb_logger.update(head='val', step=epoch, **{key: value})

        if (epoch + 1) % config.SAVE_INTERVAL == 0:
            is_best = val_metrics.get('loss', float('inf')) < best_val_loss
            if is_best:
                best_val_loss = val_metrics.get('loss', float('inf'))

            save_checkpoint(
                exp_dir, epoch, planner, discriminator,
                amp_planner_opt.optimizer, amp_disc_opt.optimizer,
                train_metrics, val_metrics, is_best, trainer
            )

        for key, value in train_metrics.items():
            tb_logger.update(head='train', step=epoch, **{key: value})

    logger.info(f"Training complete. Results saved to: {exp_dir}")

    return exp_dir

def train_one_epoch(
    trainer: PlannerTrainer,
    train_loader: DataLoader,
    epoch: int,
    config: SingleClassTrainingConfig,
    metric_logger: MetricLogger,
    tb_logger: TensorboardLogger
) -> Dict[str, float]:

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

    num_batches = len(train_loader)

    for batch_idx, (images, labels) in enumerate(train_loader):
        step_metrics = trainer.train_step(images, labels, metric_logger)

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

        if hasattr(trainer, 'collapse_history') and len(trainer.collapse_history) > 0:
            epoch_metrics['collapse_count'] = len(trainer.collapse_history)

        current_guidance_weight = step_metrics.get('guidance_weight', 0.0)
        epoch_metrics['guidance_weight'] = current_guidance_weight

        if batch_idx % config.LOG_INTERVAL == 0:
            warmup_phase = step_metrics.get('warmup_phase', False)
            phase_str = "warmup" if warmup_phase else "joint"

            logger.info(f"  Batch {batch_idx}/{num_batches}: [{phase_str}] "
                       f"Loss_D={step_metrics.get('loss_D', 0):.4f}, "
                       f"Loss_P={step_metrics.get('loss_P', 0):.4f}, "
                       f"Acc_D={step_metrics.get('acc_D', 0):.3f}, "
                       f"real_acc={step_metrics.get('acc_real', 0):.3f}, "
                       f"fake_acc={step_metrics.get('acc_fake', 0):.3f}, "
                       f"guidance_weight={current_guidance_weight:.4f}")

            if hasattr(trainer, 'collapse_history') and len(trainer.collapse_history) > 0:
                recent_collapses = [c for c in trainer.collapse_history if c['collapsed']]
                if recent_collapses:
                    logger.warning(f"Detected {len(recent_collapses)} discriminator collapse(s)")

    if num_batches > 0:
        for key in epoch_metrics:
            if key not in ['warmup_phase', 'collapse_count']:
                epoch_metrics[key] /= num_batches

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
    checkpoint = {
        'epoch': epoch,
        'planner_state_dict': planner.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'planner_optimizer_state_dict': planner_optimizer.state_dict(),
        'discriminator_optimizer_state_dict': discriminator_optimizer.state_dict(),
        'train_metrics': train_metrics,
        'val_metrics': val_metrics
    }

    if trainer is not None:
        trainer_state = {
            'current_step': getattr(trainer, 'current_step', 0),
            'is_warmup_phase': getattr(trainer, 'is_warmup_phase', True),
            'collapse_history': getattr(trainer, 'collapse_history', []),
            'recovery_mode': getattr(trainer, 'recovery_mode', False),
        }

        if hasattr(trainer, 'alternating_manager'):
            alternating_state = {
                'step_count': trainer.alternating_manager.step_count,
                'disc_updates': trainer.alternating_manager.disc_updates,
                'planner_updates': trainer.alternating_manager.planner_updates,
                'last_metrics': trainer.alternating_manager.last_metrics,
            }
            trainer_state['alternating_manager'] = alternating_state

        checkpoint['trainer_state'] = trainer_state

    checkpoint_path = os.path.join(exp_dir, "checkpoints", f"checkpoint_epoch_{epoch+1}.pth")
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint: {checkpoint_path}")

    if is_best:
        best_path = os.path.join(exp_dir, "checkpoints", "best_checkpoint.pth")
        torch.save(checkpoint, best_path)
        logger.info(f"Saved best checkpoint: {best_path}")

def load_checkpoint(
    checkpoint_path: str,
    planner: nn.Module,
    discriminator: nn.Module,
    planner_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    device: str = 'cuda',
    trainer: Optional['PlannerTrainer'] = None
) -> Tuple[int, Dict[str, float], Dict[str, float]]:
    """
    Load training state from a checkpoint.

    Args:
        checkpoint_path: path to checkpoint file
        planner: GuidanceInjector model
        discriminator: discriminator model
        planner_optimizer: GuidanceInjector optimizer
        discriminator_optimizer: discriminator optimizer
        device: device string
        trainer: PlannerTrainer instance (optional, to restore internal state)

    Returns:
        Tuple[start_epoch, train_metrics, val_metrics]
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    try:
        missing_keys, unexpected_keys = planner.load_state_dict(checkpoint['planner_state_dict'], strict=False)
        if unexpected_keys:
            pos_keys = [k for k in unexpected_keys if k.startswith('pos_')]
            other_keys = [k for k in unexpected_keys if not k.startswith('pos_')]
            if other_keys:
                logger.warning(f"Unexpected keys in planner state dict: {other_keys}")
        if missing_keys:
            logger.warning(f"Missing keys in planner state dict: {missing_keys}")
    except Exception as e:
        logger.error(f"Failed to load GuidanceInjector state: {e}")
        raise

    try:
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'], strict=True)
    except Exception as e:
        logger.error(f"Failed to load discriminator state: {e}")
        raise

    try:
        planner_optimizer.load_state_dict(checkpoint['planner_optimizer_state_dict'])
    except Exception as e:
        logger.warning(f"Failed to load planner optimizer state, using defaults: {e}")

    try:
        discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])
    except Exception as e:
        logger.warning(f"Failed to load discriminator optimizer state, using defaults: {e}")

    start_epoch = checkpoint.get('epoch', 0)
    train_metrics = checkpoint.get('train_metrics', {})
    val_metrics = checkpoint.get('val_metrics', {})

    if trainer is not None and 'trainer_state' in checkpoint:
        trainer_state = checkpoint['trainer_state']

        if hasattr(trainer, 'current_step'):
            trainer.current_step = trainer_state.get('current_step', 0)
        if hasattr(trainer, 'is_warmup_phase'):
            trainer.is_warmup_phase = trainer_state.get('is_warmup_phase', True)
        if hasattr(trainer, 'collapse_history'):
            trainer.collapse_history = trainer_state.get('collapse_history', [])
        if hasattr(trainer, 'recovery_mode'):
            trainer.recovery_mode = trainer_state.get('recovery_mode', False)

        if hasattr(trainer, 'alternating_manager') and 'alternating_manager' in trainer_state:
            alt_state = trainer_state['alternating_manager']
            trainer.alternating_manager.step_count = alt_state.get('step_count', 0)
            trainer.alternating_manager.disc_updates = alt_state.get('disc_updates', 0)
            trainer.alternating_manager.planner_updates = alt_state.get('planner_updates', 0)
            trainer.alternating_manager.last_metrics = alt_state.get('last_metrics', {})

        logger.info(f"Restored trainer state: step={trainer.current_step}, warmup={trainer.is_warmup_phase}")
    elif trainer is not None:
        logger.warning("No trainer state found in checkpoint; using default state")

    logger.info(f"Resuming from epoch {start_epoch + 1}")

    return start_epoch, train_metrics, val_metrics

def main():
    parser = argparse.ArgumentParser(description='AID-VAR single-class training')

    parser.add_argument('--data_root', type=str, default=SingleClassTrainingConfig.DATA_ROOT)
    parser.add_argument('--class_id', type=int, default=SingleClassTrainingConfig.CLASS_ID)
    parser.add_argument('--batch_size', type=int, default=SingleClassTrainingConfig.BATCH_SIZE)
    parser.add_argument('--num_epochs', type=int, default=SingleClassTrainingConfig.EPOCHS)
    parser.add_argument('--max_train_samples', type=int, default=SingleClassTrainingConfig.MAX_SAMPLES_PER_CLASS)
    parser.add_argument('--device', type=str, default=SingleClassTrainingConfig.DEVICE)

    parser.add_argument('--var_depth', type=int, default=SingleClassTrainingConfig.VAR_DEPTH)
    parser.add_argument('--warmup_steps', type=int, default=SingleClassTrainingConfig.WARMUP_STEPS)
    parser.add_argument('--enable_staged_training', type=bool, default=SingleClassTrainingConfig.ENABLE_STAGED_TRAINING)
    parser.add_argument('--lambda_rec', type=float, default=SingleClassTrainingConfig.LAMBDA_REC)
    parser.add_argument('--lr_discriminator', type=float, default=SingleClassTrainingConfig.LEARNING_RATE_DISCRIMINATOR)

    parser.add_argument('--resume_checkpoint', type=str, default=None)

    args = parser.parse_args()

    config = SingleClassTrainingConfig()

    if hasattr(args, 'var_depth'):
        config.update_for_var_depth(args.var_depth)

    config.DATA_ROOT = args.data_root
    config.CLASS_ID = args.class_id
    config.BATCH_SIZE = args.batch_size
    config.EPOCHS = args.num_epochs
    config.MAX_SAMPLES_PER_CLASS = args.max_train_samples
    config.DEVICE = args.device
    config.WARMUP_STEPS = args.warmup_steps
    config.ENABLE_STAGED_TRAINING = args.enable_staged_training
    config.LAMBDA_REC = args.lambda_rec
    config.LEARNING_RATE_DISCRIMINATOR = args.lr_discriminator
    config.RESUME_CHECKPOINT = args.resume_checkpoint

    try:
        exp_dir = train_single_class(config)
        print(f"\nTraining complete.")
        print(f"Results: {exp_dir}")
        print(f"Logs: tail -f {exp_dir}/logs/*.log")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
