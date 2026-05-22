"""
LoRA Trainer for AID-VAR Ablation Study

This trainer replaces the guidance injector (GuidanceInjector) with LoRA fine-tuning
while preserving all discriminator training and adversarial loss components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import logging

from aid_helpers import (
    AlternatingTrainingManager,
    calculate_mse_loss,
    save_visual_debugging_samples,
    create_multi_scale_composite_images,
    should_save_visual_debug,
    detect_discriminator_collapse,
    log_alternating_training_step
)

logger = logging.getLogger(__name__)

# Type aliases
Ten = torch.Tensor
ITen = torch.LongTensor


class LoRATrainer:
    """
    LoRA-based trainer for AID-VAR ablation study.

    Replaces GuidanceInjector with LoRA adapters on VAR model while keeping:
    - Discriminator adversarial training
    - Multi-scale discrimination
    - Alternating training strategy
    - All loss functions and metrics
    """

    def __init__(
        self,
        device: torch.device,
        patch_nums: Tuple[int, ...],
        vae: nn.Module,
        lora_var: nn.Module,
        discriminator: nn.Module,
        opt_var: 'AmpOptimizer',
        opt_discriminator: 'AmpOptimizer',
        lambda_rec: float = 0.1,
        lambda_adv: float = 1.0,
        enable_staged_training: bool = True,
        warmup_steps: int = 150,
        alternating_strategy: str = "adaptive",
        guidance_weight_max: float = 0.01,
        Cvae: int = 32,
        vocab_size: int = 4096,
        enable_visual_debug: bool = True,
        visual_debug_interval: int = 50
    ):
        """
        Initialize LoRA trainer.

        Args:
            device: Training device
            patch_nums: Multi-scale patch numbers (e.g., (1,2,3,4,5,6,8,10,13,16))
            vae: Frozen VQ-VAE model
            lora_var: VAR model with LoRA adapters
            discriminator: StyleGAN discriminator
            opt_var: Optimizer for LoRA parameters
            opt_discriminator: Optimizer for discriminator
            lambda_rec: Reconstruction loss weight
            lambda_adv: Adversarial loss weight
            enable_staged_training: Enable warmup phase
            warmup_steps: Number of warmup steps
            alternating_strategy: Strategy for alternating updates
            guidance_weight_max: Maximum guidance weight (repurposed as LoRA scaling)
            Cvae: VAE channel dimension
            vocab_size: Vocabulary size
            enable_visual_debug: Enable visual debugging
            visual_debug_interval: Interval for visual debugging
        """
        self.device = device
        self.patch_nums = patch_nums
        self.vae = vae
        self.lora_var = lora_var
        self.discriminator = discriminator
        self.opt_var = opt_var
        self.opt_discriminator = opt_discriminator

        # Loss weights
        self.lambda_rec = lambda_rec
        self.lambda_adv = lambda_adv

        # Training configuration
        self.enable_staged_training = enable_staged_training
        self.warmup_steps = warmup_steps
        self.Cvae = Cvae
        self.vocab_size = vocab_size

        # Visual debugging
        self.enable_visual_debug = enable_visual_debug
        self.visual_debug_interval = visual_debug_interval

        # Alternating training manager
        self.alternating_manager = AlternatingTrainingManager(
            strategy=alternating_strategy,
            warmup_steps=warmup_steps,
            guidance_weight_max=guidance_weight_max,
            progressive_steps=50
        )

        # Training state
        self.current_step = 0
        self.is_warmup_phase = True
        self.collapse_history = []

        # Metrics tracking
        self.train_metrics = {
            'loss_discriminator': 0.0,
            'loss_var': 0.0,
            'loss_rec': 0.0,
            'loss_adv': 0.0,
            'acc_real': 0.0,
            'acc_fake': 0.0,
            'discriminator_accuracy': 0.0,
            'guidance_weight': 0.0
        }

        logger.info("=" * 80)
        logger.info("🚀 LoRATrainer 初始化")
        logger.info("=" * 80)
        logger.info(f"📊 配置:")
        logger.info(f"   - 设备: {device}")
        logger.info(f"   - 多尺度patch数: {patch_nums}")
        logger.info(f"   - 重建损失权重: {lambda_rec}")
        logger.info(f"   - 对抗损失权重: {lambda_adv}")
        logger.info(f"   - 分阶段训练: {enable_staged_training}")
        logger.info(f"   - 预热步数: {warmup_steps}")
        logger.info(f"   - 交替策略: {alternating_strategy}")
        logger.info(f"   - 最大引导权重: {guidance_weight_max}")
        logger.info("=" * 80)

    def train_step(
        self,
        img_B3HW: Ten,
        label_B: ITen,
        epoch: int,
        batch_idx: int,
        exp_dir: str
    ) -> Dict[str, float]:
        """
        Execute one training step.

        Args:
            img_B3HW: Input images (B, 3, H, W)
            label_B: Class labels (B,)
            epoch: Current epoch
            batch_idx: Current batch index
            exp_dir: Experiment directory for saving debug samples

        Returns:
            Dictionary of training metrics
        """
        B = img_B3HW.shape[0]

        # Update warmup phase status
        if self.enable_staged_training:
            self.is_warmup_phase = self.current_step < self.warmup_steps
        else:
            self.is_warmup_phase = False

        # Get guidance weight (repurposed as LoRA scaling factor)
        guidance_weight = self.alternating_manager.get_progressive_guidance_weight(self.current_step)

        # ============================================================================
        # Step 1: Get ground truth tokens from VAE
        # ============================================================================
        with torch.no_grad():
            gt_idx_Bl = self.vae.img_to_idxBl(img_B3HW)  # List of (B, pn^2) for each scale

        # ============================================================================
        # Step 2: Generate fake tokens with LoRA-adapted VAR (no guidance)
        # ============================================================================
        # During warmup: freeze LoRA adapters
        if self.is_warmup_phase:
            for param in self.lora_var.parameters():
                if 'lora' in param.name if hasattr(param, 'name') else False:
                    param.requires_grad = False
        else:
            for param in self.lora_var.parameters():
                if 'lora' in param.name if hasattr(param, 'name') else False:
                    param.requires_grad = True

        # Generate tokens with LoRA-adapted VAR
        fake_tokens_list, fake_logits_list = self.lora_var.autoregressive_infer(
            B=B,
            label_B=label_B,
            g_seed=None,
            cfg=1.0,
            top_k=0,
            top_p=1.0,
            more_smooth=False,
            return_all_scales=True,
            return_logits=True
        )

        # Also generate standard VAR tokens for comparison (without LoRA)
        with torch.no_grad():
            std_tokens_list, _ = self.lora_var.autoregressive_infer(
                B=B,
                label_B=label_B,
                g_seed=None,
                cfg=1.0,
                top_k=0,
                top_p=1.0,
                more_smooth=False,
                return_all_scales=True,
                return_logits=False
            )

        # ============================================================================
        # Step 3: Multi-scale discrimination and loss computation
        # ============================================================================
        loss_D_total = 0.0
        loss_VAR_total = 0.0
        loss_rec_total = 0.0
        loss_adv_total = 0.0

        acc_real_list = []
        acc_fake_list = []

        real_tokens_history = []
        fake_tokens_history = []

        for si, pn in enumerate(self.patch_nums):
            L_current = pn * pn

            # Get tokens for current scale
            gt_tokens_k = gt_idx_Bl[si]  # (B, L_current)
            fake_tokens_k = fake_tokens_list[si]  # (B, L_current)
            fake_logits_k = fake_logits_list[si]  # (B, L_current, vocab_size)

            # Convert tokens to RGB for discrimination
            # Real RGB: cumulative reconstruction up to current scale
            real_rgb_k = self.tokens_to_rgb_cumulative(gt_idx_Bl, si)  # (B, 3, H, W)

            # Fake RGB: soft reconstruction from logits
            fake_probs_k = F.softmax(fake_logits_k, dim=-1)  # (B, L_current, vocab_size)
            fake_rgb_k = self.soft_tokens_to_rgb(fake_tokens_list[:si+1], fake_probs_k, si)  # (B, 3, H, W)

            # Discriminator forward pass
            pred_real = self.discriminator.forward_rgb(real_rgb_k)  # (B,)
            pred_fake = self.discriminator.forward_soft_rgb(fake_rgb_k)  # (B,)

            # Discriminator loss (hinge loss)
            loss_D_real = F.relu(1.0 - pred_real).mean()
            loss_D_fake = F.relu(1.0 + pred_fake).mean()
            loss_D_scale = loss_D_real + loss_D_fake

            # VAR adversarial loss
            loss_adv_scale = -pred_fake.mean()

            # VAR reconstruction loss
            loss_rec_scale = F.cross_entropy(
                fake_logits_k.reshape(-1, self.vocab_size),
                gt_tokens_k.reshape(-1)
            )

            # Combined VAR loss
            loss_VAR_scale = self.lambda_adv * loss_adv_scale + self.lambda_rec * loss_rec_scale

            # Accumulate losses
            loss_D_total += loss_D_scale
            loss_VAR_total += loss_VAR_scale
            loss_rec_total += loss_rec_scale
            loss_adv_total += loss_adv_scale

            # Compute accuracies
            with torch.no_grad():
                acc_real = (pred_real > 0).float().mean().item()
                acc_fake = (pred_fake < 0).float().mean().item()
                acc_real_list.append(acc_real)
                acc_fake_list.append(acc_fake)

            # Store token history for visual debugging
            real_tokens_history.append(gt_tokens_k.detach())
            fake_tokens_history.append(fake_tokens_k.detach())

            # Visual debugging
            if self.enable_visual_debug and batch_idx % self.visual_debug_interval == 0:
                if should_save_visual_debug(epoch, batch_idx, si, acc_real, acc_fake):
                    save_visual_debugging_samples(
                        vae=self.vae,
                        patch_nums=self.patch_nums,
                        Cvae=self.Cvae,
                        device=self.device,
                        epoch=epoch,
                        batch_idx=batch_idx,
                        scale_idx=si,
                        real_tokens=gt_tokens_k,
                        fake_tokens=fake_tokens_k,
                        std_tokens=std_tokens_list,
                        save_dir=exp_dir,
                        real_tokens_history=real_tokens_history,
                        fake_tokens_history=fake_tokens_history,
                        max_samples=4,
                        guidance_tokens_list=None  # No guidance in LoRA version
                    )

        # Average losses across scales
        num_scales = len(self.patch_nums)
        loss_D_total /= num_scales
        loss_VAR_total /= num_scales
        loss_rec_total /= num_scales
        loss_adv_total /= num_scales

        # Average accuracies
        acc_real_avg = sum(acc_real_list) / len(acc_real_list)
        acc_fake_avg = sum(acc_fake_list) / len(acc_fake_list)
        acc_D_avg = (acc_real_avg + acc_fake_avg) / 2.0

        # ============================================================================
        # Step 4: Alternating updates
        # ============================================================================
        current_metrics = {
            'acc_real': acc_real_avg,
            'acc_fake': acc_fake_avg,
            'loss_D': loss_D_total.item(),
            'loss_VAR': loss_VAR_total.item()
        }

        should_update_disc = self.alternating_manager.should_update_discriminator(
            self.current_step, current_metrics
        )
        should_update_var = self.alternating_manager.should_update_planner(
            self.current_step, current_metrics
        )

        # Update discriminator
        if should_update_disc:
            self.opt_discriminator.zero_grad()
            self.opt_discriminator.backward_clip_step(loss=loss_D_total, retain_graph=True)

        # Update LoRA adapters
        if should_update_var and not self.is_warmup_phase:
            self.opt_var.zero_grad()
            self.opt_var.backward_clip_step(loss=loss_VAR_total, retain_graph=False)

        # Update alternating manager
        self.alternating_manager.update_step_count(
            updated_discriminator=should_update_disc,
            updated_planner=should_update_var,
            metrics=current_metrics
        )

        # Detect discriminator collapse
        is_collapsed, should_recover = detect_discriminator_collapse(
            acc_real=acc_real_avg,
            acc_fake=acc_fake_avg,
            current_step=self.current_step,
            collapse_history=self.collapse_history
        )

        # Log training step
        if batch_idx % 10 == 0:
            phase = "预热" if self.is_warmup_phase else "联合训练"
            log_alternating_training_step(
                current_step=self.current_step,
                phase=phase,
                updated_discriminator=should_update_disc,
                updated_planner=should_update_var,
                guidance_weight=guidance_weight,
                acc_real=acc_real_avg,
                acc_fake=acc_fake_avg,
                strategy=self.alternating_manager.strategy
            )

        # Update step counter
        self.current_step += 1

        # Update metrics
        self.train_metrics = {
            'loss_discriminator': loss_D_total.item(),
            'loss_var': loss_VAR_total.item(),
            'loss_rec': loss_rec_total.item(),
            'loss_adv': loss_adv_total.item(),
            'acc_real': acc_real_avg,
            'acc_fake': acc_fake_avg,
            'discriminator_accuracy': acc_D_avg,
            'guidance_weight': guidance_weight,
            'is_warmup': self.is_warmup_phase,
            'updated_discriminator': should_update_disc,
            'updated_var': should_update_var
        }

        return self.train_metrics

    def tokens_to_rgb_cumulative(self, tokens_list: List[ITen], scale_idx: int) -> Ten:
        """
        Convert tokens to RGB image cumulatively up to scale_idx.

        Args:
            tokens_list: List of token tensors for all scales
            scale_idx: Current scale index

        Returns:
            RGB image tensor (B, 3, H, W)
        """
        with torch.no_grad():
            # Build cumulative token sequence
            cumulative_tokens = []
            for si in range(scale_idx + 1):
                cumulative_tokens.append(tokens_list[si])

            # Pad with zeros for future scales
            for si in range(scale_idx + 1, len(self.patch_nums)):
                pn = self.patch_nums[si]
                L_future = pn * pn
                B = tokens_list[0].shape[0]
                zero_tokens = torch.zeros(B, L_future, dtype=tokens_list[0].dtype, device=self.device)
                cumulative_tokens.append(zero_tokens)

            # Reconstruct image
            rgb = self.vae.idxBl_to_img(cumulative_tokens, same_shape=True, last_one=True)
            return rgb

    def soft_tokens_to_rgb(
        self,
        tokens_history: List[ITen],
        soft_probs: Ten,
        scale_idx: int
    ) -> Ten:
        """
        Convert soft token probabilities to RGB image.

        Args:
            tokens_history: List of hard tokens for previous scales
            soft_probs: Soft probabilities for current scale (B, L, vocab_size)
            scale_idx: Current scale index

        Returns:
            RGB image tensor (B, 3, H, W)
        """
        with torch.no_grad():
            # Use hard tokens for previous scales
            cumulative_tokens = []
            for si in range(scale_idx):
                cumulative_tokens.append(tokens_history[si])

            # Use soft probabilities for current scale (convert to hard tokens)
            hard_tokens_current = soft_probs.argmax(dim=-1)  # (B, L)
            cumulative_tokens.append(hard_tokens_current)

            # Pad with zeros for future scales
            for si in range(scale_idx + 1, len(self.patch_nums)):
                pn = self.patch_nums[si]
                L_future = pn * pn
                B = tokens_history[0].shape[0]
                zero_tokens = torch.zeros(B, L_future, dtype=tokens_history[0].dtype, device=self.device)
                cumulative_tokens.append(zero_tokens)

            # Reconstruct image
            rgb = self.vae.idxBl_to_img(cumulative_tokens, same_shape=True, last_one=True)
            return rgb

    def validate_one_epoch(
        self,
        val_loader,
        epoch: int,
        exp_dir: str,
        max_batches: int = 10
    ) -> Dict[str, float]:
        """
        Validate for one epoch.

        Args:
            val_loader: Validation data loader
            epoch: Current epoch
            exp_dir: Experiment directory
            max_batches: Maximum number of batches to validate

        Returns:
            Dictionary of validation metrics
        """
        logger.info(f"🔍 开始验证 Epoch {epoch+1}")

        self.lora_var.eval()
        self.discriminator.eval()

        val_metrics = {
            'loss_discriminator': 0.0,
            'loss_var': 0.0,
            'acc_real': 0.0,
            'acc_fake': 0.0
        }

        num_batches = 0

        with torch.no_grad():
            for batch_idx, (img_B3HW, label_B) in enumerate(val_loader):
                if batch_idx >= max_batches:
                    break

                img_B3HW = img_B3HW.to(self.device)
                label_B = label_B.to(self.device)

                # Similar to train_step but without updates
                # (Simplified version for validation)
                B = img_B3HW.shape[0]

                # Get ground truth
                gt_idx_Bl = self.vae.img_to_idxBl(img_B3HW)

                # Generate fake tokens
                fake_tokens_list, fake_logits_list = self.lora_var.autoregressive_infer(
                    B=B,
                    label_B=label_B,
                    g_seed=None,
                    cfg=1.0,
                    top_k=0,
                    top_p=1.0,
                    more_smooth=False,
                    return_all_scales=True,
                    return_logits=True
                )

                # Compute losses for last scale only (for efficiency)
                si = len(self.patch_nums) - 1
                gt_tokens_k = gt_idx_Bl[si]
                fake_logits_k = fake_logits_list[si]

                real_rgb_k = self.tokens_to_rgb_cumulative(gt_idx_Bl, si)
                fake_probs_k = F.softmax(fake_logits_k, dim=-1)
                fake_rgb_k = self.soft_tokens_to_rgb(fake_tokens_list[:si+1], fake_probs_k, si)

                pred_real = self.discriminator.forward_rgb(real_rgb_k)
                pred_fake = self.discriminator.forward_soft_rgb(fake_rgb_k)

                loss_D = F.relu(1.0 - pred_real).mean() + F.relu(1.0 + pred_fake).mean()
                loss_rec = F.cross_entropy(
                    fake_logits_k.reshape(-1, self.vocab_size),
                    gt_tokens_k.reshape(-1)
                )
                loss_adv = -pred_fake.mean()
                loss_VAR = self.lambda_adv * loss_adv + self.lambda_rec * loss_rec

                acc_real = (pred_real > 0).float().mean().item()
                acc_fake = (pred_fake < 0).float().mean().item()

                val_metrics['loss_discriminator'] += loss_D.item()
                val_metrics['loss_var'] += loss_VAR.item()
                val_metrics['acc_real'] += acc_real
                val_metrics['acc_fake'] += acc_fake

                num_batches += 1

        # Average metrics
        for key in val_metrics:
            val_metrics[key] /= max(num_batches, 1)

        self.lora_var.train()
        self.discriminator.train()

        logger.info(f"✅ 验证完成: loss_D={val_metrics['loss_discriminator']:.4f}, "
                   f"loss_VAR={val_metrics['loss_var']:.4f}, "
                   f"acc_real={val_metrics['acc_real']:.3f}, "
                   f"acc_fake={val_metrics['acc_fake']:.3f}")

        return val_metrics

    def state_dict(self) -> Dict:
        """Get trainer state for checkpointing."""
        return {
            'current_step': self.current_step,
            'is_warmup_phase': self.is_warmup_phase,
            'collapse_history': self.collapse_history,
            'alternating_manager_state': {
                'step_count': self.alternating_manager.step_count,
                'disc_updates': self.alternating_manager.disc_updates,
                'planner_updates': self.alternating_manager.planner_updates,
                'last_metrics': self.alternating_manager.last_metrics
            }
        }

    def load_state_dict(self, state: Dict):
        """Load trainer state from checkpoint."""
        self.current_step = state.get('current_step', 0)
        self.is_warmup_phase = state.get('is_warmup_phase', True)
        self.collapse_history = state.get('collapse_history', [])

        if 'alternating_manager_state' in state:
            alt_state = state['alternating_manager_state']
            self.alternating_manager.step_count = alt_state.get('step_count', 0)
            self.alternating_manager.disc_updates = alt_state.get('disc_updates', 0)
            self.alternating_manager.planner_updates = alt_state.get('planner_updates', 0)
            self.alternating_manager.last_metrics = alt_state.get('last_metrics', {})
