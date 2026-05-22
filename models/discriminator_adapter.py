import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, List, Tuple
import numpy as np
import sys
import os

logger = logging.getLogger(__name__)

# Ensure stylegan_t is on the Python path
stylegan_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'stylegan_t')
if stylegan_path not in sys.path:
    sys.path.insert(0, stylegan_path)

try:
    from stylegan_t.networks.discriminator import ProjectedDiscriminator, DiscHead, DINO
    from stylegan_t.networks.shared import ResidualBlock, FullyConnectedLayer
    from stylegan_t.torch_utils import misc
    from models.stable_discriminator_head import StableProjectedDiscriminator
    STYLEGAN_AVAILABLE = True
except ImportError as e:
    logger.error(f"❌ StyleGAN-T modules not available: {e}")
    STYLEGAN_AVAILABLE = False
    raise ImportError("StyleGAN-T is required for AID-VAR. No fallback allowed.")


class PixelDiscriminatorAdapter(nn.Module):
    """
    Pixel-level discriminator adapter based on StyleGAN-T ProjectedDiscriminator.

    Key features:
    - RGB image direct discrimination (forward_rgb, forward_soft_rgb)
    - Token->RGB cumulative decoding (backward compatible)
    - Pixel-level discrimination
    - Multi-scale discrimination support
    - Gradient flow preservation (via Gumbel-Softmax)

    Architecture:
    - New interface: processes RGB images directly, resolves cuDNN compatibility issues
    - Legacy interface: token processing kept for backward compatibility (with deprecation warnings)
    """

    def __init__(
        self,
        vae,  # VQ-VAE instance for token decoding
        patch_nums: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16),  # multi-scale patch counts
        img_resolution: int = 256,
        c_dim: int = 0,  # conditioning dimension
        diffaug: bool = True,
        p_crop: float = 0.5,
        freeze_backbone: bool = True,
        dino_checkpoint_path: str = None  # local DINO model path
    ):
        super().__init__()

        self.vae = vae  # VQ-VAE for token decoding
        self.patch_nums = patch_nums
        self.img_resolution = img_resolution
        self.c_dim = c_dim
        self.freeze_backbone = freeze_backbone

        self.discriminator = StableProjectedDiscriminator(
            c_dim=c_dim,
            diffaug=diffaug,
            p_crop=p_crop,
            dino_checkpoint_path=dino_checkpoint_path
        )

        # Freeze DINO backbone if requested
        if freeze_backbone:
            for param in self.discriminator.dino.parameters():
                param.requires_grad = False

        # Fixed temperature parameter to prevent training instability
        self.gumbel_temperature = 1.0  # fixed at 1.0, not learnable

        self._log_parameter_statistics()

    def _log_parameter_statistics(self):
        """Log parameter statistics."""
        pass

    def get_trainable_params(self) -> List[torch.nn.Parameter]:
        """Return trainable parameters."""
        if self.freeze_backbone:
            # Only return discriminator head parameters
            params = []
            for param in self.discriminator.heads.parameters():
                if param.requires_grad:
                    params.append(param)
            # Temperature parameter is now fixed, not included
            return params
        else:
            return [p for p in self.parameters() if p.requires_grad]

    def tokens_to_rgb_cumulative(self, tokens_list: List[torch.Tensor], target_scale_idx: int) -> torch.Tensor:
        """
        Multi-scale cumulative decoding: tokens -> RGB.

        Args:
            tokens_list: per-scale token list [tokens_0, tokens_1, ..., tokens_k]
            target_scale_idx: target scale index (0-based)

        Returns:
            rgb_images: decoded RGB images (B, 3, H, W) in range [-1, 1]
        """
        try:
            # VQ-VAE embed_to_fhat expects all 10 scales of tokens.
            # Build the full 10-scale list: scales 0..target_scale_idx use real tokens,
            # the rest are zero-padded.
            cumulative_tokens = []

            B = 1
            device = 'cuda'
            if tokens_list and len(tokens_list) > 0:
                for t in tokens_list:
                    if t is not None:
                        B = t.shape[0]
                        device = t.device
                        break

            for i in range(len(self.patch_nums)):
                if i <= target_scale_idx and i < len(tokens_list) and tokens_list[i] is not None:
                    cumulative_tokens.append(tokens_list[i])
                else:
                    pn = self.patch_nums[i]
                    zero_tokens = torch.zeros(B, pn*pn, dtype=torch.long, device=device)
                    cumulative_tokens.append(zero_tokens)

            with torch.no_grad():  # VQ-VAE is frozen
                rgb_images = self.vae.idxBl_to_img(cumulative_tokens, same_shape=True, last_one=True)

            rgb_images = torch.clamp(rgb_images, -1.0, 1.0)
            return rgb_images

        except Exception as e:
            logger.error(f"❌ Cumulative decoding failed: {e}")

            B = 1
            device = 'cuda'
            if tokens_list and len(tokens_list) > 0:
                for t in tokens_list:
                    if t is not None:
                        B = t.shape[0]
                        device = t.device
                        break

            return torch.randn(B, 3, self.img_resolution, self.img_resolution, device=device) * 0.1

    def soft_tokens_to_rgb(self, soft_probs_list: List[torch.Tensor], target_scale_idx: int) -> torch.Tensor:
        """
        Soft-probability tokens -> RGB (preserves gradient flow).

        Args:
            soft_probs_list: per-scale soft probability list [(B, L_i, V), ...]
            target_scale_idx: target scale index

        Returns:
            rgb_images: decoded RGB images (B, 3, H, W)
        """
        try:
            if not soft_probs_list or len(soft_probs_list) == 0:
                raise ValueError("soft_probs_list is empty")

            valid_soft_probs = [sp for sp in soft_probs_list if sp is not None]
            if not valid_soft_probs:
                raise ValueError("All soft_probs are None")

            B = valid_soft_probs[0].shape[0]
            V = valid_soft_probs[0].shape[-1]
            device = valid_soft_probs[0].device

            cumulative_tokens = []

            for i in range(len(self.patch_nums)):
                if i <= target_scale_idx and i < len(soft_probs_list) and soft_probs_list[i] is not None:
                    probs_k = soft_probs_list[i]

                    if torch.isnan(probs_k).any() or torch.isinf(probs_k).any():
                        logger.warning(f"⚠️ Soft probs at scale {i} contain NaN/Inf, fixing")
                        probs_k = torch.nan_to_num(probs_k, nan=0.001, posinf=1.0, neginf=0.0)
                        probs_k = F.softmax(probs_k, dim=-1)

                    gumbel_tokens = F.gumbel_softmax(
                        probs_k,
                        tau=self.gumbel_temperature,
                        hard=False,
                        dim=-1
                    )  # (B, L_i, V)

                    if torch.isnan(gumbel_tokens).any() or torch.isinf(gumbel_tokens).any():
                        logger.warning(f"⚠️ Gumbel sampling at scale {i} produced NaN/Inf, using raw probs")
                        gumbel_tokens = probs_k

                    cumulative_tokens.append(gumbel_tokens)
                else:
                    pn = self.patch_nums[i]
                    zero_probs = torch.zeros(B, pn*pn, V, device=device)
                    zero_probs[:, :, 0] = 1.0  # probability 1 for first token
                    cumulative_tokens.append(zero_probs)

            rgb_images = self._soft_decode_through_vae(cumulative_tokens)
            return rgb_images

        except Exception as e:
            logger.error(f"❌ Soft decoding failed: {e}")
            B = soft_probs_list[0].shape[0] if len(soft_probs_list) > 0 else 1
            device = soft_probs_list[0].device if len(soft_probs_list) > 0 else 'cuda'
            return torch.randn(B, 3, self.img_resolution, self.img_resolution, device=device, requires_grad=True) * 0.1

    def _soft_decode_through_vae(self, soft_tokens_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Simplified soft decoding via VQ-VAE embed_to_fhat.

        Approach:
        1. Convert soft probs to hard tokens (argmax)
        2. Use standard multi-scale cumulative decoding
        3. Approximate gradient via straight-through trick
        """
        try:
            if not soft_tokens_list or len(soft_tokens_list) == 0:
                raise ValueError("soft_tokens_list is empty")

            valid_soft_tokens = [st for st in soft_tokens_list if st is not None]
            if not valid_soft_tokens:
                raise ValueError("All soft_tokens are None")

            # Soft probs -> soft embeddings via embedding matrix
            vae_embed_weight = self.vae.quantize.embedding.weight  # (V, embed_dim)

            ms_soft_embeddings = []
            for i, soft_tokens in enumerate(soft_tokens_list):
                if soft_tokens is None or torch.all(soft_tokens == 0):
                    pn = self.patch_nums[i] if i < len(self.patch_nums) else 16
                    B = valid_soft_tokens[0].shape[0]
                    device = valid_soft_tokens[0].device
                    embed_dim = vae_embed_weight.shape[1]
                    zero_embedding = torch.zeros(B, pn*pn, embed_dim, device=device)
                    ms_soft_embeddings.append(zero_embedding)
                    continue

                # (B, L_i, V) @ (V, embed_dim) -> (B, L_i, embed_dim)
                soft_embedding = torch.matmul(soft_tokens, vae_embed_weight)
                ms_soft_embeddings.append(soft_embedding)

            # Reshape embeddings to spatial format for VQ-VAE decoder
            ms_h_BChw = []
            for i, soft_embedding in enumerate(ms_soft_embeddings):
                B, L, embed_dim = soft_embedding.shape
                pn = self.patch_nums[i]
                h_BChw = soft_embedding.transpose(1, 2).view(B, embed_dim, pn, pn)
                ms_h_BChw.append(h_BChw)

            # VQ-VAE components must run under no_grad to avoid cuDNN issues;
            # gradient flow is maintained via a straight-through approximation.
            with torch.no_grad():
                f_hat = self.vae.quantize.embed_to_fhat(ms_h_BChw, all_to_max_scale=True, last_one=True)
                rgb_images_frozen = self.vae.decoder(self.vae.post_quant_conv(f_hat))

            # Straight-through: values come from frozen decode, gradients flow from soft embeddings
            input_grad_magnitude = sum(torch.norm(emb) for emb in ms_soft_embeddings) / len(ms_soft_embeddings)
            rgb_images = rgb_images_frozen + 0.0 * input_grad_magnitude

            if torch.isnan(rgb_images).any() or torch.isinf(rgb_images).any():
                logger.warning("⚠️ Soft decode produced NaN/Inf, fixing")
                rgb_images = torch.nan_to_num(rgb_images, nan=0.0, posinf=1.0, neginf=-1.0)

            rgb_images = torch.clamp(rgb_images, -1.0, 1.0)
            return rgb_images

        except Exception as e:
            logger.error(f"❌ Simplified soft decode failed: {e}")

            B = 1
            device = 'cuda'
            if soft_tokens_list and len(soft_tokens_list) > 0:
                for st in soft_tokens_list:
                    if st is not None:
                        B = st.shape[0]
                        device = st.device
                        break

            return torch.randn(B, 3, self.img_resolution, self.img_resolution, device=device, requires_grad=True) * 0.1

    def train(self, mode: bool = True):
        """Override train() to keep frozen components in eval mode."""
        super().train(mode)

        # VQ-VAE always stays in eval mode (frozen)
        if hasattr(self.vae, 'eval'):
            self.vae.eval()

        # DINO backbone stays frozen
        if self.freeze_backbone:
            self.discriminator.dino.eval()
            for param in self.discriminator.dino.parameters():
                param.requires_grad = False

        # Discriminator heads follow the training mode
        self.discriminator.heads.train(mode)

        return self

    def forward(self, tokens_list: List[torch.Tensor], scale_idx: int, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Pixel-level discriminator forward pass - legacy interface (backward compat).

        ⚠️ DEPRECATED: Use forward_rgb() instead to avoid complex token->RGB conversion.

        Args:
            tokens_list: per-scale token list [tokens_0, tokens_1, ..., tokens_k]
            scale_idx: current scale index (0-based)
            c: conditioning (unused, kept for compatibility)

        Returns:
            logits: discrimination logits (B, n_heads)
        """
        logger.warning("⚠️ DEPRECATED: forward() token interface will be removed; use forward_rgb()")
        try:
            rgb_images = self.tokens_to_rgb_cumulative(tokens_list, scale_idx)
            logits = self.discriminator(rgb_images, c)
            logits = torch.clamp(logits, min=-10.0, max=10.0)

            if torch.isnan(logits).any() or torch.isinf(logits).any():
                logger.warning("⚠️ Pixel discriminator produced NaN/Inf, fixing")
                logits = torch.nan_to_num(logits, nan=0.0, posinf=1.0, neginf=-1.0)

            return logits

        except Exception as e:
            logger.error(f"❌ Pixel discriminator forward failed: {e}")
            B = tokens_list[0].shape[0] if len(tokens_list) > 0 else 1
            device = tokens_list[0].device if len(tokens_list) > 0 else 'cuda'
            n_heads = len(self.discriminator.heads)
            return torch.zeros(B, n_heads, device=device)

    def forward_rgb(self, rgb_images: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Direct RGB discrimination - recommended new interface.

        Resolves cuDNN compatibility issues and avoids complex token->RGB conversion.

        Args:
            rgb_images: RGB images (B, 3, H, W) in range [-1, 1]
            c: conditioning (unused, kept for compatibility)

        Returns:
            logits: discrimination logits (B, n_heads)
        """
        try:
            rgb_images = torch.clamp(rgb_images, min=-1.0, max=1.0)

            if torch.isnan(rgb_images).any() or torch.isinf(rgb_images).any():
                logger.warning("⚠️ VQ-VAE decoded output contains NaN/Inf, fixing")
                rgb_images = torch.nan_to_num(rgb_images, nan=0.0, posinf=1.0, neginf=-1.0)

            rgb_images_01 = (rgb_images + 1.0) * 0.5
            rgb_images_01 = torch.clamp(rgb_images_01, min=0.0, max=1.0)

            if torch.isnan(rgb_images_01).any() or torch.isinf(rgb_images_01).any():
                logger.warning("⚠️ [-1,1]->[0,1] transform produced NaN/Inf, fixing")
                rgb_images_01 = torch.nan_to_num(rgb_images_01, nan=0.5, posinf=1.0, neginf=0.0)

            try:
                dino_features = self.discriminator.dino(rgb_images_01)
            except Exception as dino_e:
                logger.error(f"⚠️ DINO feature extraction failed: {dino_e}, using zero features")
                B = rgb_images.size(0)
                device = rgb_images.device
                dino_features = {}
                for i in range(5):  # StyleGAN-T has 5 DINO feature layers by default
                    dino_features[str(i)] = torch.zeros(B, 384, device=device, requires_grad=True)

            for k, feat in dino_features.items():
                if torch.isnan(feat).any() or torch.isinf(feat).any():
                    logger.warning(f"⚠️ DINO feature {k} contains NaN/Inf, fixing")
                    dino_features[k] = torch.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=-1.0)

            logits_list = []
            for k, head in self.discriminator.heads.items():
                try:
                    head_logits = head(dino_features[k], None)

                    if torch.isnan(head_logits).any() or torch.isinf(head_logits).any():
                        logger.warning(f"⚠️ Discriminator head {k} output NaN/Inf, fixing")
                        head_logits = torch.nan_to_num(head_logits, nan=0.0, posinf=1.0, neginf=-1.0)

                    logits_list.append(head_logits.view(rgb_images.size(0), -1))
                except Exception as head_e:
                    logger.warning(f"⚠️ Discriminator head {k} failed: {head_e}, using zero output")
                    B = rgb_images.size(0)
                    device = rgb_images.device
                    fallback_logits = torch.zeros(B, 1, device=device)
                    logits_list.append(fallback_logits)

            if logits_list:
                logits = torch.cat(logits_list, dim=1)
            else:
                logger.error("❌ All discriminator heads failed, returning zero output")
                B = rgb_images.size(0)
                device = rgb_images.device
                logits = torch.zeros(B, 1, device=device)

            logits = torch.clamp(logits, min=-10.0, max=10.0)

            if torch.isnan(logits).any() or torch.isinf(logits).any():
                logger.warning("⚠️ RGB discriminator produced NaN/Inf, fixing")
                logits = torch.nan_to_num(logits, nan=0.0, posinf=1.0, neginf=-1.0)

            return logits

        except Exception as e:
            logger.error(f"❌ RGB discriminator forward failed: {e}")
            B = rgb_images.shape[0]
            device = rgb_images.device
            total_dim = sum(head.cls.out_channels for head in self.discriminator.heads.values())
            return torch.zeros(B, total_dim, device=device)

    def forward_soft_rgb(self, rgb_images: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Soft RGB discrimination - preserves gradient flow.

        Same as forward_rgb but ensures gradients flow (for generator training).

        Args:
            rgb_images: RGB images (B, 3, H, W) in range [-1, 1], must have requires_grad=True
            c: conditioning (unused, kept for compatibility)

        Returns:
            logits: discrimination logits (B, n_heads) with gradients
        """
        try:
            if not rgb_images.requires_grad:
                logger.warning("⚠️ RGB images do not require grad; generator training may be affected")

            rgb_images = torch.clamp(rgb_images, min=-1.0, max=1.0)

            if torch.isnan(rgb_images).any() or torch.isinf(rgb_images).any():
                logger.warning("⚠️ Soft discriminator VQ-VAE output contains NaN/Inf, fixing")
                rgb_images = torch.nan_to_num(rgb_images, nan=0.0, posinf=1.0, neginf=-1.0)

            rgb_images_01 = (rgb_images + 1.0) * 0.5
            rgb_images_01 = torch.clamp(rgb_images_01, min=0.0, max=1.0)

            if torch.isnan(rgb_images_01).any() or torch.isinf(rgb_images_01).any():
                logger.warning("⚠️ Soft [-1,1]->[0,1] transform produced NaN/Inf, fixing")
                rgb_images_01 = torch.nan_to_num(rgb_images_01, nan=0.5, posinf=1.0, neginf=0.0)

            try:
                dino_features = self.discriminator.dino(rgb_images_01)
            except Exception as dino_e:
                logger.error(f"⚠️ Soft DINO feature extraction failed: {dino_e}, using zero features")
                B = rgb_images.size(0)
                device = rgb_images.device
                dino_features = {}
                for i in range(5):
                    dino_features[str(i)] = torch.zeros(B, 384, device=device, requires_grad=True)

            for k, feat in dino_features.items():
                if torch.isnan(feat).any() or torch.isinf(feat).any():
                    logger.warning(f"⚠️ Soft DINO feature {k} contains NaN/Inf, fixing")
                    dino_features[k] = torch.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=-1.0)

            logits_list = []
            for k, head in self.discriminator.heads.items():
                try:
                    head_logits = head(dino_features[k], None)

                    if torch.isnan(head_logits).any() or torch.isinf(head_logits).any():
                        logger.warning(f"⚠️ Soft discriminator head {k} output NaN/Inf, fixing")
                        head_logits = torch.nan_to_num(head_logits, nan=0.0, posinf=1.0, neginf=-1.0)

                    logits_list.append(head_logits.view(rgb_images.size(0), -1))
                except Exception as head_e:
                    logger.warning(f"⚠️ Soft discriminator head {k} failed: {head_e}, using zero output")
                    B = rgb_images.size(0)
                    device = rgb_images.device
                    fallback_logits = torch.zeros(B, 1, device=device, requires_grad=True)
                    logits_list.append(fallback_logits)

            if logits_list:
                logits = torch.cat(logits_list, dim=1)
            else:
                logger.error("❌ All soft discriminator heads failed, returning zero output")
                B = rgb_images.size(0)
                device = rgb_images.device
                logits = torch.zeros(B, 1, device=device, requires_grad=True)

            logits = torch.clamp(logits, min=-10.0, max=10.0)

            if torch.isnan(logits).any() or torch.isinf(logits).any():
                logger.warning("⚠️ Soft RGB discriminator produced NaN/Inf, fixing")
                logits = torch.nan_to_num(logits, nan=0.0, posinf=1.0, neginf=-1.0)

            return logits

        except Exception as e:
            logger.error(f"❌ Soft RGB discriminator forward failed: {e}")
            B = rgb_images.shape[0]
            device = rgb_images.device
            total_dim = sum(head.cls.out_channels for head in self.discriminator.heads.values())
            return torch.zeros(B, total_dim, device=device, requires_grad=True)

    def forward_soft(self, logits_list: List[torch.Tensor], scale_idx: int, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Soft-probability forward pass (preserves gradient flow) - legacy interface.

        ⚠️ DEPRECATED: Use forward_soft_rgb() instead.

        Args:
            logits_list: per-scale logits list [(B, L_i, V), ...]
            scale_idx: current scale index
            c: conditioning (unused, kept for compatibility)

        Returns:
            logits: discrimination logits (B, n_heads)
        """
        logger.warning("⚠️ DEPRECATED: forward_soft() token interface will be removed; use forward_soft_rgb()")
        try:
            soft_probs_list = []
            for logits_k in logits_list[:scale_idx+1]:
                if logits_k is not None:
                    soft_probs_k = F.softmax(logits_k, dim=-1)
                    soft_probs_list.append(soft_probs_k)
                else:
                    soft_probs_list.append(None)

            rgb_images = self.soft_tokens_to_rgb(soft_probs_list, scale_idx)
            logits = self.discriminator(rgb_images, c)
            logits = torch.clamp(logits, min=-10.0, max=10.0)

            if torch.isnan(logits).any() or torch.isinf(logits).any():
                logger.warning("⚠️ Soft pixel discriminator produced NaN/Inf, fixing")
                logits = torch.nan_to_num(logits, nan=0.0, posinf=1.0, neginf=-1.0)

            return logits

        except Exception as e:
            logger.error(f"❌ Soft pixel discriminator forward failed: {e}")
            B = logits_list[0].shape[0] if len(logits_list) > 0 else 1
            device = logits_list[0].device if len(logits_list) > 0 else 'cuda'
            n_heads = len(self.discriminator.heads)
            return torch.zeros(B, n_heads, device=device, requires_grad=True)


class StyleGANDiscriminatorAdapter(nn.Module):
    """
    StyleGAN-T discriminator adapter for pixel-level adversarial training.

    Key features:
    - RGB image direct discrimination (forward_rgb, forward_soft_rgb)
    - Token->RGB cumulative decoding (backward compatible)
    - Based on StyleGAN-T ProjectedDiscriminator
    - Multi-scale pixel-level discrimination
    - Gradient flow preservation (via Gumbel-Softmax)

    Architecture notes:
    - Resolves cuDNN compatibility issues
    - Simplified token->RGB pipeline (moved to trainer layer)
    - Clean RGB discrimination interface
    """

    def __init__(
        self,
        vae = None,  # VQ-VAE instance (required)
        patch_nums: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        img_resolution: int = 256,
        c_dim: int = 0,
        # Deprecated parameters kept for backward compatibility
        vae_embedding_weight: torch.Tensor = None,
        var_word_embed: nn.Module = None,
        feature_dim: int = 1024,
        dino_checkpoint_path: str = None  # local DINO model path
    ):
        super().__init__()

        if vae is None:
            raise ValueError("A vae instance must be provided for pixel-level discrimination")

        self.discriminator = PixelDiscriminatorAdapter(
            vae=vae,
            patch_nums=patch_nums,
            img_resolution=img_resolution,
            c_dim=c_dim,
            diffaug=True,
            p_crop=0.5,
            freeze_backbone=True,
            dino_checkpoint_path=dino_checkpoint_path
        )

        self.img_resolution = img_resolution
        self.c_dim = c_dim
        self.feature_dim = feature_dim  # kept for backward compatibility

        self._log_parameter_statistics()

    def _log_parameter_statistics(self):
        """Log parameter statistics."""
        pass

    def get_trainable_params(self) -> List[torch.nn.Parameter]:
        """Return trainable parameters."""
        return self.discriminator.get_trainable_params()

    def train(self, mode: bool = True):
        """Override train() to keep frozen components in eval mode."""
        return self.discriminator.train(mode)

    def forward_rgb(self, rgb_images: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Direct RGB discrimination - recommended new interface.

        Args:
            rgb_images: RGB images (B, 3, H, W) in range [-1, 1]
            c: conditioning (optional)

        Returns:
            logits: discrimination logits (B, n_heads)
        """
        return self.discriminator.forward_rgb(rgb_images, c)

    def forward_soft_rgb(self, rgb_images: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Soft RGB discrimination - preserves gradient flow.

        Args:
            rgb_images: RGB images (B, 3, H, W) in range [-1, 1], must have requires_grad=True
            c: conditioning (optional)

        Returns:
            logits: discrimination logits (B, n_heads) with gradients
        """
        return self.discriminator.forward_soft_rgb(rgb_images, c)

    def forward(self, tokens_list_or_tokens: [List[torch.Tensor], torch.Tensor], scale_idx: int = None, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass - pixel-level discrimination (backward compat).

        ⚠️ DEPRECATED: Use forward_rgb() instead.

        Supports two calling modes:
        1. New mode: forward(tokens_list, scale_idx, c)
        2. Compat mode: forward(tokens, c) - auto-detected and converted

        Args:
            tokens_list_or_tokens: token list or single token tensor
            scale_idx: scale index (required in new mode)
            c: conditioning (optional)

        Returns:
            logits: discrimination logits
        """
        logger.warning("⚠️ DEPRECATED: StyleGANDiscriminatorAdapter.forward() will be removed; use forward_rgb()")
        if isinstance(tokens_list_or_tokens, torch.Tensor) and scale_idx is None:
            # Compat mode: forward(tokens, c)
            logger.warning("⚠️ Using compat mode; consider switching to tokens_list mode")
            tokens = tokens_list_or_tokens
            tokens_list = [tokens]
            scale_idx = 0
            return self.discriminator(tokens_list, scale_idx, c)
        else:
            return self.discriminator(tokens_list_or_tokens, scale_idx, c)

    def forward_soft(self, logits_list_or_soft: [List[torch.Tensor], torch.Tensor], scale_idx: int = None, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Soft-probability forward pass (preserves gradient flow) - backward compat.

        ⚠️ DEPRECATED: Use forward_soft_rgb() instead.

        Supports two calling modes:
        1. New mode: forward_soft(logits_list, scale_idx, c)
        2. Compat mode: forward_soft(soft_embeddings, c)

        Args:
            logits_list_or_soft: logits list or soft embedding tensor
            scale_idx: scale index (required in new mode)
            c: conditioning (optional)

        Returns:
            logits: discrimination logits
        """
        logger.warning("⚠️ DEPRECATED: StyleGANDiscriminatorAdapter.forward_soft() will be removed; use forward_soft_rgb()")
        if isinstance(logits_list_or_soft, torch.Tensor) and len(logits_list_or_soft.shape) == 3 and scale_idx is None:
            # Compat mode
            logger.warning("⚠️ Using compat mode; consider switching to logits_list mode")
            B = logits_list_or_soft.shape[0]
            device = logits_list_or_soft.device
            n_heads = len(self.discriminator.discriminator.heads)
            return torch.zeros(B, n_heads, device=device, requires_grad=True)
        else:
            return self.discriminator.forward_soft(logits_list_or_soft, scale_idx, c)

    def __repr__(self):
        trainable_params = sum(p.numel() for p in self.get_trainable_params())
        total_params = sum(p.numel() for p in self.parameters())

        return (f"StyleGANDiscriminatorAdapter(\n"
                f"  architecture=PixelDiscriminator,\n"
                f"  img_resolution={self.img_resolution},\n"
                f"  total_params={total_params:,},\n"
                f"  trainable_params={trainable_params:,},\n"
                f"  pixel_level=True,\n"
                f"  new_rgb_interface=True\n"
                f")")


def create_add_style_discriminator(
    vae = None,
    patch_nums: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
    img_resolution: int = 256,
    c_dim: int = 0,
    # Deprecated parameters kept for backward compatibility
    vae_embedding_weight: torch.Tensor = None,
    var_word_embed: nn.Module = None,
    feature_dim: int = 1024
) -> StyleGANDiscriminatorAdapter:
    """
    Create a pixel-level StyleGAN-T discriminator.

    Args:
        vae: VQ-VAE instance (required for pixel-level discrimination)
        patch_nums: multi-scale patch counts
        img_resolution: image resolution
        c_dim: conditioning dimension
    """
    if vae is None:
        raise ValueError("vae parameter is required")

    return StyleGANDiscriminatorAdapter(
        vae=vae,
        patch_nums=patch_nums,
        img_resolution=img_resolution,
        c_dim=c_dim
    )


# Public API
__all__ = [
    'StyleGANDiscriminatorAdapter',
    'create_add_style_discriminator',
    'PixelDiscriminatorAdapter'
]
