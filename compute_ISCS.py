#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ISCS (Inter-Scale Consistency Score) computation script.

ISCS is a metric designed for AID-VAR that quantifies the quality of
conditional predictions p(r_k|r_{<k'}). Implementation uses Frechet Joint
Distribution (FJD) divergence applied to VAR's multi-scale generation process.

For each scale k, a joint feature vector is constructed and Frechet distance
is computed between real and generated distributions.
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("seaborn not installed, falling back to basic matplotlib")

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.var import VAR
from models.vqvae import VQVAE
from models.guidance_injector import GuidanceInjector
from models import build_vae_var
from utils.iscs_utils import (
    AdvancedFeatureExtractor,
    NumericalStabilityTools,
    ISCSAnalyzer,
    load_checkpoint_safely,
    setup_logging
)
import dist

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('iscs_computation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DINOv2FeatureExtractor(nn.Module):
    """
    DINO feature extractor using local model checkpoint.
    Extracts semantic features for building joint feature vectors.
    """

    def __init__(self, model_name='dinov2_vits14', device='cuda'):
        super().__init__()
        self.device = device
        self.model_name = model_name

        try:
            import timm
            from stylegan_t.networks.discriminator import DINO

            local_dino_path = "/home/intern/Ligong/VAR/checkpoints/dino/dino_deitsmall16_pretrain.pth"

            self.model = DINO(
                hooks=[2, 5, 8, 11],
                hook_patch=True,
                local_checkpoint_path=local_dino_path
            ).to(device)

            self.model.eval()

            for param in self.model.parameters():
                param.requires_grad = False

        except Exception as e:
            logger.error(f"Failed to load local DINO model: {e}")
            raise

        # Feature dimension for DINO ViT-S
        self.feature_dim = 384

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract image features.

        Args:
            images: Input images (B, 3, H, W) in range [0, 1]

        Returns:
            features: Image features (B, feature_dim)
        """
        if images.max() > 1.0:
            images = images / 255.0

        # Convert [0, 1] to [-1, 1] for StyleGAN-T DINO
        images = images * 2.0 - 1.0

        with torch.no_grad():
            features_dict = self.model(images)

            # Use the last-layer features (key '4')
            if '4' in features_dict:
                features = features_dict['4']
            elif len(features_dict) > 0:
                features = list(features_dict.values())[-1]
            else:
                raise ValueError("DINO model returned no valid features")

            # Ensure 2D output (B, feature_dim)
            if len(features.shape) > 2:
                features = features.mean(dim=list(range(2, len(features.shape))))

        return features

class ImageDataset(Dataset):
    """
    Image dataset supporting loading from directory or npz file.
    """

    def __init__(self, data_source: Union[str, np.ndarray], transform=None):
        self.transform = transform

        if isinstance(data_source, str):
            self.images = []
            for img_path in Path(data_source).glob('*.png'):
                self.images.append(str(img_path))
        elif isinstance(data_source, np.ndarray):
            self.images = data_source

        else:
            raise ValueError("data_source must be a directory path or numpy array")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if isinstance(self.images[idx], str):
            image = Image.open(self.images[idx]).convert('RGB')
            image = transforms.ToTensor()(image)
        else:
            image = torch.from_numpy(self.images[idx]).float()
            if image.max() > 1.0:
                image = image / 255.0

        if self.transform:
            image = self.transform(image)

        return image

class ISCSCalculator:
    """
    ISCS (Inter-Scale Consistency Score) calculator.
    Implements joint distribution divergence to evaluate multi-scale generation quality.
    """

    def __init__(self,
                 feature_extractor: DINOv2FeatureExtractor,
                 patch_nums: Tuple[int] = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
                 device: str = 'cuda'):

        self.feature_extractor = feature_extractor
        self.patch_nums = patch_nums
        self.device = device

        # Coarser scales get higher weights
        self.weights = {k: 2**k for k in range(len(patch_nums))}
        total_weight = sum(self.weights.values())
        self.weights = {k: w/total_weight for k, w in self.weights.items()}


    def extract_multiscale_features(self,
                                  images: torch.Tensor,
                                  var_model: VAR) -> Dict[int, torch.Tensor]:
        """
        Extract multi-scale feature representations.

        Args:
            images: Input images (B, 3, H, W)
            var_model: VAR model for multi-scale token representations

        Returns:
            scale_features: Feature dict per scale {scale_idx: features}
        """
        scale_features = {}

        with torch.no_grad():
            vae = var_model.vae_proxy[0]

            # Normalize to [-1, 1]
            if images.max() <= 1.0:
                images = images * 2.0 - 1.0

            images = images.to(dtype=torch.float32)

            f = vae.quant_conv(vae.encoder(images))

            ms_idx_Bl = vae.quantize.f_to_idxBl_or_fhat(f, to_fhat=False, v_patch_nums=self.patch_nums)

            for scale_idx, patch_num in enumerate(self.patch_nums):
                if scale_idx < len(ms_idx_Bl):
                    idx_Bl = ms_idx_Bl[scale_idx]  # (B, L_k)

                    # Get continuous features via VAE embedding, then project to VAR space
                    vae_embeddings = vae.quantize.embedding(idx_Bl)  # (B, L_k, Cvae)
                    scale_embeddings = var_model.word_embed(vae_embeddings)  # (B, L_k, C)

                    # Global average pooling to fixed-length feature
                    scale_features[scale_idx] = scale_embeddings.mean(dim=1)  # (B, C)
                else:
                    logger.warning(f"Scale {scale_idx} out of range, skipping")
                    continue

        return scale_features

    def build_joint_features(self,
                           scale_features: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        """
        Build joint feature vectors [feat(s_{k-1}), feat(s_k)].

        Args:
            scale_features: Per-scale features

        Returns:
            joint_features: Joint feature dict {scale_idx: joint_feat}
        """
        joint_features = {}

        for scale_idx in range(1, len(self.patch_nums)):
            prev_feat = scale_features[scale_idx - 1]  # s_{k-1}
            curr_feat = scale_features[scale_idx]       # s_k

            joint_feat = torch.cat([prev_feat, curr_feat], dim=1)  # (B, 2*C)
            joint_features[scale_idx] = joint_feat

        return joint_features

    def compute_frechet_distance(self,
                               real_features: torch.Tensor,
                               generated_features: torch.Tensor) -> float:
        """
        Compute Frechet distance between two feature sets.

        Args:
            real_features: Real data features (N, D)
            generated_features: Generated data features (M, D)

        Returns:
            frechet_distance: FJD distance
        """
        if isinstance(real_features, torch.Tensor):
            real_features = real_features.cpu().numpy()
        if isinstance(generated_features, torch.Tensor):
            generated_features = generated_features.cpu().numpy()

        mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
        mu2, sigma2 = generated_features.mean(axis=0), np.cov(generated_features, rowvar=False)

        # Numerical stability
        eps = 1e-6
        sigma1 += eps * np.eye(sigma1.shape[0])
        sigma2 += eps * np.eye(sigma2.shape[0])

        diff = mu1 - mu2
        covmean = self._sqrtm_newton(sigma1.dot(sigma2))

        fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)

        return float(fid)

    def _sqrtm_newton(self, A: np.ndarray, num_iters: int = 50) -> np.ndarray:
        """
        Compute matrix square root via Newton iteration (numerically stable).
        """
        dim = A.shape[0]
        X = A.copy()

        for _ in range(num_iters):
            X_inv = np.linalg.pinv(X)
            X_new = 0.5 * (X + A.dot(X_inv))

            if np.allclose(X, X_new, rtol=1e-6):
                break
            X = X_new

        return X

    def compute_scale_iscs(self,
                          real_joint_features: Dict[int, torch.Tensor],
                          generated_joint_features: Dict[int, torch.Tensor]) -> Dict[int, float]:
        """
        Compute per-scale ISCS scores (lower FJD distance = better).

        Args:
            real_joint_features: Joint features from real data
            generated_joint_features: Joint features from generated data

        Returns:
            scale_scores: Per-scale ISCS scores
        """
        scale_scores = {}

        for scale_idx in real_joint_features.keys():
            real_feat = real_joint_features[scale_idx]
            gen_feat = generated_joint_features[scale_idx]

            fjd = self.compute_frechet_distance(real_feat, gen_feat)
            iscs_score = fjd
            scale_scores[scale_idx] = iscs_score

            logger.info(f"Scale {scale_idx}: FJD={fjd:.4f}, ISCS={iscs_score:.4f}")

        return scale_scores

    def compute_aggregated_iscs(self, scale_scores: Dict[int, float]) -> float:
        """
        Compute weighted aggregate ISCS score.

        Args:
            scale_scores: Per-scale scores

        Returns:
            total_iscs: Weighted total score
        """
        total_score = 0.0
        total_weight = 0.0

        for scale_idx, score in scale_scores.items():
            weight = self.weights.get(scale_idx, 0.0)
            total_score += weight * score
            total_weight += weight

        if total_weight > 0:
            total_iscs = total_score / total_weight
        else:
            total_iscs = 0.0

        logger.info(f"Aggregated ISCS: {total_iscs:.4f}")
        return total_iscs

    def compute_sample_iscs(self,
                           real_joint_features: Dict[int, torch.Tensor],
                           sample_scale_features: Dict[int, torch.Tensor]) -> float:
        """
        Compute ISCS score for a single sample.

        Args:
            real_joint_features: Joint features from real data (reference distribution)
            sample_scale_features: Multi-scale features for one sample

        Returns:
            sample_iscs: ISCS score for the sample
        """
        sample_joint_features = self.build_joint_features(sample_scale_features)

        scale_scores = {}

        for scale_idx in sample_joint_features.keys():
            if scale_idx in real_joint_features:
                real_feat = real_joint_features[scale_idx]
                sample_feat = sample_joint_features[scale_idx]  # (1, D)

                device = self.device
                real_feat = real_feat.to(device)
                sample_feat = sample_feat.to(device)

                # Euclidean distance from sample to real distribution mean
                real_mean = real_feat.mean(dim=0)  # (D,)
                sample_vec = sample_feat.squeeze(0)  # (D,)
                distance = torch.norm(sample_vec - real_mean).item()
                scale_scores[scale_idx] = distance

        total_score = 0.0
        total_weight = 0.0

        for scale_idx, score in scale_scores.items():
            weight = self.weights.get(scale_idx, 0.0)
            total_score += weight * score
            total_weight += weight

        if total_weight > 0:
            sample_iscs = total_score / total_weight
        else:
            sample_iscs = 0.0

        return sample_iscs

class ISCSExperiment:
    """
    ISCS experiment manager.
    Manages the full evaluation pipeline: data loading, model inference, metric computation.
    """

    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

        self._setup_optimization()

        self.feature_extractor = DINOv2FeatureExtractor(
            model_name=args.dinov2_model,
            device=self.device
        )

        self.iscs_calculator = ISCSCalculator(
            feature_extractor=self.feature_extractor,
            patch_nums=tuple(args.patch_nums),
            device=self.device
        )

        self._load_models()

        """Validate checkpoint files exist and provide download info if missing."""
        missing_files = []

        if not os.path.exists(self.args.vae_ckpt):
            missing_files.append(('VAE', self.args.vae_ckpt))

        if not os.path.exists(self.args.var_ckpt):
            missing_files.append(('VAR', self.args.var_ckpt))

        if missing_files:
            logger.error(f"Missing required checkpoint files:")
            for model_type, path in missing_files:
                logger.error(f"  - {model_type}: {path}")

            logger.error(f"Download from HuggingFace:")
            logger.error(f"  wget https://huggingface.co/FoundationVision/var/resolve/main/{self.args.vae_ckpt}")
            logger.error(f"  wget https://huggingface.co/FoundationVision/var/resolve/main/{self.args.var_ckpt}")

            raise FileNotFoundError("Missing required checkpoint files")

    def _setup_optimization(self):
        """Configure optimization settings (consistent with generate_aid_fid_samples.py)."""
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        tf32 = True
        torch.backends.cudnn.allow_tf32 = bool(tf32)
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.set_float32_matmul_precision('high' if tf32 else 'highest')

        if self.args.dtype == 'float16':
            self.dtype = torch.float16
        else:
            self.dtype = torch.bfloat16

        logger.info(f"Optimization config: TF32={tf32}, dtype={self.args.dtype}")

        if self.args.var_ckpt is None:
            self.args.var_ckpt = f'checkpoints/var_d{self.args.model_depth}.pth'
        elif not os.path.exists(self.args.var_ckpt):
            auto_var_ckpt = f'checkpoints/var_d{self.args.model_depth}.pth'
            if os.path.exists(auto_var_ckpt):
                self.args.var_ckpt = auto_var_ckpt

        self._validate_checkpoints()

        patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
        vae, var = build_vae_var(
            V=4096, Cvae=32, ch=160, share_quant_resi=4,
            device=self.device, patch_nums=patch_nums,
            num_classes=1000, depth=self.args.model_depth, shared_aln=False,
        )

        logger.info(f"Loading VAE checkpoint: {self.args.vae_ckpt}")
        vae.load_state_dict(torch.load(self.args.vae_ckpt, map_location='cpu'), strict=True)

        logger.info(f"Loading VAR checkpoint: {self.args.var_ckpt}")
        var.load_state_dict(torch.load(self.args.var_ckpt, map_location='cpu'), strict=True)

        vae.eval()
        var.eval()
        for p in vae.parameters(): p.requires_grad_(False)
        for p in var.parameters(): p.requires_grad_(False)

        self.vae = vae
        self.var = var

        if self.args.planner_ckpt and os.path.exists(self.args.planner_ckpt):
            var_embed_dim = self.args.model_depth * 64  # width = depth * 64

            self.planner = GuidanceInjector(
                input_dim=var_embed_dim,
                embed_dim=var_embed_dim,
                num_layers=2,
                num_heads=8
            ).to(self.device)

            logger.info(f"Loading GuidanceInjector checkpoint: {self.args.planner_ckpt}")
            planner_state = torch.load(self.args.planner_ckpt, map_location='cpu')

            if 'planner_state_dict' in planner_state:
                missing_keys, unexpected_keys = self.planner.load_state_dict(
                    planner_state['planner_state_dict'], strict=False
                )
                if unexpected_keys:
                    pos_keys = [k for k in unexpected_keys if k.startswith('pos_')]
                    other_keys = [k for k in unexpected_keys if not k.startswith('pos_')]
                    if pos_keys:
                        logger.info(f"Ignoring position encoding buffers: {pos_keys}")
                    if other_keys:
                        logger.warning(f"Unexpected keys: {other_keys}")
                if missing_keys:
                    logger.warning(f"Missing keys: {missing_keys}")
            elif 'planner' in planner_state:
                self.planner.load_state_dict(planner_state['planner'], strict=True)
            else:
                self.planner.load_state_dict(planner_state, strict=True)

            self.planner.eval()
            for p in self.planner.parameters(): p.requires_grad_(False)

            logger.info(f"GuidanceInjector loaded successfully")
        else:
            self.planner = None
            logger.info(f"No GuidanceInjector loaded, evaluating base VAR only")

        logger.info(f"Models loaded:")
        logger.info(f"  VAE params: {sum(p.numel() for p in vae.parameters()):,}")
        logger.info(f"  VAR params: {sum(p.numel() for p in var.parameters()):,}")
        if self.planner is not None:
            logger.info(f"  GuidanceInjector params: {sum(p.numel() for p in self.planner.parameters()):,}")

    def save_sample_images_with_scores(self,
                                     images: torch.Tensor,
                                     iscs_scores: List[float],
                                     output_dir: str,
                                     prefix: str = "sample"):
        """
        Save sample images annotated with ISCS scores.

        Args:
            images: Generated image tensors (B, 3, H, W) in range [0, 1]
            iscs_scores: ISCS score per sample
            output_dir: Output directory
            prefix: Filename prefix
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for i, (img_tensor, score) in enumerate(zip(images, iscs_scores)):
            img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)

            img_with_score = self._add_score_annotation(img_pil, score)

            filename = f"{prefix}_{i:02d}_iscs_{score:.4f}.png"
            save_path = output_path / filename
            img_with_score.save(save_path)

    def _add_score_annotation(self, image: Image.Image, score: float) -> Image.Image:
        """
        Annotate an image with its ISCS score in the top-right corner.

        Args:
            image: PIL image
            score: ISCS score

        Returns:
            annotated_image: Image with score annotation
        """
        img_copy = image.copy()
        draw = ImageDraw.Draw(img_copy)

        try:
            font_size = max(16, min(image.width // 20, 24))
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
            except:
                font = ImageFont.load_default()

        score_text = f"ISCS: {score:.4f}"

        text_bbox = draw.textbbox((0, 0), score_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        margin = 10
        x = image.width - text_width - margin
        y = margin

        bg_padding = 5
        bg_bbox = [
            x - bg_padding,
            y - bg_padding,
            x + text_width + bg_padding,
            y + text_height + bg_padding
        ]
        draw.rectangle(bg_bbox, fill=(0, 0, 0, 180))
        draw.text((x, y), score_text, fill=(255, 255, 255), font=font)

        return img_copy

    def generate_and_extract_features_streaming(self, num_samples: int, with_planner: bool = False):
        """
        Stream-generate samples and extract features.
        Saves the first 20 images annotated with their ISCS scores.

        Generation strategy:
        - 1000 classes, 50 samples per class
        - Fixed batch size of 50 (one full class per batch)
        - Class ID used as random seed for reproducibility

        Args:
            num_samples: Number of samples to generate (should be 50000)
            with_planner: Whether to use GuidanceInjector

        Returns:
            all_scale_features: Accumulated multi-scale features
        """
        expected_samples = 50000
        if num_samples != expected_samples:
            logger.warning(f"num_samples ({num_samples}) != expected ({expected_samples}), generating by class")

        accumulated_features = {}

        batch_size = 50
        samples_per_class = 50
        num_classes = 1000
        total_samples = 0

        saved_images_count = 0
        max_save_images = 20
        output_dir = Path(self.args.output_dir) / "sample_images_with_scores"

        # Pre-compute reference distribution from real data for per-sample ISCS
        real_images = self.load_real_images()
        real_scale_features = self.extract_real_features_streaming(real_images)
        real_joint_features = self.iscs_calculator.build_joint_features(real_scale_features)
        del real_images

        for class_id in tqdm(range(num_classes), desc=f"Generating {'(AID-VAR)' if with_planner else '(base VAR)'}"):
            torch.manual_seed(class_id)
            import random
            random.seed(class_id)
            np.random.seed(class_id)

            class_labels = [class_id] * samples_per_class
            current_batch_size = batch_size
            label_B = torch.tensor(class_labels, device=self.device)

            with torch.inference_mode():
                autocast_enabled = False
                with torch.autocast(self.device.type, enabled=autocast_enabled, dtype=self.dtype, cache_enabled=True):
                    if with_planner and self.planner is not None:
                        images = self.var.aid_guided_inference(
                            planner=self.planner,
                            B=current_batch_size,
                            label_B=label_B,
                            cfg=self.args.cfg,
                            top_k=self.args.top_k,
                            top_p=self.args.top_p,
                            g_seed=class_id,
                            more_smooth=self.args.more_smooth
                        )
                    else:
                        images = self.var.autoregressive_infer_cfg(
                            B=current_batch_size,
                            label_B=label_B,
                            cfg=self.args.cfg,
                            top_k=self.args.top_k,
                            top_p=self.args.top_p,
                            g_seed=class_id,
                            more_smooth=self.args.more_smooth
                        )

                    batch_features = self.iscs_calculator.extract_multiscale_features(images, self.var)

                    # Save first max_save_images images with ISCS annotations
                    if saved_images_count < max_save_images:
                        num_to_save = min(max_save_images - saved_images_count, current_batch_size)

                        sample_iscs_scores = []
                        for i in range(num_to_save):
                            sample_features = {}
                            for scale_idx, scale_feat in batch_features.items():
                                sample_features[scale_idx] = scale_feat[i:i+1]

                            sample_iscs = self.iscs_calculator.compute_sample_iscs(
                                real_joint_features, sample_features
                            )
                            sample_iscs_scores.append(sample_iscs)

                        self.save_sample_images_with_scores(
                            images[:num_to_save],
                            sample_iscs_scores,
                            output_dir,
                            prefix=f"{'aid_var' if with_planner else 'base_var'}_class{class_id}"
                        )

                        saved_images_count += num_to_save

                    # Accumulate features in CPU memory
                    for scale_idx, features in batch_features.items():
                        features_cpu = features.cpu()
                        if scale_idx not in accumulated_features:
                            accumulated_features[scale_idx] = []
                        accumulated_features[scale_idx].append(features_cpu)

                    total_samples += current_batch_size

                    del images, batch_features
                    torch.cuda.empty_cache()

        del real_scale_features, real_joint_features
        torch.cuda.empty_cache()

        # Concatenate features from all batches
        final_features = {}
        for scale_idx, feature_list in accumulated_features.items():
            final_features[scale_idx] = torch.cat(feature_list, dim=0)

        return final_features

    def load_real_images(self) -> torch.Tensor:
        """Load real image data from npz file or directory."""
        if self.args.real_images_path.endswith('.npz'):
            data = np.load(self.args.real_images_path)

            if 'images' in data:
                real_images = torch.from_numpy(data['images']).float()
            elif 'arr_0' in data:
                real_images = torch.from_numpy(data['arr_0']).float()
            else:
                key = list(data.keys())[0]
                real_images = torch.from_numpy(data[key]).float()

            # Ensure (N, C, H, W) format
            if len(real_images.shape) == 4:
                if real_images.shape[1] == 3:
                    pass  # Already (N, C, H, W)
                elif real_images.shape[3] == 3:
                    real_images = real_images.permute(0, 3, 1, 2)
                    logger.info(f"Converted format from (N,H,W,C) to (N,C,H,W)")
            elif len(real_images.shape) == 3:
                # Grayscale to RGB
                real_images = real_images.unsqueeze(1).repeat(1, 3, 1, 1)
                logger.info(f"Added channel dim, converted grayscale to RGB")
            else:
                raise ValueError(f"Unsupported image data shape: {real_images.shape}")

            if real_images.max() > 1.0:
                real_images = real_images / 255.0
                logger.info(f"Normalized data range from [0,255] to [0,1]")

        else:
            dataset = ImageDataset(self.args.real_images_path)
            dataloader = DataLoader(
                dataset, batch_size=self.args.batch_size,
                shuffle=False, num_workers=4
            )

            real_images = []
            for batch in tqdm(dataloader, desc="Loading real images"):
                real_images.append(batch)
            real_images = torch.cat(real_images, dim=0)

        if len(real_images) > self.args.num_samples:
            indices = torch.randperm(len(real_images))[:self.args.num_samples]
            real_images = real_images[indices]

        if len(real_images.shape) != 4 or real_images.shape[1] != 3:
            raise ValueError(f"Image data shape error: {real_images.shape}, expected (N, 3, H, W)")

        return real_images

    def run_experiment(self):
        """Run the full ISCS evaluation experiment (memory-efficient)."""
        results = {}

        # 1. Extract real image features
        real_images = self.load_real_images()
        real_scale_features = self.extract_real_features_streaming(real_images)

        real_joint_features = self.iscs_calculator.build_joint_features(real_scale_features)

        del real_images
        torch.cuda.empty_cache()

        # 2. Evaluate AID-VAR (our method)
        if self.planner is not None:
            gen_scale_features_aid = self.generate_and_extract_features_streaming(
                self.args.num_samples, with_planner=True
            )

            gen_joint_features_aid = self.iscs_calculator.build_joint_features(gen_scale_features_aid)

            scale_scores_aid = self.iscs_calculator.compute_scale_iscs(
                real_joint_features, gen_joint_features_aid
            )
            total_iscs_aid = self.iscs_calculator.compute_aggregated_iscs(scale_scores_aid)

            results['aid_var'] = {
                'scale_scores': scale_scores_aid,
                'total_iscs': total_iscs_aid
            }

            logger.info(f"AID-VAR ISCS (our method): {total_iscs_aid:.6f}")

            del gen_scale_features_aid, gen_joint_features_aid
            torch.cuda.empty_cache()

        # 3. Evaluate base VAR (baseline)
        gen_scale_features_base = self.generate_and_extract_features_streaming(
            self.args.num_samples, with_planner=False
        )

        gen_joint_features_base = self.iscs_calculator.build_joint_features(gen_scale_features_base)

        scale_scores_base = self.iscs_calculator.compute_scale_iscs(
            real_joint_features, gen_joint_features_base
        )
        total_iscs_base = self.iscs_calculator.compute_aggregated_iscs(scale_scores_base)

        results['base_var'] = {
            'scale_scores': scale_scores_base,
            'total_iscs': total_iscs_base
        }

        logger.info(f"Base VAR ISCS (baseline): {total_iscs_base:.6f}")

        if 'aid_var' in results:
            improvement = total_iscs_base - results['aid_var']['total_iscs']
            improvement_pct = (improvement / total_iscs_base) * 100

            logger.info(f"ISCS improvement: {improvement:.6f} ({improvement_pct:.2f}%)")
            logger.info(f"Our method vs baseline: {'better' if improvement > 0 else 'worse'}")

        del gen_scale_features_base, gen_joint_features_base
        torch.cuda.empty_cache()

        del real_joint_features
        torch.cuda.empty_cache()

        self._save_results(results)
        self._visualize_results(results)

        return results

    def extract_real_features_streaming(self, real_images: torch.Tensor):
        """
        Stream-extract real image features (memory-efficient).

        Args:
            real_images: Real image tensor

        Returns:
            real_scale_features: Multi-scale features for real images
        """
        accumulated_features = {}
        batch_size = self.args.batch_size
        num_samples = min(len(real_images), self.args.num_samples)
        total_batches = (num_samples + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(total_batches), desc="Extracting real image features"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)

            batch_images = real_images[start_idx:end_idx].to(self.device)

            if len(batch_images.shape) != 4 or batch_images.shape[1] != 3:
                logger.error(f"Batch image shape error: {batch_images.shape}")
                raise ValueError(f"Batch image shape error: {batch_images.shape}, expected (B, 3, H, W)")

            with torch.no_grad():
                batch_images = batch_images.to(dtype=torch.float32)

                batch_features = self.iscs_calculator.extract_multiscale_features(batch_images, self.var)

                for scale_idx, features in batch_features.items():
                    features_cpu = features.cpu()
                    if scale_idx not in accumulated_features:
                        accumulated_features[scale_idx] = []
                    accumulated_features[scale_idx].append(features_cpu)

                del batch_images, batch_features
                torch.cuda.empty_cache()

        final_features = {}
        for scale_idx, feature_list in accumulated_features.items():
            final_features[scale_idx] = torch.cat(feature_list, dim=0)

        return final_features

    def _save_results(self, results: Dict):
        """Save experiment results to JSON."""
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(exist_ok=True)

        results_path = output_dir / 'iscs_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved: {results_path}")

    def _visualize_results(self, results: Dict):
        """Visualize ISCS results."""
        output_dir = Path(self.args.output_dir)

        plt.figure(figsize=(12, 6))

        scales = list(results['base_var']['scale_scores'].keys())
        base_scores = [results['base_var']['scale_scores'][s] for s in scales]

        plt.subplot(1, 2, 1)
        plt.plot(scales, base_scores, 'o-', label='Base VAR', linewidth=2, markersize=6)

        if 'aid_var' in results:
            aid_scores = [results['aid_var']['scale_scores'][s] for s in scales]
            plt.plot(scales, aid_scores, 's-', label='AID-VAR', linewidth=2, markersize=6)

        plt.xlabel('Scale index')
        plt.ylabel('ISCS score')
        plt.title('Per-scale ISCS comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        models = ['Base VAR']
        total_scores = [results['base_var']['total_iscs']]

        if 'aid_var' in results:
            models.append('AID-VAR')
            total_scores.append(results['aid_var']['total_iscs'])

        bars = plt.bar(models, total_scores, color=['skyblue', 'lightcoral'][:len(models)])
        plt.ylabel('Total ISCS score')
        plt.title('Total ISCS comparison')

        for bar, score in zip(bars, total_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{score:.4f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(output_dir / 'iscs_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Visualization saved: {output_dir}/iscs_comparison.png")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='ISCS (Inter-Scale Consistency Score) computation')

    parser.add_argument('--vae_ckpt', type=str, default='vae_ch160v4096z32.pth',
                       help='VAE model checkpoint path')
    parser.add_argument('--var_ckpt', type=str, default=None,
                       help='VAR model checkpoint path (auto-determined from model_depth if None)')
    parser.add_argument('--planner_ckpt', type=str, default=None,
                       help='GuidanceInjector checkpoint path (optional)')
    parser.add_argument('--model_depth', type=int, default=16, choices=[16, 20, 24, 30],
                       help='VAR model depth')

    parser.add_argument('--real_images_path', type=str, required=True,
                       help='Real image data path (directory or npz file)')
    parser.add_argument('--num_samples', type=int, default=5000,
                       help='Number of samples to evaluate')

    parser.add_argument('--cfg', type=float, default=1.5,
                       help='Classifier-free guidance scale')
    parser.add_argument('--top_k', type=int, default=900,
                       help='Top-k sampling parameter')
    parser.add_argument('--top_p', type=float, default=0.96,
                       help='Top-p sampling parameter')
    parser.add_argument('--more_smooth', action='store_true',
                       help='Enable more_smooth for better visual quality')
    parser.add_argument('--dtype', type=str, default='float16', choices=['float16', 'bfloat16'],
                       help='Inference data type')

    parser.add_argument('--dinov2_model', type=str, default='dinov2_vits14',
                       choices=['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14'],
                       help='DINOv2 model version')

    parser.add_argument('--device', type=str, default='cuda',
                       help='Compute device')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--output_dir', type=str, default='./iscs_results',
                       help='Results output directory')

    parser.add_argument('--patch_nums', type=int, nargs='+',
                       default=[1, 2, 3, 4, 5, 6, 8, 10, 13, 16],
                       help='VAR patch number sequence')

    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)

    if not dist.initialized():
        dist.initialize()

    logger.info("Starting ISCS evaluation experiment")
    logger.info(f"Config: depth={args.model_depth}, cfg={args.cfg}, top_k={args.top_k}, top_p={args.top_p}")
    logger.info(f"Data: real_images={args.real_images_path}, num_samples={args.num_samples}")
    logger.info(f"System: device={args.device}, dtype={args.dtype}, planner={args.planner_ckpt or 'none'}")

    try:
        experiment = ISCSExperiment(args)
        results = experiment.run_experiment()

        logger.info("="*60)
        logger.info("ISCS Evaluation Results:")

        if 'aid_var' in results:
            logger.info(f"AID-VAR ISCS (ours): {results['aid_var']['total_iscs']:.4f}")

        logger.info(f"Base VAR ISCS (baseline): {results['base_var']['total_iscs']:.4f}")

        if 'aid_var' in results:
            improvement = results['base_var']['total_iscs'] - results['aid_var']['total_iscs']
            logger.info(f"ISCS improvement: {improvement:.4f} ({improvement/results['base_var']['total_iscs']*100:.2f}%)")
            logger.info(f"Conclusion: our method is {'better' if improvement > 0 else 'worse'} than baseline")

        logger.info("="*60)

    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    main()
