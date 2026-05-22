#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EB-C (Exposure Bias - Consistency) computation script.

EB-C(M, l, f_div) = CGD(M|M, l, f_div) / CGD(M|D, l, f_div)

- Numerator CGD(M|M, l): divergence between the distribution obtained by
  autoregressive inference (using the model's own generated prefix) and the
  real data distribution.
- Denominator CGD(M|D, l): divergence between the distribution obtained by
  a single forward pass (using the real data as prefix) and the real data
  distribution.
- f_div: JS divergence.
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.var import VAR
from models.vqvae import VQVAE
from models.guidance_injector import GuidanceInjector
from models import build_vae_var
from models.helpers import sample_with_top_k_top_p_
import dist

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ebc_computation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EBCCalculator:

    def __init__(self,
                 var_model: VAR,
                 planner: Optional[GuidanceInjector] = None,
                 patch_nums: Tuple[int] = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
                 device: str = 'cuda',
                 cfg: float = 1.5,
                 top_k: int = 900,
                 top_p: float = 0.96,
                 more_smooth: bool = False):
        self.var_model = var_model
        self.planner = planner
        self.patch_nums = patch_nums
        self.device = device

        self.cfg = cfg
        self.top_k = top_k
        self.top_p = top_p
        self.more_smooth = more_smooth

        self.vae = var_model.vae_proxy[0]
        self.vocab_size = self.vae.vocab_size

        self.real_images_dict = {}

    def extract_multiscale_tokens_from_images(self, images: torch.Tensor) -> List[torch.LongTensor]:
        """Extract multi-scale VAR tokens from images.

        Args:
            images: Input images (B, 3, H, W) in range [0, 1].

        Returns:
            tokens_list: Per-scale token list [(B, L_0), (B, L_1), ..., (B, L_K)].
        """
        with torch.no_grad():
            if images.max() <= 1.0:
                images = images * 2.0 - 1.0

            images = images.to(dtype=torch.float32, device=self.device)

            f = self.vae.quant_conv(self.vae.encoder(images))

            ms_idx_Bl = self.vae.quantize.f_to_idxBl_or_fhat(
                f, to_fhat=False, v_patch_nums=self.patch_nums
            )

            return ms_idx_Bl

    def compute_js_divergence(self,
                            logits1: torch.Tensor,
                            logits2: torch.Tensor,
                            eps: float = 1e-8) -> float:
        """Compute JS divergence between two logit distributions.

        Args:
            logits1: Logits of the first distribution (B, L, V).
            logits2: Logits of the second distribution (B, L, V).
            eps: Numerical stability constant.

        Returns:
            js_div: JS divergence value.
        """
        p = F.softmax(logits1, dim=-1)
        q = F.softmax(logits2, dim=-1)

        m = 0.5 * (p + q)

        kl_pm = F.kl_div(m.log(), p, reduction='none').sum(dim=-1)
        kl_qm = F.kl_div(m.log(), q, reduction='none').sum(dim=-1)

        js_div = 0.5 * (kl_pm + kl_qm)

        js_div_mean = js_div.mean().item()

        return js_div_mean

    def compute_token_distribution_divergence(self,
                                             generated_tokens: torch.LongTensor,
                                             real_tokens: torch.LongTensor) -> float:
        """Compute distribution divergence between generated and real tokens.

        Args:
            generated_tokens: Generated tokens (B, L).
            real_tokens: Real tokens (B, L).

        Returns:
            divergence: Distribution divergence value.
        """
        B, L = generated_tokens.shape

        gen_hist = torch.histc(
            generated_tokens.float(),
            bins=self.vocab_size,
            min=0,
            max=self.vocab_size-1
        )
        gen_dist = gen_hist / gen_hist.sum()

        real_hist = torch.histc(
            real_tokens.float(),
            bins=self.vocab_size,
            min=0,
            max=self.vocab_size-1
        )
        real_dist = real_hist / real_hist.sum()

        eps = 1e-8
        gen_dist = gen_dist + eps
        real_dist = real_dist + eps

        # Re-normalize
        gen_dist = gen_dist / gen_dist.sum()
        real_dist = real_dist / real_dist.sum()

        m = 0.5 * (gen_dist + real_dist)
        kl_gen = F.kl_div(m.log(), gen_dist, reduction='sum')
        kl_real = F.kl_div(m.log(), real_dist, reduction='sum')
        js_div = 0.5 * (kl_gen + kl_real)

        return js_div.item()

    def generate_with_real_prefix_forward(self,
                                         real_tokens_list: List[torch.LongTensor],
                                         class_labels: torch.LongTensor,
                                         use_planner: bool = False) -> List[torch.LongTensor]:
        """Generate tokens for all scales using real tokens as prefix (forward pass).

        Used to compute CGD(M|D, l) — the denominator.
        Matches the train_step logic in trainer.py exactly.

        Args:
            real_tokens_list: Real token list for all scales [(B, L_0), ..., (B, L_K)].
            class_labels: Class labels (B,).
            use_planner: Whether to use GuidanceInjector (not used in forward mode).

        Returns:
            all_scale_generated_tokens: Generated tokens for all scales [(B, L_0), ..., (B, L_K)].
        """
        B = class_labels.shape[0]

        x_BLCv_wo_first_l = self.vae.quantize.idxBl_to_var_input(real_tokens_list)

        with torch.no_grad():
            logits_BLV = self.var_model(class_labels, x_BLCv_wo_first_l)

            all_scale_generated_tokens = []
            start_idx = 0
            for scale_idx, pn in enumerate(self.patch_nums):
                scale_L = pn * pn
                end_idx = start_idx + scale_L

                scale_logits = logits_BLV[:, start_idx:end_idx, :]  # (B, L_scale, V)

                scale_tokens = sample_with_top_k_top_p_(
                    scale_logits,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    num_samples=1
                )[:, :, 0]  # (B, L_scale)

                all_scale_generated_tokens.append(scale_tokens)
                start_idx = end_idx

        return all_scale_generated_tokens

    def generate_with_model_prefix_inference(self,
                                            class_labels: torch.LongTensor,
                                            use_planner: bool = False,
                                            g_seed: Optional[int] = None) -> List[torch.LongTensor]:
        """Generate tokens for all scales via autoregressive inference.

        Used to compute CGD(M|M, l) — the numerator.
        Follows VAR's aid_guided_inference implementation with CFG.

        Args:
            class_labels: Class labels (B,).
            use_planner: Whether to use GuidanceInjector guidance.
            g_seed: Random seed for reproducibility.

        Returns:
            all_scale_generated_tokens: Generated tokens for all scales [(B, L_0), ..., (B, L_K)].
        """
        B = class_labels.shape[0]

        if g_seed is not None:
            rng = torch.Generator(device=self.device)
            rng.manual_seed(g_seed)
        else:
            rng = None

        with torch.no_grad():
            sos = cond_BD = self.var_model.class_emb(
                torch.cat((class_labels,
                          torch.full_like(class_labels, fill_value=self.var_model.num_classes)),
                          dim=0)
            )

            lvl_pos = self.var_model.lvl_embed(self.var_model.lvl_1L) + self.var_model.pos_1LC
            next_token_map = sos.unsqueeze(1).expand(2 * B, self.var_model.first_l, -1) + \
                            self.var_model.pos_start.expand(2 * B, self.var_model.first_l, -1) + \
                            lvl_pos[:, :self.var_model.first_l]

            cur_L = 0
            f_hat = sos.new_zeros(B, self.vae.Cvae, self.patch_nums[-1], self.patch_nums[-1])

            all_scale_generated_tokens = []

            # Enable KV cache
            for b in self.var_model.blocks:
                b.attn.kv_caching(True)

            for si, pn in enumerate(self.patch_nums):
                ratio = si / self.var_model.num_stages_minus_1
                cur_L += pn * pn

                # Generate guidance tokens (starting from second scale)
                guidance_tokens = None
                if use_planner and self.planner is not None and si > 0:
                    prev_tokens = all_scale_generated_tokens[si - 1]

                    prev_embed = F.embedding(prev_tokens, self.vae.quantize.embedding.weight.detach())
                    prev_features = self.var_model.word_embed(prev_embed)

                    guidance_tokens = self.planner(prev_features, target_patch_num=pn)
                    guidance_tokens = 0.001 * guidance_tokens

                    guidance_tokens = torch.cat([guidance_tokens, guidance_tokens], dim=0)

                cond_BD_or_gss = self.var_model.shared_ada_lin(cond_BD)
                x = next_token_map

                if guidance_tokens is not None:
                    if guidance_tokens.shape[0] == x.shape[0] and guidance_tokens.shape[1] == x.shape[1]:
                        x = x + guidance_tokens

                # Transformer forward
                for b in self.var_model.blocks:
                    x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)

                logits_BlV = self.var_model.get_logits(x, cond_BD)

                t = self.cfg * ratio
                logits_BlV = (1 + t) * logits_BlV[:B] - t * logits_BlV[B:]

                idx_Bl = sample_with_top_k_top_p_(
                    logits_BlV,
                    rng=rng,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    num_samples=1
                )[:, :, 0]  # (B, pn*pn)

                all_scale_generated_tokens.append(idx_Bl)

                if si < len(self.patch_nums) - 1:
                    if not self.more_smooth:
                        h_BChw = self.vae.quantize.embedding(idx_Bl)
                    else:
                        gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)
                        h_BChw = self.vae.quantize.embedding(idx_Bl)

                    h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.vae.Cvae, pn, pn)
                    f_hat, next_token_map = self.vae.quantize.get_next_autoregressive_input(
                        si, len(self.patch_nums), f_hat, h_BChw
                    )

                    next_token_map = next_token_map.view(B, self.vae.Cvae, -1).transpose(1, 2)
                    next_token_map = self.var_model.word_embed(next_token_map) + \
                                   lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                    next_token_map = next_token_map.repeat(2, 1, 1)

            for b in self.var_model.blocks:
                b.attn.kv_caching(False)

        return all_scale_generated_tokens

    def compute_cgd_for_scale(self,
                            generated_tokens_list: List[torch.LongTensor],
                            real_tokens_list: List[torch.LongTensor],
                            target_scale_idx: int) -> float:
        """Compute Conditional Generation Deviation (CGD) at a specific scale.

        Args:
            generated_tokens_list: Generated token list for all scales [(B, L_0), ..., (B, L_K)].
            real_tokens_list: Real token list for all scales [(B, L_0), ..., (B, L_K)].
            target_scale_idx: Target scale index.

        Returns:
            cgd: CGD value.
        """
        generated_tokens_at_scale = generated_tokens_list[target_scale_idx]  # (B, L_scale)
        real_tokens_at_scale = real_tokens_list[target_scale_idx]  # (B, L_scale)

        cgd = self.compute_token_distribution_divergence(
            generated_tokens_at_scale, real_tokens_at_scale
        )

        return cgd

    def compute_ebc_for_model(self,
                             real_images: torch.Tensor,
                             real_class_labels: torch.LongTensor,
                             num_model_samples: int,
                             use_planner: bool = False,
                             batch_size: int = 10) -> Dict:
        """Compute the full EB-C metric for a model.

        Steps:
        1. Extract multi-scale tokens from real images (for the denominator CGD(M|D)).
        2. Generate CGD(M|D) tokens using real prefix + forward pass.
        3. Generate CGD(M|M) tokens using autoregressive inference.
        4. Compute CGD per scale via compute_cgd_for_scale.
        5. Compute EB-C = CGD(M|M, l) / CGD(M|D, l).

        Args:
            real_images: Real images (N_real, 3, H, W) used for the denominator.
            real_class_labels: Class labels for real images (N_real,).
            num_model_samples: Number of samples to generate for the numerator CGD(M|M).
            use_planner: Whether to use the planner.
            batch_size: Batch processing size.

        Returns:
            results: EB-C result dictionary.
        """
        model_name = "AID-VAR" if use_planner else "VAR"

        num_real_samples = len(real_images)

        all_real_tokens = []
        for i in tqdm(range(0, num_real_samples, batch_size), desc="Extracting real tokens"):
            batch_images = real_images[i:i+batch_size]
            batch_tokens = self.extract_multiscale_tokens_from_images(batch_images)
            all_real_tokens.append(batch_tokens)

        real_tokens_list = []
        for scale_idx in range(len(self.patch_nums)):
            scale_tokens = torch.cat([batch_tokens[scale_idx] for batch_tokens in all_real_tokens], dim=0)
            real_tokens_list.append(scale_tokens)

        all_cgd_md_tokens = []
        for i in tqdm(range(0, num_real_samples, batch_size), desc="Generating CGD(M|D) tokens"):
            batch_real_tokens = [tokens[i:i+batch_size] for tokens in real_tokens_list]
            batch_labels = real_class_labels[i:i+batch_size]

            batch_cgd_md_tokens = self.generate_with_real_prefix_forward(
                real_tokens_list=batch_real_tokens,
                class_labels=batch_labels,
                use_planner=False,
            )  # [(B, L_0), ..., (B, L_K)]

            all_cgd_md_tokens.append(batch_cgd_md_tokens)

        cgd_md_tokens_list = []
        for scale_idx in range(len(self.patch_nums)):
            scale_tokens = torch.cat([batch_tokens[scale_idx] for batch_tokens in all_cgd_md_tokens], dim=0)
            cgd_md_tokens_list.append(scale_tokens)

        model_class_labels = torch.arange(num_model_samples, device=self.device) % 1000

        all_cgd_mm_tokens = []
        for i in tqdm(range(0, num_model_samples, batch_size), desc="Generating CGD(M|M) tokens"):
            batch_labels = model_class_labels[i:i+batch_size]

            g_seed = batch_labels[0].item()

            batch_cgd_mm_tokens = self.generate_with_model_prefix_inference(
                class_labels=batch_labels,
                use_planner=use_planner,
                g_seed=g_seed
            )  # [(B, L_0), ..., (B, L_K)]

            all_cgd_mm_tokens.append(batch_cgd_mm_tokens)

        cgd_mm_tokens_list = []
        for scale_idx in range(len(self.patch_nums)):
            scale_tokens = torch.cat([batch_tokens[scale_idx] for batch_tokens in all_cgd_mm_tokens], dim=0)
            cgd_mm_tokens_list.append(scale_tokens)

        results = {
            'scale_cgd_real_prefix': {},
            'scale_cgd_model_prefix': {},
            'scale_ebc': {},
            'model_name': model_name
        }

        for scale_idx in range(len(self.patch_nums)):
            cgd_real_prefix = self.compute_cgd_for_scale(
                generated_tokens_list=cgd_md_tokens_list,
                real_tokens_list=real_tokens_list,
                target_scale_idx=scale_idx
            )
            results['scale_cgd_real_prefix'][scale_idx] = cgd_real_prefix

            if scale_idx > 0:
                cgd_model_prefix = self.compute_cgd_for_scale(
                    generated_tokens_list=cgd_mm_tokens_list,
                    real_tokens_list=real_tokens_list,
                    target_scale_idx=scale_idx
                )
                results['scale_cgd_model_prefix'][scale_idx] = cgd_model_prefix

                if cgd_real_prefix > 1e-8:
                    ebc = cgd_model_prefix / cgd_real_prefix
                else:
                    ebc = float('inf')

                results['scale_ebc'][scale_idx] = ebc

                logger.info(f"Scale {scale_idx}: "
                          f"CGD(M|D)={cgd_real_prefix:.6f}, "
                          f"CGD(M|M)={cgd_model_prefix:.6f}, "
                          f"EB-C={ebc:.4f}")
            else:
                results['scale_cgd_model_prefix'][scale_idx] = None
                results['scale_ebc'][scale_idx] = None

        valid_ebc_scores = [ebc for ebc in results['scale_ebc'].values() if ebc is not None and ebc != float('inf')]
        if len(valid_ebc_scores) > 0:
            results['mean_ebc'] = np.mean(valid_ebc_scores)
            results['std_ebc'] = np.std(valid_ebc_scores)
        else:
            results['mean_ebc'] = None
            results['std_ebc'] = None

        logger.info(f"{model_name} mean EB-C: {results['mean_ebc']:.4f} +/- {results['std_ebc']:.4f}")

        return results


class EBCExperiment:

    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

        self._setup_optimization()
        self._load_models()

        self.ebc_calculator = EBCCalculator(
            var_model=self.var,
            planner=self.planner,
            patch_nums=tuple(args.patch_nums),
            device=self.device,
            cfg=args.cfg,
            top_k=args.top_k,
            top_p=args.top_p,
            more_smooth=args.more_smooth
        )

    def _setup_optimization(self):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        tf32 = True
        torch.backends.cudnn.allow_tf32 = bool(tf32)
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.set_float32_matmul_precision('high' if tf32 else 'highest')

    def _load_models(self):

        if self.args.var_ckpt is None:
            self.args.var_ckpt = f'checkpoints/var_d{self.args.model_depth}.pth'

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
            logger.info(f"Loading GuidanceInjector from: {self.args.planner_ckpt}")

            var_embed_dim = self.args.model_depth * 64
            self.planner = GuidanceInjector(
                input_dim=var_embed_dim,
                embed_dim=var_embed_dim,
                num_layers=2,
                num_heads=8
            ).to(self.device)

            planner_state = torch.load(self.args.planner_ckpt, map_location='cpu')

            if 'planner_state_dict' in planner_state:
                self.planner.load_state_dict(planner_state['planner_state_dict'], strict=False)
            elif 'planner' in planner_state:
                self.planner.load_state_dict(planner_state['planner'], strict=True)
            else:
                self.planner.load_state_dict(planner_state, strict=True)

            self.planner.eval()
            for p in self.planner.parameters(): p.requires_grad_(False)

        else:
            self.planner = None
            logger.warning("No GuidanceInjector checkpoint provided; evaluating base VAR only.")

    def load_real_images(self) -> Tuple[torch.Tensor, torch.LongTensor]:
        """Load real image data for the denominator CGD(M|D) computation."""
        logger.info(f"Loading real images from: {self.args.real_images_path}")

        if self.args.real_images_path.endswith('.npz'):
            data = np.load(self.args.real_images_path)

            if 'images' in data:
                real_images = torch.from_numpy(data['images']).float()
            elif 'arr_0' in data:
                real_images = torch.from_numpy(data['arr_0']).float()
            else:
                key = list(data.keys())[0]
                real_images = torch.from_numpy(data[key]).float()

            if len(real_images.shape) == 4:
                if real_images.shape[3] == 3:
                    real_images = real_images.permute(0, 3, 1, 2)

            if real_images.max() > 1.0:
                real_images = real_images / 255.0
        else:
            raise ValueError("Please provide real image data in .npz format.")

        if len(real_images) > self.args.num_real_samples:
            torch.manual_seed(42)
            indices = torch.randperm(len(real_images))[:self.args.num_real_samples]
            real_images = real_images[indices]

        num_images = len(real_images)
        class_labels = torch.arange(num_images) % 1000

        return real_images, class_labels

    def run_experiment(self):
        """Run the full EB-C evaluation experiment."""
        real_images, real_class_labels = self.load_real_images()
        real_images = real_images.to(self.device)
        real_class_labels = real_class_labels.to(self.device)

        results = {}

        results['var'] = self.ebc_calculator.compute_ebc_for_model(
            real_images=real_images,
            real_class_labels=real_class_labels,
            num_model_samples=self.args.num_samples,
            use_planner=False,
            batch_size=self.args.batch_size
        )

        if self.planner is not None:
            results['aid_var'] = self.ebc_calculator.compute_ebc_for_model(
                real_images=real_images,
                real_class_labels=real_class_labels,
                num_model_samples=self.args.num_samples,
                use_planner=True,
                batch_size=self.args.batch_size
            )

            var_ebc = results['var']['mean_ebc']
            aid_ebc = results['aid_var']['mean_ebc']
            improvement = ((var_ebc - aid_ebc) / var_ebc) * 100

            logger.info(f"EB-C comparison: VAR={var_ebc:.4f}, AID-VAR={aid_ebc:.4f}, improvement={improvement:.2f}%")

        self._save_results(results)
        self._visualize_results(results)

        return results

    def _save_results(self, results: Dict):
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        results_path = output_dir / 'ebc_results.json'

        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        results_serializable = convert_to_serializable(results)

        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_serializable, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to: {results_path}")

    def _visualize_results(self, results: Dict):
        output_dir = Path(self.args.output_dir)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        ax1 = axes[0]

        var_scales = sorted([k for k in results['var']['scale_ebc'].keys()
                           if results['var']['scale_ebc'][k] is not None])
        var_ebc_scores = [results['var']['scale_ebc'][s] for s in var_scales]

        ax1.plot(var_scales, var_ebc_scores, 'o-', label='VAR', linewidth=2, markersize=8)

        if 'aid_var' in results:
            aid_ebc_scores = [results['aid_var']['scale_ebc'][s] for s in var_scales]
            ax1.plot(var_scales, aid_ebc_scores, 's-', label='AID-VAR', linewidth=2, markersize=8)

        ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='EB-C=1 (baseline)')
        ax1.set_xlabel('Scale index', fontsize=12)
        ax1.set_ylabel('EB-C', fontsize=12)
        ax1.set_title('Per-scale EB-C comparison', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = axes[1]

        models = ['VAR']
        mean_ebcs = [results['var']['mean_ebc']]

        if 'aid_var' in results:
            models.append('AID-VAR')
            mean_ebcs.append(results['aid_var']['mean_ebc'])

        bars = ax2.bar(models, mean_ebcs, color=['skyblue', 'lightcoral'][:len(models)], alpha=0.7)
        ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)

        for bar, score in zip(bars, mean_ebcs):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{score:.4f}', ha='center', va='bottom', fontweight='bold')

        ax2.set_ylabel('Mean EB-C', fontsize=12)
        ax2.set_title('Mean EB-C comparison (lower is better)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(output_dir / 'ebc_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Visualization saved to: {output_dir}/ebc_comparison.png")


def parse_args():
    parser = argparse.ArgumentParser(description='EB-C (Exposure Bias - Consistency) computation')

    parser.add_argument('--vae_ckpt', type=str, default='vae_ch160v4096z32.pth',
                       help='VAE model checkpoint path')
    parser.add_argument('--var_ckpt', type=str, default=None,
                       help='VAR model checkpoint path')
    parser.add_argument('--planner_ckpt', type=str, default=None,
                       help='GuidanceInjector checkpoint path (optional)')
    parser.add_argument('--model_depth', type=int, default=16, choices=[16, 20, 24, 30],
                       help='VAR model depth')

    parser.add_argument('--real_images_path', type=str, required=True,
                       help='Path to real images (.npz file)')
    parser.add_argument('--num_samples', type=int, default=50000,
                       help='Number of samples for numerator CGD(M|M)')
    parser.add_argument('--num_real_samples', type=int, default=10000,
                       help='Number of real images for denominator CGD(M|D)')
    parser.add_argument('--batch_size', type=int, default=50,
                       help='Batch size')

    parser.add_argument('--cfg', type=float, default=1.5,
                       help='Classifier-free guidance scale')
    parser.add_argument('--top_k', type=int, default=900,
                       help='Top-k sampling parameter')
    parser.add_argument('--top_p', type=float, default=0.96,
                       help='Top-p sampling parameter')
    parser.add_argument('--more_smooth', action='store_true',
                       help='Enable more_smooth sampling')

    parser.add_argument('--device', type=str, default='cuda',
                       help='Compute device')
    parser.add_argument('--output_dir', type=str, default='./ebc_results',
                       help='Output directory')

    parser.add_argument('--patch_nums', type=int, nargs='+',
                       default=[1, 2, 3, 4, 5, 6, 8, 10, 13, 16],
                       help='VAR patch number sequence')

    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)

    if not dist.initialized():
        dist.initialize()

    logger.info("="*80)
    logger.info("EB-C (Exposure Bias - Consistency) Evaluation")
    logger.info("="*80)
    logger.info(f"model_depth={args.model_depth}, "
                f"num_real_samples={args.num_real_samples}, "
                f"num_samples={args.num_samples}, "
                f"cfg={args.cfg}, top_k={args.top_k}, top_p={args.top_p}")
    logger.info("="*80)

    try:
        experiment = EBCExperiment(args)
        results = experiment.run_experiment()
        logger.info("EB-C evaluation completed successfully.")

    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
