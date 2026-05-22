#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ISCS utility module.

Provides helper functions for ISCS computation, including feature extraction,
data preprocessing, result analysis, and numerical stability tools.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import sqrtm
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AdvancedFeatureExtractor:
    """Advanced feature extractor providing multiple methods for building joint feature vectors."""
    
    def __init__(self, device='cuda'):
        self.device = device
        
    def extract_hierarchical_features(self,
                                    images: torch.Tensor,
                                    var_model,
                                    method='vae_tokens') -> Dict[str, torch.Tensor]:
        """
        Extract hierarchical features.

        Args:
            images: input images (B, 3, H, W)
            var_model: VAR model
            method: feature extraction method

        Returns:
            features: feature dictionary
        """
        features = {}

        if method == 'vae_tokens':
            with torch.no_grad():
                vae = var_model.vae_proxy[0]
                f_hat, _ = vae.img_to_fhat(images)

                _, _, token_indices = vae.quantize(f_hat)

                B, H, W = token_indices.shape
                token_flat = token_indices.view(B, -1)

                token_onehot = F.one_hot(token_flat, num_classes=vae.vocab_size).float()
                token_features = var_model.word_embed(token_onehot)  # (B, L, C)

                features['vae_tokens'] = token_features

        elif method == 'multiscale_pooling':
            with torch.no_grad():
                vae = var_model.vae_proxy[0]
                f_hat, _ = vae.img_to_fhat(images)

                pooled_features = []
                for scale in [1, 2, 4, 8]:
                    if scale == 1:
                        pooled = f_hat.mean(dim=[2, 3])  # Global average pooling
                    else:
                        pooled = F.adaptive_avg_pool2d(f_hat, output_size=(scale, scale))
                        pooled = pooled.view(pooled.size(0), -1)
                    pooled_features.append(pooled)

                features['multiscale_pooling'] = torch.cat(pooled_features, dim=1)

        return features
    
    def compute_statistical_features(self,
                                   tokens: torch.Tensor) -> torch.Tensor:
        """
        Compute statistical features of a token sequence.

        Args:
            tokens: token sequence (B, L, C)

        Returns:
            stat_features: statistical features (B, 4*C)
        """
        mean_feat = tokens.mean(dim=1)  # (B, C)
        var_feat = tokens.var(dim=1)    # (B, C)

        tokens_norm = (tokens - mean_feat.unsqueeze(1)) / (var_feat.unsqueeze(1).sqrt() + 1e-8)

        skew_feat = (tokens_norm ** 3).mean(dim=1)  # (B, C)

        kurt_feat = (tokens_norm ** 4).mean(dim=1)  # (B, C)

        stat_features = torch.cat([mean_feat, var_feat, skew_feat, kurt_feat], dim=1)

        return stat_features

class NumericalStabilityTools:
    """Numerical stability utilities."""
    
    @staticmethod
    def safe_frechet_distance(mu1: np.ndarray, sigma1: np.ndarray,
                            mu2: np.ndarray, sigma2: np.ndarray,
                            eps: float = 1e-6) -> float:
        """
        Numerically stable Fréchet distance computation.

        Args:
            mu1, sigma1: mean and covariance of the first distribution
            mu2, sigma2: mean and covariance of the second distribution
            eps: numerical stability parameter

        Returns:
            frechet_distance: FJD distance
        """
        sigma1_reg = sigma1 + eps * np.eye(sigma1.shape[0])
        sigma2_reg = sigma2 + eps * np.eye(sigma2.shape[0])

        diff = mu1 - mu2

        try:
            covmean = sqrtm(sigma1_reg @ sigma2_reg)

            if np.iscomplexobj(covmean):
                logger.warning("Matrix square root produced complex values; taking real part.")
                covmean = covmean.real

        except Exception as e:
            logger.warning(f"sqrtm failed: {e}; falling back to eigendecomposition.")
            try:
                product = sigma1_reg @ sigma2_reg
                eigenvals, eigenvecs = np.linalg.eigh(product)
                eigenvals = np.maximum(eigenvals, 0)
                sqrt_eigenvals = np.sqrt(eigenvals)
                covmean = eigenvecs @ np.diag(sqrt_eigenvals) @ eigenvecs.T
            except Exception as e2:
                logger.error(f"Eigendecomposition also failed: {e2}; using diagonal approximation.")
                diag1 = np.diag(np.diag(sigma1_reg))
                diag2 = np.diag(np.diag(sigma2_reg))
                covmean = np.sqrt(diag1 @ diag2)

        fid = (diff @ diff +
               np.trace(sigma1_reg) +
               np.trace(sigma2_reg) -
               2 * np.trace(covmean))

        fid = max(0.0, float(np.real(fid)))

        return fid
    
    @staticmethod
    def robust_covariance_estimation(features: np.ndarray,
                                   regularization: float = 1e-6) -> np.ndarray:
        """
        Robust covariance matrix estimation.

        Args:
            features: feature matrix (N, D)
            regularization: regularization parameter

        Returns:
            cov_matrix: covariance matrix
        """
        N, D = features.shape

        if N < D:
            logger.warning(f"Sample count ({N}) is less than feature dimension ({D}); using shrinkage estimator.")

            sample_cov = np.cov(features, rowvar=False)
            identity = np.eye(D)

            shrinkage = min(1.0, (D - N) / (N * D)) if N > 1 else 0.9
            cov_matrix = (1 - shrinkage) * sample_cov + shrinkage * identity

        else:
            cov_matrix = np.cov(features, rowvar=False)

        cov_matrix += regularization * np.eye(D)

        return cov_matrix

class ISCSAnalyzer:
    """ISCS result analyzer providing in-depth analysis and visualization."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def analyze_scale_progression(self,
                                scale_scores: Dict[int, float],
                                patch_nums: List[int]) -> Dict[str, float]:
        """
        Analyze scale progression patterns.

        Args:
            scale_scores: ISCS scores per scale
            patch_nums: patch number sequence

        Returns:
            analysis: analysis results
        """
        scores = [scale_scores[i] for i in sorted(scale_scores.keys())]

        analysis = {}

        if len(scores) > 1:
            x = np.arange(len(scores))
            slope = np.polyfit(x, scores, 1)[0]
            analysis['linear_trend'] = slope

            diffs = np.diff(scores)
            analysis['monotonic_increase'] = all(d >= 0 for d in diffs)
            analysis['monotonic_decrease'] = all(d <= 0 for d in diffs)

            analysis['coefficient_variation'] = np.std(scores) / np.mean(scores) if np.mean(scores) > 0 else float('inf')

        best_scale = max(scale_scores.keys(), key=lambda k: scale_scores[k])
        worst_scale = min(scale_scores.keys(), key=lambda k: scale_scores[k])

        analysis['best_scale'] = best_scale
        analysis['worst_scale'] = worst_scale
        analysis['score_range'] = scale_scores[best_scale] - scale_scores[worst_scale]

        mid_point = len(scores) // 2
        early_scores = scores[:mid_point] if mid_point > 0 else scores[:1]
        late_scores = scores[mid_point:] if mid_point < len(scores) else scores[-1:]

        analysis['early_performance'] = np.mean(early_scores)
        analysis['late_performance'] = np.mean(late_scores)
        analysis['performance_shift'] = np.mean(late_scores) - np.mean(early_scores)

        return analysis
    
    def create_detailed_visualization(self,
                                    results: Dict,
                                    patch_nums: List[int]):
        """
        Create a detailed visualization analysis chart.

        Args:
            results: ISCS results
            patch_nums: patch number sequence
        """
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('ISCS Detailed Analysis Report', fontsize=16, fontweight='bold')

        ax1 = axes[0, 0]
        scales = list(results['base_var']['scale_scores'].keys())
        base_scores = [results['base_var']['scale_scores'][s] for s in scales]

        ax1.plot(scales, base_scores, 'o-', label='Base VAR', linewidth=2, markersize=6)

        if 'aid_var' in results:
            aid_scores = [results['aid_var']['scale_scores'][s] for s in scales]
            ax1.plot(scales, aid_scores, 's-', label='AID-VAR', linewidth=2, markersize=6)

        ax1.set_xlabel('Scale Index')
        ax1.set_ylabel('ISCS Score')
        ax1.set_title('Per-Scale ISCS Score Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = axes[0, 1]
        if 'aid_var' in results:
            improvements = [results['aid_var']['scale_scores'][s] - results['base_var']['scale_scores'][s]
                          for s in scales]
            colors = ['green' if imp > 0 else 'red' for imp in improvements]
            bars = ax2.bar(scales, improvements, color=colors, alpha=0.7)

            for bar, imp in zip(bars, improvements):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001 if height >= 0 else height - 0.001,
                        f'{imp:.3f}', ha='center', va='bottom' if height >= 0 else 'top')

        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_xlabel('Scale Index')
        ax2.set_ylabel('ISCS Improvement')
        ax2.set_title('Per-Scale Improvement from GuidanceInjector')
        ax2.grid(True, alpha=0.3)

        ax3 = axes[0, 2]
        models = ['Base VAR']
        total_scores = [results['base_var']['total_iscs']]
        colors = ['skyblue']

        if 'aid_var' in results:
            models.append('AID-VAR')
            total_scores.append(results['aid_var']['total_iscs'])
            colors.append('lightcoral')

        bars = ax3.bar(models, total_scores, color=colors, alpha=0.8)

        for bar, score in zip(bars, total_scores):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{score:.4f}', ha='center', va='bottom', fontweight='bold')

        ax3.set_ylabel('Total ISCS Score')
        ax3.set_title('Total ISCS Score Comparison')
        ax3.grid(True, alpha=0.3)

        ax4 = axes[1, 0]
        patch_nums_plot = [patch_nums[s] for s in scales]

        ax4.scatter(patch_nums_plot, base_scores, s=60, alpha=0.7, label='Base VAR')
        if 'aid_var' in results:
            ax4.scatter(patch_nums_plot, aid_scores, s=60, alpha=0.7, label='AID-VAR')

        ax4.set_xlabel('Patch Count')
        ax4.set_ylabel('ISCS Score')
        ax4.set_title('Patch Complexity vs ISCS Performance')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        ax5 = axes[1, 1]
        weights = [2**k for k in range(len(scales))]
        total_weight = sum(weights)
        weights_norm = [w/total_weight for w in weights]

        bars = ax5.bar(scales, weights_norm, alpha=0.7, color='orange')
        ax5.set_xlabel('Scale Index')
        ax5.set_ylabel('Normalized Weight')
        ax5.set_title('Scale Weight Distribution (Exponential)')
        ax5.grid(True, alpha=0.3)

        ax6 = axes[1, 2]
        ax6.axis('off')

        base_analysis = self.analyze_scale_progression(results['base_var']['scale_scores'], patch_nums)

        summary_text = f"""
ISCS Performance Summary

Base VAR:
- Total: {results['base_var']['total_iscs']:.4f}
- Best scale: {base_analysis['best_scale']} (score: {results['base_var']['scale_scores'][base_analysis['best_scale']]:.4f})
- Worst scale: {base_analysis['worst_scale']} (score: {results['base_var']['scale_scores'][base_analysis['worst_scale']]:.4f})
- Score range: {base_analysis['score_range']:.4f}
"""

        if 'aid_var' in results:
            aid_analysis = self.analyze_scale_progression(results['aid_var']['scale_scores'], patch_nums)
            improvement = results['aid_var']['total_iscs'] - results['base_var']['total_iscs']
            improvement_pct = (improvement / results['base_var']['total_iscs']) * 100

            summary_text += f"""
AID-VAR:
- Total: {results['aid_var']['total_iscs']:.4f}
- Improvement: {improvement:.4f} ({improvement_pct:.2f}%)
- Best scale: {aid_analysis['best_scale']} (score: {results['aid_var']['scale_scores'][aid_analysis['best_scale']]:.4f})

Conclusion:
"""
            if improvement > 0.01:
                summary_text += "GuidanceInjector significantly improves scale consistency."
            elif improvement > 0:
                summary_text += "GuidanceInjector slightly improves scale consistency."
            else:
                summary_text += "GuidanceInjector did not improve scale consistency."

        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        plt.tight_layout()
        plt.savefig(self.output_dir / 'iscs_detailed_analysis.png',
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def generate_analysis_report(self,
                               results: Dict,
                               patch_nums: List[int]) -> str:
        """
        Generate a detailed analysis report.

        Args:
            results: ISCS results
            patch_nums: patch number sequence

        Returns:
            report: analysis report text
        """
        report = []
        report.append("=" * 80)
        report.append("ISCS (Inter-Scale Consistency Score) Detailed Analysis Report")
        report.append("=" * 80)
        report.append("")

        report.append("Experiment Configuration:")
        report.append(f"  - Number of evaluation scales: {len(patch_nums)}")
        report.append(f"  - Patch number sequence: {patch_nums}")
        report.append(f"  - Weighting scheme: exponential (w_k ~ 2^k)")
        report.append("")

        base_analysis = self.analyze_scale_progression(results['base_var']['scale_scores'], patch_nums)
        report.append("Base VAR Performance:")
        report.append(f"  - Total ISCS score: {results['base_var']['total_iscs']:.6f}")
        report.append(f"  - Best scale: {base_analysis['best_scale']} (score: {results['base_var']['scale_scores'][base_analysis['best_scale']]:.6f})")
        report.append(f"  - Worst scale: {base_analysis['worst_scale']} (score: {results['base_var']['scale_scores'][base_analysis['worst_scale']]:.6f})")
        report.append(f"  - Score range: {base_analysis['score_range']:.6f}")
        report.append(f"  - Linear trend slope: {base_analysis.get('linear_trend', 'N/A'):.6f}")
        report.append(f"  - Coefficient of variation: {base_analysis.get('coefficient_variation', 'N/A'):.6f}")
        report.append("")

        if 'aid_var' in results:
            aid_analysis = self.analyze_scale_progression(results['aid_var']['scale_scores'], patch_nums)
            improvement = results['aid_var']['total_iscs'] - results['base_var']['total_iscs']
            improvement_pct = (improvement / results['base_var']['total_iscs']) * 100

            report.append("AID-VAR Performance:")
            report.append(f"  - Total ISCS score: {results['aid_var']['total_iscs']:.6f}")
            report.append(f"  - Improvement over Base VAR: {improvement:.6f} ({improvement_pct:.2f}%)")
            report.append(f"  - Best scale: {aid_analysis['best_scale']} (score: {results['aid_var']['scale_scores'][aid_analysis['best_scale']]:.6f})")
            report.append(f"  - Worst scale: {aid_analysis['worst_scale']} (score: {results['aid_var']['scale_scores'][aid_analysis['worst_scale']]:.6f})")
            report.append("")

            report.append("Per-Scale Improvement Details:")
            for scale in sorted(results['base_var']['scale_scores'].keys()):
                base_score = results['base_var']['scale_scores'][scale]
                aid_score = results['aid_var']['scale_scores'][scale]
                scale_improvement = aid_score - base_score
                scale_improvement_pct = (scale_improvement / base_score) * 100

                status = "[+]" if scale_improvement > 0.001 else "[~]" if scale_improvement > 0 else "[-]"
                report.append(f"  - Scale {scale}: {base_score:.6f} -> {aid_score:.6f} "
                            f"({scale_improvement:+.6f}, {scale_improvement_pct:+.2f}%) {status}")
            report.append("")

        report.append("Conclusions:")

        if 'aid_var' in results:
            improvement = results['aid_var']['total_iscs'] - results['base_var']['total_iscs']

            if improvement > 0.01:
                report.append("  GuidanceInjector significantly improves inter-scale consistency.")
                report.append("  Recommend using the AID-VAR configuration.")
            elif improvement > 0:
                report.append("  GuidanceInjector slightly improves inter-scale consistency.")
                report.append("  Consider further tuning the GuidanceInjector training strategy.")
            else:
                report.append("  GuidanceInjector did not improve inter-scale consistency.")
                report.append("  Recommend revisiting GuidanceInjector design and training.")

            improvements_by_scale = {}
            for scale in results['base_var']['scale_scores'].keys():
                improvements_by_scale[scale] = (
                    results['aid_var']['scale_scores'][scale] -
                    results['base_var']['scale_scores'][scale]
                )

            best_improved_scale = max(improvements_by_scale.keys(),
                                    key=lambda k: improvements_by_scale[k])
            worst_improved_scale = min(improvements_by_scale.keys(),
                                     key=lambda k: improvements_by_scale[k])

            report.append(f"  GuidanceInjector is most effective at scale {best_improved_scale} "
                         f"(improvement {improvements_by_scale[best_improved_scale]:.6f})")
            report.append(f"  GuidanceInjector is least effective at scale {worst_improved_scale} "
                         f"(improvement {improvements_by_scale[worst_improved_scale]:.6f})")
        else:
            report.append("  Only the Base VAR model was evaluated.")
            report.append("  Consider training GuidanceInjector for comparison.")

        report.append("")
        report.append("=" * 80)

        report_text = "\n".join(report)

        report_path = self.output_dir / 'iscs_analysis_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)

        return report_text

def load_checkpoint_safely(checkpoint_path: str,
                          model: nn.Module,
                          map_location: str = 'cpu') -> bool:
    """
    Safely load a model checkpoint.

    Args:
        checkpoint_path: path to checkpoint file
        model: model instance
        map_location: device to load onto

    Returns:
        success: whether loading succeeded
    """
    try:
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint file not found: {checkpoint_path}")
            return False

        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=False)
        return True

    except Exception as e:
        logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
        return False

def setup_logging(output_dir: str, level: int = logging.INFO):
    """
    Configure the logging system.

    Args:
        output_dir: output directory
        level: log level
    """
    os.makedirs(output_dir, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(level)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler = logging.FileHandler(
        os.path.join(output_dir, 'iscs_computation.log'),
        encoding='utf-8'
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger