#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 ISCS工具模块

提供ISCS计算所需的辅助函数，包括：
- 高级特征提取方法
- 数据预处理工具
- 结果分析和可视化
- 数值稳定性工具

作者：Expert AI Developer
日期：2025-09-23
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
    """
    🎯 高级特征提取器
    
    提供多种特征提取方法来构建更robust的联合特征向量
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        
    def extract_hierarchical_features(self, 
                                    images: torch.Tensor,
                                    var_model,
                                    method='vae_tokens') -> Dict[str, torch.Tensor]:
        """
        提取层次化特征
        
        Args:
            images: 输入图像 (B, 3, H, W)
            var_model: VAR模型
            method: 特征提取方法
            
        Returns:
            features: 特征字典
        """
        features = {}
        
        if method == 'vae_tokens':
            # 使用VAE tokens作为特征
            with torch.no_grad():
                vae = var_model.vae_proxy[0]
                f_hat, _ = vae.img_to_fhat(images)
                
                # 量化得到离散tokens
                _, _, token_indices = vae.quantize(f_hat)
                
                # 转换为one-hot并通过embedding获取特征
                B, H, W = token_indices.shape
                token_flat = token_indices.view(B, -1)
                
                token_onehot = F.one_hot(token_flat, num_classes=vae.vocab_size).float()
                token_features = var_model.word_embed(token_onehot)  # (B, L, C)
                
                features['vae_tokens'] = token_features
                
        elif method == 'multiscale_pooling':
            # 多尺度池化特征
            with torch.no_grad():
                vae = var_model.vae_proxy[0]
                f_hat, _ = vae.img_to_fhat(images)
                
                # 不同尺度的池化
                pooled_features = []
                for scale in [1, 2, 4, 8]:
                    if scale == 1:
                        pooled = f_hat.mean(dim=[2, 3])  # 全局平均池化
                    else:
                        pooled = F.adaptive_avg_pool2d(f_hat, output_size=(scale, scale))
                        pooled = pooled.view(pooled.size(0), -1)
                    pooled_features.append(pooled)
                
                features['multiscale_pooling'] = torch.cat(pooled_features, dim=1)
        
        return features
    
    def compute_statistical_features(self, 
                                   tokens: torch.Tensor) -> torch.Tensor:
        """
        计算token序列的统计特征
        
        Args:
            tokens: token序列 (B, L, C)
            
        Returns:
            stat_features: 统计特征 (B, 4*C)
        """
        # 计算均值、方差、偏度、峰度
        mean_feat = tokens.mean(dim=1)  # (B, C)
        var_feat = tokens.var(dim=1)    # (B, C)
        
        # 标准化tokens用于计算高阶矩
        tokens_norm = (tokens - mean_feat.unsqueeze(1)) / (var_feat.unsqueeze(1).sqrt() + 1e-8)
        
        # 偏度 (三阶矩)
        skew_feat = (tokens_norm ** 3).mean(dim=1)  # (B, C)
        
        # 峰度 (四阶矩)
        kurt_feat = (tokens_norm ** 4).mean(dim=1)  # (B, C)
        
        # 拼接所有统计特征
        stat_features = torch.cat([mean_feat, var_feat, skew_feat, kurt_feat], dim=1)
        
        return stat_features

class NumericalStabilityTools:
    """
    🎯 数值稳定性工具
    
    提供各种数值稳定性保证的方法
    """
    
    @staticmethod
    def safe_frechet_distance(mu1: np.ndarray, sigma1: np.ndarray,
                            mu2: np.ndarray, sigma2: np.ndarray,
                            eps: float = 1e-6) -> float:
        """
        数值稳定的Fréchet距离计算
        
        Args:
            mu1, sigma1: 第一个分布的均值和协方差
            mu2, sigma2: 第二个分布的均值和协方差
            eps: 数值稳定性参数
            
        Returns:
            frechet_distance: FJD距离
        """
        # 添加正则化项保证数值稳定性
        sigma1_reg = sigma1 + eps * np.eye(sigma1.shape[0])
        sigma2_reg = sigma2 + eps * np.eye(sigma2.shape[0])
        
        # 计算均值差
        diff = mu1 - mu2
        
        # 使用更稳定的矩阵平方根计算
        try:
            # 方法1：使用scipy的sqrtm
            covmean = sqrtm(sigma1_reg @ sigma2_reg)
            
            # 检查是否为复数结果
            if np.iscomplexobj(covmean):
                logger.warning("矩阵平方根产生复数结果，取实部")
                covmean = covmean.real
                
        except Exception as e:
            logger.warning(f"sqrtm计算失败: {e}，使用替代方法")
            # 方法2：特征值分解
            try:
                product = sigma1_reg @ sigma2_reg
                eigenvals, eigenvecs = np.linalg.eigh(product)
                eigenvals = np.maximum(eigenvals, 0)  # 确保非负
                sqrt_eigenvals = np.sqrt(eigenvals)
                covmean = eigenvecs @ np.diag(sqrt_eigenvals) @ eigenvecs.T
            except Exception as e2:
                logger.error(f"特征值分解也失败: {e2}，使用对角近似")
                # 方法3：对角近似（最后的fallback）
                diag1 = np.diag(np.diag(sigma1_reg))
                diag2 = np.diag(np.diag(sigma2_reg))
                covmean = np.sqrt(diag1 @ diag2)
        
        # 计算Fréchet距离
        fid = (diff @ diff + 
               np.trace(sigma1_reg) + 
               np.trace(sigma2_reg) - 
               2 * np.trace(covmean))
        
        # 确保结果为非负实数
        fid = max(0.0, float(np.real(fid)))
        
        return fid
    
    @staticmethod
    def robust_covariance_estimation(features: np.ndarray, 
                                   regularization: float = 1e-6) -> np.ndarray:
        """
        鲁棒的协方差矩阵估计
        
        Args:
            features: 特征矩阵 (N, D)
            regularization: 正则化参数
            
        Returns:
            cov_matrix: 协方差矩阵
        """
        N, D = features.shape
        
        if N < D:
            logger.warning(f"样本数量({N})少于特征维度({D})，使用收缩估计器")
            
            # 使用收缩估计器
            sample_cov = np.cov(features, rowvar=False)
            identity = np.eye(D)
            
            # 优化收缩参数
            shrinkage = min(1.0, (D - N) / (N * D)) if N > 1 else 0.9
            cov_matrix = (1 - shrinkage) * sample_cov + shrinkage * identity
            
        else:
            # 标准协方差估计
            cov_matrix = np.cov(features, rowvar=False)
        
        # 添加正则化
        cov_matrix += regularization * np.eye(D)
        
        return cov_matrix

class ISCSAnalyzer:
    """
    🎯 ISCS结果分析器
    
    提供深入的ISCS结果分析和可视化功能
    """
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def analyze_scale_progression(self, 
                                scale_scores: Dict[int, float],
                                patch_nums: List[int]) -> Dict[str, float]:
        """
        分析尺度进展模式
        
        Args:
            scale_scores: 各尺度ISCS分数
            patch_nums: patch数量序列
            
        Returns:
            analysis: 分析结果
        """
        scores = [scale_scores[i] for i in sorted(scale_scores.keys())]
        
        analysis = {}
        
        # 1. 总体趋势
        if len(scores) > 1:
            # 线性趋势
            x = np.arange(len(scores))
            slope = np.polyfit(x, scores, 1)[0]
            analysis['linear_trend'] = slope
            
            # 单调性检查
            diffs = np.diff(scores)
            analysis['monotonic_increase'] = all(d >= 0 for d in diffs)
            analysis['monotonic_decrease'] = all(d <= 0 for d in diffs)
            
            # 变异系数
            analysis['coefficient_variation'] = np.std(scores) / np.mean(scores) if np.mean(scores) > 0 else float('inf')
        
        # 2. 关键尺度识别
        best_scale = max(scale_scores.keys(), key=lambda k: scale_scores[k])
        worst_scale = min(scale_scores.keys(), key=lambda k: scale_scores[k])
        
        analysis['best_scale'] = best_scale
        analysis['worst_scale'] = worst_scale
        analysis['score_range'] = scale_scores[best_scale] - scale_scores[worst_scale]
        
        # 3. 早期vs后期性能
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
        创建详细的可视化分析图
        
        Args:
            results: ISCS结果
            patch_nums: patch数量序列
        """
        # 设置图表样式
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('ISCS详细分析报告', fontsize=16, fontweight='bold')
        
        # 1. 尺度分数对比 (左上)
        ax1 = axes[0, 0]
        scales = list(results['base_var']['scale_scores'].keys())
        base_scores = [results['base_var']['scale_scores'][s] for s in scales]
        
        ax1.plot(scales, base_scores, 'o-', label='基础VAR', linewidth=2, markersize=6)
        
        if 'aid_var' in results:
            aid_scores = [results['aid_var']['scale_scores'][s] for s in scales]
            ax1.plot(scales, aid_scores, 's-', label='AID-VAR', linewidth=2, markersize=6)
        
        ax1.set_xlabel('尺度索引')
        ax1.set_ylabel('ISCS分数')
        ax1.set_title('各尺度ISCS分数对比')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 分数改善情况 (右上)
        ax2 = axes[0, 1]
        if 'aid_var' in results:
            improvements = [results['aid_var']['scale_scores'][s] - results['base_var']['scale_scores'][s] 
                          for s in scales]
            colors = ['green' if imp > 0 else 'red' for imp in improvements]
            bars = ax2.bar(scales, improvements, color=colors, alpha=0.7)
            
            # 添加数值标签
            for bar, imp in zip(bars, improvements):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001 if height >= 0 else height - 0.001,
                        f'{imp:.3f}', ha='center', va='bottom' if height >= 0 else 'top')
        
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_xlabel('尺度索引')
        ax2.set_ylabel('ISCS改善量')
        ax2.set_title('GuidanceInjector的各尺度改善效果')
        ax2.grid(True, alpha=0.3)
        
        # 3. 总分对比 (左中)
        ax3 = axes[0, 2]
        models = ['基础VAR']
        total_scores = [results['base_var']['total_iscs']]
        colors = ['skyblue']
        
        if 'aid_var' in results:
            models.append('AID-VAR')
            total_scores.append(results['aid_var']['total_iscs'])
            colors.append('lightcoral')
        
        bars = ax3.bar(models, total_scores, color=colors, alpha=0.8)
        
        # 添加数值标签
        for bar, score in zip(bars, total_scores):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{score:.4f}', ha='center', va='bottom', fontweight='bold')
        
        ax3.set_ylabel('总ISCS分数')
        ax3.set_title('总ISCS分数对比')
        ax3.grid(True, alpha=0.3)
        
        # 4. Patch数量vs性能 (右中)
        ax4 = axes[1, 0]
        patch_nums_plot = [patch_nums[s] for s in scales]
        
        ax4.scatter(patch_nums_plot, base_scores, s=60, alpha=0.7, label='基础VAR')
        if 'aid_var' in results:
            ax4.scatter(patch_nums_plot, aid_scores, s=60, alpha=0.7, label='AID-VAR')
        
        ax4.set_xlabel('Patch数量')
        ax4.set_ylabel('ISCS分数')
        ax4.set_title('Patch复杂度vs ISCS性能')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. 权重分布 (左下)
        ax5 = axes[1, 1]
        weights = [2**k for k in range(len(scales))]
        total_weight = sum(weights)
        weights_norm = [w/total_weight for w in weights]
        
        bars = ax5.bar(scales, weights_norm, alpha=0.7, color='orange')
        ax5.set_xlabel('尺度索引')
        ax5.set_ylabel('归一化权重')
        ax5.set_title('尺度权重分布 (指数加权)')
        ax5.grid(True, alpha=0.3)
        
        # 6. 性能分析摘要 (右下)
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # 计算分析指标
        base_analysis = self.analyze_scale_progression(results['base_var']['scale_scores'], patch_nums)
        
        summary_text = f"""
ISCS性能分析摘要

基础VAR:
• 总分: {results['base_var']['total_iscs']:.4f}
• 最佳尺度: {base_analysis['best_scale']} (分数: {results['base_var']['scale_scores'][base_analysis['best_scale']]:.4f})
• 最差尺度: {base_analysis['worst_scale']} (分数: {results['base_var']['scale_scores'][base_analysis['worst_scale']]:.4f})
• 分数范围: {base_analysis['score_range']:.4f}
"""
        
        if 'aid_var' in results:
            aid_analysis = self.analyze_scale_progression(results['aid_var']['scale_scores'], patch_nums)
            improvement = results['aid_var']['total_iscs'] - results['base_var']['total_iscs']
            improvement_pct = (improvement / results['base_var']['total_iscs']) * 100
            
            summary_text += f"""
AID-VAR:
• 总分: {results['aid_var']['total_iscs']:.4f}
• 改善: {improvement:.4f} ({improvement_pct:.2f}%)
• 最佳尺度: {aid_analysis['best_scale']} (分数: {results['aid_var']['scale_scores'][aid_analysis['best_scale']]:.4f})

结论:
"""
            if improvement > 0.01:
                summary_text += "✅ GuidanceInjector显著提升尺度一致性"
            elif improvement > 0:
                summary_text += "🔶 GuidanceInjector轻微提升尺度一致性"
            else:
                summary_text += "❌ GuidanceInjector未提升尺度一致性"
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'iscs_detailed_analysis.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"📊 详细分析图已保存: {self.output_dir}/iscs_detailed_analysis.png")
    
    def generate_analysis_report(self, 
                               results: Dict,
                               patch_nums: List[int]) -> str:
        """
        生成详细的分析报告
        
        Args:
            results: ISCS结果
            patch_nums: patch数量序列
            
        Returns:
            report: 分析报告文本
        """
        report = []
        report.append("=" * 80)
        report.append("🎯 ISCS (Inter-Scale Consistency Score) 详细分析报告")
        report.append("=" * 80)
        report.append("")
        
        # 1. 实验配置
        report.append("📋 实验配置:")
        report.append(f"  • 评估尺度数量: {len(patch_nums)}")
        report.append(f"  • Patch数量序列: {patch_nums}")
        report.append(f"  • 权重方案: 指数加权 (w_k ∝ 2^k)")
        report.append("")
        
        # 2. 基础VAR分析
        base_analysis = self.analyze_scale_progression(results['base_var']['scale_scores'], patch_nums)
        report.append("📊 基础VAR性能分析:")
        report.append(f"  • 总ISCS分数: {results['base_var']['total_iscs']:.6f}")
        report.append(f"  • 最佳尺度: {base_analysis['best_scale']} (分数: {results['base_var']['scale_scores'][base_analysis['best_scale']]:.6f})")
        report.append(f"  • 最差尺度: {base_analysis['worst_scale']} (分数: {results['base_var']['scale_scores'][base_analysis['worst_scale']]:.6f})")
        report.append(f"  • 分数范围: {base_analysis['score_range']:.6f}")
        report.append(f"  • 线性趋势斜率: {base_analysis.get('linear_trend', 'N/A'):.6f}")
        report.append(f"  • 变异系数: {base_analysis.get('coefficient_variation', 'N/A'):.6f}")
        report.append("")
        
        # 3. AID-VAR分析（如果可用）
        if 'aid_var' in results:
            aid_analysis = self.analyze_scale_progression(results['aid_var']['scale_scores'], patch_nums)
            improvement = results['aid_var']['total_iscs'] - results['base_var']['total_iscs']
            improvement_pct = (improvement / results['base_var']['total_iscs']) * 100
            
            report.append("🚀 AID-VAR性能分析:")
            report.append(f"  • 总ISCS分数: {results['aid_var']['total_iscs']:.6f}")
            report.append(f"  • 相对基础VAR改善: {improvement:.6f} ({improvement_pct:.2f}%)")
            report.append(f"  • 最佳尺度: {aid_analysis['best_scale']} (分数: {results['aid_var']['scale_scores'][aid_analysis['best_scale']]:.6f})")
            report.append(f"  • 最差尺度: {aid_analysis['worst_scale']} (分数: {results['aid_var']['scale_scores'][aid_analysis['worst_scale']]:.6f})")
            report.append("")
            
            # 4. 尺度级别改善分析
            report.append("🔍 各尺度改善详情:")
            for scale in sorted(results['base_var']['scale_scores'].keys()):
                base_score = results['base_var']['scale_scores'][scale]
                aid_score = results['aid_var']['scale_scores'][scale]
                scale_improvement = aid_score - base_score
                scale_improvement_pct = (scale_improvement / base_score) * 100
                
                status = "✅" if scale_improvement > 0.001 else "🔶" if scale_improvement > 0 else "❌"
                report.append(f"  • 尺度 {scale}: {base_score:.6f} → {aid_score:.6f} "
                            f"({scale_improvement:+.6f}, {scale_improvement_pct:+.2f}%) {status}")
            report.append("")
        
        # 5. 结论与建议
        report.append("💡 结论与建议:")
        
        if 'aid_var' in results:
            improvement = results['aid_var']['total_iscs'] - results['base_var']['total_iscs']
            
            if improvement > 0.01:
                report.append("  ✅ GuidanceInjector显著提升了模型的尺度间一致性")
                report.append("  📈 建议在实际应用中采用AID-VAR配置")
            elif improvement > 0:
                report.append("  🔶 GuidanceInjector轻微提升了尺度间一致性")
                report.append("  🤔 建议进一步优化GuidanceInjector的训练策略")
            else:
                report.append("  ❌ GuidanceInjector未能提升尺度间一致性")
                report.append("  🔧 建议重新审视GuidanceInjector的设计和训练过程")
            
            # 分析哪些尺度受益最多
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
            
            report.append(f"  🎯 GuidanceInjector对尺度{best_improved_scale}效果最佳 "
                         f"(改善{improvements_by_scale[best_improved_scale]:.6f})")
            report.append(f"  ⚠️ GuidanceInjector对尺度{worst_improved_scale}效果最差 "
                         f"(改善{improvements_by_scale[worst_improved_scale]:.6f})")
        else:
            report.append("  📝 仅评估了基础VAR模型")
            report.append("  💡 建议训练GuidanceInjector并进行对比实验")
        
        report.append("")
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        # 保存报告
        report_path = self.output_dir / 'iscs_analysis_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"📄 分析报告已保存: {report_path}")
        
        return report_text

def load_checkpoint_safely(checkpoint_path: str, 
                          model: nn.Module,
                          map_location: str = 'cpu') -> bool:
    """
    安全地加载模型检查点
    
    Args:
        checkpoint_path: 检查点路径
        model: 模型实例
        map_location: 加载位置
        
    Returns:
        success: 是否成功加载
    """
    try:
        if not os.path.exists(checkpoint_path):
            logger.error(f"检查点文件不存在: {checkpoint_path}")
            return False
        
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        # 处理不同的检查点格式
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # 尝试加载状态字典
        model.load_state_dict(state_dict, strict=False)
        logger.info(f"✅ 成功加载检查点: {checkpoint_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ 加载检查点失败: {checkpoint_path}, 错误: {e}")
        return False

def setup_logging(output_dir: str, level: int = logging.INFO):
    """
    设置日志系统
    
    Args:
        output_dir: 输出目录
        level: 日志级别
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # 清除现有的handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 创建formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 文件handler
    file_handler = logging.FileHandler(
        os.path.join(output_dir, 'iscs_computation.log'),
        encoding='utf-8'
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger 