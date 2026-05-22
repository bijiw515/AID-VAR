#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 ISCS (Inter-Scale Consistency Score) 计算脚本

ISCS是专门为AID-VAR设计的新指标，用于量化GuidanceInjector解决的特定"误差"。
核心论点：误差主要表现为条件预测p(r_k|r_{<k'})质量的下降。

实现方案：联合分布散度 (ISCS-FJD)
- 使用FJD概念应用于VAR的多尺度生成过程
- 评估模型在群体水平上是否正确学习了条件分布
- 对每个尺度k构建联合特征向量并计算Fréchet距离

作者：Expert AI Developer
日期：2025-09-23
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
    print("⚠️ seaborn未安装，将使用基础matplotlib进行可视化")

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入项目模块
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

# 设置日志
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
    🎯 DINO特征提取器（使用本地模型）
    
    用于提取图像的语义特征，构建联合特征向量。
    支持多尺度特征提取以匹配VAR的层次结构。
    """
    
    def __init__(self, model_name='dinov2_vits14', device='cuda'):
        super().__init__()
        self.device = device
        self.model_name = model_name
        
        logger.info(f"🔄 初始化DINO特征提取器: {model_name}")
        
        # 使用本地DINO模型
        try:
            import timm
            from stylegan_t.networks.discriminator import DINO
            
            # 使用本地DINO模型路径
            local_dino_path = "/home/intern/Ligong/VAR/checkpoints/dino/dino_deitsmall16_pretrain.pth"
            
            # 创建DINO模型（使用StyleGAN-T中的实现）
            self.model = DINO(
                hooks=[2, 5, 8, 11], 
                hook_patch=True,
                local_checkpoint_path=local_dino_path
            ).to(device)
            
            self.model.eval()
            
            # 冻结参数
            for param in self.model.parameters():
                param.requires_grad = False
                
            logger.info(f"✅ 本地DINO模型加载成功: {local_dino_path}")
            
        except Exception as e:
            logger.error(f"❌ 本地DINO模型加载失败: {e}")
            raise
        
        # 特征维度（DINO ViT-S）
        self.feature_dim = 384
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        提取图像特征
        
        Args:
            images: 输入图像 (B, 3, H, W) 范围[0,1]
            
        Returns:
            features: 图像特征 (B, feature_dim)
        """
        # 确保图像在正确的范围内
        if images.max() > 1.0:
            images = images / 255.0
        
        # 将图像范围从[0,1]转换为[-1,1]（StyleGAN-T DINO的期望输入）
        images = images * 2.0 - 1.0
        
        # 提取特征
        with torch.no_grad():
            features_dict = self.model(images)
            
            # StyleGAN-T DINO返回字典，我们使用最后一层特征
            # 通常是 '4' 键对应最高层特征
            if '4' in features_dict:
                features = features_dict['4']
            elif len(features_dict) > 0:
                # 使用最后一个特征
                features = list(features_dict.values())[-1]
            else:
                raise ValueError("DINO模型未返回有效特征")
                
            # 确保特征是2D的 (B, feature_dim)
            if len(features.shape) > 2:
                features = features.mean(dim=list(range(2, len(features.shape))))
            
        return features

class ImageDataset(Dataset):
    """
    🎯 图像数据集类
    
    支持从目录或npz文件加载图像数据
    """
    
    def __init__(self, data_source: Union[str, np.ndarray], transform=None):
        self.transform = transform
        
        if isinstance(data_source, str):
            # 从目录加载
            self.images = []
            for img_path in Path(data_source).glob('*.png'):
                self.images.append(str(img_path))
            logger.info(f"📁 从目录加载 {len(self.images)} 张图像")
            
        elif isinstance(data_source, np.ndarray):
            # 从numpy数组加载
            self.images = data_source
            logger.info(f"📊 从numpy数组加载 {len(self.images)} 张图像")
            
        else:
            raise ValueError("数据源必须是目录路径或numpy数组")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        if isinstance(self.images[idx], str):
            # 从文件加载
            image = Image.open(self.images[idx]).convert('RGB')
            image = transforms.ToTensor()(image)
        else:
            # 从numpy数组加载
            image = torch.from_numpy(self.images[idx]).float()
            if image.max() > 1.0:
                image = image / 255.0
        
        if self.transform:
            image = self.transform(image)
            
        return image

class ISCSCalculator:
    """
    🎯 ISCS (Inter-Scale Consistency Score) 计算器
    
    实现联合分布散度方法来评估多尺度生成质量
    """
    
    def __init__(self, 
                 feature_extractor: DINOv2FeatureExtractor,
                 patch_nums: Tuple[int] = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
                 device: str = 'cuda'):
        
        self.feature_extractor = feature_extractor
        self.patch_nums = patch_nums
        self.device = device
        
        # 权重系数：粗糙尺度权重更高
        self.weights = {k: 2**k for k in range(len(patch_nums))}
        total_weight = sum(self.weights.values())
        self.weights = {k: w/total_weight for k, w in self.weights.items()}
        
        logger.info(f"🎯 ISCS计算器初始化完成")
        logger.info(f"📏 尺度数量: {len(patch_nums)}")
        logger.info(f"⚖️ 权重分布: {self.weights}")
        
    def extract_multiscale_features(self, 
                                  images: torch.Tensor, 
                                  var_model: VAR) -> Dict[int, torch.Tensor]:
        """
        提取多尺度特征表示
        
        Args:
            images: 输入图像 (B, 3, H, W)
            var_model: VAR模型用于获取多尺度token表示
            
        Returns:
            scale_features: 每个尺度的特征字典 {scale_idx: features}
        """
        scale_features = {}
        
        with torch.no_grad():
            # 1. 通过VAE编码器获取特征表示
            vae = var_model.vae_proxy[0]
            
            # 调整图像范围：从[0,1]到[-1,1]
            if images.max() <= 1.0:
                images = images * 2.0 - 1.0
            
            # 确保数据类型一致性
            images = images.to(dtype=torch.float32)
            
            # 通过编码器获取特征
            f = vae.quant_conv(vae.encoder(images))
            
            # 获取多尺度的token indices
            ms_idx_Bl = vae.quantize.f_to_idxBl_or_fhat(f, to_fhat=False, v_patch_nums=self.patch_nums)
            
            # 3. 为每个尺度提取特征
            for scale_idx, patch_num in enumerate(self.patch_nums):
                # 获取对应尺度的token indices
                if scale_idx < len(ms_idx_Bl):
                    idx_Bl = ms_idx_Bl[scale_idx]  # (B, L_k)
                    
                    # 通过VAE的quantize embedding获取特征，然后用VAR的word_embed
                    # 首先通过VAE的embedding获取连续特征
                    vae_embeddings = vae.quantize.embedding(idx_Bl)  # (B, L_k, Cvae)
                    
                    # 然后通过VAR的word_embed转换为VAR特征空间
                    scale_embeddings = var_model.word_embed(vae_embeddings)  # (B, L_k, C)
                    
                    # 全局平均池化得到固定长度特征
                    scale_features[scale_idx] = scale_embeddings.mean(dim=1)  # (B, C)
                else:
                    # 如果没有对应尺度，跳过
                    logger.warning(f"⚠️ 尺度 {scale_idx} 超出可用范围，跳过")
                    continue
                
                logger.debug(f"🔍 尺度 {scale_idx} (patch_num={patch_num}): "
                           f"特征形状 {scale_features[scale_idx].shape}")
        
        return scale_features
    
    def build_joint_features(self, 
                           scale_features: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        """
        构建联合特征向量 [feat(s_{k-1}), feat(s_k)]
        
        Args:
            scale_features: 每个尺度的特征
            
        Returns:
            joint_features: 联合特征字典 {scale_idx: joint_feat}
        """
        joint_features = {}
        
        for scale_idx in range(1, len(self.patch_nums)):  # 从第2个尺度开始
            prev_feat = scale_features[scale_idx - 1]  # s_{k-1}
            curr_feat = scale_features[scale_idx]       # s_k
            
            # 拼接构建联合特征
            joint_feat = torch.cat([prev_feat, curr_feat], dim=1)  # (B, 2*C)
            joint_features[scale_idx] = joint_feat
            
            logger.debug(f"🔗 联合特征 尺度{scale_idx}: {joint_feat.shape}")
        
        return joint_features
    
    def compute_frechet_distance(self, 
                               real_features: torch.Tensor, 
                               generated_features: torch.Tensor) -> float:
        """
        计算两组特征之间的Fréchet距离
        
        Args:
            real_features: 真实数据特征 (N, D)
            generated_features: 生成数据特征 (M, D)
            
        Returns:
            frechet_distance: FJD距离
        """
        # 转换为numpy数组
        if isinstance(real_features, torch.Tensor):
            real_features = real_features.cpu().numpy()
        if isinstance(generated_features, torch.Tensor):
            generated_features = generated_features.cpu().numpy()
        
        # 计算均值和协方差
        mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
        mu2, sigma2 = generated_features.mean(axis=0), np.cov(generated_features, rowvar=False)
        
        # 数值稳定性处理
        eps = 1e-6
        sigma1 += eps * np.eye(sigma1.shape[0])
        sigma2 += eps * np.eye(sigma2.shape[0])
        
        # 计算Fréchet距离
        diff = mu1 - mu2
        covmean = self._sqrtm_newton(sigma1.dot(sigma2))
        
        fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
        
        return float(fid)
    
    def _sqrtm_newton(self, A: np.ndarray, num_iters: int = 50) -> np.ndarray:
        """
        使用Newton迭代法计算矩阵平方根（数值稳定版本）
        """
        dim = A.shape[0]
        # 初始化
        X = A.copy()
        I = np.eye(dim)
        
        for _ in range(num_iters):
            X_inv = np.linalg.pinv(X)
            X_new = 0.5 * (X + A.dot(X_inv))
            
            # 检查收敛
            if np.allclose(X, X_new, rtol=1e-6):
                break
            X = X_new
        
        return X
    
    def compute_scale_iscs(self, 
                          real_joint_features: Dict[int, torch.Tensor],
                          generated_joint_features: Dict[int, torch.Tensor]) -> Dict[int, float]:
        """
        计算每个尺度的ISCS分数
        
        Args:
            real_joint_features: 真实数据的联合特征
            generated_joint_features: 生成数据的联合特征
            
        Returns:
            scale_scores: 每个尺度的ISCS分数（直接使用FJD距离）
        """
        scale_scores = {}
        
        for scale_idx in real_joint_features.keys():
            real_feat = real_joint_features[scale_idx]
            gen_feat = generated_joint_features[scale_idx]
            
            logger.info(f"🔄 计算尺度 {scale_idx} 的ISCS分数...")
            
            # 计算FJD距离
            fjd = self.compute_frechet_distance(real_feat, gen_feat)
            
            # ISCS直接使用FJD距离值（距离越小越好）
            iscs_score = fjd
            scale_scores[scale_idx] = iscs_score
            
            logger.info(f"✅ 尺度 {scale_idx}: FJD={fjd:.4f}, ISCS={iscs_score:.4f}")
        
        return scale_scores
    
    def compute_aggregated_iscs(self, scale_scores: Dict[int, float]) -> float:
        """
        计算聚合的ISCS总分
        
        Args:
            scale_scores: 每个尺度的分数
            
        Returns:
            total_iscs: 加权总分
        """
        total_score = 0.0
        total_weight = 0.0
        
        for scale_idx, score in scale_scores.items():
            weight = self.weights.get(scale_idx, 0.0)
            total_score += weight * score
            total_weight += weight
        
        # 归一化
        if total_weight > 0:
            total_iscs = total_score / total_weight
        else:
            total_iscs = 0.0
        
        logger.info(f"🎯 聚合ISCS总分: {total_iscs:.4f}")
        return total_iscs

    def compute_sample_iscs(self, 
                           real_joint_features: Dict[int, torch.Tensor],
                           sample_scale_features: Dict[int, torch.Tensor]) -> float:
        """
        计算单个样本的ISCS分数
        
        Args:
            real_joint_features: 真实数据的联合特征（用作参考分布）
            sample_scale_features: 单个样本的多尺度特征
            
        Returns:
            sample_iscs: 单个样本的ISCS分数
        """
        # 构建样本的联合特征
        sample_joint_features = self.build_joint_features(sample_scale_features)
        
        scale_scores = {}
        
        for scale_idx in sample_joint_features.keys():
            if scale_idx in real_joint_features:
                # 获取真实数据分布特征（参考）
                real_feat = real_joint_features[scale_idx]
                
                # 单个样本特征
                sample_feat = sample_joint_features[scale_idx]  # (1, D)
                
                # 确保张量在同一设备上进行计算
                device = self.device
                real_feat = real_feat.to(device)
                sample_feat = sample_feat.to(device)
                
                # 计算样本到真实分布的平均距离（简化版FJD）
                real_mean = real_feat.mean(dim=0)  # (D,)
                sample_vec = sample_feat.squeeze(0)  # (D,)
                
                # 使用欧几里得距离作为样本级ISCS分数
                distance = torch.norm(sample_vec - real_mean).item()
                scale_scores[scale_idx] = distance
        
        # 计算加权总分
        total_score = 0.0
        total_weight = 0.0
        
        for scale_idx, score in scale_scores.items():
            weight = self.weights.get(scale_idx, 0.0)
            total_score += weight * score
            total_weight += weight
        
        # 归一化
        if total_weight > 0:
            sample_iscs = total_score / total_weight
        else:
            sample_iscs = 0.0
        
        return sample_iscs

class ISCSExperiment:
    """
    🎯 ISCS实验管理器
    
    管理完整的ISCS评估流程，包括数据加载、模型推理和指标计算
    """
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        
        # 设置优化配置（与generate_aid_fid_samples.py一致）
        self._setup_optimization()
        
        # 初始化特征提取器
        self.feature_extractor = DINOv2FeatureExtractor(
            model_name=args.dinov2_model,
            device=self.device
        )
        
        # 初始化ISCS计算器
        self.iscs_calculator = ISCSCalculator(
            feature_extractor=self.feature_extractor,
            patch_nums=tuple(args.patch_nums),
            device=self.device
        )
        
        # 加载模型
        self._load_models()
        
        logger.info(f"🚀 ISCS实验初始化完成")
    
    def _validate_checkpoints(self):
        """验证检查点文件是否存在，提供下载信息"""
        missing_files = []
        
        # 检查VAE检查点
        if not os.path.exists(self.args.vae_ckpt):
            missing_files.append(('VAE', self.args.vae_ckpt))
        
        # 检查VAR检查点
        if not os.path.exists(self.args.var_ckpt):
            missing_files.append(('VAR', self.args.var_ckpt))
        
        if missing_files:
            logger.error(f"❌ 缺失必要的检查点文件:")
            for model_type, path in missing_files:
                logger.error(f"  • {model_type}: {path}")
            
            logger.error(f"🔄 请下载必要的检查点文件:")
            logger.error(f"  💾 从HuggingFace下载基础模型检查点:")
            logger.error(f"    wget https://huggingface.co/FoundationVision/var/resolve/main/{self.args.vae_ckpt}")
            logger.error(f"    wget https://huggingface.co/FoundationVision/var/resolve/main/{self.args.var_ckpt}")
            logger.error(f"  📁 或确保检查点文件位于正确路径")
            
            raise FileNotFoundError(f"缺失必要的检查点文件，请按照上述提示下载")
        
        logger.info(f"✅ 所有必要的检查点文件已就位")
    
    def _setup_optimization(self):
        """设置优化配置（与generate_aid_fid_samples.py一致）"""
        # 随机种子设置（确保可重现性）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # 启用TF32优化
        tf32 = True
        torch.backends.cudnn.allow_tf32 = bool(tf32)
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.set_float32_matmul_precision('high' if tf32 else 'highest')
        
        # 确定数据类型
        if self.args.dtype == 'float16':
            self.dtype = torch.float16
        else:
            self.dtype = torch.bfloat16
        
        logger.info(f"🔧 优化配置:")
        logger.info(f"  • TF32优化: {tf32}")
        logger.info(f"  • 数据类型: {self.args.dtype}")
        logger.info(f"  • CUDNN确定性: {torch.backends.cudnn.deterministic}")
    
    def _load_models(self):
        """加载VAR模型和GuidanceInjector（与generate_aid_fid_samples.py保持一致）"""
        logger.info(f"🔄 加载模型...")
        
        # 确定VAR检查点路径（与generate_aid_fid_samples.py一致）
        if self.args.var_ckpt is None:
            # 自动确定路径，与generate_aid_fid_samples.py完全一致
            self.args.var_ckpt = f'checkpoints/var_d{self.args.model_depth}.pth'
            logger.info(f"🔄 自动确定VAR检查点路径: {self.args.var_ckpt}")
        elif not os.path.exists(self.args.var_ckpt):
            # 尝试自动确定路径
            auto_var_ckpt = f'checkpoints/var_d{self.args.model_depth}.pth'
            if os.path.exists(auto_var_ckpt):
                self.args.var_ckpt = auto_var_ckpt
                logger.info(f"🔄 自动确定VAR检查点路径: {auto_var_ckpt}")
        
        # 检查检查点文件是否存在，如果不存在提供下载信息
        self._validate_checkpoints()
        
        # 构建VAE和VAR模型（使用与generate_aid_fid_samples.py相同的配置）
        patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
        logger.info(f"🏗️ 构建VAE和VAR模型 (深度={self.args.model_depth})...")
        vae, var = build_vae_var(
            V=4096, Cvae=32, ch=160, share_quant_resi=4,    # VQVAE超参数
            device=self.device, patch_nums=patch_nums,
            num_classes=1000, depth=self.args.model_depth, shared_aln=False,
        )
        
        # 加载VAE检查点
        logger.info(f"🔄 加载VAE检查点: {self.args.vae_ckpt}")
        vae.load_state_dict(torch.load(self.args.vae_ckpt, map_location='cpu'), strict=True)
        logger.info(f"✅ VAE检查点加载成功")
        
        # 加载VAR检查点
        logger.info(f"🔄 加载VAR检查点: {self.args.var_ckpt}")
        var.load_state_dict(torch.load(self.args.var_ckpt, map_location='cpu'), strict=True)
        logger.info(f"✅ VAR检查点加载成功")
        
        # 设置为评估模式并冻结参数（与generate_aid_fid_samples.py一致）
        vae.eval()
        var.eval()
        for p in vae.parameters(): p.requires_grad_(False)
        for p in var.parameters(): p.requires_grad_(False)
        
        self.vae = vae
        self.var = var
        
        # 加载GuidanceInjector（与generate_aid_fid_samples.py一致的加载方式）
        if self.args.planner_ckpt and os.path.exists(self.args.planner_ckpt):
            logger.info(f"🔄 构建GuidanceInjector...")
            
            # 根据VAR模型深度计算正确的嵌入维度（与build_vae_var保持一致）
            var_embed_dim = self.args.model_depth * 64  # width = depth * 64
            
            logger.info(f"🎯 GuidanceInjector配置: VAR深度={self.args.model_depth}, 嵌入维度={var_embed_dim}")
            
            self.planner = GuidanceInjector(
                input_dim=var_embed_dim,    # VAR的嵌入维度（根据深度动态计算）
                embed_dim=var_embed_dim,    # 保持一致
                num_layers=2,
                num_heads=8
            ).to(self.device)
            
            logger.info(f"🔄 加载GuidanceInjector检查点: {self.args.planner_ckpt}")
            planner_state = torch.load(self.args.planner_ckpt, map_location='cpu')
            
            # 使用与generate_aid_fid_samples.py相同的加载逻辑
            if 'planner_state_dict' in planner_state:
                # 从完整的训练检查点加载
                missing_keys, unexpected_keys = self.planner.load_state_dict(
                    planner_state['planner_state_dict'], strict=False
                )
                if unexpected_keys:
                    pos_keys = [k for k in unexpected_keys if k.startswith('pos_')]
                    other_keys = [k for k in unexpected_keys if not k.startswith('pos_')]
                    if pos_keys:
                        logger.info(f"🔄 忽略位置编码buffer: {pos_keys}")
                    if other_keys:
                        logger.warning(f"⚠️ 其他未预期的键: {other_keys}")
                if missing_keys:
                    logger.warning(f"⚠️ 缺失的键: {missing_keys}")
                logger.info("✅ GuidanceInjector模型状态从训练检查点加载成功")
            elif 'planner' in planner_state:
                # 从旧格式的检查点加载
                self.planner.load_state_dict(planner_state['planner'], strict=True)
                logger.info("✅ GuidanceInjector模型状态从旧格式检查点加载成功")
            else:
                # 直接加载模型权重
                self.planner.load_state_dict(planner_state, strict=True)
                logger.info("✅ GuidanceInjector模型状态直接加载成功")
            
            self.planner.eval()
            for p in self.planner.parameters(): p.requires_grad_(False)
            
            logger.info(f"✅ GuidanceInjector加载完成")
        else:
            self.planner = None
            logger.info(f"⚠️ 未加载GuidanceInjector，仅评估基础VAR模型")
        
        # 输出模型参数统计（与generate_aid_fid_samples.py一致）
        logger.info(f"🎯 模型加载完成:")
        logger.info(f"  📦 VAE参数: {sum(p.numel() for p in vae.parameters()):,}")
        logger.info(f"  🧠 VAR参数: {sum(p.numel() for p in var.parameters()):,}")
        if self.planner is not None:
            logger.info(f"  🎯 GuidanceInjector参数: {sum(p.numel() for p in self.planner.parameters()):,}")
    
    def save_sample_images_with_scores(self, 
                                     images: torch.Tensor, 
                                     iscs_scores: List[float], 
                                     output_dir: str, 
                                     prefix: str = "sample"):
        """
        保存样本图像并在图像上标注ISCS分数
        
        Args:
            images: 生成的图像张量 (B, 3, H, W)，范围[0,1]
            iscs_scores: 每个样本的ISCS分数列表
            output_dir: 输出目录
            prefix: 文件名前缀
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 转换图像格式并保存
        for i, (img_tensor, score) in enumerate(zip(images, iscs_scores)):
            # 将张量转换为PIL图像
            img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)
            
            # 在图像上添加ISCS分数标注
            img_with_score = self._add_score_annotation(img_pil, score)
            
            # 保存图像
            filename = f"{prefix}_{i:02d}_iscs_{score:.4f}.png"
            save_path = output_path / filename
            img_with_score.save(save_path)
            
            logger.info(f"💾 保存样本图像: {save_path}")
    
    def _add_score_annotation(self, image: Image.Image, score: float) -> Image.Image:
        """
        在图像上添加ISCS分数标注
        
        Args:
            image: PIL图像
            score: ISCS分数
            
        Returns:
            annotated_image: 标注后的图像
        """
        # 创建图像副本
        img_copy = image.copy()
        draw = ImageDraw.Draw(img_copy)
        
        # 设置字体（尝试使用系统字体，如果不存在则使用默认字体）
        try:
            # 尝试加载中文字体
            font_size = max(16, min(image.width // 20, 24))
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            try:
                # 尝试备用字体
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
            except:
                # 使用默认字体
                font = ImageFont.load_default()
        
        # 准备文本
        score_text = f"ISCS: {score:.4f}"
        
        # 计算文本位置（右上角）
        text_bbox = draw.textbbox((0, 0), score_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # 设置位置（右上角，留一些边距）
        margin = 10
        x = image.width - text_width - margin
        y = margin
        
        # 绘制半透明背景
        bg_padding = 5
        bg_bbox = [
            x - bg_padding, 
            y - bg_padding, 
            x + text_width + bg_padding, 
            y + text_height + bg_padding
        ]
        draw.rectangle(bg_bbox, fill=(0, 0, 0, 180))  # 半透明黑色背景
        
        # 绘制文本
        draw.text((x, y), score_text, fill=(255, 255, 255), font=font)  # 白色文字
        
        return img_copy
    
    def generate_and_extract_features_streaming(self, num_samples: int, with_planner: bool = False):
        """
        流式生成样本并提取特征（增强版本：保存前20张图片并标注ISCS分数）
        
        生成策略：
        - 1000个类别，每个类别50个样本
        - 批次大小固定为50（每次处理一个完整类别）
        - 以类别ID作为随机种子确保可重现性
        - 保存前20张图片并标注其ISCS分数
        
        Args:
            num_samples: 生成样本数量（应为50000）
            with_planner: 是否使用GuidanceInjector引导
            
        Returns:
            all_scale_features: 累积的多尺度特征
        """
        logger.info(f"🎨 流式生成 {num_samples} 个样本并提取特征 (with_planner={with_planner})...")
        logger.info(f"📋 生成策略：1000类 × 50样本/类，批次大小=50，以类别ID为种子")
        logger.info(f"📸 将保存前20张生成图片并标注ISCS分数")
        
        # 验证样本数量
        expected_samples = 50000  # 1000类 × 50样本
        if num_samples != expected_samples:
            logger.warning(f"⚠️ 样本数量({num_samples})不等于期望值({expected_samples})，将按类别生成")
        
        # 初始化特征累积器
        accumulated_features = {}
        
        # 固定配置（与generate_aid_fid_samples.py一致）
        batch_size = 50  # 固定批次大小
        samples_per_class = 50  # 每类固定样本数
        num_classes = 1000
        total_samples = 0
        
        # 图像保存配置
        saved_images_count = 0
        max_save_images = 20
        output_dir = Path(self.args.output_dir) / "sample_images_with_scores"
        
        # 为了计算单个样本的ISCS分数，我们需要预先计算真实数据的参考分布
        logger.info(f"🔄 预计算真实数据参考分布用于样本级ISCS计算...")
        real_images = self.load_real_images()
        real_scale_features = self.extract_real_features_streaming(real_images)
        real_joint_features = self.iscs_calculator.build_joint_features(real_scale_features)
        del real_images  # 释放内存
        
        # 逐类生成样本（与generate_aid_fid_samples.py完全一致）
        for class_id in tqdm(range(num_classes), desc=f"生成样本{'(AID-VAR)' if with_planner else '(基础VAR)'}"):
            # 为每个类设置种子以确保可重现性（与generate_aid_fid_samples.py一致）
            torch.manual_seed(class_id)
            import random
            random.seed(class_id)
            np.random.seed(class_id)
            
            # 为当前类生成确切50个样本
            class_labels = [class_id] * samples_per_class
            current_batch_size = batch_size  # 始终50
            label_B = torch.tensor(class_labels, device=self.device)
            
            with torch.inference_mode():
                # 注意：VAE特征提取需要float32，生成时才使用mixed precision
                autocast_enabled = False  # 暂时禁用autocast避免dtype冲突
                with torch.autocast(self.device.type, enabled=autocast_enabled, dtype=self.dtype, cache_enabled=True):
                    # 1. 生成当前类别的图像
                    if with_planner and self.planner is not None:
                        # 使用AID-VAR引导生成（与generate_aid_fid_samples.py完全一致）
                        images = self.var.aid_guided_inference(
                            planner=self.planner,
                            B=current_batch_size,
                            label_B=label_B,
                            cfg=self.args.cfg,
                            top_k=self.args.top_k,
                            top_p=self.args.top_p,
                            g_seed=class_id,  # 使用class_id作为种子
                            more_smooth=self.args.more_smooth
                        )
                    else:
                        # 基础VAR生成（使用相同参数）
                        images = self.var.autoregressive_infer_cfg(
                            B=current_batch_size,
                            label_B=label_B,
                            cfg=self.args.cfg,
                            top_k=self.args.top_k,
                            top_p=self.args.top_p,
                            g_seed=class_id,  # 使用class_id作为种子
                            more_smooth=self.args.more_smooth
                        )
                    
                    # 2. 立即提取多尺度特征
                    batch_features = self.iscs_calculator.extract_multiscale_features(images, self.var)
                    
                    # 3. 如果还没保存够20张图片，计算单个样本的ISCS分数并保存图片
                    if saved_images_count < max_save_images:
                        num_to_save = min(max_save_images - saved_images_count, current_batch_size)
                        
                        # 计算每个样本的ISCS分数
                        sample_iscs_scores = []
                        for i in range(num_to_save):
                            # 提取单个样本的特征
                            sample_features = {}
                            for scale_idx, scale_feat in batch_features.items():
                                sample_features[scale_idx] = scale_feat[i:i+1]  # 保持批次维度
                            
                            # 计算单个样本的ISCS分数
                            sample_iscs = self.iscs_calculator.compute_sample_iscs(
                                real_joint_features, sample_features
                            )
                            sample_iscs_scores.append(sample_iscs)
                        
                        # 保存带分数标注的图片
                        self.save_sample_images_with_scores(
                            images[:num_to_save],
                            sample_iscs_scores,
                            output_dir,
                            prefix=f"{'aid_var' if with_planner else 'base_var'}_class{class_id}"
                        )
                        
                        saved_images_count += num_to_save
                        logger.info(f"📸 已保存 {saved_images_count}/{max_save_images} 张图片")
                    
                    # 4. 累积特征到CPU内存
                    for scale_idx, features in batch_features.items():
                        features_cpu = features.cpu()
                        if scale_idx not in accumulated_features:
                            accumulated_features[scale_idx] = []
                        accumulated_features[scale_idx].append(features_cpu)
                    
                    total_samples += current_batch_size
                    
                    # 5. 清理GPU内存
                    del images, batch_features
                    torch.cuda.empty_cache()
        
        # 清理参考特征
        del real_scale_features, real_joint_features
        torch.cuda.empty_cache()
        
        # 6. 合并所有批次的特征
        final_features = {}
        for scale_idx, feature_list in accumulated_features.items():
            final_features[scale_idx] = torch.cat(feature_list, dim=0)
            logger.info(f"🔍 尺度 {scale_idx} 最终特征形状: {final_features[scale_idx].shape}")
        
        logger.info(f"✅ 特征提取完成，总样本数: {total_samples}")
        logger.info(f"📊 生成统计: {num_classes}类 × {samples_per_class}样本/类 = {total_samples}总样本")
        logger.info(f"📸 图片保存统计: 成功保存 {saved_images_count} 张带ISCS分数标注的图片")
        
        return final_features
    
    def load_real_images(self) -> torch.Tensor:
        """加载真实图像数据"""
        logger.info(f"📁 加载真实图像数据: {self.args.real_images_path}")
        
        if self.args.real_images_path.endswith('.npz'):
            # 从npz文件加载
            data = np.load(self.args.real_images_path)
            
            # 查找图像数据
            if 'images' in data:
                real_images = torch.from_numpy(data['images']).float()
            elif 'arr_0' in data:
                real_images = torch.from_numpy(data['arr_0']).float()
            else:
                # 使用第一个数组
                key = list(data.keys())[0]
                real_images = torch.from_numpy(data[key]).float()
            
            logger.info(f"🔍 原始数据形状: {real_images.shape}")
            
            # 确保数据是4D张量 (N, C, H, W)
            if len(real_images.shape) == 4:
                # 检查通道维度位置
                if real_images.shape[1] == 3:
                    # 已经是 (N, C, H, W) 格式
                    pass
                elif real_images.shape[3] == 3:
                    # 是 (N, H, W, C) 格式，需要转换
                    real_images = real_images.permute(0, 3, 1, 2)
                    logger.info(f"🔄 转换图像格式从(N,H,W,C)到(N,C,H,W)")
            elif len(real_images.shape) == 3:
                # 可能是 (N, H, W) 灰度图，需要添加通道维度
                real_images = real_images.unsqueeze(1).repeat(1, 3, 1, 1)
                logger.info(f"🔄 添加通道维度，从灰度转为RGB")
            else:
                raise ValueError(f"不支持的图像数据形状: {real_images.shape}")
            
            # 调整数据范围到[0,1]
            if real_images.max() > 1.0:
                real_images = real_images / 255.0
                logger.info(f"🔄 将数据范围从[0,255]调整到[0,1]")
                
        else:
            # 从目录加载
            dataset = ImageDataset(self.args.real_images_path)
            dataloader = DataLoader(
                dataset, batch_size=self.args.batch_size, 
                shuffle=False, num_workers=4
            )
            
            real_images = []
            for batch in tqdm(dataloader, desc="加载真实图像"):
                real_images.append(batch)
            real_images = torch.cat(real_images, dim=0)
        
        # 限制数量
        if len(real_images) > self.args.num_samples:
            indices = torch.randperm(len(real_images))[:self.args.num_samples]
            real_images = real_images[indices]
        
        logger.info(f"✅ 真实图像加载完成: {real_images.shape}")
        
        # 最终验证数据形状
        if len(real_images.shape) != 4 or real_images.shape[1] != 3:
            raise ValueError(f"图像数据形状错误: {real_images.shape}，期望 (N, 3, H, W)")
        
        return real_images
    
    def run_experiment(self):
        """运行完整的ISCS评估实验（内存高效版本）"""
        logger.info(f"🚀 开始ISCS评估实验 - 将生成 {self.args.num_samples} 个样本进行ISCS计算")
        
        results = {}
        
        # 1. 提取真实图像的多尺度特征
        logger.info(f"📊 提取真实图像特征...")
        real_images = self.load_real_images()
        real_scale_features = self.extract_real_features_streaming(real_images)
        
        # 构建真实图像的联合特征
        real_joint_features = self.iscs_calculator.build_joint_features(real_scale_features)
        logger.info(f"✅ 真实图像联合特征构建完成")
        
        # 清理真实图像内存
        del real_images
        torch.cuda.empty_cache()
        
        # 2. 评估AID-VAR模型（我们的方法，优先评估）
        if self.planner is not None:
            logger.info(f"🚀 评估AID-VAR模型（我们的方法）（流式生成 {self.args.num_samples} 个样本）...")
            gen_scale_features_aid = self.generate_and_extract_features_streaming(
                self.args.num_samples, with_planner=True
            )
            
            # 构建AID-VAR的联合特征
            gen_joint_features_aid = self.iscs_calculator.build_joint_features(gen_scale_features_aid)
            
            # 计算AID-VAR的ISCS分数
            scale_scores_aid = self.iscs_calculator.compute_scale_iscs(
                real_joint_features, gen_joint_features_aid
            )
            total_iscs_aid = self.iscs_calculator.compute_aggregated_iscs(scale_scores_aid)
            
            results['aid_var'] = {
                'scale_scores': scale_scores_aid,
                'total_iscs': total_iscs_aid
            }
            
            logger.info(f"📈 AID-VAR ISCS（我们的方法）: {total_iscs_aid:.6f}")
            
            # 清理AID-VAR特征内存
            del gen_scale_features_aid, gen_joint_features_aid
            torch.cuda.empty_cache()
        
        # 3. 评估基础VAR模型（基线模型）
        logger.info(f"📊 评估基础VAR模型（基线模型）（流式生成 {self.args.num_samples} 个样本）...")
        gen_scale_features_base = self.generate_and_extract_features_streaming(
            self.args.num_samples, with_planner=False
        )
        
        # 构建基础VAR的联合特征
        gen_joint_features_base = self.iscs_calculator.build_joint_features(gen_scale_features_base)
        
        # 计算基础VAR的ISCS分数
        scale_scores_base = self.iscs_calculator.compute_scale_iscs(
            real_joint_features, gen_joint_features_base
        )
        total_iscs_base = self.iscs_calculator.compute_aggregated_iscs(scale_scores_base)
        
        results['base_var'] = {
            'scale_scores': scale_scores_base,
            'total_iscs': total_iscs_base
        }
        
        logger.info(f"📈 基础VAR ISCS（基线模型）: {total_iscs_base:.6f}")
        
        # 计算改善程度（如果有AID-VAR结果）
        if 'aid_var' in results:
            improvement = total_iscs_base - results['aid_var']['total_iscs']  # 距离减少为正改善
            improvement_pct = (improvement / total_iscs_base) * 100
            
            logger.info(f"📊 ISCS改善: {improvement:.6f} ({improvement_pct:.2f}%)")
            logger.info(f"🎯 我们的方法相比基线模型的表现: {'✅ 更好' if improvement > 0 else '❌ 更差'}")
        
        # 清理基础VAR特征内存
        del gen_scale_features_base, gen_joint_features_base
        torch.cuda.empty_cache()
        
        # 清理真实图像联合特征
        del real_joint_features
        torch.cuda.empty_cache()
        
        # 4. 保存结果
        self._save_results(results)
        
        # 5. 可视化结果
        self._visualize_results(results)
        
        logger.info(f"✅ ISCS评估实验完成")
        
        return results
    
    def extract_real_features_streaming(self, real_images: torch.Tensor):
        """
        流式提取真实图像特征（内存高效）
        
        Args:
            real_images: 真实图像张量
            
        Returns:
            real_scale_features: 真实图像的多尺度特征
        """
        logger.info(f"🔄 流式提取真实图像特征...")
        
        accumulated_features = {}
        batch_size = self.args.batch_size
        num_samples = min(len(real_images), self.args.num_samples)
        total_batches = (num_samples + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(total_batches), desc="提取真实图像特征"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            
            batch_images = real_images[start_idx:end_idx].to(self.device)
            
            # 验证批次数据形状
            if len(batch_images.shape) != 4 or batch_images.shape[1] != 3:
                logger.error(f"❌ 批次图像形状错误: {batch_images.shape}")
                raise ValueError(f"批次图像形状错误: {batch_images.shape}，期望 (B, 3, H, W)")
            
            with torch.no_grad():
                # 确保批次图像是float32类型
                batch_images = batch_images.to(dtype=torch.float32)
                
                # 提取当前批次的多尺度特征
                batch_features = self.iscs_calculator.extract_multiscale_features(batch_images, self.var)
                
                # 累积特征到CPU内存
                for scale_idx, features in batch_features.items():
                    features_cpu = features.cpu()
                    if scale_idx not in accumulated_features:
                        accumulated_features[scale_idx] = []
                    accumulated_features[scale_idx].append(features_cpu)
                
                # 清理GPU内存
                del batch_images, batch_features
                torch.cuda.empty_cache()
        
        # 合并所有批次的特征
        final_features = {}
        for scale_idx, feature_list in accumulated_features.items():
            final_features[scale_idx] = torch.cat(feature_list, dim=0)
            logger.info(f"🔍 真实图像尺度 {scale_idx} 特征形状: {final_features[scale_idx].shape}")
        
        return final_features
    
    def _save_results(self, results: Dict):
        """保存实验结果"""
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 保存JSON结果
        results_path = output_dir / 'iscs_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 结果已保存: {results_path}")
    
    def _visualize_results(self, results: Dict):
        """可视化ISCS结果"""
        output_dir = Path(self.args.output_dir)
        
        # 1. 尺度分数对比图
        plt.figure(figsize=(12, 6))
        
        scales = list(results['base_var']['scale_scores'].keys())
        base_scores = [results['base_var']['scale_scores'][s] for s in scales]
        
        plt.subplot(1, 2, 1)
        plt.plot(scales, base_scores, 'o-', label='基础VAR', linewidth=2, markersize=6)
        
        if 'aid_var' in results:
            aid_scores = [results['aid_var']['scale_scores'][s] for s in scales]
            plt.plot(scales, aid_scores, 's-', label='AID-VAR', linewidth=2, markersize=6)
        
        plt.xlabel('尺度索引')
        plt.ylabel('ISCS分数')
        plt.title('各尺度ISCS分数对比')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. 总分对比
        plt.subplot(1, 2, 2)
        models = ['基础VAR']
        total_scores = [results['base_var']['total_iscs']]
        
        if 'aid_var' in results:
            models.append('AID-VAR')
            total_scores.append(results['aid_var']['total_iscs'])
        
        bars = plt.bar(models, total_scores, color=['skyblue', 'lightcoral'][:len(models)])
        plt.ylabel('总ISCS分数')
        plt.title('总ISCS分数对比')
        
        # 添加数值标签
        for bar, score in zip(bars, total_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{score:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'iscs_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 可视化结果已保存: {output_dir}/iscs_comparison.png")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='ISCS (Inter-Scale Consistency Score) 计算脚本')
    
    # 模型参数
    parser.add_argument('--vae_ckpt', type=str, default='vae_ch160v4096z32.pth',
                       help='VAE模型检查点路径')
    parser.add_argument('--var_ckpt', type=str, default=None,
                       help='VAR模型检查点路径（如果未指定，自动根据model_depth确定）')
    parser.add_argument('--planner_ckpt', type=str, default=None,
                       help='GuidanceInjector检查点路径（可选）')
    parser.add_argument('--model_depth', type=int, default=16, choices=[16, 20, 24, 30],
                       help='VAR模型深度（支持16, 20, 24, 30）')
    
    # 数据参数
    parser.add_argument('--real_images_path', type=str, required=True,
                       help='真实图像数据路径（目录或npz文件）')
    parser.add_argument('--num_samples', type=int, default=5000,
                       help='评估样本数量')
    
    # 生成参数（与generate_aid_fid_samples.py保持一致）
    parser.add_argument('--cfg', type=float, default=1.5,
                       help='分类器无关引导强度')
    parser.add_argument('--top_k', type=int, default=900,
                       help='Top-k采样参数（与generate_aid_fid_samples.py一致）')
    parser.add_argument('--top_p', type=float, default=0.96,
                       help='Top-p采样参数（与generate_aid_fid_samples.py一致）')
    parser.add_argument('--more_smooth', action='store_true',
                       help='启用more_smooth提升视觉质量（与generate_aid_fid_samples.py一致）')
    parser.add_argument('--dtype', type=str, default='float16', choices=['float16', 'bfloat16'],
                       help='推理数据类型（与generate_aid_fid_samples.py一致）')
    
    # 特征提取参数
    parser.add_argument('--dinov2_model', type=str, default='dinov2_vits14',
                       choices=['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14'],
                       help='DINOv2模型版本')
    
    # 系统参数
    parser.add_argument('--device', type=str, default='cuda',
                       help='计算设备')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='批次大小')
    parser.add_argument('--output_dir', type=str, default='./iscs_results',
                       help='结果输出目录')
    
    # VAR架构参数
    parser.add_argument('--patch_nums', type=int, nargs='+', 
                       default=[1, 2, 3, 4, 5, 6, 8, 10, 13, 16],
                       help='VAR的patch数量序列')
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 初始化分布式（如果需要）
    if not dist.initialized():
        dist.initialize()
    
    logger.info("🚀 开始ISCS评估实验")
    logger.info(f"📋 实验配置:")
    logger.info(f"  🎯 模型配置:")
    logger.info(f"    • VAR模型深度: {args.model_depth}")
    logger.info(f"    • VAE检查点: {args.vae_ckpt}")
    logger.info(f"    • VAR检查点: {args.var_ckpt if args.var_ckpt else f'checkpoints/var_d{args.model_depth}.pth (自动确定)'}")
    logger.info(f"    • GuidanceInjector检查点: {args.planner_ckpt if args.planner_ckpt else '未使用'}")
    logger.info(f"  🎮 生成参数:")
    logger.info(f"    • CFG引导强度: {args.cfg}")
    logger.info(f"    • Top-k采样: {args.top_k}")
    logger.info(f"    • Top-p采样: {args.top_p}")
    logger.info(f"    • More smooth: {args.more_smooth}")
    logger.info(f"    • 数据类型: {args.dtype}")
    logger.info(f"  📊 数据配置:")
    logger.info(f"    • 真实图像路径: {args.real_images_path}")
    logger.info(f"    • 评估样本数量: {args.num_samples}")
    logger.info(f"    • 批次大小: {args.batch_size}")
    logger.info(f"    • 输出目录: {args.output_dir}")
    logger.info(f"  🖥️ 系统配置:")
    logger.info(f"    • 设备: {args.device}")
    logger.info(f"    • 特征提取器: {args.dinov2_model}")
    logger.info(f"    • VAR尺度序列: {args.patch_nums}")
    
    try:
        # 创建实验管理器
        experiment = ISCSExperiment(args)
        
        # 运行实验
        results = experiment.run_experiment()
        
        # 打印最终结果
        logger.info("="*60)
        logger.info("🎯 ISCS评估结果汇总:")
        
        if 'aid_var' in results:
            logger.info(f"🚀 AID-VAR ISCS（我们的方法）: {results['aid_var']['total_iscs']:.4f}")
        
        logger.info(f"📊 基础VAR ISCS（基线模型）: {results['base_var']['total_iscs']:.4f}")
        
        if 'aid_var' in results:
            improvement = results['base_var']['total_iscs'] - results['aid_var']['total_iscs']  # 距离减少为正改善
            logger.info(f"📈 ISCS改善: {improvement:.4f} ({improvement/results['base_var']['total_iscs']*100:.2f}%)")
            logger.info(f"🎯 结论: 我们的方法相比基线{'✅ 更优秀' if improvement > 0 else '❌ 表现不佳'}")
        
        logger.info("="*60)
        logger.info("✅ ISCS评估实验成功完成!")
        
    except Exception as e:
        logger.error(f"❌ 实验失败: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    main() 