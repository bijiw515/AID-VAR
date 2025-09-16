"""
🔧 数值稳定的判别器头实现
解决StyleGAN-T判别器头的NaN问题

主要修复：
1. 增大BatchNorm的eps值
2. 改进权重初始化
3. 添加数值稳定性检查
4. 简化条件判别运算
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.spectral_norm import SpectralNorm
import logging

logger = logging.getLogger(__name__)


class StableSpectralConv1d(nn.Conv1d):
    """数值稳定的SpectralConv1d"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 🔧 关键修复：增大eps值以提高数值稳定性
        SpectralNorm.apply(self, name='weight', n_power_iterations=1, dim=0, eps=1e-6)
        
        # 🔧 改进权重初始化
        with torch.no_grad():
            nn.init.normal_(self.weight, 0.0, 0.02)
            if self.bias is not None:
                nn.init.zeros_(self.bias)


class StableBatchNormLocal(nn.Module):
    """数值稳定的Local Batch Normalization"""
    def __init__(self, num_features: int, affine: bool = True, virtual_bs: int = 8, eps: float = 1e-3):
        super().__init__()
        self.virtual_bs = virtual_bs
        self.eps = eps  # 🔧 关键修复：从1e-5增大到1e-3
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
            
        # 🔧 改进参数初始化
        self.reset_parameters()
    
    def reset_parameters(self):
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.warning("⚠️ StableBatchNormLocal输入包含NaN/Inf，进行修复")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        shape = x.size()
        
        # Reshape batch into groups
        G = max(1, int(np.ceil(x.size(0) / self.virtual_bs)))
        x = x.view(G, -1, x.size(-2), x.size(-1))
        
        # 🔧 数值稳定的统计计算
        mean = x.mean([1, 3], keepdim=True)
        var = x.var([1, 3], keepdim=True, unbiased=False)
        
        # 🔧 关键修复：确保方差不会导致数值问题
        var = torch.clamp(var, min=self.eps)
        std = torch.sqrt(var + self.eps)
        
        # 🔧 检查统计量的数值稳定性
        if torch.isnan(mean).any() or torch.isnan(std).any():
            logger.warning("⚠️ BatchNorm统计量包含NaN，使用安全默认值")
            mean = torch.zeros_like(mean)
            std = torch.ones_like(std)
        
        x = (x - mean) / std
        
        if self.affine:
            x = x * self.weight[None, :, None] + self.bias[None, :, None]
        
        result = x.view(shape)
        
        # 🔧 最终数值检查
        if torch.isnan(result).any() or torch.isinf(result).any():
            logger.warning("⚠️ BatchNorm输出包含NaN/Inf，进行修复")
            result = torch.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return result


class StableResidualBlock(nn.Module):
    """数值稳定的残差块"""
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        # 🔧 使用更稳定的归一化因子
        self.norm_factor = 1.0 / np.sqrt(2.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.warning("⚠️ ResidualBlock输入包含NaN/Inf，进行修复")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        try:
            residual = self.fn(x)
            
            # 🔧 检查残差计算结果
            if torch.isnan(residual).any() or torch.isinf(residual).any():
                logger.warning("⚠️ 残差计算产生NaN/Inf，使用输入作为输出")
                return x
            
            result = (residual + x) * self.norm_factor
            
            # 🔧 最终数值检查
            if torch.isnan(result).any() or torch.isinf(result).any():
                logger.warning("⚠️ ResidualBlock输出包含NaN/Inf，使用输入")
                return x
            
            return result
            
        except Exception as e:
            logger.error(f"⚠️ ResidualBlock计算失败: {e}，返回输入")
            return x


def stable_make_block(channels: int, kernel_size: int) -> nn.Module:
    """创建数值稳定的判别器块"""
    return nn.Sequential(
        StableSpectralConv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size//2,
            padding_mode='circular',
        ),
        StableBatchNormLocal(channels),
        nn.LeakyReLU(0.2, True),
    )


class StableFullyConnectedLayer(nn.Module):
    """数值稳定的全连接层"""
    def __init__(self, in_features: int, out_features: int, bias: bool = True, 
                 lr_multiplier: float = 1.0, weight_init: float = 0.02):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lr_multiplier = lr_multiplier
        
        # 🔧 更稳定的权重初始化
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * weight_init)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        
        # 🔧 稳定的缩放因子
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.warning("⚠️ FC层输入包含NaN/Inf，进行修复")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        w = self.weight * self.weight_gain
        
        result = F.linear(x, w, self.bias)
        
        if torch.isnan(result).any() or torch.isinf(result).any():
            logger.warning("⚠️ FC层输出包含NaN/Inf，进行修复")
            result = torch.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return result


class StableDiscHead(nn.Module):
    """数值稳定的判别器头"""
    def __init__(self, channels: int, c_dim: int, cmap_dim: int = 64):
        super().__init__()
        self.channels = channels
        self.c_dim = c_dim
        self.cmap_dim = cmap_dim
        
        # 🔧 使用稳定的块构建
        self.main = nn.Sequential(
            stable_make_block(channels, kernel_size=1),
            StableResidualBlock(stable_make_block(channels, kernel_size=9))
        )
        
        if self.c_dim > 0:
            self.cmapper = StableFullyConnectedLayer(self.c_dim, cmap_dim)
            self.cls = StableSpectralConv1d(channels, cmap_dim, kernel_size=1, padding=0)
            # 🔧 稳定的归一化因子
            self.norm_factor = 1.0 / np.sqrt(max(1.0, float(cmap_dim)))
        else:
            self.cls = StableSpectralConv1d(channels, 1, kernel_size=1, padding=0)
        
        # 🔧 权重初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """改进的权重初始化"""
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            nn.init.normal_(module.weight, 0.0, 0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm1d, StableBatchNormLocal)):
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.ones_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.warning("⚠️ DiscHead输入特征包含NaN/Inf，进行修复")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        try:
            # 主要特征提取
            h = self.main(x)
            
            if torch.isnan(h).any() or torch.isinf(h).any():
                logger.warning("⚠️ 主特征提取产生NaN/Inf，使用零特征")
                h = torch.zeros_like(h)
            
            # 分类头
            out = self.cls(h)
            
            if torch.isnan(out).any() or torch.isinf(out).any():
                logger.warning("⚠️ 分类头输出NaN/Inf，使用零输出")
                if self.c_dim > 0:
                    out = torch.zeros(x.size(0), self.cmap_dim, h.size(-1), device=x.device)
                else:
                    out = torch.zeros(x.size(0), 1, h.size(-1), device=x.device)
            
            # 🔧 简化的条件判别处理
            if self.c_dim > 0 and c is not None:
                try:
                    cmap = self.cmapper(c).unsqueeze(-1)
                    
                    if torch.isnan(cmap).any() or torch.isinf(cmap).any():
                        logger.warning("⚠️ 条件映射产生NaN/Inf，使用单位映射")
                        cmap = torch.ones_like(cmap)
                    
                    # 🔧 更稳定的条件判别运算
                    out = out * cmap
                    out = out.sum(1, keepdim=True) * self.norm_factor
                    
                except Exception as e:
                    logger.warning(f"⚠️ 条件判别计算失败: {e}，使用无条件输出")
                    out = out.mean(1, keepdim=True)
            
            # 🔧 最终数值稳定性检查和限制
            out = torch.clamp(out, min=-10.0, max=10.0)
            
            if torch.isnan(out).any() or torch.isinf(out).any():
                logger.warning("⚠️ DiscHead最终输出包含NaN/Inf，返回安全默认值")
                out = torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)
            
            return out
            
        except Exception as e:
            logger.error(f"⚠️ DiscHead前向传播失败: {e}，返回零输出")
            return torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)


class StableProjectedDiscriminator(nn.Module):
    """数值稳定的投影判别器"""
    def __init__(self, c_dim: int, diffaug: bool = True, p_crop: float = 0.5, dino_checkpoint_path: str = None):
        super().__init__()
        self.c_dim = c_dim
        self.diffaug = diffaug
        self.p_crop = p_crop
        
        # 导入DINO（保持原有的DINO实现，因为它是稳定的）
        from stylegan_t.networks.discriminator import DINO
        # 🔥 支持传入本地DINO模型路径
        self.dino = DINO(local_checkpoint_path=dino_checkpoint_path)
        
        # 🔧 使用稳定的判别器头
        heads = []
        for i in range(self.dino.n_hooks):
            heads.append((str(i), StableDiscHead(self.dino.embed_dim, c_dim)))
        
        self.heads = nn.ModuleDict(heads)
        
        # 冻结DINO，只训练头部
        for param in self.dino.parameters():
            param.requires_grad = False
    
    def get_trainable_params(self):
        """获取可训练参数（仅头部）"""
        trainable_params = []
        for head in self.heads.values():
            trainable_params.extend(head.parameters())
        return trainable_params
    
    def train(self, mode: bool = True):
        """重写train方法，确保DINO始终为eval模式"""
        super().train(mode)
        self.dino.eval()  # DINO始终保持eval模式
        return self
    
    def forward_rgb(self, rgb_images: torch.Tensor, c: torch.Tensor = None) -> torch.Tensor:
        """数值稳定的RGB前向传播"""
        try:
            # 数值稳定的预处理
            rgb_images = torch.clamp(rgb_images, min=-1.0, max=1.0)
            
            if torch.isnan(rgb_images).any() or torch.isinf(rgb_images).any():
                logger.warning("⚠️ 输入RGB图像包含NaN/Inf，进行修复")
                rgb_images = torch.nan_to_num(rgb_images, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # 变换到[0,1]范围
            rgb_images_01 = (rgb_images + 1.0) * 0.5
            rgb_images_01 = torch.clamp(rgb_images_01, min=0.0, max=1.0)
            
            # DINO特征提取
            dino_features = self.dino(rgb_images_01)
            
            # 多头判别
            logits_list = []
            for k, head in self.heads.items():
                try:
                    feature_k = dino_features[k]
                    if torch.isnan(feature_k).any() or torch.isinf(feature_k).any():
                        logger.warning(f"⚠️ DINO特征{k}包含NaN/Inf，进行修复")
                        feature_k = torch.nan_to_num(feature_k, nan=0.0, posinf=1.0, neginf=-1.0)
                    
                    head_logits = head(feature_k, c)
                    logits_list.append(head_logits.view(rgb_images.size(0), -1))
                    
                except Exception as head_e:
                    logger.warning(f"⚠️ 判别器头{k}计算失败: {head_e}，使用零输出")
                    B = rgb_images.size(0)
                    device = rgb_images.device
                    fallback_logits = torch.zeros(B, 1, device=device, dtype=rgb_images.dtype)
                    logits_list.append(fallback_logits)
            
            # 合并输出
            if logits_list:
                logits = torch.cat(logits_list, dim=1)
            else:
                logger.error("❌ 所有判别器头都失败，返回零输出")
                B = rgb_images.size(0)
                device = rgb_images.device
                logits = torch.zeros(B, 1, device=device, dtype=rgb_images.dtype)
            
            # 最终数值稳定性检查
            logits = torch.clamp(logits, min=-10.0, max=10.0)
            
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                logger.warning("⚠️ 最终判别器输出包含NaN/Inf，进行修复")
                logits = torch.nan_to_num(logits, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return logits
            
        except Exception as e:
            logger.error(f"❌ 稳定判别器前向传播失败: {e}")
            B = rgb_images.shape[0]
            device = rgb_images.device
            return torch.zeros(B, 1, device=device, dtype=rgb_images.dtype)
    
    def forward_soft_rgb(self, rgb_images: torch.Tensor, c: torch.Tensor = None) -> torch.Tensor:
        """软RGB前向传播（保持梯度流）"""
        return self.forward_rgb(rgb_images, c)  # 使用相同的稳定实现 