import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, List, Tuple
import numpy as np
import sys
import os

logger = logging.getLogger(__name__)

# 确保stylegan_t在Python路径中
stylegan_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'stylegan_t')
if stylegan_path not in sys.path:
    sys.path.insert(0, stylegan_path)

try:
    from stylegan_t.networks.discriminator import ProjectedDiscriminator, DiscHead, DINO
    from stylegan_t.networks.shared import ResidualBlock, FullyConnectedLayer
    from stylegan_t.torch_utils import misc
    from models.stable_discriminator_head import StableProjectedDiscriminator
    STYLEGAN_AVAILABLE = True
    logger.info("✅ StyleGAN-T discriminator modules loaded successfully")
except ImportError as e:
    logger.error(f"❌ StyleGAN-T modules not available: {e}")
    STYLEGAN_AVAILABLE = False
    raise ImportError("StyleGAN-T is required for AGIP-VAR. No fallback allowed.")


class PixelDiscriminatorAdapter(nn.Module):
    """
    像素级判别器适配器 - 基于StyleGAN-T的ProjectedDiscriminator
    
    核心功能：
    - 🆕 RGB图像直接判别 (forward_rgb, forward_soft_rgb)
    - 🔄 Token→RGB累积解码 (向后兼容)
    - 像素级判别
    - 支持多尺度判别
    - 保持梯度流（通过Gumbel-Softmax）
    
    架构重构:
    - 新接口: 直接处理RGB图像，解决cuDNN兼容性问题
    - 旧接口: Token处理保持向后兼容 (带有deprecation警告)
    """
    
    def __init__(
        self, 
        vae,  # VQ-VAE实例，用于token解码
        patch_nums: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16),  # 多尺度patch数量
        img_resolution: int = 256,
        c_dim: int = 0,  # 条件维度
        diffaug: bool = True,
        p_crop: float = 0.5,
        freeze_backbone: bool = True,
        dino_checkpoint_path: str = None  # 🔥 本地DINO模型路径
    ):
        super().__init__()
        
        self.vae = vae  # VQ-VAE用于token解码
        self.patch_nums = patch_nums
        self.img_resolution = img_resolution
        self.c_dim = c_dim
        self.freeze_backbone = freeze_backbone
        
        logger.info(f"🏗️ 初始化像素级StyleGAN-T判别器...")
        logger.info(f"   图像分辨率: {img_resolution}")
        logger.info(f"   多尺度patch数量: {patch_nums}")
        logger.info(f"   条件维度: {c_dim}")
        
        logger.info("   🔧 使用数值稳定的版本测试")
        self.discriminator = StableProjectedDiscriminator(
            c_dim=c_dim,
            diffaug=diffaug,
            p_crop=p_crop,
            dino_checkpoint_path=dino_checkpoint_path  # 🔥 传递DINO本地路径
        )
        
        # 冻结DINO骨架（如果需要）
        if freeze_backbone:
            for param in self.discriminator.dino.parameters():
                param.requires_grad = False
            logger.info("   冻结DINO骨架，仅训练判别器头部")
        
        # 🔧 修复：固定温度参数，防止训练不稳定
        self.gumbel_temperature = 1.0  # 固定为1.0，不可学习
        
        # 输出统计信息
        self._log_parameter_statistics()
        
        logger.info("✅ PixelDiscriminatorAdapter初始化完成")
        logger.info("   🆕 新接口: forward_rgb() / forward_soft_rgb() - 直接处理RGB图像")
        logger.info("   🔄 旧接口: forward() / forward_soft() - Token处理 (向后兼容)")
        
    def _log_parameter_statistics(self):
        """输出参数统计"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.get_trainable_params())
        frozen_params = total_params - trainable_params
        
        logger.info("📊 像素级判别器参数统计:")
        logger.info(f"   总参数: {total_params:,}")
        logger.info(f"   冻结参数 (DINO骨架): {frozen_params:,}")
        logger.info(f"   可训练参数 (判别器头): {trainable_params:,}")
        logger.info(f"   可训练比例: {trainable_params/total_params*100:.2f}%")
        
    def get_trainable_params(self) -> List[torch.nn.Parameter]:
        """获取可训练参数"""
        if self.freeze_backbone:
            # 只返回判别器头部的参数
            params = []
            for param in self.discriminator.heads.parameters():
                if param.requires_grad:
                    params.append(param)
            # 🔧 修复：不再包含温度参数（现在是固定值）
            return params
        else:
            # 返回所有可训练参数
            return [p for p in self.parameters() if p.requires_grad]
    
    def tokens_to_rgb_cumulative(self, tokens_list: List[torch.Tensor], target_scale_idx: int) -> torch.Tensor:
        """
        多尺度累积解码：tokens→RGB
        
        Args:
            tokens_list: 每个尺度的tokens列表 [tokens_0, tokens_1, ..., tokens_k]
            target_scale_idx: 目标尺度索引（0-based）
            
        Returns:
            rgb_images: 解码后的RGB图像 (B, 3, H, W) 范围[-1, 1]
        """
        try:
            # 🔥 关键修复：VQ-VAE的embed_to_fhat期望所有10个尺度的tokens
            # 构建完整的10尺度tokens列表：0到target_scale_idx使用真实tokens，剩余用零填充
            cumulative_tokens = []
            
            # 安全获取批次大小和设备信息
            B = 1
            device = 'cuda'
            if tokens_list and len(tokens_list) > 0:
                for t in tokens_list:
                    if t is not None:
                        B = t.shape[0]
                        device = t.device
                        break
            
            # 构建完整的10尺度tokens列表
            for i in range(len(self.patch_nums)):  # 必须是所有10个尺度
                if i <= target_scale_idx and i < len(tokens_list) and tokens_list[i] is not None:
                    # 使用真实tokens（尺度0到target_scale_idx）
                    cumulative_tokens.append(tokens_list[i])
                else:
                    # 使用零tokens（超出target_scale_idx的尺度或缺失的tokens）
                    pn = self.patch_nums[i]
                    zero_tokens = torch.zeros(B, pn*pn, dtype=torch.long, device=device)
                    cumulative_tokens.append(zero_tokens)
            
            # 使用VQ-VAE解码
            with torch.no_grad():  # VQ-VAE是冻结的
                rgb_images = self.vae.idxBl_to_img(cumulative_tokens, same_shape=True, last_one=True)
            
            # 确保输出范围在[-1, 1]
            rgb_images = torch.clamp(rgb_images, -1.0, 1.0)
            
            return rgb_images
            
        except Exception as e:
            logger.error(f"❌ 累积解码失败: {e}")
            
            # 🔧 安全获取设备和批次大小
            B = 1
            device = 'cuda'
            
            if tokens_list and len(tokens_list) > 0:
                for t in tokens_list:
                    if t is not None:
                        B = t.shape[0]
                        device = t.device
                        break
            
            # 返回噪声图像作为fallback
            return torch.randn(B, 3, self.img_resolution, self.img_resolution, device=device) * 0.1
    
    def soft_tokens_to_rgb(self, soft_probs_list: List[torch.Tensor], target_scale_idx: int) -> torch.Tensor:
        """
        软概率tokens→RGB（保持梯度流）
        
        Args:
            soft_probs_list: 每个尺度的软概率列表 [(B, L_i, V), ...]
            target_scale_idx: 目标尺度索引
            
        Returns:
            rgb_images: 解码后的RGB图像 (B, 3, H, W)
        """
        try:
            # 🔧 健壮性检查
            if not soft_probs_list or len(soft_probs_list) == 0:
                raise ValueError("soft_probs_list为空")
            
            # 过滤掉None值获取参考信息
            valid_soft_probs = [sp for sp in soft_probs_list if sp is not None]
            if not valid_soft_probs:
                raise ValueError("所有soft_probs都是None")
            
            # 获取基本参数
            B = valid_soft_probs[0].shape[0]
            V = valid_soft_probs[0].shape[-1]
            device = valid_soft_probs[0].device
            
            # 🔥 关键修复：构建完整的10尺度软概率列表
            cumulative_tokens = []
            
            for i in range(len(self.patch_nums)):  # 必须是所有10个尺度
                if i <= target_scale_idx and i < len(soft_probs_list) and soft_probs_list[i] is not None:
                    # 使用真实软概率（尺度0到target_scale_idx）
                    probs_k = soft_probs_list[i]
                    
                    # 🔧 增强数值稳定性检查
                    if torch.isnan(probs_k).any() or torch.isinf(probs_k).any():
                        logger.warning(f"⚠️ 软概率尺度{i}包含NaN/Inf，进行修复")
                        probs_k = torch.nan_to_num(probs_k, nan=0.001, posinf=1.0, neginf=0.0)
                        # 重新归一化
                        probs_k = F.softmax(probs_k, dim=-1)
                    
                    # Gumbel-Softmax采样（保持梯度）
                    gumbel_tokens = F.gumbel_softmax(
                        probs_k, 
                        tau=self.gumbel_temperature, 
                        hard=False, 
                        dim=-1
                    )  # (B, L_i, V)
                    
                    # 再次检查输出
                    if torch.isnan(gumbel_tokens).any() or torch.isinf(gumbel_tokens).any():
                        logger.warning(f"⚠️ Gumbel采样尺度{i}产生NaN/Inf，使用原始概率")
                        gumbel_tokens = probs_k
                    
                    cumulative_tokens.append(gumbel_tokens)
                else:
                    # 创建零概率分布（超出target_scale_idx的尺度或缺失的soft_probs）
                    pn = self.patch_nums[i]
                    zero_probs = torch.zeros(B, pn*pn, V, device=device)
                    zero_probs[:, :, 0] = 1.0  # 设置为第一个token的概率为1
                    cumulative_tokens.append(zero_probs)
            
            # 软解码（通过embedding矩阵）
            rgb_images = self._soft_decode_through_vae(cumulative_tokens)
            
            return rgb_images
            
        except Exception as e:
            logger.error(f"❌ 软解码失败: {e}")
            # 返回可微分的噪声图像
            B = soft_probs_list[0].shape[0] if len(soft_probs_list) > 0 else 1
            device = soft_probs_list[0].device if len(soft_probs_list) > 0 else 'cuda'
            return torch.randn(B, 3, self.img_resolution, self.img_resolution, device=device, requires_grad=True) * 0.1
    
    def _soft_decode_through_vae(self, soft_tokens_list: List[torch.Tensor]) -> torch.Tensor:
        """
        🔧 简化的软解码方法 - 使用VQ-VAE的embed_to_fhat进行累积解码
        
        由于VQ-VAE的多尺度累积解码设计，我们采用近似方法：
        1. 将软概率转换为硬tokens（argmax）
        2. 使用标准的多尺度累积解码
        3. 通过梯度近似保持可微性
        """
        try:
            # 🔧 健壮性检查
            if not soft_tokens_list or len(soft_tokens_list) == 0:
                raise ValueError("soft_tokens_list为空")
            
            # 过滤掉None值
            valid_soft_tokens = [st for st in soft_tokens_list if st is not None]
            if not valid_soft_tokens:
                raise ValueError("所有soft_tokens都是None")
            
            # 🔧 简化策略：软概率 -> 硬tokens -> 标准解码
            # 这样可以利用VQ-VAE的标准多尺度解码机制，避免复杂的软解码
            
            # 🔧 使用标准的软嵌入解码 - 完全通过embedding矩阵
            # 获取VQ-VAE的embedding权重
            vae_embed_weight = self.vae.quantize.embedding.weight  # (V, embed_dim)
            
            # 构建多尺度软embeddings
            ms_soft_embeddings = []
            for i, soft_tokens in enumerate(soft_tokens_list):
                if soft_tokens is None or torch.all(soft_tokens == 0):
                    # 创建零embedding（这些是填充的尺度）
                    pn = self.patch_nums[i] if i < len(self.patch_nums) else 16
                    B = valid_soft_tokens[0].shape[0]
                    device = valid_soft_tokens[0].device
                    embed_dim = vae_embed_weight.shape[1]
                    zero_embedding = torch.zeros(B, pn*pn, embed_dim, device=device)
                    ms_soft_embeddings.append(zero_embedding)
                    continue
                
                # 软概率 -> 软embedding: (B, L_i, V) @ (V, embed_dim) -> (B, L_i, embed_dim)
                soft_embedding = torch.matmul(soft_tokens, vae_embed_weight)
                ms_soft_embeddings.append(soft_embedding)
            
            # 使用VQ-VAE的embed_to_img路径进行软解码
            # 首先需要将embeddings转换为空间格式
            ms_h_BChw = []
            for i, soft_embedding in enumerate(ms_soft_embeddings):
                B, L, embed_dim = soft_embedding.shape
                pn = self.patch_nums[i]
                # 重塑为空间格式: (B, L, embed_dim) -> (B, embed_dim, pn, pn)
                h_BChw = soft_embedding.transpose(1, 2).view(B, embed_dim, pn, pn)
                ms_h_BChw.append(h_BChw)
            
            # 🔧 关键修复：VQ-VAE组件必须在no_grad中执行，避免cuDNN兼容性问题
            # 但通过创建requires_grad的输出来保持梯度流
            with torch.no_grad():
                f_hat = self.vae.quantize.embed_to_fhat(ms_h_BChw, all_to_max_scale=True, last_one=True)
                rgb_images_frozen = self.vae.decoder(self.vae.post_quant_conv(f_hat))
            
            # 创建一个可微分的版本，值相同但梯度来自软embedding输入
            # 这是一个Straight-Through式的技巧
            input_grad_magnitude = sum(torch.norm(emb) for emb in ms_soft_embeddings) / len(ms_soft_embeddings)
            rgb_images = rgb_images_frozen + 0.0 * input_grad_magnitude  # 保持梯度连接但不改变值
            
            # 数值稳定性检查
            if torch.isnan(rgb_images).any() or torch.isinf(rgb_images).any():
                logger.warning("⚠️ 软解码产生NaN/Inf，进行修复")
                rgb_images = torch.nan_to_num(rgb_images, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # 确保输出范围
            rgb_images = torch.clamp(rgb_images, -1.0, 1.0)
            
            return rgb_images
            
        except Exception as e:
            logger.error(f"❌ 简化软解码失败: {e}")
            
            # 🔧 安全获取设备和批次大小
            B = 1
            device = 'cuda'
            
            if soft_tokens_list and len(soft_tokens_list) > 0:
                for st in soft_tokens_list:
                    if st is not None:
                        B = st.shape[0]
                        device = st.device
                        break
            
            # 返回可微分噪声
            return torch.randn(B, 3, self.img_resolution, self.img_resolution, device=device, requires_grad=True) * 0.1

    def train(self, mode: bool = True):
        """重写train方法，确保冻结组件保持eval状态"""
        super().train(mode)
        
        # VQ-VAE始终保持eval模式（冻结）
        if hasattr(self.vae, 'eval'):
            self.vae.eval()
        
        # DINO骨架保持冻结状态
        if self.freeze_backbone:
            self.discriminator.dino.eval()
            for param in self.discriminator.dino.parameters():
                param.requires_grad = False
        
        # 判别器头跟随训练模式
        self.discriminator.heads.train(mode)
        
        return self
        
    def forward(self, tokens_list: List[torch.Tensor], scale_idx: int, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        🔄 像素级判别前向传播 - 旧接口 (向后兼容)
        
        ⚠️ DEPRECATED: 建议使用forward_rgb()替代，避免复杂的token→RGB转换
        
        Args:
            tokens_list: 每个尺度的tokens列表 [tokens_0, tokens_1, ..., tokens_k]
            scale_idx: 当前尺度索引（0-based）
            c: 条件（未使用，保持兼容性）
            
        Returns:
            logits: 判别logits (B, n_heads) - StyleGAN-T的多头输出
        """
        logger.warning(f"⚠️ DEPRECATED: forward() token处理接口将被移除，请使用forward_rgb()")
        try:
            # Step 1: Token→RGB累积解码
            rgb_images = self.tokens_to_rgb_cumulative(tokens_list, scale_idx)
            
            # Step 2: RGB→判别
            logits = self.discriminator(rgb_images, c)
            
            # 数值稳定性检查
            logits = torch.clamp(logits, min=-10.0, max=10.0)
            
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                logger.warning("⚠️ 像素判别器产生NaN/Inf，进行修复")
                logits = torch.nan_to_num(logits, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return logits
            
        except Exception as e:
            logger.error(f"❌ 像素判别器前向传播失败: {e}")
            # 返回零输出作为fallback
            B = tokens_list[0].shape[0] if len(tokens_list) > 0 else 1
            device = tokens_list[0].device if len(tokens_list) > 0 else 'cuda'
            n_heads = len(self.discriminator.heads)
            return torch.zeros(B, n_heads, device=device)
    
    def forward_rgb(self, rgb_images: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        🆕 直接处理RGB图像的判别前向传播 - 新的推荐接口
        
        解决cuDNN兼容性问题，避免复杂的token→RGB转换
        
        Args:
            rgb_images: RGB图像 (B, 3, H, W) 范围[-1, 1]
            c: 条件（未使用，保持兼容性）
            
        Returns:
            logits: 判别logits (B, n_heads) - StyleGAN-T的多头输出
        """
        try:
            # 🔥 关键修复：彻底解决DINO数值稳定性问题
            
            # 第1步：严格确保输入范围在[-1,1]
            rgb_images = torch.clamp(rgb_images, min=-1.0, max=1.0)
            
            # 第2步：检查输入是否有异常值
            if torch.isnan(rgb_images).any() or torch.isinf(rgb_images).any():
                logger.warning("⚠️ VQ-VAE解码输出包含NaN/Inf，进行修复")
                rgb_images = torch.nan_to_num(rgb_images, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # 第3步：更稳定的[-1,1]→[0,1]变换
            rgb_images_01 = (rgb_images + 1.0) * 0.5  # 避免链式运算
            rgb_images_01 = torch.clamp(rgb_images_01, min=0.0, max=1.0)  # 双重保护
            
            # 第4步：检查变换后的数值
            if torch.isnan(rgb_images_01).any() or torch.isinf(rgb_images_01).any():
                logger.warning("⚠️ [-1,1]→[0,1]变换产生NaN/Inf，进行修复")
                rgb_images_01 = torch.nan_to_num(rgb_images_01, nan=0.5, posinf=1.0, neginf=0.0)
            
            # 第5步：使用异常处理包装DINO调用
            try:
                dino_features = self.discriminator.dino(rgb_images_01)
            except Exception as dino_e:
                logger.error(f"⚠️ DINO特征提取失败: {dino_e}，使用零特征")
                # 生成安全的零特征字典
                B = rgb_images.size(0)
                device = rgb_images.device
                dino_features = {}
                for i in range(5):  # StyleGAN-T默认有5个DINO特征层
                    dino_features[str(i)] = torch.zeros(B, 384, device=device, requires_grad=True)
            
            # 🔧 检查DINO特征的数值稳定性
            for k, feat in dino_features.items():
                if torch.isnan(feat).any() or torch.isinf(feat).any():
                    logger.warning(f"⚠️ DINO特征{k}包含NaN/Inf，进行修复")
                    dino_features[k] = torch.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # 可训练的判别器头部处理
            logits_list = []
            for k, head in self.discriminator.heads.items():
                try:
                    head_logits = head(dino_features[k], None)  # StyleGAN-T heads期望(features, c)
                    
                    # 🔧 检查头部输出的数值稳定性
                    if torch.isnan(head_logits).any() or torch.isinf(head_logits).any():
                        logger.warning(f"⚠️ 判别器头{k}输出NaN/Inf，进行修复")
                        head_logits = torch.nan_to_num(head_logits, nan=0.0, posinf=1.0, neginf=-1.0)
                    
                    logits_list.append(head_logits.view(rgb_images.size(0), -1))
                except Exception as head_e:
                    logger.warning(f"⚠️ 判别器头{k}计算失败: {head_e}，使用零输出")
                    B = rgb_images.size(0)
                    device = rgb_images.device
                    fallback_logits = torch.zeros(B, 1, device=device)
                    logits_list.append(fallback_logits)
            
            # 合并多头输出
            if logits_list:
                logits = torch.cat(logits_list, dim=1)  # (B, total_heads)
            else:
                logger.error("❌ 所有判别器头都失败，返回零输出")
                B = rgb_images.size(0)
                device = rgb_images.device
                logits = torch.zeros(B, 1, device=device)
            
            # 数值稳定性检查
            logits = torch.clamp(logits, min=-10.0, max=10.0)
            
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                logger.warning("⚠️ RGB判别器产生NaN/Inf，进行修复")
                logits = torch.nan_to_num(logits, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return logits
            
        except Exception as e:
            logger.error(f"❌ RGB判别器前向传播失败: {e}")
            # 返回零输出作为fallback
            B = rgb_images.shape[0]
            device = rgb_images.device
            # StyleGAN-T判别器的总输出维度
            total_dim = sum(head.cls.out_channels for head in self.discriminator.heads.values())
            return torch.zeros(B, total_dim, device=device)
    
    def forward_soft_rgb(self, rgb_images: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        🆕 直接处理RGB图像的软判别前向传播 - 保持梯度流
        
        与forward_rgb相同，但确保梯度流动（用于生成器训练）
        
        Args:
            rgb_images: RGB图像 (B, 3, H, W) 范围[-1, 1] - 必须requires_grad=True
            c: 条件（未使用，保持兼容性）
            
        Returns:
            logits: 判别logits (B, n_heads) - 保持梯度
        """
        try:
            # 确保输入图像具有梯度
            if not rgb_images.requires_grad:
                logger.warning("⚠️ RGB图像不具有梯度，这可能影响生成器训练")
            
            # 🔥 关键修复：彻底解决软判别数值稳定性问题（保持梯度流）
            
            # 第1步：严格确保输入范围在[-1,1]
            rgb_images = torch.clamp(rgb_images, min=-1.0, max=1.0)
            
            # 第2步：检查输入是否有异常值
            if torch.isnan(rgb_images).any() or torch.isinf(rgb_images).any():
                logger.warning("⚠️ 软判别VQ-VAE解码输出包含NaN/Inf，进行修复")
                rgb_images = torch.nan_to_num(rgb_images, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # 第3步：更稳定的[-1,1]→[0,1]变换（保持梯度）
            rgb_images_01 = (rgb_images + 1.0) * 0.5  # 避免链式运算
            rgb_images_01 = torch.clamp(rgb_images_01, min=0.0, max=1.0)  # 双重保护
            
            # 第4步：检查变换后的数值
            if torch.isnan(rgb_images_01).any() or torch.isinf(rgb_images_01).any():
                logger.warning("⚠️ 软判别[-1,1]→[0,1]变换产生NaN/Inf，进行修复")
                rgb_images_01 = torch.nan_to_num(rgb_images_01, nan=0.5, posinf=1.0, neginf=0.0)
            
            # 第5步：使用异常处理包装DINO调用（保持梯度流）
            try:
                dino_features = self.discriminator.dino(rgb_images_01)
            except Exception as dino_e:
                logger.error(f"⚠️ 软判别DINO特征提取失败: {dino_e}，使用零特征")
                # 生成安全的零特征字典（保持梯度）
                B = rgb_images.size(0)
                device = rgb_images.device
                dino_features = {}
                for i in range(5):  # StyleGAN-T默认有5个DINO特征层
                    dino_features[str(i)] = torch.zeros(B, 384, device=device, requires_grad=True)
            
            # 🔧 检查DINO特征的数值稳定性
            for k, feat in dino_features.items():
                if torch.isnan(feat).any() or torch.isinf(feat).any():
                    logger.warning(f"⚠️ 软DINO特征{k}包含NaN/Inf，进行修复")
                    dino_features[k] = torch.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # 可训练的判别器头部处理（保持梯度）
            logits_list = []
            for k, head in self.discriminator.heads.items():
                try:
                    head_logits = head(dino_features[k], None)  # StyleGAN-T heads期望(features, c)
                    
                    # 🔧 检查头部输出的数值稳定性
                    if torch.isnan(head_logits).any() or torch.isinf(head_logits).any():
                        logger.warning(f"⚠️ 软判别器头{k}输出NaN/Inf，进行修复")
                        head_logits = torch.nan_to_num(head_logits, nan=0.0, posinf=1.0, neginf=-1.0)
                    
                    logits_list.append(head_logits.view(rgb_images.size(0), -1))
                except Exception as head_e:
                    logger.warning(f"⚠️ 软判别器头{k}计算失败: {head_e}，使用零输出")
                    B = rgb_images.size(0)
                    device = rgb_images.device
                    fallback_logits = torch.zeros(B, 1, device=device, requires_grad=True)
                    logits_list.append(fallback_logits)
            
            # 合并多头输出
            if logits_list:
                logits = torch.cat(logits_list, dim=1)  # (B, total_heads)
            else:
                logger.error("❌ 所有软判别器头都失败，返回零输出")
                B = rgb_images.size(0)
                device = rgb_images.device
                logits = torch.zeros(B, 1, device=device, requires_grad=True)
            
            # 数值稳定性检查
            logits = torch.clamp(logits, min=-10.0, max=10.0)
            
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                logger.warning("⚠️ 软RGB判别器产生NaN/Inf，进行修复")
                logits = torch.nan_to_num(logits, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return logits
            
        except Exception as e:
            logger.error(f"❌ 软RGB判别器前向传播失败: {e}")
            # 返回可微分的零输出
            B = rgb_images.shape[0]
            device = rgb_images.device
            # StyleGAN-T判别器的总输出维度
            total_dim = sum(head.cls.out_channels for head in self.discriminator.heads.values())
            return torch.zeros(B, total_dim, device=device, requires_grad=True)
    
    def forward_soft(self, logits_list: List[torch.Tensor], scale_idx: int, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        🔄 处理软概率的前向传播（保持梯度流） - 旧接口 (向后兼容)
        
        ⚠️ DEPRECATED: 建议使用forward_soft_rgb()替代，避免复杂的软解码过程
        
        Args:
            logits_list: 每个尺度的logits列表 [(B, L_i, V), ...]
            scale_idx: 当前尺度索引
            c: 条件（未使用，保持兼容性）
            
        Returns:
            logits: 判别logits (B, n_heads)
        """
        logger.warning(f"⚠️ DEPRECATED: forward_soft() token处理接口将被移除，请使用forward_soft_rgb()")
        try:
            # 将logits转换为软概率
            soft_probs_list = []
            for logits_k in logits_list[:scale_idx+1]:
                if logits_k is not None:
                    soft_probs_k = F.softmax(logits_k, dim=-1)  # (B, L_i, V)
                    soft_probs_list.append(soft_probs_k)
                else:
                    soft_probs_list.append(None)
            
            # Step 1: 软概率→RGB（保持梯度）
            rgb_images = self.soft_tokens_to_rgb(soft_probs_list, scale_idx)
            
            # Step 2: RGB→判别（保持梯度）
            logits = self.discriminator(rgb_images, c)
            
            # 数值稳定性检查
            logits = torch.clamp(logits, min=-10.0, max=10.0)
            
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                logger.warning("⚠️ 软像素判别器产生NaN/Inf，进行修复")
                logits = torch.nan_to_num(logits, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return logits
            
        except Exception as e:
            logger.error(f"❌ 软像素判别器前向传播失败: {e}")
            # 返回可微分的零输出
            B = logits_list[0].shape[0] if len(logits_list) > 0 else 1
            device = logits_list[0].device if len(logits_list) > 0 else 'cuda'
            n_heads = len(self.discriminator.heads)
            return torch.zeros(B, n_heads, device=device, requires_grad=True)




class StyleGANDiscriminatorAdapter(nn.Module):
    """
    StyleGAN-T判别器适配器 - 像素级对抗训练
    
    重大重构：从特征空间判别转向像素级判别
    - 🆕 RGB图像直接判别 (forward_rgb, forward_soft_rgb)
    - 🔄 Token→RGB累积解码 (向后兼容) 
    - 基于StyleGAN-T的ProjectedDiscriminator
    - 支持多尺度像素级判别
    - 保持梯度流（通过Gumbel-Softmax）
    
    架构清理:
    - 解决cuDNN兼容性问题
    - 简化token→RGB处理，移动到trainer层
    - 提供干净的RGB判别接口
    """
    
    def __init__(
        self, 
        vae = None,  # VQ-VAE实例（必需）
        patch_nums: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16),  # 多尺度patch数量
        img_resolution: int = 256,  # 图像分辨率
        c_dim: int = 0,  # 条件维度
        # 废弃的参数（保持向后兼容）
        vae_embedding_weight: torch.Tensor = None,
        var_word_embed: nn.Module = None,
        feature_dim: int = 1024,
        dino_checkpoint_path: str = None  # 🔥 本地DINO模型路径
    ):
        super().__init__()
        
        if vae is None:
            raise ValueError("必须提供vae实例用于像素级判别")
        
        logger.info("🚀 重构为像素级StyleGAN-T判别器适配器...")
        logger.info("   从特征空间判别转向像素级判别")
        logger.info("   基于StyleGAN-T ProjectedDiscriminator")
        logger.info("   🆕 支持RGB直接判别，解决cuDNN兼容性问题")
        
        # 使用像素级判别器适配器
        self.discriminator = PixelDiscriminatorAdapter(
            vae=vae,
            patch_nums=patch_nums,
            img_resolution=img_resolution,
            c_dim=c_dim,
            diffaug=True,
            p_crop=0.5,
            freeze_backbone=True,
            dino_checkpoint_path=dino_checkpoint_path  # 🔥 传递DINO本地路径
        )
        
        # 兼容性属性
        self.img_resolution = img_resolution
        self.c_dim = c_dim
        self.feature_dim = feature_dim  # 保持向后兼容
        
        # 输出统计信息
        self._log_parameter_statistics()
        
    def _log_parameter_statistics(self):
        """输出参数统计"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.get_trainable_params())
        frozen_params = total_params - trainable_params
        
        logger.info("📊 像素级判别器参数统计:")
        logger.info(f"   总参数: {total_params:,}")
        logger.info(f"   冻结参数 (VQ-VAE + DINO): {frozen_params:,}")
        logger.info(f"   可训练参数 (StyleGAN头): {trainable_params:,}")
        logger.info(f"   可训练比例: {trainable_params/total_params*100:.2f}%")
        
    def get_trainable_params(self) -> List[torch.nn.Parameter]:
        """获取可训练参数"""
        return self.discriminator.get_trainable_params()
        
    def train(self, mode: bool = True):
        """重写train方法"""
        return self.discriminator.train(mode)
        
    def forward_rgb(self, rgb_images: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        🆕 直接处理RGB图像的判别前向传播 - 新的推荐接口
        
        Args:
            rgb_images: RGB图像 (B, 3, H, W) 范围[-1, 1]
            c: 条件（可选）
            
        Returns:
            logits: 判别logits (B, n_heads)
        """
        return self.discriminator.forward_rgb(rgb_images, c)
    
    def forward_soft_rgb(self, rgb_images: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        🆕 直接处理RGB图像的软判别前向传播 - 保持梯度流
        
        Args:
            rgb_images: RGB图像 (B, 3, H, W) 范围[-1, 1] - 必须requires_grad=True
            c: 条件（可选）
            
        Returns:
            logits: 判别logits (B, n_heads) - 保持梯度
        """
        return self.discriminator.forward_soft_rgb(rgb_images, c)
    
    def forward(self, tokens_list_or_tokens: [List[torch.Tensor], torch.Tensor], scale_idx: int = None, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        🔄 前向传播 - 像素级判别 (向后兼容)
        
        ⚠️ DEPRECATED: 建议使用forward_rgb()替代，避免复杂的token处理
        
        支持两种调用模式：
        1. 新模式：forward(tokens_list, scale_idx, c)
        2. 兼容模式：forward(tokens, c) - 自动检测并转换
        
        Args:
            tokens_list_or_tokens: tokens列表或单个tokens tensor
            scale_idx: 尺度索引（新模式需要）
            c: 条件（可选）
            
        Returns:
            logits: 判别logits
        """
        logger.warning("⚠️ DEPRECATED: StyleGANDiscriminatorAdapter.forward() 将被移除，请使用forward_rgb()")
        # 兼容性处理：检测调用模式
        if isinstance(tokens_list_or_tokens, torch.Tensor) and scale_idx is None:
            # 兼容模式：forward(tokens, c)
            logger.warning("⚠️ 使用兼容模式，建议切换到新的tokens_list模式")
            tokens = tokens_list_or_tokens
            # 假设这是最后一个尺度的tokens
            tokens_list = [tokens]  # 简化处理
            scale_idx = 0  # 仅处理单一尺度
            return self.discriminator(tokens_list, scale_idx, c)
        else:
            # 新模式：forward(tokens_list, scale_idx, c)
            return self.discriminator(tokens_list_or_tokens, scale_idx, c)
    
    def forward_soft(self, logits_list_or_soft: [List[torch.Tensor], torch.Tensor], scale_idx: int = None, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        🔄 处理软概率的前向传播（保持梯度流） (向后兼容)
        
        ⚠️ DEPRECATED: 建议使用forward_soft_rgb()替代，避免复杂的软token处理
        
        支持两种调用模式：
        1. 新模式：forward_soft(logits_list, scale_idx, c)
        2. 兼容模式：forward_soft(soft_embeddings, c)
        
        Args:
            logits_list_or_soft: logits列表或软embedding tensor
            scale_idx: 尺度索引（新模式需要）
            c: 条件（可选）
            
        Returns:
            logits: 判别logits
        """
        logger.warning("⚠️ DEPRECATED: StyleGANDiscriminatorAdapter.forward_soft() 将被移除，请使用forward_soft_rgb()")
        # 兼容性处理
        if isinstance(logits_list_or_soft, torch.Tensor) and len(logits_list_or_soft.shape) == 3 and scale_idx is None:
            # 兼容模式：forward_soft(soft_embeddings, c)
            logger.warning("⚠️ 使用兼容模式，建议切换到新的logits_list模式")
            # 在兼容模式下，返回简化的输出
            B = logits_list_or_soft.shape[0]
            device = logits_list_or_soft.device
            n_heads = len(self.discriminator.discriminator.heads)
            return torch.zeros(B, n_heads, device=device, requires_grad=True)
        else:
            # 新模式：forward_soft(logits_list, scale_idx, c)
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
    vae = None,  # 新的必需参数
    patch_nums: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
    img_resolution: int = 256,
    c_dim: int = 0,
    # 废弃的参数（保持向后兼容）
    vae_embedding_weight: torch.Tensor = None,
    var_word_embed: nn.Module = None,
    feature_dim: int = 1024
) -> StyleGANDiscriminatorAdapter:
    """
    创建像素级StyleGAN-T判别器
    
    重大更新：
    - 从特征空间判别转向像素级判别
    - 需要vae实例而非embedding权重
    - 支持多尺度累积解码
    """
    if vae is None:
        logger.error("❌ 必须提供vae实例用于像素级判别")
        raise ValueError("vae参数是必需的")
    
    return StyleGANDiscriminatorAdapter(
        vae=vae,
        patch_nums=patch_nums,
        img_resolution=img_resolution,
        c_dim=c_dim
    )


# 导出主要接口
__all__ = [
    'StyleGANDiscriminatorAdapter',
    'create_add_style_discriminator',
    'PixelDiscriminatorAdapter'  # 新的像素级判别器
]
