#!/usr/bin/env python3
"""
🎯 AGIP-VAR 训练辅助函数模块

包含AGIP-VAR训练过程中使用的各种辅助函数，
从trainer_planner.py中分离出来以保持代码组织的清晰性。

包含功能：
- 视觉调试和样本保存
- MSE损失计算
- 视觉调试摘要生成
- 其他训练辅助工具
"""

import os
import time
import json
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Tuple
from PIL import Image
import numpy as np

# 获取logger
import logging
logger = logging.getLogger(__name__)

# 类型定义
Ten = torch.Tensor
ITen = torch.LongTensor


def calculate_mse_loss(pred_tokens: ITen, gt_tokens: ITen) -> float:
    """计算MSE损失
    
    Args:
        pred_tokens: 预测tokens (B, L) 或 (L,)
        gt_tokens: 真实tokens (B, L) 或 (L,)
        
    Returns:
        MSE损失值
    """
    try:
        # 🔥 关键修复：确保输入和目标张量形状匹配
        pred_tokens = pred_tokens.float()
        gt_tokens = gt_tokens.float()
        
        # 记录原始形状
        original_pred_shape = pred_tokens.shape
        original_gt_shape = gt_tokens.shape
        
        # 确保两个张量都是2D的
        if pred_tokens.dim() == 1:
            pred_tokens = pred_tokens.unsqueeze(0)  # (L,) -> (1, L)
            logger.debug(f"   🔧 pred_tokens从1D扩展为2D: {original_pred_shape} -> {pred_tokens.shape}")
        if gt_tokens.dim() == 1:
            gt_tokens = gt_tokens.unsqueeze(0)  # (L,) -> (1, L)
            logger.debug(f"   🔧 gt_tokens从1D扩展为2D: {original_gt_shape} -> {gt_tokens.shape}")
        
        # 确保形状匹配
        if pred_tokens.shape != gt_tokens.shape:
            logger.debug(f"   🔧 形状不匹配，进行调整:")
            logger.debug(f"      pred_tokens: {pred_tokens.shape}")
            logger.debug(f"      gt_tokens: {gt_tokens.shape}")
            
            # 如果形状不匹配，尝试广播或调整
            if pred_tokens.shape[0] != gt_tokens.shape[0]:
                # 批次大小不匹配，取较小的
                min_batch = min(pred_tokens.shape[0], gt_tokens.shape[0])
                pred_tokens = pred_tokens[:min_batch]
                gt_tokens = gt_tokens[:min_batch]
                logger.debug(f"      🔧 调整批次大小: {min_batch}")
            
            if pred_tokens.shape[1] != gt_tokens.shape[1]:
                # 序列长度不匹配，取较小的
                min_length = min(pred_tokens.shape[1], gt_tokens.shape[1])
                pred_tokens = pred_tokens[:, :min_length]
                gt_tokens = gt_tokens[:, :min_length]
                logger.debug(f"      🔧 调整序列长度: {min_length}")
            
            logger.debug(f"   ✅ 调整后形状: pred={pred_tokens.shape}, gt={gt_tokens.shape}")
        
        mse_value = F.mse_loss(pred_tokens, gt_tokens).item()
        logger.debug(f"   ✅ MSE计算成功: {mse_value:.6f}")
        return mse_value
        
    except Exception as e:
        logger.warning(f"⚠️ MSE损失计算失败: {e}")
        logger.warning(f"   原始形状: pred={original_pred_shape if 'original_pred_shape' in locals() else 'N/A'}, gt={original_gt_shape if 'original_gt_shape' in locals() else 'N/A'}")
        logger.warning(f"   当前形状: pred={pred_tokens.shape if 'pred_tokens' in locals() else 'N/A'}, gt={gt_tokens.shape if 'gt_tokens' in locals() else 'N/A'}")
        return 0.0


def save_visual_debugging_samples(
    vae: nn.Module,
    patch_nums: Tuple[int],
    Cvae: int,
    device: torch.device,
    epoch: int,
    batch_idx: int,
    scale_idx: int,
    real_tokens: torch.Tensor,  # 当前尺度的真实tokens
    fake_tokens: torch.Tensor,  # 当前尺度的假tokens
    std_tokens: List[torch.Tensor],  # 标准VAR预测的多尺度tokens
    save_dir: str,
    real_tokens_history: List[torch.Tensor],  # 真实tokens的多尺度历史（必需）
    fake_tokens_history: List[torch.Tensor],  # 假tokens的多尺度历史（必需）
    max_samples: int = 4,
    guidance_tokens_list: Optional[List[torch.Tensor]] = None  # 🆕 I_predictor生成的引导tokens (注意力特征向量)
):
    """保存视觉调试样本
    
    Args:
        vae: VQ-VAE模型，用于token到图像的转换
        patch_nums: 多尺度patch数量
        Cvae: VAE通道数
        device: 设备
        epoch: 当前epoch
        batch_idx: 当前batch索引
        scale_idx: 当前尺度索引
        real_tokens: 当前尺度的真实tokens (B, pn^2)
        fake_tokens: 当前尺度的假tokens (B, pn^2)
        std_tokens: 标准VAR预测的多尺度tokens列表
        save_dir: 保存目录
        real_tokens_history: 真实tokens的多尺度历史列表（必需，用于构建真实样本的完整序列）
        fake_tokens_history: 假tokens的多尺度历史列表（必需，用于构建假样本的完整序列）
        max_samples: 最大保存样本数
        guidance_tokens_list: 🆕 I_predictor生成的多尺度引导tokens列表 (注意力特征向量形式)
        
    Note:
        - 真实样本：使用real_tokens_history作为历史 + 当前real_tokens
        - 假样本：使用fake_tokens_history作为历史 + 当前fake_tokens  
        - 标准VAR样本：使用std_tokens作为历史 + 当前std_tokens
        - 🆕 引导可视化：将guidance_tokens_list可视化为注意力热力图叠加在std图像上
    """
    logger.info(f"🔬 保存视觉调试样本: Epoch {epoch+1}, Batch {batch_idx+1}, 尺度 {scale_idx+1}({patch_nums[scale_idx]}x{patch_nums[scale_idx]})")
    
    # 🔍 详细的调试信息
    logger.info(f"   🔍 详细调试信息:")
    logger.info(f"      - patch_nums: {patch_nums}")
    logger.info(f"      - scale_idx: {scale_idx}")
    logger.info(f"      - real_tokens: {real_tokens.shape if real_tokens is not None else 'None'}")
    logger.info(f"      - fake_tokens: {fake_tokens.shape if fake_tokens is not None else 'None'}")
    logger.info(f"      - std_tokens type: {type(std_tokens)}")
    logger.info(f"      - std_tokens length: {len(std_tokens) if std_tokens is not None else 'None'}")
    
    if std_tokens is not None:
        for i, token in enumerate(std_tokens):
            if token is not None:
                logger.info(f"      - std_tokens[{i}]: {token.shape}")
            else:
                logger.info(f"      - std_tokens[{i}]: None")
        
        # 创建保存目录
    debug_dir = os.path.join(save_dir, "visual_debug", f"epoch_{epoch+1}", f"batch_{batch_idx+1}", f"scale_{scale_idx+1}")
    os.makedirs(debug_dir, exist_ok=True)
    
    B = min(real_tokens.shape[0], max_samples)
    save_stats = {"real": 0, "fake": 0, "std": 0, "guidance": 0}  # 🆕 添加guidance统计
    
    # 🎯 关键修复：检查数据完整性
    logger.info(f"   📊 数据检查: real_tokens={real_tokens.shape}, fake_tokens={fake_tokens.shape}")
    logger.info(f"   📊 std_tokens长度: {len(std_tokens)}, 需要访问尺度0到{scale_idx}")
    
    # 检查std_tokens是否有足够的数据
    if len(std_tokens) <= scale_idx:
        logger.warning(f"   ⚠️ std_tokens长度不足: {len(std_tokens)} <= {scale_idx}")
        return
    
    # 检查每个需要的std_tokens是否存在且有足够的样本
    for si in range(scale_idx + 1):
        if si >= len(std_tokens) or std_tokens[si] is None:
            logger.warning(f"   ⚠️ std_tokens[{si}] 不存在或为None")
            return
        if std_tokens[si].shape[0] < B:
            logger.warning(f"   ⚠️ std_tokens[{si}] 样本数不足: {std_tokens[si].shape[0]} < {B}")
            B = std_tokens[si].shape[0]  # 调整到最小可用样本数
    
    logger.info(f"   ✅ 数据检查通过，将处理{B}个样本")
    
    # 🎯 关键修复：使用正确的VQ-VAE多尺度重建流程
    for i in range(B):
        try:
            logger.info(f"   🔄 处理样本 {i+1}/{B}")
            
            # 保存真实样本的重建图像
            try:
                # 🔥 关键修复：构建完整的10个尺度tokens列表
                # 使用真实tokens的历史，当前尺度使用真实tokens，未来尺度用零填充
                ms_real_tokens = []
                for si in range(len(patch_nums)):  # 完整的10个尺度
                    if si < scale_idx:
                        # 之前的尺度：强制使用真实tokens的历史
                        if real_tokens_history is None or si >= len(real_tokens_history):
                            raise ValueError(f"真实tokens历史数据不足: 需要尺度{si}，但只有{len(real_tokens_history) if real_tokens_history else 0}个尺度")
                        token = real_tokens_history[si][i:i+1]  # (1, L_si)
                        # 🔧 确保是2D张量
                        if token.dim() == 1:
                            token = token.unsqueeze(0)  # (1, L_si)
                        ms_real_tokens.append(token)
                    elif si == scale_idx:
                        # 当前尺度：使用真实tokens
                        token = real_tokens[i:i+1]  # (1, L_current)
                        # 🔧 确保是2D张量
                        if token.dim() == 1:
                            token = token.unsqueeze(0)  # (1, L_current)
                        ms_real_tokens.append(token)
                    else:
                        # 未来尺度：用零填充
                        pn = patch_nums[si]
                        L_future = pn * pn
                        zero_tokens = torch.zeros(1, L_future, dtype=real_tokens.dtype, device=real_tokens.device)
                        ms_real_tokens.append(zero_tokens)
                
                logger.info(f"       📊 构建真实样本多尺度序列: {[t.shape for t in ms_real_tokens]}")
                
                # 🔥 关键修复：使用VQ-VAE的正确重建方法
                real_img = vae.idxBl_to_img(ms_real_tokens, same_shape=True, last_one=True)
                
                # 转换为PIL图像并保存
                if real_img is not None and real_img.numel() > 0:
                    # 确保图像在[0,1]范围内 (从[-1,1]转换)
                    real_img_np = real_img[0].detach().cpu().add(1).mul(0.5).clamp(0, 1).permute(1, 2, 0).numpy()
                    real_img_pil = Image.fromarray((real_img_np * 255).astype(np.uint8))
                    real_img_pil.save(os.path.join(debug_dir, f"real_sample_{i+1}_scale_{scale_idx+1}.png"))
                    save_stats["real"] += 1
                    logger.info(f"       ✅ 保存 real 样本 {i+1} 成功")
                    
            except Exception as e:
                logger.warning(f"       ⚠️ 保存 real 样本 {i+1} 失败: {e}")
                import traceback
                logger.warning(f"       详细错误: {traceback.format_exc()}")
            
            # 保存假样本的重建图像
            try:
                # 🔥 关键修复：构建完整的10个尺度tokens列表
                # 使用假tokens的历史，当前尺度使用假tokens，未来尺度用零填充
                ms_fake_tokens = []
                for si in range(len(patch_nums)):  # 完整的10个尺度
                    if si < scale_idx:
                        # 之前的尺度：强制使用假tokens的历史
                        if fake_tokens_history is None or si >= len(fake_tokens_history):
                            raise ValueError(f"假tokens历史数据不足: 需要尺度{si}，但只有{len(fake_tokens_history) if fake_tokens_history else 0}个尺度")
                        token = fake_tokens_history[si][i:i+1]  # (1, L_si)
                        # 🔧 确保是2D张量
                        if token.dim() == 1:
                            token = token.unsqueeze(0)  # (1, L_si)
                        ms_fake_tokens.append(token)
                    elif si == scale_idx:
                        # 当前尺度：使用生成的假tokens
                        token = fake_tokens[i:i+1]  # (1, L_current)
                        # 🔧 确保是2D张量
                        if token.dim() == 1:
                            token = token.unsqueeze(0)  # (1, L_current)
                        ms_fake_tokens.append(token)
                    else:
                        # 未来尺度：用零填充
                        pn = patch_nums[si]
                        L_future = pn * pn
                        zero_tokens = torch.zeros(1, L_future, dtype=fake_tokens.dtype, device=fake_tokens.device)
                        ms_fake_tokens.append(zero_tokens)
                
                logger.info(f"       📊 构建假样本多尺度序列: {[t.shape for t in ms_fake_tokens]}")
                
                # 🔥 关键修复：使用VQ-VAE的正确重建方法
                fake_img = vae.idxBl_to_img(ms_fake_tokens, same_shape=True, last_one=True)
                
                # 转换为PIL图像并保存
                if fake_img is not None and fake_img.numel() > 0:
                    # 确保图像在[0,1]范围内 (从[-1,1]转换)
                    fake_img_np = fake_img[0].detach().cpu().add(1).mul(0.5).clamp(0, 1).permute(1, 2, 0).numpy()
                    fake_img_pil = Image.fromarray((fake_img_np * 255).astype(np.uint8))
                    fake_img_pil.save(os.path.join(debug_dir, f"fake_sample_{i+1}_scale_{scale_idx+1}.png"))
                    save_stats["fake"] += 1
                    logger.info(f"       ✅ 保存 fake 样本 {i+1} 成功")
                    
            except Exception as e:
                logger.warning(f"       ⚠️ 保存 fake 样本 {i+1} 失败: {e}")
                import traceback
                logger.warning(f"       详细错误: {traceback.format_exc()}")
                
            # 保存标准VAR样本的重建图像
            try:
                # 🔥 关键修复：构建完整的10个尺度tokens列表
                ms_std_tokens = []
                for si in range(len(patch_nums)):  # 完整的10个尺度
                    if si <= scale_idx:
                        # 已有的尺度：使用标准VAR预测
                        token = std_tokens[si][i:i+1]  # (1, L_si)
                        # 🔧 确保是2D张量
                        if token.dim() == 1:
                            token = token.unsqueeze(0)  # (1, L_si)
                        ms_std_tokens.append(token)
                    else:
                        # 未来尺度：用零填充
                        pn = patch_nums[si]
                        L_future = pn * pn
                        zero_tokens = torch.zeros(1, L_future, dtype=std_tokens[0].dtype, device=std_tokens[0].device)
                        ms_std_tokens.append(zero_tokens)
                
                logger.info(f"       📊 构建标准样本多尺度序列: {[t.shape for t in ms_std_tokens]}")
                
                # 🔥 关键修复：使用VQ-VAE的正确重建方法
                std_img = vae.idxBl_to_img(ms_std_tokens, same_shape=True, last_one=True)
                
                # 转换为PIL图像并保存
                if std_img is not None and std_img.numel() > 0:
                    # 确保图像在[0,1]范围内 (从[-1,1]转换)
                    std_img_np = std_img[0].detach().cpu().add(1).mul(0.5).clamp(0, 1).permute(1, 2, 0).numpy()
                    std_img_pil = Image.fromarray((std_img_np * 255).astype(np.uint8))
                    std_img_pil.save(os.path.join(debug_dir, f"std_sample_{i+1}_scale_{scale_idx+1}.png"))
                    save_stats["std"] += 1
                    logger.info(f"       ✅ 保存 std 样本 {i+1} 成功")
                    
            except Exception as e:
                logger.warning(f"       ⚠️ 保存 std 样本 {i+1} 失败: {e}")
                import traceback
                logger.warning(f"       详细错误: {traceback.format_exc()}")
            
            # 🆕 保存引导注意力可视化
            if guidance_tokens_list is not None:
                try:
                    logger.info(f"       🎯 处理引导tokens样本 {i+1}: 生成注意力热力图叠加可视化")
                    
                    # 🔥 重新设计：将guidance tokens可视化为注意力热力图
                    # 设计思路：
                    # 1. 背景图：所有尺度都使用同一张完整的fake tokens重构图片（无零填充）
                    # 2. 热力图：每个尺度使用对应尺度的guidance tokens生成注意力热力图
                    
                    # 🔥 修改：使用完整的fake tokens图像作为背景（没有零填充）
                    # 构建完整的fake tokens多尺度序列，使用所有可用的fake tokens
                    ms_complete_fake_tokens = []
                    for si in range(len(patch_nums)):
                        if si < scale_idx:
                            # 之前的尺度：使用fake_tokens_history
                            if fake_tokens_history is None or si >= len(fake_tokens_history):
                                # 如果没有历史，fallback到std_tokens
                                token = std_tokens[si][i:i+1]
                            else:
                                token = fake_tokens_history[si][i:i+1]
                        elif si == scale_idx:
                            # 当前尺度：使用当前fake_tokens
                            token = fake_tokens[i:i+1]
                        else:
                            # 未来尺度：使用fake_tokens（避免零填充）
                            token = fake_tokens[si][i:i+1]
                        
                        if token.dim() == 1:
                            token = token.unsqueeze(0)
                        ms_complete_fake_tokens.append(token)
                    
                    # 生成完整的fake背景图像（无零填充的完整图像）
                    fake_base_img = vae.idxBl_to_img(ms_complete_fake_tokens, same_shape=True, last_one=True)
                    
                    if fake_base_img is not None and scale_idx < len(guidance_tokens_list) and guidance_tokens_list[scale_idx] is not None:
                        # 获取当前尺度的guidance特征
                        guidance_features = guidance_tokens_list[scale_idx][i:i+1]  # (1, patch_size^2, C)
                        
                        logger.info(f"         🔧 guidance特征形状: {guidance_features.shape}")
                        
                        with torch.no_grad():
                            # 计算注意力权重：使用特征的L2范数或均值作为注意力强度
                            B, L_patch, C = guidance_features.shape
                            patch_size = int(L_patch ** 0.5)  # patch_nums[scale_idx]
                            
                            # 方法1：使用L2范数作为注意力强度
                            attention_weights = torch.norm(guidance_features, dim=-1)  # (1, L_patch)
                            
                            # 归一化到[0,1]
                            attention_weights = attention_weights.squeeze(0)  # (L_patch,)
                            attention_min = attention_weights.min()
                            attention_max = attention_weights.max()
                            if attention_max > attention_min:
                                attention_weights = (attention_weights - attention_min) / (attention_max - attention_min)
                            else:
                                attention_weights = torch.zeros_like(attention_weights)
                            
                            # 重塑为2D热力图
                            attention_heatmap = attention_weights.view(patch_size, patch_size)  # (patch_size, patch_size)
                            
                            logger.info(f"         🔧 注意力热力图形状: {attention_heatmap.shape}, 值范围: [{attention_weights.min():.4f}, {attention_weights.max():.4f}]")
                            
                            # 将背景图像转换为numpy
                            fake_img_np = fake_base_img[0].detach().cpu().add(1).mul(0.5).clamp(0, 1).permute(1, 2, 0).numpy()
                            
                            # 将注意力热力图上采样到图像尺寸
                            import torch.nn.functional as F
                            import matplotlib.pyplot as plt
                            import matplotlib.cm as cm
                            
                            # 上采样热力图到图像尺寸
                            img_size = fake_img_np.shape[0]  # 假设是正方形图像
                            attention_heatmap_upsampled = F.interpolate(
                                attention_heatmap.unsqueeze(0).unsqueeze(0).float(),  # (1, 1, patch_size, patch_size)
                                size=(img_size, img_size),
                                mode='bilinear',
                                align_corners=False
                            ).squeeze(0).squeeze(0)  # (img_size, img_size)
                            
                            # 转换为numpy
                            attention_np = attention_heatmap_upsampled.cpu().numpy()
                            
                            # 使用matplotlib的热力图colormap
                            colormap = cm.get_cmap('jet')  # 使用jet colormap：蓝色(低) -> 红色(高)
                            attention_colored = colormap(attention_np)[:, :, :3]  # RGB，去掉alpha通道
                            
                            # 创建叠加图像：背景图 + 透明度调制的热力图
                            # 高注意力区域：不透明，低注意力区域：透明
                            alpha = attention_np * 0.6  # 最大透明度60%
                            
                            # 混合图像
                            overlay_img = fake_img_np * (1 - alpha[:, :, np.newaxis]) + attention_colored * alpha[:, :, np.newaxis]
                            overlay_img = np.clip(overlay_img, 0, 1)
                            
                            # 转换为PIL图像并保存
                            overlay_img_pil = Image.fromarray((overlay_img * 255).astype(np.uint8))
                            overlay_img_pil.save(os.path.join(debug_dir, f"guidance_overlay_sample_{i+1}_scale_{scale_idx+1}.png"))
                            
                            # 同时保存纯热力图用于调试
                            heatmap_img_pil = Image.fromarray((attention_colored * 255).astype(np.uint8))
                            heatmap_img_pil.save(os.path.join(debug_dir, f"guidance_heatmap_sample_{i+1}_scale_{scale_idx+1}.png"))
                            
                            save_stats["guidance"] += 1
                            logger.info(f"       ✅ 保存 guidance 注意力可视化样本 {i+1} 成功")
                            logger.info(f"         - 叠加图: guidance_overlay_sample_{i+1}_scale_{scale_idx+1}.png (尺度{scale_idx+1}热力图 + 完整fake背景)")
                            logger.info(f"         - 热力图: guidance_heatmap_sample_{i+1}_scale_{scale_idx+1}.png (纯尺度{scale_idx+1}热力图)")
                    else:
                        logger.info(f"       ⚠️ 样本 {i+1} 无guidance数据或fake背景图生成失败")
                        
                except Exception as e:
                    logger.warning(f"       ⚠️ 保存 guidance 注意力可视化样本 {i+1} 失败: {e}")
                    import traceback
                    logger.warning(f"       详细错误: {traceback.format_exc()}")
                
        except Exception as e:
            logger.warning(f"       ⚠️ 处理样本 {i+1} 时发生错误: {e}")
            import traceback
            logger.warning(f"       详细错误: {traceback.format_exc()}")
    
    logger.info(f"🔬 视觉调试保存完成: Real={save_stats['real']}, Fake={save_stats['fake']}, Std={save_stats['std']}, Guidance={save_stats['guidance']}")
    
    # 保存元数据
    metadata = {
        "epoch": epoch + 1,
        "batch_idx": batch_idx + 1,
        "scale_idx": scale_idx + 1,
        "patch_size": f"{patch_nums[scale_idx]}x{patch_nums[scale_idx]}",
        "save_stats": save_stats,
        "save_time": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(os.path.join(debug_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)


def create_multi_scale_composite_images(
    save_dir: str,
    epoch: int,
    batch_idx: int,
    max_samples: int = 2
):
    """
    从已保存的单独scale图像创建综合比较图
    布局：10列（每个scale一列），每列3行（fake, real, std）
    
    Args:
        save_dir: 保存目录基础路径
        epoch: epoch编号
        batch_idx: batch编号  
        max_samples: 最大样本数量
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
        import os
        
        logger.info(f"🎨 创建多尺度综合比较图: Epoch {epoch+1}, Batch {batch_idx+1}")
        
        # 基础目录路径
        base_debug_dir = os.path.join(save_dir, "visual_debug", f"epoch_{epoch+1}", f"batch_{batch_idx+1}")
        
        if not os.path.exists(base_debug_dir):
            logger.warning(f"调试目录不存在: {base_debug_dir}")
            return
        
        # 获取所有scale目录
        scale_dirs = []
        for scale_idx in range(1, 11):  # scale_1 到 scale_10
            scale_dir = os.path.join(base_debug_dir, f"scale_{scale_idx}")
            if os.path.exists(scale_dir):
                scale_dirs.append((scale_idx, scale_dir))
        
        if not scale_dirs:
            logger.warning(f"未找到任何scale目录在: {base_debug_dir}")
            return
        
        logger.info(f"找到 {len(scale_dirs)} 个scale目录")
        
        # 为每个样本创建综合比较图
        for sample_idx in range(1, max_samples + 1):
            logger.info(f"处理样本 {sample_idx}/{max_samples}")
            
            # 收集所有scale的图像
            scale_images = {}  # {scale_idx: {'fake': img, 'real': img, 'std': img}}
            
            for scale_idx, scale_dir in scale_dirs:
                images = {}
                
                # 尝试加载各类型图像
                for img_type in ['fake', 'real', 'std', 'guidance']:  # 🆕 添加guidance类型
                    if img_type == 'guidance':
                        # guidance类型使用叠加图像
                        img_path = os.path.join(scale_dir, f"guidance_overlay_sample_{sample_idx}_scale_{scale_idx}.png")
                    else:
                        img_path = os.path.join(scale_dir, f"{img_type}_sample_{sample_idx}_scale_{scale_idx}.png")
                    
                    if os.path.exists(img_path):
                        try:
                            img = Image.open(img_path)
                            # 调整图像大小以保持一致性
                            img = img.resize((96, 96), Image.LANCZOS)
                            images[img_type] = img
                        except Exception as e:
                            logger.warning(f"加载图像失败 {img_path}: {e}")
                            # 创建占位图像
                            images[img_type] = Image.new('RGB', (96, 96), color='gray')
                    else:
                        # 创建占位图像
                        images[img_type] = Image.new('RGB', (96, 96), color='lightgray')
                
                scale_images[scale_idx] = images
            
            if not scale_images:
                logger.warning(f"样本 {sample_idx} 未找到任何图像")
                continue
            
            # 创建综合比较图
            try:
                # 计算画布尺寸
                img_size = 96
                header_height = 50
                label_height = 20
                num_scales = len(scale_images)
                
                # 布局：10列（scales），4行（fake, real, std, guidance）🆕
                canvas_width = num_scales * img_size + 50  # 额外空间用于标签
                canvas_height = header_height + label_height + 4 * img_size + 30  # 4行图像 + 标签空间🆕
                
                # 创建画布
                composite_img = Image.new('RGB', (canvas_width, canvas_height), color='white')
                draw = ImageDraw.Draw(composite_img)
                
                # 尝试加载字体
                try:
                    font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 10)
                    title_font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 14)
                except:
                    font = ImageFont.load_default()
                    title_font = ImageFont.load_default()
                
                # 绘制标题
                title = f"Sample {sample_idx} - Multi-Scale Comparison (Epoch {epoch+1}, Batch {batch_idx+1})"
                draw.text((10, 10), title, fill='black', font=title_font)
                
                # 绘制行标签
                row_labels = ['Fake', 'Real', 'Std', 'Guidance']  # 🆕 添加Guidance行
                for row_idx, label in enumerate(row_labels):
                    y_pos = header_height + label_height + row_idx * img_size + img_size // 2
                    draw.text((5, y_pos), label, fill='black', font=font)
                
                # 绘制列标签和图像
                sorted_scales = sorted(scale_images.keys())
                for col_idx, scale_idx in enumerate(sorted_scales):
                    # 绘制scale标签
                    x_center = 25 + col_idx * img_size + img_size // 2
                    draw.text((x_center - 15, header_height), f"Scale {scale_idx}", fill='black', font=font)
                    
                    # 绘制该scale的四张图像🆕
                    images = scale_images[scale_idx]
                    for row_idx, img_type in enumerate(['fake', 'real', 'std', 'guidance']):  # 🆕 添加guidance
                        if img_type in images:
                            x_pos = 25 + col_idx * img_size
                            y_pos = header_height + label_height + row_idx * img_size
                            composite_img.paste(images[img_type], (x_pos, y_pos))
                
                # 保存综合比较图
                composite_dir = os.path.join(base_debug_dir, "composite")
                os.makedirs(composite_dir, exist_ok=True)
                composite_path = os.path.join(composite_dir, f"sample_{sample_idx}_multi_scale_comparison.jpg")
                composite_img.save(composite_path, quality=90)
                
                logger.info(f"综合比较图已保存: {composite_path}")
                
            except Exception as e:
                logger.error(f"创建样本 {sample_idx} 综合比较图失败: {e}")
                import traceback
                logger.error(f"详细错误: {traceback.format_exc()}")
        
        logger.info(f"🎨 多尺度综合比较图创建完成")
        
    except Exception as e:
        logger.error(f"创建多尺度综合比较图时发生错误: {e}")
        import traceback
        logger.error(f"详细错误信息: {traceback.format_exc()}")


def generate_visual_debug_summary(save_dir: str, epoch: int):
    """🔬 生成视觉调试摘要报告
    
    分析保存的视觉调试样本，生成摘要报告帮助理解判别器行为
    """
    try:
        debug_dir = os.path.join(save_dir, "validation", f"epoch_{epoch+1}", "visual_debug")
        
        if not os.path.exists(debug_dir):
            return
        
        # 收集调试信息文件
        debug_info_files = glob.glob(os.path.join(debug_dir, "debug_info_*.json"))
        
        if not debug_info_files:
            return
        
        logger.info(f"🔬 生成Epoch {epoch+1}视觉调试摘要...")
        
        summary = {
            "epoch": epoch + 1,
            "total_debug_sessions": len(debug_info_files),
            "scales_analyzed": [],
            "discriminator_analysis": {
                "samples_saved": {"real": 0, "fake": 0, "std": 0},
                "notes": []
            },
            "image_files": []
        }
        
        for info_file in debug_info_files:
            try:
                with open(info_file, 'r') as f:
                    info = json.load(f)
                
                scale_info = f"Scale_{info['scale_idx']}_{info['patch_size']}"
                if scale_info not in summary["scales_analyzed"]:
                    summary["scales_analyzed"].append(scale_info)
                
                # 累积样本统计
                for sample_type, count in info["samples_saved"].items():
                    summary["discriminator_analysis"]["samples_saved"][sample_type] += count
            
            except Exception as e:
                logger.warning(f"处理调试信息文件失败 {info_file}: {e}")
        
        # 收集图像文件列表
        image_files = glob.glob(os.path.join(debug_dir, "*.png"))
        summary["image_files"] = [os.path.basename(f) for f in image_files]
        
        # 生成分析说明
        total_samples = sum(summary["discriminator_analysis"]["samples_saved"].values())
        if total_samples > 0:
            summary["discriminator_analysis"]["notes"].append(
                f"总共保存了 {total_samples} 个解码样本用于视觉调试"
            )
            summary["discriminator_analysis"]["notes"].append(
                "请检查以下文件来分析判别器崩溃原因："
            )
            summary["discriminator_analysis"]["notes"].append(
                "- 'real' 样本：真实图像(判别器正样本)"
            )
            summary["discriminator_analysis"]["notes"].append(
                "- 'fake' 样本：AGIP-VAR生成图像(判别器负样本)"
            )
            summary["discriminator_analysis"]["notes"].append(
                "- 'std' 样本：标准VAR生成图像(对比基准)"
            )
            summary["discriminator_analysis"]["notes"].append(
                "- 'guidance' 样本：🆕 I_predictor引导tokens可视化"
            )
            summary["discriminator_analysis"]["notes"].append(
                "分析要点：1)fake样本是否明显劣于real样本 2)fake样本是否与std样本差异过大 3)🆕 guidance样本展现的规划意图"
            )
        
        # 保存摘要
        summary_file = os.path.join(debug_dir, f"visual_debug_summary_epoch_{epoch+1}.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📋 视觉调试摘要已生成: {summary_file}")
        logger.info(f"   分析尺度: {', '.join(summary['scales_analyzed'])}")
        logger.info(f"   保存样本: Real={summary['discriminator_analysis']['samples_saved']['real']}, " +
                   f"Fake={summary['discriminator_analysis']['samples_saved']['fake']}, " +
                   f"Std={summary['discriminator_analysis']['samples_saved']['std']}, " +
                   f"Guidance={summary['discriminator_analysis']['samples_saved'].get('guidance', 0)}")
        logger.info(f"   图像文件数: {len(summary['image_files'])}")
        
    except Exception as e:
        logger.warning(f"🔬 生成视觉调试摘要失败: {e}")


def should_save_visual_debug(
    epoch: int,
    batch_idx: int, 
    scale_idx: int,
    acc_real: float,
    acc_fake: float
) -> bool:
    """判断是否应该保存视觉调试样本
    
    Args:
        epoch: 当前epoch
        batch_idx: 当前batch索引
        scale_idx: 当前尺度索引
        acc_real: 真实样本准确率
        acc_fake: 假样本准确率
    
    Returns:
        是否应该保存视觉调试样本
    """
    # 🎯 修改：每次验证都保存真假词元图对应的图片，不要特殊条件
    return True


# 训练状态辅助函数
def is_warmup_phase(enable_staged_training: bool, step: int, warmup_steps: int) -> bool:
    """判断是否处于预热阶段"""
    return enable_staged_training and step < warmup_steps


def format_training_metrics(metrics: Dict[str, float], phase: str = "") -> str:
    """格式化训练指标输出"""
    phase_prefix = f"[{phase}] " if phase else ""
    
    return (f"{phase_prefix}D_loss: {metrics.get('loss_discriminator', 0):.4f}, "
            f"P_loss: {metrics.get('loss_planner', 0):.4f}, "
            f"D_acc: {metrics.get('discriminator_accuracy', 0):.3f}")


def log_discriminator_collapse_warning(epoch: int, acc_real: float, acc_fake: float):
    """记录判别器崩溃警告"""
    if acc_real > 0.95 and acc_fake < 0.05:
        logger.warning(f"🚨 Epoch {epoch+1}: 检测到判别器崩溃 - 总是判断为真实 (Real: {acc_real:.3f}, Fake: {acc_fake:.3f})")
    elif acc_real < 0.05 and acc_fake > 0.95:
        logger.warning(f"🚨 Epoch {epoch+1}: 检测到判别器崩溃 - 总是判断为假 (Real: {acc_real:.3f}, Fake: {acc_fake:.3f})")
    elif abs(acc_real - 0.5) < 0.1 and abs(acc_fake - 0.5) < 0.1:
        logger.warning(f"⚠️ Epoch {epoch+1}: 判别器接近随机水平 (Real: {acc_real:.3f}, Fake: {acc_fake:.3f})")


def create_debug_info_dict(
    epoch: int,
    batch_idx: int,
    scale_idx: int,
    patch_size: str,
    success_count: Dict[str, int],
    token_shapes: Dict[str, List[int]]
) -> Dict:
    """创建调试信息字典"""
    return {
        "epoch": epoch + 1,
        "batch_idx": batch_idx + 1,
        "scale_idx": scale_idx + 1,
        "patch_size": patch_size,
        "samples_saved": success_count,
        "token_shapes": token_shapes,
        "notes": "Visual debugging samples for discriminator collapse analysis"
    }


def ensure_debug_dir_exists(save_dir: str, epoch: int) -> str:
    """确保调试目录存在"""
    debug_dir = os.path.join(save_dir, "visual_debug", f"epoch_{epoch+1}")
    os.makedirs(debug_dir, exist_ok=True)
    return debug_dir 


# ============================================================================
# 🔄 交替训练相关辅助函数
# ============================================================================

class AlternatingTrainingManager:
    """交替训练管理器 - 管理discriminator和I_predictor的交替更新策略"""
    
    def __init__(
        self,
        strategy: str = "adaptive",      # 交替策略
        warmup_steps: int = 150,         # 预热步数
        guidance_weight_max: float = 0.01, # 最大引导权重
        progressive_steps: int = 50,     # 渐进式训练步数
    ):
        self.strategy = strategy
        self.warmup_steps = warmup_steps
        self.guidance_weight_max = guidance_weight_max
        self.progressive_steps = progressive_steps
        
        # 状态跟踪
        self.step_count = 0
        self.disc_updates = 0
        self.planner_updates = 0
        self.last_metrics = {}
        
        logger.info(f"🔄 交替训练管理器初始化: 策略={strategy}, 预热={warmup_steps}步")
    
    def should_update_discriminator(self, current_step: int, metrics: Optional[Dict] = None) -> bool:
        """判断当前步是否应该更新discriminator"""
        
        # 预热阶段：只训练discriminator
        if current_step < self.warmup_steps:
            return True
        
        # 联合训练阶段的策略
        if self.strategy == "simple_1_1":
            # 简单1:1交替
            return (current_step - self.warmup_steps) % 2 == 0
            
        elif self.strategy == "adaptive":
            # 自适应策略：基于性能调整
            if metrics:
                acc_real = metrics.get('acc_real', 0.5)
                acc_fake = metrics.get('acc_fake', 0.5)
                
                # 如果discriminator太弱，多训练discriminator
                if acc_real < 0.6 or acc_fake < 0.6:
                    return (current_step - self.warmup_steps) % 3 != 2  # 2/3时间训练discriminator
                
                # 如果discriminator太强，多训练I_predictor
                elif acc_real > 0.9 and acc_fake > 0.9:
                    return (current_step - self.warmup_steps) % 4 == 0  # 1/4时间训练discriminator
            
            # 默认1:1交替
            return (current_step - self.warmup_steps) % 2 == 0
            
        elif self.strategy == "discriminator_priority":
            # 每3步中前2步训练discriminator
            return (current_step - self.warmup_steps) % 3 != 2
        
        else:
            # 默认简单交替
            return (current_step - self.warmup_steps) % 2 == 0
    
    def should_update_planner(self, current_step: int, metrics: Optional[Dict] = None) -> bool:
        """判断当前步是否应该更新I_predictor"""
        
        # 预热阶段：不训练I_predictor
        if current_step < self.warmup_steps:
            return False
        
        # 联合训练阶段：与discriminator互斥
        return not self.should_update_discriminator(current_step, metrics)
    
    def get_progressive_guidance_weight(self, current_step: int) -> float:
        """计算引导权重 - 修改为固定权重"""
        if current_step < self.warmup_steps:
            return 0.0
        
        # 🔥 修改：使用固定引导权重，避免渐进增长导致的判别器崩溃
        return self.guidance_weight_max
        
        # # 原始渐进式增长逻辑（已注释）
        # # 联合训练阶段：渐进式增长
        # steps_since_joint = current_step - self.warmup_steps
        # if steps_since_joint < self.progressive_steps:
        #     # 线性增长到最大值
        #     progress = steps_since_joint / self.progressive_steps
        #     return self.guidance_weight_max * progress
        # else:
        #     return self.guidance_weight_max
    
    def update_step_count(self, updated_discriminator: bool, updated_planner: bool, metrics: Optional[Dict] = None):
        """更新步数统计"""
        self.step_count += 1
        if updated_discriminator:
            self.disc_updates += 1
        if updated_planner:
            self.planner_updates += 1
        
        # 保存指标用于下次决策
        if metrics:
            self.last_metrics = {
                'acc_real': metrics.get('acc_real', 0.5),
                'acc_fake': metrics.get('acc_fake', 0.5),
                'loss_D': metrics.get('loss_D', 0.0),
                'loss_P': metrics.get('loss_P', 0.0)
            }
    
    def get_training_stats(self) -> Dict:
        """获取训练统计信息"""
        total_updates = self.disc_updates + self.planner_updates
        return {
            'total_steps': self.step_count,
            'discriminator_updates': self.disc_updates,
            'planner_updates': self.planner_updates,
            'disc_update_ratio': self.disc_updates / max(total_updates, 1),
            'planner_update_ratio': self.planner_updates / max(total_updates, 1),
            'current_strategy': self.strategy,
            'warmup_completed': self.step_count >= self.warmup_steps
        }


def detect_discriminator_collapse(
    acc_real: float, 
    acc_fake: float, 
    current_step: int,
    collapse_history: List[Dict],
    threshold_real: float = 0.95,
    threshold_fake: float = 0.05
) -> Tuple[bool, bool]:
    """
    检测discriminator崩溃
    
    Args:
        acc_real: 真实样本准确率
        acc_fake: 假样本准确率
        current_step: 当前步数
        collapse_history: 崩溃历史记录
        threshold_real: 真实样本准确率阈值
        threshold_fake: 假样本准确率阈值
        
    Returns:
        (is_collapsed, should_recover): 是否崩溃，是否应该恢复
    """
    # 判别器崩溃的特征：对真实样本过拟合，无法识别假样本
    is_collapsed = acc_real > threshold_real and acc_fake < threshold_fake
    
    # 记录当前状态
    collapse_history.append({
        'step': current_step,
        'acc_real': acc_real,
        'acc_fake': acc_fake,
        'collapsed': is_collapsed
    })
    
    # 保持历史长度
    if len(collapse_history) > 10:
        collapse_history.pop(0)
    
    # 检测持续崩溃
    recent_collapses = sum(1 for h in collapse_history[-3:] if h['collapsed'])
    is_persistent_collapse = recent_collapses >= 2
    
    # 检测恢复
    should_recover = acc_real < 0.8 and acc_fake > 0.3
    
    if is_persistent_collapse:
        logger.warning(f"🚨 检测到持续discriminator崩溃！步数={current_step}, 连续崩溃={recent_collapses}")
    
    return is_collapsed, should_recover


def apply_learning_rate_fix(
    opt_planner, 
    opt_disc, 
    base_lr: float = 5e-6,
    recovery_mode: bool = False
):
    """
    应用学习率修复
    
    Args:
        opt_planner: I_predictor优化器
        opt_disc: discriminator优化器
        base_lr: 基础学习率
        recovery_mode: 是否为恢复模式（降低学习率）
    """
    target_lr_planner = base_lr
    target_lr_disc = base_lr
    
    # 恢复模式：降低学习率
    if recovery_mode:
        target_lr_planner *= 0.8
        target_lr_disc *= 0.5
        logger.info(f"🔧 恢复模式：降低学习率 planner={target_lr_planner:.2e}, disc={target_lr_disc:.2e}")
    
    # 更新学习率
    for param_group in opt_planner.optimizer.param_groups:
        param_group['lr'] = target_lr_planner
    for param_group in opt_disc.optimizer.param_groups:
        param_group['lr'] = target_lr_disc


class NoOpOptimizer:
    """空操作优化器包装器 - 用于交替训练时临时禁用某个优化器"""
    
    def __init__(self, real_opt):
        self.real_opt = real_opt
        self.amp_ctx = real_opt.amp_ctx
        self.optimizer = real_opt.optimizer  # 保持接口兼容性
        
    def backward_clip_step(self, stepping=True, loss=None, retain_graph=False):
        """执行反向传播但不更新参数"""
        # 让loss计算正常进行，但不执行参数更新
        # 这样可以保持梯度计算的正确性，但避免参数更新
        return 


def log_alternating_training_step(
    current_step: int,
    phase: str,
    updated_discriminator: bool,
    updated_planner: bool,
    guidance_weight: float,
    acc_real: float,
    acc_fake: float,
    strategy: str
):
    """记录交替训练步骤信息"""
    
    update_info = []
    if updated_discriminator:
        update_info.append("判别器")
    if updated_planner:
        update_info.append("规划器")
    if not update_info:
        update_info.append("无更新")
    
    update_str = "/".join(update_info)
    
    logger.info(f"🔄 步数{current_step} ({phase}): 更新{update_str}, "
               f"引导权重={guidance_weight:.4f}, "
               f"准确率 real={acc_real:.3f}/fake={acc_fake:.3f}, "
               f"策略={strategy}")


def format_alternating_training_metrics(metrics: Dict, alt_manager: AlternatingTrainingManager) -> str:
    """格式化交替训练指标显示"""
    
    stats = alt_manager.get_training_stats()
    
    lines = [
        f"📊 交替训练统计:",
        f"   总步数: {stats['total_steps']}",
        f"   判别器更新: {stats['discriminator_updates']} ({stats['disc_update_ratio']*100:.1f}%)",
        f"   规划器更新: {stats['planner_updates']} ({stats['planner_update_ratio']*100:.1f}%)",
        f"   策略: {stats['current_strategy']}",
        f"   预热完成: {'是' if stats['warmup_completed'] else '否'}",
        f"   当前准确率: real={metrics.get('acc_real', 0):.3f}, fake={metrics.get('acc_fake', 0):.3f}",
        f"   引导权重: {metrics.get('guidance_weight', 0):.4f}"
    ]
    
    return "\n".join(lines) 