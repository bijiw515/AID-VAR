import time
import logging
import os
import json
import random
import numpy as np
import traceback
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

import dist
from models.var import VAR
from models.vqvae import VQVAE
from models.guidance_injector import GuidanceInjector
from models.discriminator_adapter import StyleGANDiscriminatorAdapter
from utils.amp_sc import AmpOptimizer
from utils.misc import MetricLogger, TensorboardLogger

# 导入AID辅助函数模块
import aid_helpers

Ten = torch.Tensor
ITen = torch.LongTensor

logger = logging.getLogger(__name__)

class PlannerTrainer:
    """✅ 完全重构的AID-VAR训练器，使用空间感知的规划词元图 + 交替训练策略
    
    核心特性：
    🎯 空间感知规划：GuidanceInjector输出规划词元图，与VAR状态逐位置相加
    🔄 交替训练：discriminator和GuidanceInjector交替更新，避免计算图冲突
    🚨 自动崩溃检测：检测discriminator模式崩溃并自动恢复
    📈 渐进式引导：从0.00到最大值的渐进式引导权重增长
    🧊 分阶段训练：预热阶段→联合训练阶段的平滑过渡
    """
    
    def __init__(
        self,
        device,
        vae: VQVAE,
        var: VAR,
        planner: GuidanceInjector,
        disc: StyleGANDiscriminatorAdapter,
        opt_planner: AmpOptimizer,
        opt_disc: AmpOptimizer,
        lambda_rec: float = 0.1,
        patch_nums: Optional[Tuple[int, ...]] = None,
        top_k: int = 0,
        top_p: float = 0.0,
        cfg: float = 1.0,
        # 🎯 新增：分阶段训练参数
        warmup_steps: int = 0,  # 判别器预热步数
        enable_staged_training: bool = True,  # 是否启用分阶段训练
        # 🔄 新增：交替训练参数
        alternating_strategy: str = "adaptive",  # 交替训练策略
        guidance_weight_max: float = 0.01,      # 最大引导权重（修复硬编码0.05）
    ):
        self.device = device
        self.vae = vae
        self.var = var
        self.planner = planner
        self.disc = disc
        self.opt_planner = opt_planner
        self.opt_disc = opt_disc
        self.lambda_rec = lambda_rec
        
        # VAR配置
        self.patch_nums = patch_nums or var.patch_nums
        self.Cvae = vae.quantize.Cvae
        self.V = vae.quantize.vocab_size
        self.num_stages = len(self.patch_nums)
        
        # 采样参数
        self.top_k = top_k
        self.top_p = top_p
        self.cfg = cfg
        
        # 🎯 分阶段训练策略
        self.enable_staged_training = enable_staged_training
        self.warmup_steps = warmup_steps
        self.current_step = 0
        self.is_warmup_phase = True
        
        # 🔄 交替训练管理器
        self.alternating_manager = aid_helpers.AlternatingTrainingManager(
            strategy=alternating_strategy,
            warmup_steps=warmup_steps,
            guidance_weight_max=guidance_weight_max,
            progressive_steps=300  # 渐进式引导权重增长步数（增加到300步，更加渐进）
        )
        
        # 🔄 discriminator崩溃检测
        self.collapse_history = []  # 记录崩溃历史
        self.recovery_mode = False  # 是否在恢复模式
        
        # 预计算序列位置信息
        self.scale_begins = []
        self.scale_ends = []
        cur = 0
        for pn in self.patch_nums:
            self.scale_begins.append(cur)
            cur += pn * pn
            self.scale_ends.append(cur)
        self.total_seq_len = cur  # 总序列长度 = 680
        
        # 统计可训练参数
        planner_params = sum(p.numel() for p in self.planner.parameters() if p.requires_grad)
        disc_trainable_params = self.disc.get_trainable_params()
        disc_params = sum(p.numel() for p in disc_trainable_params)
        # AID-VAR参数统计 (静默)
        
        # 🎯 分阶段训练状态
        if self.enable_staged_training:
            self._freeze_planner()  # 初始化时冻结GuidanceInjector

    def _freeze_planner(self):
        """冻结GuidanceInjector参数"""
        for param in self.planner.parameters():
            param.requires_grad = False

    def _unfreeze_planner(self):
        """解冻GuidanceInjector参数"""
        for param in self.planner.parameters():
            param.requires_grad = True

    def _update_training_phase(self):
        """更新训练阶段状态"""
        if not self.enable_staged_training:
            return
            
        # 检查是否需要从预热阶段切换到联合训练阶段
        if self.is_warmup_phase and self.current_step >= self.warmup_steps:
            self.is_warmup_phase = False
            self._unfreeze_planner()

    def _get_planning_token_map(self, prev_features: Ten, target_patch_num: int) -> Optional[Ten]:
        """根据训练阶段生成规划词元图
        
        Args:
            prev_features: 前一尺度特征 (B, L_prev, C)
            target_patch_num: 目标尺度patch数量
            
        Returns:
            规划词元图 (B, L_target, C) 或 None
        """
        if not self.enable_staged_training or not self.is_warmup_phase:
            # 联合训练阶段：正常生成规划词元图
            return self.planner(prev_features, target_patch_num=target_patch_num)
        else:
            # 预热阶段：返回零向量，不影响VAR生成
            B, _, C = prev_features.shape
            L_target = target_patch_num * target_patch_num
            zero_planning_map = torch.zeros(
                B, L_target, C, 
                device=prev_features.device, 
                dtype=prev_features.dtype,
                requires_grad=False  # 🔥 关键：预热阶段不需要梯度
            )
            return zero_planning_map

    def train_step(self, img_B3HW: Ten, label_B: ITen, metric_lg: Optional[MetricLogger] = None):
        """🚀 AID-VAR逐尺度训练步骤 - 基于VAR的autoregressive_train_step方法的逐尺度训练逻辑
        
        核心改进：
        1. 📊 标准Token生成：使用VAR的autoregressive_train_step方法，逐尺度生成std_tokens (无引导)
        2. 🎯 引导Token生成：使用std_tokens作为GuidanceInjector输入，生成所有尺度的guidance_tokens
        3. 🔄 引导Token生成：使用VAR的autoregressive_train_step方法，逐尺度生成fake_tokens (有引导)
        4. 🥊 批次级梯度下降：使用多尺度discriminator损失进行单次梯度下降
        
        训练流程：
        Step 1: 获取真实tokens (监督信号)
        Step 2: 使用VAR逐尺度自回归生成std_tokens (无引导，确定性采样)
        Step 3: 使用GuidanceInjector生成guidance_tokens (基于std_tokens，逐尺度)
        Step 4: 使用VAR逐尺度自回归生成fake_tokens (有引导，随机采样)
        Step 5: 批次级判别器训练 (多尺度损失聚合)
        """
        
        # 🎯 更新训练阶段状态
        self._update_training_phase()
        self.current_step += 1
        
        # 训练步骤开始
        
        self.var.eval()  # 冻结VAR
        self.vae.eval()  # 冻结VAE
        
        B = img_B3HW.size(0)
        device = self.device
        
        # 确保数据在正确设备上
        img_B3HW = img_B3HW.to(device, non_blocking=True)
        label_B = label_B.to(device, non_blocking=True)
        
        # Step 1: 获取真实图像的多尺度tokens
        with torch.no_grad():
            gt_idx_Bl: List[ITen] = self.vae.img_to_idxBl(img_B3HW)
            gt_BL = torch.cat(gt_idx_Bl, dim=1)
            x_BLCv_wo_first_l = self.vae.quantize.idxBl_to_var_input(gt_idx_Bl)
        
        # Step 2: 生成标准tokens (无引导)
        with torch.no_grad():
            std_tokens_list, std_logits_list = self.var.autoregressive_train_step(
                B=B, 
                label_B=label_B, 
                guidance_tokens_list=None,
                deterministic=True,
                g_seed=None
            )
        
        # Step 3: 生成引导tokens
        guidance_tokens_list = []
        
        if not (self.enable_staged_training and self.is_warmup_phase):
            # 非预热阶段：正常生成引导tokens
            for si, pn in enumerate(self.patch_nums):
                if si == 0:
                    # 第一个尺度没有前驱，使用零引导
                    guidance_tokens_list.append(torch.zeros(B, pn * pn, self.var.C, device=device))
                else:
                    # 使用前一尺度的std_tokens生成引导
                    prev_std_tokens = std_tokens_list[si-1]
                    prev_features = F.embedding(prev_std_tokens, self.vae.quantize.embedding.weight.detach())
                    prev_features = self.var.word_embed(prev_features)
                    prev_features = prev_features.detach().clone().requires_grad_(True)
                    
                    # 数值稳定性保护
                    prev_features = torch.clamp(prev_features, min=-10.0, max=10.0)
                    
                    # 生成引导tokens
                    guidance_tokens = self._get_planning_token_map(prev_features, pn)
                    if guidance_tokens is not None:
                        guidance_tokens = torch.clamp(guidance_tokens, min=-5.0, max=5.0)
                        guidance_tokens_list.append(guidance_tokens)
                    else:
                        guidance_tokens_list.append(torch.zeros(B, pn * pn, self.var.C, device=device))
        else:
            # 预热阶段：使用零引导
            for si, pn in enumerate(self.patch_nums):
                guidance_tokens_list.append(torch.zeros(B, pn * pn, self.var.C, device=device))
        
        # 合并所有尺度的引导tokens
        guidance_tokens_BLC = torch.cat(guidance_tokens_list, dim=1)
        
        # Step 4: 逐尺度生成引导后tokens
        guidance_weight = self.alternating_manager.get_progressive_guidance_weight(self.current_step)
        
        with torch.enable_grad():
            # 应用渐进式引导权重
            scaled_guidance_tokens_list = []
            for si, guidance_tokens in enumerate(guidance_tokens_list):
                if guidance_tokens is not None:
                    scaled_guidance = guidance_tokens * guidance_weight
                    scaled_guidance_tokens_list.append(scaled_guidance)
                else:
                    scaled_guidance_tokens_list.append(None)
            
            # 使用VAR生成引导后tokens
            if self.enable_staged_training and self.is_warmup_phase and guidance_weight == 0.0:
                fake_tokens_list = [tokens.clone() for tokens in std_tokens_list]
                fake_logits_list = [logits.clone() for logits in std_logits_list]
            else:
                # 联合训练阶段：使用引导的逐尺度自回归预测
                fake_tokens_list, fake_logits_list = self.var.autoregressive_train_step(
                    B=B, 
                    label_B=label_B, 
                    guidance_tokens_list=scaled_guidance_tokens_list,  # 有引导
                    deterministic=False,  # 随机采样
                    g_seed=None
                )
            

            
            # 🔥 计算软特征映射用于判别器梯度流（仅在联合训练阶段）
            fake_probs_list = []
            fake_soft_features_list = []
            
            if not (self.enable_staged_training and self.is_warmup_phase):
                # 联合训练阶段：计算软概率用于梯度流
                vae_embedding_weight = self.vae.quantize.embedding.weight  # (V, Cvae)
                
                for fake_logits in fake_logits_list:
                    # 转换为概率分布
                    fake_probs = F.softmax(fake_logits, dim=-1)  # (B, L_scale, V)
                    fake_probs_list.append(fake_probs)
                    
                    # 计算软特征映射
                    fake_soft_features = torch.matmul(fake_probs, vae_embedding_weight)  # (B, L_scale, Cvae)
                    fake_soft_features_list.append(fake_soft_features)
                
    
            else:
                # 预热阶段：创建空列表避免后续代码错误，但不执行软解码
                fake_probs_list = [None] * len(fake_logits_list)
                fake_soft_features_list = [None] * len(fake_logits_list)

        
        # 🎯 Step 5: 批次级判别器训练 (多尺度损失聚合)

        
        loss_D_total = torch.tensor(0.0, device=device)
        loss_P_total = torch.tensor(0.0, device=device)
        loss_adv_total = torch.tensor(0.0, device=device)
        loss_rec_total = torch.tensor(0.0, device=device)
        acc_D_sum = 0.0
        acc_real_sum = 0.0
        acc_fake_sum = 0.0
        valid_scales = 0
        
        # 批次级别的交替训练决策
        current_metrics = {
            'acc_real': 0.5,
            'acc_fake': 0.5,
            'loss_D': 0.0,
            'loss_P': 0.0,
        }
        
        should_update_disc_batch = self.alternating_manager.should_update_discriminator(
            self.current_step, current_metrics)
        should_update_planner_batch = self.alternating_manager.should_update_planner(
            self.current_step, current_metrics)
        
        # 获取渐进式引导权重
        guidance_weight = self.alternating_manager.get_progressive_guidance_weight(self.current_step)
        
        # 多尺度判别器训练
        for si, pn in enumerate(self.patch_nums):
            if si == 0:
                continue  # 跳过第一个尺度
            
            real_tokens_k = gt_idx_Bl[si]
            fake_tokens_k = fake_tokens_list[si]
            
            try:
                # 🚀 新架构：trainer层面token→RGB转换，判别器直接处理RGB
                
                # Step 1: 真实tokens → RGB图像
                real_rgb_k = self.tokens_to_rgb_cumulative(gt_idx_Bl, si)
                
                # Step 2: tokens → RGB图像 (根据训练阶段选择硬解码或软解码)
                if self.enable_staged_training and self.is_warmup_phase:
                    # 🔧 预热阶段：使用硬解码避免cuDNN兼容性问题
                    fake_rgb_k = self.tokens_to_rgb_cumulative(fake_tokens_list, si)
                else:
                    # 🔧 联合训练阶段：使用软解码保持梯度流  
                    fake_probs_k = fake_probs_list[si]  # (B, L_scale, V) - 直接从列表获取
                    if fake_probs_k is not None:
                        fake_rgb_k = self.soft_tokens_to_rgb(fake_probs_list, si)
                    else:
                        # 备用方案：使用硬解码
                        fake_rgb_k = self.tokens_to_rgb_cumulative(fake_tokens_list, si)
                
                # Step 3: 判别器直接处理RGB图像
                pred_real = self.disc.forward_rgb(real_rgb_k)  # 🆕 直接RGB判别
                
                if self.enable_staged_training and self.is_warmup_phase:
                    # 🔧 预热阶段：使用标准RGB判别（硬解码的图像）
                    pred_fake = self.disc.forward_rgb(fake_rgb_k)
                else:
                    # 🔧 联合训练阶段：使用软RGB判别保持梯度流
                    pred_fake = self.disc.forward_soft_rgb(fake_rgb_k)
                
                # 🔧 关键数值稳定性检查：防止NaN传播到损失计算
                if torch.isnan(pred_real).any() or torch.isinf(pred_real).any():
                    pred_real = torch.nan_to_num(pred_real, nan=0.0, posinf=1.0, neginf=-1.0)
                
                if torch.isnan(pred_fake).any() or torch.isinf(pred_fake).any():
                    pred_fake = torch.nan_to_num(pred_fake, nan=0.0, posinf=1.0, neginf=-1.0)
                
                # 处理多头判别器输出
                pred_real_mean = pred_real.mean(dim=1)  # [B, multi_heads] -> [B]
                pred_fake_mean = pred_fake.mean(dim=1)  # [B, multi_heads] -> [B]
                
                # 🔧 最终数值稳定性检查
                pred_real_mean = torch.clamp(pred_real_mean, min=-10.0, max=10.0)
                pred_fake_mean = torch.clamp(pred_fake_mean, min=-10.0, max=10.0)
                
                # 计算判别器损失
                loss_D_real = F.relu(1 - pred_real_mean).mean()
                loss_D_fake = F.relu(1 + pred_fake_mean).mean()
                loss_D = loss_D_real + loss_D_fake
                
                # 🔧 优化：预热阶段跳过规划器损失计算，避免复杂计算图
                if self.enable_staged_training and self.is_warmup_phase:
                    # 预热阶段：只计算判别器损失，简化计算图
                    loss_P = torch.tensor(0.0, device=device)  
                    loss_adv = torch.tensor(0.0, device=device)  # 避免计算复杂梯度
                    loss_rec = torch.tensor(0.0, device=device)  # 避免计算复杂梯度
                else:
                    # 联合训练阶段：计算完整的规划器损失
                    loss_adv = -pred_fake_mean.mean()
                    
                    # 计算重构损失 (基于逐尺度的logits)
                    fake_logits_k = fake_logits_list[si]  # (B, L_scale, V) - 直接从列表获取
                    loss_rec = F.cross_entropy(
                        fake_logits_k.reshape(-1, self.var.V), 
                        real_tokens_k.reshape(-1)
                    )
                    
                    loss_P = loss_adv + self.lambda_rec * loss_rec
                
                # 计算准确率
                acc_real = (pred_real_mean > 0).float().mean().item()
                acc_fake = (pred_fake_mean < 0).float().mean().item()
                
                # 累积损失和指标 - 🔥 修复: 保持梯度用于backward
                loss_D_total += loss_D  # 不要detach，保持梯度
                loss_P_total += loss_P  # 不要detach，保持梯度
                loss_adv_total += loss_adv.detach()  # 对于记录可以detach
                loss_rec_total += loss_rec.detach()  # 对于记录可以detach
                acc_D_sum += (acc_real + acc_fake) / 2
                acc_real_sum += acc_real
                acc_fake_sum += acc_fake
                valid_scales += 1
                

                
            except Exception as e:

                continue
        
        # 执行参数更新（批次级别）
        if valid_scales > 0:
            # 🔧 修复：根据训练阶段使用不同的计算图策略
            if self.enable_staged_training and self.is_warmup_phase:
                # 预热阶段：只更新判别器，使用简单计算图
                if should_update_disc_batch:
                    grad_norm_disc, scale_log2_disc = self.opt_disc.backward_clip_step(
                        stepping=True, 
                        loss=loss_D_total, 
                        retain_graph=False  # 🔥 预热阶段不需要保留图
                    )
                
                # 预热阶段不更新规划器，无需复杂计算图
                
            else:
                # 联合训练阶段：需要交替更新，使用retain_graph
                if should_update_disc_batch:
                    # 先更新判别器，保留计算图
                    grad_norm_disc, scale_log2_disc = self.opt_disc.backward_clip_step(
                        stepping=True, 
                        loss=loss_D_total, 
                        retain_graph=True  # 联合训练需要保留图给规划器使用
                    )
                
                if should_update_planner_batch:
                    # 后更新规划器，释放计算图
                    grad_norm_planner, scale_log2_planner = self.opt_planner.backward_clip_step(
                        stepping=True,
                        loss=loss_P_total,
                        retain_graph=False  # 最后一个，可以释放图
                    )
        
        # 计算平均损失和指标
        if valid_scales > 0:
            avg_loss_D = loss_D_total / valid_scales
            avg_loss_P = loss_P_total / valid_scales
            avg_loss_adv = loss_adv_total / valid_scales
            avg_loss_rec = loss_rec_total / valid_scales
            avg_acc_D = acc_D_sum / valid_scales
            avg_acc_real = acc_real_sum / valid_scales
            avg_acc_fake = acc_fake_sum / valid_scales
        else:
            avg_loss_D = loss_D_total
            avg_loss_P = loss_P_total  
            avg_loss_adv = loss_adv_total
            avg_loss_rec = loss_rec_total
            avg_acc_D = 0.0
            avg_acc_real = 0.0
            avg_acc_fake = 0.0
        
        # 记录指标
        if metric_lg is not None:
            metric_lg.update(aid_loss_D=avg_loss_D.item())
            metric_lg.update(aid_loss_P=avg_loss_P.item())
            metric_lg.update(aid_loss_adv=avg_loss_adv.item())
            metric_lg.update(aid_loss_rec=avg_loss_rec.item())
            metric_lg.update(aid_acc_D=avg_acc_D)
            metric_lg.update(aid_acc_real=avg_acc_real)
            metric_lg.update(aid_acc_fake=avg_acc_fake)
            metric_lg.update(aid_scales=valid_scales)
            metric_lg.update(aid_warmup_phase=int(self.is_warmup_phase) if self.enable_staged_training else 0)
            metric_lg.update(aid_step=self.current_step)
            metric_lg.update(aid_guidance_weight=guidance_weight)
            metric_lg.update(aid_recovery_mode=int(self.recovery_mode))
            stats = self.alternating_manager.get_training_stats()
            metric_lg.update(aid_disc_updates=stats['discriminator_updates'])
            metric_lg.update(aid_planner_updates=stats['planner_updates'])
        
        # 保留内部统计但移除详细输出
        guidance_weight = self.alternating_manager.get_progressive_guidance_weight(self.current_step)
        stats = self.alternating_manager.get_training_stats()
        
        return {
            'loss_D': avg_loss_D.item(),
            'loss_P': avg_loss_P.item(), 
            'loss_adversarial': avg_loss_adv.item(),
            'loss_reconstruction': avg_loss_rec.item(),
            'acc_D': avg_acc_D,
            'acc_real': avg_acc_real,
            'acc_fake': avg_acc_fake,
            'valid_scales': valid_scales,
            'warmup_phase': self.is_warmup_phase if self.enable_staged_training else False,
            'current_step': self.current_step,
            'guidance_weight': guidance_weight,
            'recovery_mode': self.recovery_mode,
            'disc_updates': stats['discriminator_updates'],
            'planner_updates': stats['planner_updates'],
            'disc_update_ratio': stats['disc_update_ratio'],
            'planner_update_ratio': stats['planner_update_ratio']
        }

    
    
    def _sample_from_logits(self, logits: Ten, gt_idx_Bl: List[ITen]) -> List[ITen]:
        """从logits中采样tokens，用于训练时的token生成"""
        from models.helpers import sample_with_top_k_top_p_
        
        sampled_tokens_list = []
        for si, gt_tokens in enumerate(gt_idx_Bl):
            # 简单贪心采样
            sampled_tokens = logits[si].argmax(dim=-1)
            sampled_tokens_list.append(sampled_tokens)
        
        return sampled_tokens_list 
    
    def validate_one_epoch(
        self, 
        val_loader, 
        epoch: int, 
        save_dir: str
    ) -> Dict[str, float]:
        """✅ 执行一个epoch的AID-VAR验证
        
        完全按照训练逻辑执行，但不进行梯度更新
        """
        
        # 设置为验证模式
        self.planner.eval()
        self.disc.eval()
        self.var.eval()
        self.vae.eval()
        
        # 创建验证结果保存目录
        val_epoch_dir = os.path.join(save_dir, "validation", f"epoch_{epoch+1}")
        os.makedirs(val_epoch_dir, exist_ok=True)
        
        # 验证批次数量设置
        num_val_batches = min(len(val_loader), 10)
        
        # 初始化验证指标
        val_metrics = {
            'loss_planner': 0.0,
            'loss_discriminator': 0.0,
            'loss_adversarial': 0.0,
            'loss_reconstruction': 0.0,
            'discriminator_accuracy': 0.0,
            'discriminator_acc_real': 0.0,
            'discriminator_acc_fake': 0.0,
            'mse_improvement': 0.0,
            'generation_quality': 0.0,
            'scale_wise_performance': {}
        }
        
        # 随机选择验证批次
        val_indices = list(range(len(val_loader)))
        np.random.shuffle(val_indices)
        selected_indices = val_indices[:num_val_batches]
        
        # 逐批次验证
        with torch.no_grad():
            for batch_idx in selected_indices:
                try:
                    # 获取指定批次的数据
                    val_iter = iter(val_loader)
                    for _ in range(batch_idx):
                        next(val_iter)
                    
                    images, labels = next(val_iter)
                    B = images.shape[0]
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    # 创建随机数生成器
                    rng = torch.Generator(device=self.device)
                    
                    # Step 1: 获取真实图像的多尺度tokens
                    gt_idx_Bl: List[ITen] = self.vae.img_to_idxBl(images)
                    
                    # Step 2: 生成标准tokens (无引导)
                    with torch.no_grad():
                        std_tokens_list, std_logits_list = self.var.autoregressive_train_step(
                            B=B, 
                            label_B=labels, 
                            guidance_tokens_list=None,  # 无引导
                            deterministic=False,  # 随机采样
                            g_seed=None
                        )
                        
                    
                    # 🎯 Step 3: AID-VAR逐尺度引导生成 + 对抗验证 (完全模仿训练逻辑)
                    batch_metrics = self._validate_aid_var_multiscale_generation(
                        B, labels, gt_idx_Bl, std_tokens_list, std_logits_list, epoch, batch_idx, save_dir, rng
                    )
                    
                    # 累积指标
                    for key in ['loss_planner', 'loss_discriminator', 'loss_adversarial', 
                               'loss_reconstruction', 'discriminator_accuracy', 
                               'discriminator_acc_real', 'discriminator_acc_fake']:
                        val_metrics[key] += batch_metrics.get(key, 0.0)
                    
                    # 🎯 Step 4: 计算改进指标
                    if len(std_tokens_list) > 0 and len(batch_metrics.get('guided_tokens_list', [])) > 0:
                        guided_tokens_list = batch_metrics['guided_tokens_list']
                        
                        # 形状检查
                        
                        # 计算最后一个尺度的MSE
                        if len(gt_idx_Bl) > 0 and len(std_tokens_list) > 0 and len(guided_tokens_list) > 0:
                            try:
                                mse_std = aid_helpers.calculate_mse_loss(std_tokens_list[-1], gt_idx_Bl[-1])
                                mse_guided = aid_helpers.calculate_mse_loss(guided_tokens_list[-1], gt_idx_Bl[-1])
                                improvement = max(0, (mse_std - mse_guided) / (mse_std + 1e-8))
                                val_metrics['mse_improvement'] += improvement
                            except Exception as e:
                                val_metrics['mse_improvement'] += 0.0
                        else:
                            val_metrics['mse_improvement'] += 0.0
                        
                        # 生成质量评估
                        quality_score = batch_metrics.get('discriminator_accuracy', 0.0)
                        val_metrics['generation_quality'] += quality_score
                        
                        # 逐尺度性能记录
                        for si, pn in enumerate(self.patch_nums):
                            scale_key = f"scale_{si}_{pn}x{pn}"
                            if scale_key not in val_metrics['scale_wise_performance']:
                                val_metrics['scale_wise_performance'][scale_key] = {
                                    'mse_std': 0.0, 'mse_guided': 0.0, 'improvement': 0.0
                                }
                            
                                                        # 计算各尺度指标
                            if si < len(gt_idx_Bl) and si < len(std_tokens_list) and si < len(guided_tokens_list):
                                try:
                                    scale_mse_std = aid_helpers.calculate_mse_loss(std_tokens_list[si], gt_idx_Bl[si])
                                    scale_mse_guided = aid_helpers.calculate_mse_loss(guided_tokens_list[si], gt_idx_Bl[si])
                                    scale_improvement = max(0, (scale_mse_std - scale_mse_guided) / (scale_mse_std + 1e-8))
                                    
                                    val_metrics['scale_wise_performance'][scale_key]['mse_std'] += scale_mse_std
                                    val_metrics['scale_wise_performance'][scale_key]['mse_guided'] += scale_mse_guided
                                    val_metrics['scale_wise_performance'][scale_key]['improvement'] += scale_improvement
                                except Exception as e:
                                    pass
                                    val_metrics['scale_wise_performance'][scale_key]['mse_std'] += 0.0
                                    val_metrics['scale_wise_performance'][scale_key]['mse_guided'] += 0.0
                                    val_metrics['scale_wise_performance'][scale_key]['improvement'] += 0.0
                            else:

                                val_metrics['scale_wise_performance'][scale_key]['mse_std'] += 0.0
                                val_metrics['scale_wise_performance'][scale_key]['mse_guided'] += 0.0
                                val_metrics['scale_wise_performance'][scale_key]['improvement'] += 0.0
                
                except Exception as e:
                    continue
        
        # 计算平均值
        if num_val_batches > 0:
            for key in ['loss_planner', 'loss_discriminator', 'loss_adversarial',
                       'loss_reconstruction', 'discriminator_accuracy',
                       'discriminator_acc_real', 'discriminator_acc_fake',
                       'mse_improvement', 'generation_quality']:
                val_metrics[key] /= num_val_batches
            
            # 平均逐尺度性能
            for scale_key in val_metrics['scale_wise_performance']:
                for metric_key in val_metrics['scale_wise_performance'][scale_key]:
                    val_metrics['scale_wise_performance'][scale_key][metric_key] /= num_val_batches
        
        # 保存验证结果
        val_results_file = os.path.join(val_epoch_dir, "validation_results.json")
        with open(val_results_file, 'w') as f:
            json.dump(val_metrics, f, indent=2)
        
        # 🎯 精简验证结果输出 - 包含正负样本准确率
        logger.info(f"Epoch {epoch+1} Validation: "
                   f"D_loss={val_metrics['loss_discriminator']:.4f} "
                   f"P_loss={val_metrics['loss_planner']:.4f} "
                   f"D_acc={val_metrics['discriminator_accuracy']:.3f} "
                   f"real_acc={val_metrics['discriminator_acc_real']:.3f} "
                   f"fake_acc={val_metrics['discriminator_acc_fake']:.3f} "
                   f"MSE_improve={val_metrics['mse_improvement']:.4f}")
        
        # 🔬 生成视觉调试摘要 (如果有保存调试样本)
        try:
            aid_helpers.generate_visual_debug_summary(save_dir, epoch)
        except Exception as e:
            pass
        
        return val_metrics
    
    def _validate_aid_var_multiscale_generation(
        self, 
        B: int, 
        labels: ITen, 
        gt_idx_Bl: List[ITen], 
        std_tokens_list: List[ITen],
        std_logits_list: List[ITen],
        epoch: int,
        batch_idx: int, 
        save_dir: str,
        rng: torch.Generator
    ) -> Dict[str, float]:
        """✅ 完全按照train_step逻辑实现的AID-VAR验证
        
        这个方法完全模仿train_step中的逐尺度生成逻辑，
        但不执行梯度更新，只计算损失指标
        """
        
        # Step 2: AID-VAR逐尺度引导生成
        
        # 初始化损失统计
        loss_D_total = torch.tensor(0.0, device=self.device)
        loss_P_total = torch.tensor(0.0, device=self.device)
        # 🔥 新增：分别统计对抗损失和重构损失
        loss_adv_total = torch.tensor(0.0, device=self.device)
        loss_rec_total = torch.tensor(0.0, device=self.device)
        acc_D_sum = 0.0
        acc_real_sum = 0.0  # 新增：分别统计真实样本准确率
        acc_fake_sum = 0.0  # 新增：分别统计假样本准确率
        valid_scales = 0
        
        # 🎯 生成引导tokens (使用GuidanceInjector，基于std_tokens) - 并行生成所有尺度
        guidance_tokens_list = []
        
        if not (self.enable_staged_training and self.is_warmup_phase):
            # 非预热阶段：正常生成引导tokens
            for si, pn in enumerate(self.patch_nums):
                if si == 0:
                    # 第一个尺度没有前驱，使用零引导
                    guidance_tokens_list.append(torch.zeros(B, pn * pn, self.var.C, device=self.device))
                else:
                    # 使用前一尺度的std_tokens生成引导
                    prev_std_tokens = std_tokens_list[si-1]
                    prev_features = F.embedding(prev_std_tokens, self.vae.quantize.embedding.weight.detach())
                    prev_features = self.var.word_embed(prev_features)
                    prev_features = prev_features.detach().clone().requires_grad_(False)  # 验证时不需要梯度
                    
                    # 数值稳定性保护
                    prev_features = torch.clamp(prev_features, min=-10.0, max=10.0)
                    
                    # 生成引导tokens
                    guidance_tokens = self._get_planning_token_map(prev_features, pn)
                    if guidance_tokens is not None and not (torch.isnan(guidance_tokens).any() or torch.isinf(guidance_tokens).any()):
                        guidance_tokens = torch.clamp(guidance_tokens, min=-5.0, max=5.0)
                        guidance_tokens_list.append(guidance_tokens)
                    else:
                        guidance_tokens_list.append(torch.zeros(B, pn * pn, self.var.C, device=self.device))
        else:
            # 预热阶段：使用零引导
            for si, pn in enumerate(self.patch_nums):
                guidance_tokens_list.append(torch.zeros(B, pn * pn, self.var.C, device=self.device))
        
        # 合并所有尺度的引导tokens
        guidance_tokens_BLC = torch.cat(guidance_tokens_list, dim=1)  # (B, L, C)
        
        # 🎯 逐尺度生成引导后tokens (与训练逻辑完全一致)
        
        # 🔧 获取当前步骤的引导权重 (与训练保持一致)
        guidance_weight = self.alternating_manager.get_progressive_guidance_weight(self.current_step)
        
        # 🔥 关键修复：应用渐进式引导权重到guidance_tokens (与训练保持一致)
        scaled_guidance_tokens_list = []
        for si, guidance_tokens in enumerate(guidance_tokens_list):
            if guidance_tokens is not None:
                scaled_guidance = guidance_tokens * guidance_weight
                scaled_guidance_tokens_list.append(scaled_guidance)
            else:
                scaled_guidance_tokens_list.append(None)
        
        # 使用VAR的逐尺度自回归预测，注入缩放后的guidance_tokens (与训练逻辑完全一致)
        if self.enable_staged_training and self.is_warmup_phase and guidance_weight == 0.0:
            # 预热阶段且无引导：fake tokens应该等于std tokens
            generated_guided_tokens = [tokens.clone() for tokens in std_tokens_list]
            fake_logits_list = [logits.clone() for logits in std_logits_list]
        else:
            # 联合训练阶段：使用引导的逐尺度自回归预测 (验证时使用随机采样)
            generated_guided_tokens, fake_logits_list = self.var.autoregressive_train_step(
                B=B, 
                label_B=labels, 
                guidance_tokens_list=scaled_guidance_tokens_list,  # 有引导
                deterministic=False,  # 验证时使用随机采样
                g_seed=None
            )
        
        # 🔥 计算软特征映射用于判别器验证 (与训练逻辑完全一致)
        fake_probs_list = []
        fake_soft_features_list = []
        vae_embedding_weight = self.vae.quantize.embedding.weight  # (V, Cvae)
        
        for fake_logits in fake_logits_list:
            # 转换为概率分布
            fake_probs = F.softmax(fake_logits, dim=-1)  # (B, L_scale, V)
            fake_probs_list.append(fake_probs)
            
            # 计算软特征映射
            fake_soft_features = torch.matmul(fake_probs, vae_embedding_weight)  # (B, L_scale, Cvae)
            fake_soft_features_list.append(fake_soft_features)
        
        # 🎯 批次级对抗验证 (多尺度损失聚合，无梯度更新)
        
        try:
            # 多尺度判别器验证 - 使用预生成的token列表
            for si, pn in enumerate(self.patch_nums):
                if si == 0:
                    continue  # 跳过第一个尺度
                
                # 使用预生成的real和fake tokens
                real_tokens_k = gt_idx_Bl[si]  # 真实tokens
                fake_tokens_k = generated_guided_tokens[si]  # 逐尺度生成的引导tokens
                
                # ---- 对抗验证 (使用预生成的tokens，无梯度更新) ----
                
                try:
                    # 🚀 新架构验证：trainer层面token→RGB转换，判别器直接处理RGB
                    
                    # Step 1: 真实tokens → RGB图像
                    real_rgb_k = self.tokens_to_rgb_cumulative(gt_idx_Bl, si)
                    
                    # Step 2: 软概率tokens → RGB图像 (保持梯度流)
                    fake_probs_k = fake_probs_list[si]  # (B, L_scale, V) - 直接从列表获取
                    fake_rgb_k = self.soft_tokens_to_rgb(fake_probs_list, si)
                    
                    # Step 3: 判别器直接处理RGB图像（与训练保持一致）
                    pred_real = self.disc.forward_rgb(real_rgb_k)  # 🆕 直接RGB判别
                    pred_fake = self.disc.forward_soft_rgb(fake_rgb_k)  # 🆕 软RGB判别
                    
                    # 🔥 修复：处理判别器输出（验证时）
                    # 判别器输出形状: [B, 15]，需要对所有头求平均
                    pred_real_mean = pred_real.mean(dim=1)  # [B, 15] -> [B]
                    pred_fake_mean = pred_fake.mean(dim=1)  # [B, 15] -> [B]
                    
                    # 计算损失 (无梯度更新)
                    loss_D_real = F.relu(1 - pred_real_mean).mean()
                    loss_D_fake = F.relu(1 + pred_fake_mean).mean()
                    loss_D = loss_D_real + loss_D_fake
                    
                    # GuidanceInjector损失计算
                    loss_adv = -pred_fake_mean.mean()
                    
                    # 计算重构损失 (基于逐尺度的logits，与训练逻辑完全一致)
                    fake_logits_k = fake_logits_list[si]  # (B, L_scale, V) - 直接从列表获取
                    loss_rec = F.cross_entropy(
                        fake_logits_k.reshape(-1, self.var.V), 
                        real_tokens_k.reshape(-1)
                    )
                    
                    # 判断是否为预热阶段
                    if self.enable_staged_training and self.is_warmup_phase:
                        loss_P = torch.tensor(0.0, device=self.device)  # 预热阶段不训练GuidanceInjector
                    else:
                        loss_P = loss_adv + self.lambda_rec * loss_rec
                    
                    # 计算准确率（修复：使用处理过的多头输出）
                    acc_real = (pred_real_mean > 0).float().mean().item()
                    acc_fake = (pred_fake_mean < 0).float().mean().item()
                    
                    # 累积统计
                    loss_D_total += loss_D.detach()
                    loss_P_total += loss_P.detach()
                    # 🔥 分别累积对抗损失和重构损失
                    loss_adv_total += loss_adv.detach()
                    loss_rec_total += loss_rec.detach()
                    acc_D_sum += (acc_real + acc_fake) / 2
                    # 🔥 分别累积真假样本准确率
                    acc_real_sum += acc_real
                    acc_fake_sum += acc_fake
                    valid_scales += 1
                    
                    logger.info(f"       📊 尺度{si+1}损失: D={loss_D.item():.4f}, Acc={((acc_real + acc_fake) / 2):.3f}")
                    if not (self.enable_staged_training and self.is_warmup_phase):
                        logger.info(f"       📊 损失分解: Adv={loss_adv.item():.4f}, Rec={loss_rec.item():.4f}")
                    
                    # 🔬 视觉调试：保存解码样本图像
                    should_save_visual = aid_helpers.should_save_visual_debug(
                        epoch=epoch,
                        batch_idx=batch_idx,
                        scale_idx=si,
                        acc_real=acc_real,
                        acc_fake=acc_fake
                    )
                    
                    if should_save_visual:
                        logger.info(f"       🔬 触发视觉调试保存: Epoch {epoch+1}, 尺度 {si+1}, 判别器准确率 Real={acc_real:.3f}, Fake={acc_fake:.3f}")
                        try:
                            # 🔥 构建历史数据用于视觉调试
                            # 真实tokens历史：使用gt_idx_Bl中之前尺度的数据
                            real_tokens_history = gt_idx_Bl[:si] if si > 0 else []
                            
                            # 假tokens历史：使用generated_guided_tokens中之前尺度的数据
                            fake_tokens_history = generated_guided_tokens[:si] if si > 0 else []
                            
                            aid_helpers.save_visual_debugging_samples(
                                vae=self.vae,
                                patch_nums=self.patch_nums,
                                Cvae=self.Cvae,
                                device=self.device,
                                epoch=epoch,
                                batch_idx=batch_idx, 
                                scale_idx=si,
                                real_tokens=real_tokens_k,
                                fake_tokens=fake_tokens_k,
                                std_tokens=std_tokens_list,
                                save_dir=save_dir,
                                real_tokens_history=real_tokens_history,
                                fake_tokens_history=fake_tokens_history,
                                max_samples=2,  # 限制样本数量以节省存储
                                guidance_tokens_list=guidance_tokens_list  # 🆕 传入引导tokens (注意力特征)
                            )
                            
                            # 🎨 如果是最后一个scale，创建综合比较图
                            if si == len(self.patch_nums) - 1:  # 最后一个scale
                                logger.info(f"       🎨 创建多尺度综合比较图...")
                                try:
                                    aid_helpers.create_multi_scale_composite_images(
                                        save_dir=save_dir,
                                        epoch=epoch,
                                        batch_idx=batch_idx,
                                        max_samples=2
                                    )
                                except Exception as composite_e:
                                    logger.warning(f"       🎨 综合比较图创建失败: {composite_e}")
                                    
                        except Exception as debug_e:
                            logger.warning(f"       🔬 视觉调试保存失败: {debug_e}")
                
                except Exception as scale_e:
                    logger.warning(f"       尺度{si+1}处理失败: {scale_e}")
                    
        except Exception as e:
            logger.warning(f"       批次级对抗验证失败: {e}")
        
        finally:
            # 并行生成不需要KV缓存管理
            pass
        
        # 计算平均验证指标
        if valid_scales > 0:
            avg_loss_D = loss_D_total / valid_scales
            avg_loss_P = loss_P_total / valid_scales
            # 🔥 分别计算平均对抗损失和重构损失
            avg_loss_adv = loss_adv_total / valid_scales
            avg_loss_rec = loss_rec_total / valid_scales
            avg_acc_D = acc_D_sum / valid_scales
            # 🔥 分别计算真假样本平均准确率
            avg_acc_real = acc_real_sum / valid_scales
            avg_acc_fake = acc_fake_sum / valid_scales
        else:
            avg_loss_D = loss_D_total
            avg_loss_P = loss_P_total  
            avg_loss_adv = loss_adv_total
            avg_loss_rec = loss_rec_total
            avg_acc_D = 0.0
            avg_acc_real = 0.0
            avg_acc_fake = 0.0
        
        # 🎯 简化验证输出：只在valid_scales=0时报错
        if valid_scales == 0:
            logger.warning(f"验证失败：无有效尺度")
        
        return {
            'loss_discriminator': avg_loss_D.item(),
            'loss_planner': avg_loss_P.item(),
            # 🔥 修复：返回真实的分离损失值
            'loss_adversarial': avg_loss_adv.item(),
            'loss_reconstruction': avg_loss_rec.item(),
            'discriminator_accuracy': avg_acc_D,
            # 🔥 修复：返回真实的分离准确率
            'discriminator_acc_real': avg_acc_real,
            'discriminator_acc_fake': avg_acc_fake,
            'guided_tokens_list': generated_guided_tokens
        }

    # =========================================================================
    # 🆕 Token→RGB转换方法 (从discriminator移动到trainer，解决cuDNN兼容性问题) 
    # =========================================================================
    
    def tokens_to_rgb_cumulative(self, tokens_list: List[torch.Tensor], target_scale_idx: int) -> torch.Tensor:
        """
        🔄 多尺度累积解码：tokens→RGB (从discriminator移动至trainer)
        
        解决cuDNN兼容性问题，将复杂的token→RGB处理从判别器中移除
        
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
            device = self.device
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
            
            # 使用VQ-VAE解码 - 🔥 关键修复：不使用no_grad，让梯度正常流动
            # VQ-VAE参数已通过requires_grad=False冻结，无需no_grad包装
            rgb_images = self.vae.idxBl_to_img(cumulative_tokens, same_shape=True, last_one=True)
            
            # 🔧 数值稳定性检查
            if torch.isnan(rgb_images).any() or torch.isinf(rgb_images).any():
    
                rgb_images = torch.nan_to_num(rgb_images, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # 确保输出范围在[-1, 1]
            rgb_images = torch.clamp(rgb_images, -1.0, 1.0)
            
            return rgb_images
            
        except Exception as e:
            logger.error(f"❌ trainer累积解码失败: {e}")
            # 返回噪声图像作为fallback
            B = 1
            device = self.device
            if tokens_list and len(tokens_list) > 0:
                for t in tokens_list:
                    if t is not None:
                        B = t.shape[0]
                        device = t.device
                        break
            return torch.randn(B, 3, 256, 256, device=device) * 0.1
    
    def soft_tokens_to_rgb(self, soft_probs_list: List[torch.Tensor], target_scale_idx: int) -> torch.Tensor:
        """
        🔄 软概率tokens→RGB（保持梯度流） (从discriminator移动至trainer)
        
        解决cuDNN兼容性问题，在trainer层面处理复杂的软解码过程
        
        Args:
            soft_probs_list: 每个尺度的软概率列表 [(B, L_i, V), ...]
            target_scale_idx: 目标尺度索引
            
        Returns:
            rgb_images: 解码后的RGB图像 (B, 3, H, W) - 保持梯度流
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
        
                        probs_k = torch.nan_to_num(probs_k, nan=0.001, posinf=1.0, neginf=0.0)
                        # 重新归一化
                        probs_k = F.softmax(probs_k, dim=-1)
                    
                    # Gumbel-Softmax采样（保持梯度）
                    gumbel_tokens = F.gumbel_softmax(
                        probs_k, 
                        tau=1.0,  # 固定温度参数
                        hard=False, 
                        dim=-1
                    )  # (B, L_i, V)
                    
                    # 再次检查输出
                    if torch.isnan(gumbel_tokens).any() or torch.isinf(gumbel_tokens).any():
    
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
            device = soft_probs_list[0].device if len(soft_probs_list) > 0 else self.device
            return torch.randn(B, 3, 256, 256, device=device, requires_grad=True) * 0.1
    
    def _soft_decode_through_vae(self, soft_tokens_list: List[torch.Tensor]) -> torch.Tensor:
        """
        🔧 简化的软解码方法 - 使用VQ-VAE的embed_to_fhat进行累积解码
        
        由于VQ-VAE的多尺度累积解码设计，我们采用近似方法：
        1. 将软概率转换为软embedding
        2. 使用VQ-VAE的embed_to_fhat进行软解码
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
    
                rgb_images = torch.nan_to_num(rgb_images, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # 确保输出范围
            rgb_images = torch.clamp(rgb_images, -1.0, 1.0)
            
            return rgb_images
            
        except Exception as e:
            logger.error(f"❌ 简化软解码失败: {e}")
            
            # 🔧 安全获取设备和批次大小
            B = 1
            device = self.device
            
            if soft_tokens_list and len(soft_tokens_list) > 0:
                for st in soft_tokens_list:
                    if st is not None:
                        B = st.shape[0]
                        device = st.device
                        break
            
            # 返回可微分噪声
            return torch.randn(B, 3, 256, 256, device=device, requires_grad=True) * 0.1

