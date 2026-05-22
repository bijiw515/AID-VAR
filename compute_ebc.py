#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 EB-C (Exposure Bias - Consistency) 计算脚本

EB-C是用于量化暴露偏差(Exposure Bias)影响的度量指标。
定义为：EB-C(M, l, f_div) = CGD(M|M, l, f_div) / CGD(M|D, l, f_div)

其中：
- 分子 CGD(M|M, l): 模型依赖自身生成的(有缺陷的)前缀时的偏离程度
  → 使用模型的 **inference 方法**逐尺度自回归预测得到的分布与真实数据分布之间的差异
- 分母 CGD(M|D, l): 模型依赖真实数据(完美)前缀时的偏离程度
  → 使用模型的 **forward 方法**以真实数据为输入得到的分布与真实数据分布之间的差异
- f_div: 散度度量函数 (JS散度)

本脚本计算VAR和AID-VAR两个模型的EB-C分数，以证明AID-VAR减少了误差累积。

作者：Expert AI Developer
日期：2025-11-19
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

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入项目模块
from models.var import VAR
from models.vqvae import VQVAE
from models.guidance_injector import GuidanceInjector
from models import build_vae_var
from models.helpers import sample_with_top_k_top_p_
import dist

# 设置日志
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
    """
    🎯 EB-C (Exposure Bias - Consistency) 计算器

    实现条件生成偏差(CGD)和暴露偏差一致性(EB-C)的计算
    """

    def __init__(self,
                 var_model: VAR,
                 planner: Optional[GuidanceInjector] = None,
                 patch_nums: Tuple[int] = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
                 device: str = 'cuda',
                 cfg: float = 1.5,
                 top_k: int = 900,
                 top_p: float = 0.96,
                 more_smooth: bool = False):
        """
        初始化EB-C计算器

        Args:
            var_model: VAR模型
            planner: GuidanceInjector模型(可选，用于AID-VAR)
            patch_nums: VAR的patch数量序列
            device: 计算设备
            cfg: 分类器无关引导强度
            top_k: Top-k采样参数
            top_p: Top-p采样参数
            more_smooth: 是否启用平滑采样
        """
        self.var_model = var_model
        self.planner = planner
        self.patch_nums = patch_nums
        self.device = device

        # 采样超参数（与generate_aid_fid_samples.py和compute_ISCS.py保持一致）
        self.cfg = cfg
        self.top_k = top_k
        self.top_p = top_p
        self.more_smooth = more_smooth

        # VAE quantizer
        self.vae = var_model.vae_proxy[0]
        self.vocab_size = self.vae.vocab_size

        # 存储真实图像数据（用于generate_with_real_prefix_forward）
        self.real_images_dict = {}  # {sample_idx: (image_tensor, class_label)}

        logger.info(f"🎯 EB-C计算器初始化完成")
        logger.info(f"📏 尺度数量: {len(patch_nums)}")
        logger.info(f"📚 词汇表大小: {self.vocab_size}")
        logger.info(f"🎮 生成超参数: cfg={cfg}, top_k={top_k}, top_p={top_p}, more_smooth={more_smooth}")

    def extract_multiscale_tokens_from_images(self, images: torch.Tensor) -> List[torch.LongTensor]:
        """
        从图像提取多尺度VAR tokens

        Args:
            images: 输入图像 (B, 3, H, W) 范围[0,1]

        Returns:
            tokens_list: 每个尺度的token列表 [(B, L_0), (B, L_1), ..., (B, L_K)]
        """
        with torch.no_grad():
            # 调整图像范围：从[0,1]到[-1,1]
            if images.max() <= 1.0:
                images = images * 2.0 - 1.0

            # 确保数据类型一致性
            images = images.to(dtype=torch.float32, device=self.device)

            # 通过VAE编码器获取特征
            f = self.vae.quant_conv(self.vae.encoder(images))

            # 获取多尺度的token indices
            ms_idx_Bl = self.vae.quantize.f_to_idxBl_or_fhat(
                f, to_fhat=False, v_patch_nums=self.patch_nums
            )

            logger.debug(f"🔍 提取了 {len(ms_idx_Bl)} 个尺度的tokens")

            return ms_idx_Bl

    def compute_js_divergence(self,
                            logits1: torch.Tensor,
                            logits2: torch.Tensor,
                            eps: float = 1e-8) -> float:
        """
        计算两个logit分布之间的JS散度

        Args:
            logits1: 第一个分布的logits (B, L, V)
            logits2: 第二个分布的logits (B, L, V)
            eps: 数值稳定性参数

        Returns:
            js_div: JS散度值
        """
        # 转换为概率分布
        p = F.softmax(logits1, dim=-1)
        q = F.softmax(logits2, dim=-1)

        # 计算混合分布
        m = 0.5 * (p + q)

        # 计算KL散度 KL(P||M) 和 KL(Q||M)
        kl_pm = F.kl_div(m.log(), p, reduction='none').sum(dim=-1)
        kl_qm = F.kl_div(m.log(), q, reduction='none').sum(dim=-1)

        # JS散度 = 0.5 * (KL(P||M) + KL(Q||M))
        js_div = 0.5 * (kl_pm + kl_qm)

        # 平均所有位置和批次
        js_div_mean = js_div.mean().item()

        return js_div_mean

    def compute_token_distribution_divergence(self,
                                             generated_tokens: torch.LongTensor,
                                             real_tokens: torch.LongTensor) -> float:
        """
        计算生成tokens和真实tokens的分布散度

        Args:
            generated_tokens: 生成的tokens (B, L)
            real_tokens: 真实的tokens (B, L)

        Returns:
            divergence: 分布散度
        """
        # 将tokens转换为分布
        B, L = generated_tokens.shape

        # 计算生成tokens的频率分布
        gen_hist = torch.histc(
            generated_tokens.float(),
            bins=self.vocab_size,
            min=0,
            max=self.vocab_size-1
        )
        gen_dist = gen_hist / gen_hist.sum()

        # 计算真实tokens的频率分布
        real_hist = torch.histc(
            real_tokens.float(),
            bins=self.vocab_size,
            min=0,
            max=self.vocab_size-1
        )
        real_dist = real_hist / real_hist.sum()

        # 添加平滑避免log(0)
        eps = 1e-8
        gen_dist = gen_dist + eps
        real_dist = real_dist + eps

        # 重新归一化
        gen_dist = gen_dist / gen_dist.sum()
        real_dist = real_dist / real_dist.sum()

        # 计算JS散度
        m = 0.5 * (gen_dist + real_dist)
        kl_gen = F.kl_div(m.log(), gen_dist, reduction='sum')
        kl_real = F.kl_div(m.log(), real_dist, reduction='sum')
        js_div = 0.5 * (kl_gen + kl_real)

        return js_div.item()

    def generate_with_real_prefix_forward(self,
                                         real_tokens_list: List[torch.LongTensor],
                                         class_labels: torch.LongTensor,
                                         use_planner: bool = False) -> List[torch.LongTensor]:
        """
        使用真实tokens和forward方法一次性生成所有尺度的tokens

        用于计算 CGD(M|D, l) - 分母
        使用真实数据作为输入，调用VAR的forward方法获得所有尺度的logits并采样

        **与trainer.py的train_step方法完全一致的处理方式**

        Args:
            real_tokens_list: 真实tokens列表 (所有尺度) [(B, L_0), (B, L_1), ..., (B, L_K)]
            class_labels: 类别标签 (B,)
            use_planner: 是否使用GuidanceInjector引导（在forward模式下不使用）

        Returns:
            all_scale_generated_tokens: 所有尺度的生成tokens列表 [(B, L_0), ..., (B, L_K)]
        """
        B = class_labels.shape[0]

        # 使用VAE的idxBl_to_var_input方法构建VAR的输入（与trainer.py完全一致）
        x_BLCv_wo_first_l = self.vae.quantize.idxBl_to_var_input(real_tokens_list)

        with torch.no_grad():
            # 调用VAR的forward方法获取logits（与trainer.py完全一致）
            logits_BLV = self.var_model(class_labels, x_BLCv_wo_first_l)

            # 对每个尺度进行采样
            all_scale_generated_tokens = []
            start_idx = 0
            for scale_idx, pn in enumerate(self.patch_nums):
                scale_L = pn * pn
                end_idx = start_idx + scale_L

                # 提取当前尺度的logits
                scale_logits = logits_BLV[:, start_idx:end_idx, :]  # (B, L_scale, V)

                # 使用标准的sample_with_top_k_top_p_采样
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
        """
        使用模型自回归inference方法一次性生成所有尺度的tokens

        用于计算 CGD(M|M, l) - 分子
        完全按照VAR的aid_guided_inference实现，自回归生成所有尺度，包括CFG

        Args:
            class_labels: 类别标签 (B,)
            use_planner: 是否使用GuidanceInjector引导
            g_seed: 随机种子（用于可重复性）

        Returns:
            all_scale_generated_tokens: 所有尺度的生成tokens列表 [(B, L_0), ..., (B, L_K)]
        """
        B = class_labels.shape[0]

        # 设置随机数生成器（与aid_guided_inference一致）
        if g_seed is not None:
            rng = torch.Generator(device=self.device)
            rng.manual_seed(g_seed)
        else:
            rng = None

        with torch.no_grad():
            # CFG准备：创建条件和无条件的双倍批次（与aid_guided_inference完全一致）
            sos = cond_BD = self.var_model.class_emb(
                torch.cat((class_labels,
                          torch.full_like(class_labels, fill_value=self.var_model.num_classes)),
                          dim=0)
            )

            # 位置和层级嵌入
            lvl_pos = self.var_model.lvl_embed(self.var_model.lvl_1L) + self.var_model.pos_1LC
            next_token_map = sos.unsqueeze(1).expand(2 * B, self.var_model.first_l, -1) + \
                            self.var_model.pos_start.expand(2 * B, self.var_model.first_l, -1) + \
                            lvl_pos[:, :self.var_model.first_l]

            cur_L = 0
            f_hat = sos.new_zeros(B, self.vae.Cvae, self.patch_nums[-1], self.patch_nums[-1])

            all_scale_generated_tokens = []

            # 启用KV缓存
            for b in self.var_model.blocks:
                b.attn.kv_caching(True)

            # 逐尺度生成所有尺度
            for si, pn in enumerate(self.patch_nums):
                ratio = si / self.var_model.num_stages_minus_1  # 与aid_guided_inference一致
                cur_L += pn * pn

                # 生成引导tokens（从第二个尺度开始）
                guidance_tokens = None
                if use_planner and self.planner is not None and si > 0:
                    prev_tokens = all_scale_generated_tokens[si - 1]

                    # 转换为embedding features
                    prev_embed = F.embedding(prev_tokens, self.vae.quantize.embedding.weight.detach())
                    prev_features = self.var_model.word_embed(prev_embed)

                    # 生成引导tokens
                    guidance_tokens = self.planner(prev_features, target_patch_num=pn)
                    guidance_tokens = 0.001 * guidance_tokens

                    # CFG：为引导tokens也复制一份（与aid_guided_inference一致）
                    guidance_tokens = torch.cat([guidance_tokens, guidance_tokens], dim=0)

                cond_BD_or_gss = self.var_model.shared_ada_lin(cond_BD)
                x = next_token_map

                # 注入引导（与aid_guided_inference一致）
                if guidance_tokens is not None:
                    if guidance_tokens.shape[0] == x.shape[0] and guidance_tokens.shape[1] == x.shape[1]:
                        x = x + guidance_tokens

                # Transformer forward
                for b in self.var_model.blocks:
                    x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)

                logits_BlV = self.var_model.get_logits(x, cond_BD)

                # 应用CFG（与aid_guided_inference完全一致）
                t = self.cfg * ratio
                logits_BlV = (1 + t) * logits_BlV[:B] - t * logits_BlV[B:]

                # 采样（与aid_guided_inference一致）
                idx_Bl = sample_with_top_k_top_p_(
                    logits_BlV,
                    rng=rng,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    num_samples=1
                )[:, :, 0]  # (B, pn*pn)

                all_scale_generated_tokens.append(idx_Bl)

                # 如果未到最后一个尺度，准备下一尺度的输入
                if si < len(self.patch_nums) - 1:
                    if not self.more_smooth:
                        h_BChw = self.vae.quantize.embedding(idx_Bl)
                    else:
                        gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)
                        # 注意：这里需要gumbel_softmax_with_rng，但为了简化，我们默认不使用more_smooth
                        h_BChw = self.vae.quantize.embedding(idx_Bl)

                    h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.vae.Cvae, pn, pn)
                    f_hat, next_token_map = self.vae.quantize.get_next_autoregressive_input(
                        si, len(self.patch_nums), f_hat, h_BChw
                    )

                    next_token_map = next_token_map.view(B, self.vae.Cvae, -1).transpose(1, 2)
                    next_token_map = self.var_model.word_embed(next_token_map) + \
                                   lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                    # CFG：双倍批次大小（与aid_guided_inference一致）
                    next_token_map = next_token_map.repeat(2, 1, 1)

            # 关闭KV缓存
            for b in self.var_model.blocks:
                b.attn.kv_caching(False)

        return all_scale_generated_tokens

    def compute_cgd_for_scale(self,
                            generated_tokens_list: List[torch.LongTensor],
                            real_tokens_list: List[torch.LongTensor],
                            target_scale_idx: int) -> float:
        """
        计算特定尺度的条件生成偏差(CGD)

        Args:
            generated_tokens_list: 生成的tokens列表（所有尺度）[(B, L_0), ..., (B, L_K)]
            real_tokens_list: 真实tokens列表（所有尺度）[(B, L_0), ..., (B, L_K)]
            target_scale_idx: 目标尺度索引

        Returns:
            cgd: 条件生成偏差值
        """
        generated_tokens_at_scale = generated_tokens_list[target_scale_idx]  # (B, L_scale)
        real_tokens_at_scale = real_tokens_list[target_scale_idx]  # (B, L_scale)

        # 计算生成分布与真实分布的散度
        cgd = self.compute_token_distribution_divergence(
            generated_tokens_at_scale, real_tokens_at_scale
        )

        logger.debug(f"🔍 尺度 {target_scale_idx}, CGD={cgd:.6f}")

        return cgd

    def compute_ebc_for_model(self,
                             real_images: torch.Tensor,
                             real_class_labels: torch.LongTensor,
                             num_model_samples: int,
                             use_planner: bool = False,
                             batch_size: int = 10) -> Dict:
        """
        计算模型的完整EB-C指标

        新逻辑：
        1. 提取真实图像的多尺度tokens（用于分母CGD(M|D)）
        2. 分批生成CGD(M|D) tokens使用真实前缀+forward
        3. 分批生成CGD(M|M) tokens使用模型自回归inference（样本数为num_model_samples）
        4. 对每个尺度调用compute_cgd_for_scale计算CGD
        5. 计算EB-C = CGD(M|M, l) / CGD(M|D, l)

        Args:
            real_images: 真实图像 (N_real, 3, H, W) - 用于分母CGD(M|D)
            real_class_labels: 真实图像的类别标签 (N_real,)
            num_model_samples: 分子CGD(M|M)要生成的样本数
            use_planner: 是否使用planner
            batch_size: 批处理大小

        Returns:
            results: EB-C结果字典
        """
        model_name = "AID-VAR" if use_planner else "VAR"
        logger.info(f"🚀 计算 {model_name} 的EB-C指标...")

        num_real_samples = len(real_images)
        logger.info(f"📊 分母CGD(M|D)样本数: {num_real_samples}, 分子CGD(M|M)样本数: {num_model_samples}, 批大小: {batch_size}")

        # ===== 第一步：提取真实图像的多尺度tokens =====
        logger.info(f"🔄 提取真实图像的多尺度tokens...")
        all_real_tokens = []
        for i in tqdm(range(0, num_real_samples, batch_size), desc="提取真实tokens"):
            batch_images = real_images[i:i+batch_size]
            batch_tokens = self.extract_multiscale_tokens_from_images(batch_images)
            all_real_tokens.append(batch_tokens)

        # 合并所有批次的tokens，按尺度组织
        real_tokens_list = []
        for scale_idx in range(len(self.patch_nums)):
            scale_tokens = torch.cat([batch_tokens[scale_idx] for batch_tokens in all_real_tokens], dim=0)
            real_tokens_list.append(scale_tokens)

        logger.info(f"✅ 提取完成，每个尺度的tokens形状: {[t.shape for t in real_tokens_list]}")

        # ===== 第二步：生成CGD(M|D, l)的tokens - 使用真实前缀 + forward =====
        logger.info(f"🔹 生成 CGD(M|D) tokens - 使用真实tokens + forward...")
        all_cgd_md_tokens = []  # 存储所有批次的CGD(M|D) tokens
        for i in tqdm(range(0, num_real_samples, batch_size), desc="生成CGD(M|D) tokens"):
            batch_real_tokens = [tokens[i:i+batch_size] for tokens in real_tokens_list]
            batch_labels = real_class_labels[i:i+batch_size]

            # 一次性生成所有尺度的tokens（与trainer.py的train_step一致）
            batch_cgd_md_tokens = self.generate_with_real_prefix_forward(
                real_tokens_list=batch_real_tokens,
                class_labels=batch_labels,
                use_planner=False,  # forward模式不使用planner
            )  # [(B, L_0), ..., (B, L_K)]

            all_cgd_md_tokens.append(batch_cgd_md_tokens)

        # 合并所有批次，按尺度组织
        cgd_md_tokens_list = []
        for scale_idx in range(len(self.patch_nums)):
            scale_tokens = torch.cat([batch_tokens[scale_idx] for batch_tokens in all_cgd_md_tokens], dim=0)
            cgd_md_tokens_list.append(scale_tokens)

        logger.info(f"✅ CGD(M|D) tokens生成完成")

        # ===== 第三步：生成CGD(M|M, l)的tokens - 使用模型自回归inference =====
        logger.info(f"🔹 生成 CGD(M|M) tokens - 使用模型自回归inference...")

        # 生成num_model_samples个类别标签（循环分配1000个类别）
        model_class_labels = torch.arange(num_model_samples, device=self.device) % 1000

        all_cgd_mm_tokens = []  # 存储所有批次的CGD(M|M) tokens
        for i in tqdm(range(0, num_model_samples, batch_size), desc="生成CGD(M|M) tokens"):
            batch_labels = model_class_labels[i:i+batch_size]

            # 使用class_id作为随机种子（以第一个样本的类别为准）
            g_seed = batch_labels[0].item()

            # 一次性生成所有尺度的tokens
            batch_cgd_mm_tokens = self.generate_with_model_prefix_inference(
                class_labels=batch_labels,
                use_planner=use_planner,
                g_seed=g_seed
            )  # [(B, L_0), ..., (B, L_K)]

            all_cgd_mm_tokens.append(batch_cgd_mm_tokens)

        # 合并所有批次，按尺度组织
        cgd_mm_tokens_list = []
        for scale_idx in range(len(self.patch_nums)):
            scale_tokens = torch.cat([batch_tokens[scale_idx] for batch_tokens in all_cgd_mm_tokens], dim=0)
            cgd_mm_tokens_list.append(scale_tokens)

        logger.info(f"✅ CGD(M|M) tokens生成完成")

        # ===== 第四步：计算每个尺度的CGD和EB-C =====
        results = {
            'scale_cgd_real_prefix': {},  # CGD(M|D, l) - 分母
            'scale_cgd_model_prefix': {},  # CGD(M|M, l) - 分子
            'scale_ebc': {},  # EB-C per scale
            'model_name': model_name
        }

        for scale_idx in range(len(self.patch_nums)):
            logger.info(f"📊 计算尺度 {scale_idx}/{len(self.patch_nums)-1} "
                       f"(patch_num={self.patch_nums[scale_idx]})...")

            # 1. 计算 CGD(M|D, l)
            cgd_real_prefix = self.compute_cgd_for_scale(
                generated_tokens_list=cgd_md_tokens_list,
                real_tokens_list=real_tokens_list,
                target_scale_idx=scale_idx
            )
            results['scale_cgd_real_prefix'][scale_idx] = cgd_real_prefix

            # 2. 计算 CGD(M|M, l) (第一个尺度除外)
            if scale_idx > 0:
                cgd_model_prefix = self.compute_cgd_for_scale(
                    generated_tokens_list=cgd_mm_tokens_list,
                    real_tokens_list=real_tokens_list,
                    target_scale_idx=scale_idx
                )
                results['scale_cgd_model_prefix'][scale_idx] = cgd_model_prefix

                # 3. 计算EB-C
                if cgd_real_prefix > 1e-8:  # 避免除零
                    ebc = cgd_model_prefix / cgd_real_prefix
                else:
                    ebc = float('inf')

                results['scale_ebc'][scale_idx] = ebc

                logger.info(f"  ✅ 尺度 {scale_idx}: "
                          f"CGD(M|D)={cgd_real_prefix:.6f}, "
                          f"CGD(M|M)={cgd_model_prefix:.6f}, "
                          f"EB-C={ebc:.4f}")
            else:
                # 第一个尺度没有前缀，CGD(M|M)不适用
                results['scale_cgd_model_prefix'][scale_idx] = None
                results['scale_ebc'][scale_idx] = None
                logger.info(f"  ℹ️ 尺度 {scale_idx}: CGD(M|D)={cgd_real_prefix:.6f} "
                          f"(首尺度，无EB-C)")

        # 计算聚合的EB-C (忽略第一个尺度)
        valid_ebc_scores = [ebc for ebc in results['scale_ebc'].values() if ebc is not None and ebc != float('inf')]
        if len(valid_ebc_scores) > 0:
            results['mean_ebc'] = np.mean(valid_ebc_scores)
            results['std_ebc'] = np.std(valid_ebc_scores)
        else:
            results['mean_ebc'] = None
            results['std_ebc'] = None

        logger.info(f"🎯 {model_name} 平均EB-C: {results['mean_ebc']:.4f} ± {results['std_ebc']:.4f}")

        return results


class EBCExperiment:
    """
    🎯 EB-C实验管理器
    """

    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

        # 设置优化配置
        self._setup_optimization()

        # 加载模型
        self._load_models()

        # 初始化EB-C计算器
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

        logger.info(f"🚀 EB-C实验初始化完成")

    def _setup_optimization(self):
        """设置优化配置"""
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        tf32 = True
        torch.backends.cudnn.allow_tf32 = bool(tf32)
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.set_float32_matmul_precision('high' if tf32 else 'highest')

        logger.info(f"🔧 优化配置完成")

    def _load_models(self):
        """加载VAR模型和GuidanceInjector"""
        logger.info(f"🔄 加载模型...")

        # 确定VAR检查点路径
        if self.args.var_ckpt is None:
            self.args.var_ckpt = f'checkpoints/var_d{self.args.model_depth}.pth'

        # 构建VAE和VAR模型
        patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
        logger.info(f"🏗️ 构建VAE和VAR模型 (深度={self.args.model_depth})...")
        vae, var = build_vae_var(
            V=4096, Cvae=32, ch=160, share_quant_resi=4,
            device=self.device, patch_nums=patch_nums,
            num_classes=1000, depth=self.args.model_depth, shared_aln=False,
        )

        # 加载检查点
        logger.info(f"🔄 加载VAE检查点: {self.args.vae_ckpt}")
        vae.load_state_dict(torch.load(self.args.vae_ckpt, map_location='cpu'), strict=True)

        logger.info(f"🔄 加载VAR检查点: {self.args.var_ckpt}")
        var.load_state_dict(torch.load(self.args.var_ckpt, map_location='cpu'), strict=True)

        vae.eval()
        var.eval()
        for p in vae.parameters(): p.requires_grad_(False)
        for p in var.parameters(): p.requires_grad_(False)

        self.vae = vae
        self.var = var

        # 加载GuidanceInjector
        if self.args.planner_ckpt and os.path.exists(self.args.planner_ckpt):
            logger.info(f"🔄 加载GuidanceInjector...")

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

            logger.info(f"✅ GuidanceInjector加载完成")
        else:
            self.planner = None
            logger.info(f"⚠️ 未加载GuidanceInjector，仅评估基础VAR模型")

    def load_real_images(self) -> Tuple[torch.Tensor, torch.LongTensor]:
        """
        加载真实图像数据（用于分母CGD(M|D)计算）

        策略：从真实图像数据中随机采样，限制为num_real_samples个样本
        """
        logger.info(f"📁 加载真实图像数据: {self.args.real_images_path}")

        if self.args.real_images_path.endswith('.npz'):
            data = np.load(self.args.real_images_path)

            if 'images' in data:
                real_images = torch.from_numpy(data['images']).float()
            elif 'arr_0' in data:
                real_images = torch.from_numpy(data['arr_0']).float()
            else:
                key = list(data.keys())[0]
                real_images = torch.from_numpy(data[key]).float()

            # 确保格式正确
            if len(real_images.shape) == 4:
                if real_images.shape[3] == 3:
                    real_images = real_images.permute(0, 3, 1, 2)

            if real_images.max() > 1.0:
                real_images = real_images / 255.0
        else:
            raise ValueError("请提供.npz格式的真实图像数据")

        # 限制数量到num_real_samples（用于分母计算）
        if len(real_images) > self.args.num_real_samples:
            # 设置随机种子以确保可重复性
            torch.manual_seed(42)
            indices = torch.randperm(len(real_images))[:self.args.num_real_samples]
            real_images = real_images[indices]

        # 生成类别标签 (假设均匀分布在1000个类别中)
        num_images = len(real_images)
        class_labels = torch.arange(num_images) % 1000  # 循环分配类别标签

        logger.info(f"✅ 加载 {len(real_images)} 张真实图像（用于分母CGD(M|D)计算）")

        return real_images, class_labels

    def run_experiment(self):
        """运行完整的EB-C评估实验"""
        logger.info(f"🚀 开始EB-C评估实验")

        # 加载真实图像（用于分母CGD(M|D)）
        real_images, real_class_labels = self.load_real_images()
        real_images = real_images.to(self.device)
        real_class_labels = real_class_labels.to(self.device)

        results = {}

        # 评估基础VAR
        logger.info(f"📊 评估基础VAR模型...")
        results['var'] = self.ebc_calculator.compute_ebc_for_model(
            real_images=real_images,
            real_class_labels=real_class_labels,
            num_model_samples=self.args.num_samples,  # 分子使用num_samples
            use_planner=False,
            batch_size=self.args.batch_size
        )

        # 评估AID-VAR (如果有planner)
        if self.planner is not None:
            logger.info(f"🚀 评估AID-VAR模型...")
            results['aid_var'] = self.ebc_calculator.compute_ebc_for_model(
                real_images=real_images,
                real_class_labels=real_class_labels,
                num_model_samples=self.args.num_samples,  # 分子使用num_samples
                use_planner=True,
                batch_size=self.args.batch_size
            )

            # 比较结果
            var_ebc = results['var']['mean_ebc']
            aid_ebc = results['aid_var']['mean_ebc']
            improvement = ((var_ebc - aid_ebc) / var_ebc) * 100

            logger.info(f"📊 EB-C对比:")
            logger.info(f"  📈 VAR: {var_ebc:.4f}")
            logger.info(f"  🚀 AID-VAR: {aid_ebc:.4f}")
            logger.info(f"  📉 改善: {improvement:.2f}%")

            if aid_ebc < var_ebc:
                logger.info(f"  ✅ AID-VAR成功减少了暴露偏差!")
            else:
                logger.info(f"  ⚠️ AID-VAR未能减少暴露偏差")

        # 保存和可视化结果
        self._save_results(results)
        self._visualize_results(results)

        logger.info(f"✅ EB-C评估实验完成")

        return results

    def _save_results(self, results: Dict):
        """保存实验结果"""
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        # 保存JSON结果
        results_path = output_dir / 'ebc_results.json'

        # 转换numpy类型为Python原生类型
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

        logger.info(f"💾 结果已保存: {results_path}")

    def _visualize_results(self, results: Dict):
        """可视化EB-C结果"""
        output_dir = Path(self.args.output_dir)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 1. 各尺度EB-C对比
        ax1 = axes[0]

        var_scales = sorted([k for k in results['var']['scale_ebc'].keys()
                           if results['var']['scale_ebc'][k] is not None])
        var_ebc_scores = [results['var']['scale_ebc'][s] for s in var_scales]

        ax1.plot(var_scales, var_ebc_scores, 'o-', label='VAR', linewidth=2, markersize=8)

        if 'aid_var' in results:
            aid_ebc_scores = [results['aid_var']['scale_ebc'][s] for s in var_scales]
            ax1.plot(var_scales, aid_ebc_scores, 's-', label='AID-VAR', linewidth=2, markersize=8)

        ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='EB-C=1 (基线)')
        ax1.set_xlabel('尺度索引', fontsize=12)
        ax1.set_ylabel('EB-C值', fontsize=12)
        ax1.set_title('各尺度EB-C对比', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 平均EB-C对比
        ax2 = axes[1]

        models = ['VAR']
        mean_ebcs = [results['var']['mean_ebc']]

        if 'aid_var' in results:
            models.append('AID-VAR')
            mean_ebcs.append(results['aid_var']['mean_ebc'])

        bars = ax2.bar(models, mean_ebcs, color=['skyblue', 'lightcoral'][:len(models)], alpha=0.7)
        ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)

        # 添加数值标签
        for bar, score in zip(bars, mean_ebcs):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{score:.4f}', ha='center', va='bottom', fontweight='bold')

        ax2.set_ylabel('平均EB-C', fontsize=12)
        ax2.set_title('平均EB-C对比 (EB-C<1表示更好)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(output_dir / 'ebc_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"📊 可视化结果已保存: {output_dir}/ebc_comparison.png")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='EB-C (Exposure Bias - Consistency) 计算脚本')

    # 模型参数
    parser.add_argument('--vae_ckpt', type=str, default='vae_ch160v4096z32.pth',
                       help='VAE模型检查点路径')
    parser.add_argument('--var_ckpt', type=str, default=None,
                       help='VAR模型检查点路径')
    parser.add_argument('--planner_ckpt', type=str, default=None,
                       help='GuidanceInjector检查点路径(可选)')
    parser.add_argument('--model_depth', type=int, default=16, choices=[16, 20, 24, 30],
                       help='VAR模型深度')

    # 数据参数
    parser.add_argument('--real_images_path', type=str, required=True,
                       help='真实图像数据路径(.npz文件)')
    parser.add_argument('--num_samples', type=int, default=50000,
                       help='分子CGD(M|M)的样本数量（默认50000，与compute_ISCS.py一致：1000类×50样本/类）')
    parser.add_argument('--num_real_samples', type=int, default=10000,
                       help='分母CGD(M|D)使用的真实图像数量（默认10000，与VIRTUAL_imagenet256_labeled.npz的样本数一致）')
    parser.add_argument('--batch_size', type=int, default=50,
                       help='批处理大小（默认50，与compute_ISCS.py一致）')

    # 生成超参数（与compute_ISCS.py保持一致）
    parser.add_argument('--cfg', type=float, default=1.5,
                       help='分类器无关引导强度（与generate_aid_fid_samples.py一致）')
    parser.add_argument('--top_k', type=int, default=900,
                       help='Top-k采样参数（与generate_aid_fid_samples.py一致）')
    parser.add_argument('--top_p', type=float, default=0.96,
                       help='Top-p采样参数（与generate_aid_fid_samples.py一致）')
    parser.add_argument('--more_smooth', action='store_true',
                       help='启用more_smooth提升视觉质量（与generate_aid_fid_samples.py一致）')

    # 系统参数
    parser.add_argument('--device', type=str, default='cuda',
                       help='计算设备')
    parser.add_argument('--output_dir', type=str, default='./ebc_results',
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

    # 初始化分布式
    if not dist.initialized():
        dist.initialize()

    logger.info("="*80)
    logger.info("🎯 EB-C (Exposure Bias - Consistency) 评估实验")
    logger.info("="*80)
    logger.info(f"📋 实验配置:")
    logger.info(f"  • VAR模型深度: {args.model_depth}")
    logger.info(f"  • VAE检查点: {args.vae_ckpt}")
    logger.info(f"  • VAR检查点: {args.var_ckpt if args.var_ckpt else f'checkpoints/var_d{args.model_depth}.pth'}")
    logger.info(f"  • GuidanceInjector检查点: {args.planner_ckpt if args.planner_ckpt else '未使用'}")
    logger.info(f"  • 真实图像路径: {args.real_images_path}")
    logger.info(f"  • 分母CGD(M|D)样本数: {args.num_real_samples} (使用真实图像)")
    logger.info(f"  • 分子CGD(M|M)样本数: {args.num_samples} (模型自回归生成)")
    logger.info(f"  • 批处理大小: {args.batch_size}")
    logger.info(f"  • 生成超参数: cfg={args.cfg}, top_k={args.top_k}, top_p={args.top_p}, more_smooth={args.more_smooth}")
    logger.info(f"  • 生成策略: 每批{args.batch_size}个样本，以class_id为随机种子")
    logger.info(f"  • 输出目录: {args.output_dir}")
    logger.info("="*80)

    try:
        # 创建实验管理器
        experiment = EBCExperiment(args)

        # 运行实验
        results = experiment.run_experiment()

        logger.info("="*80)
        logger.info("✅ EB-C评估实验成功完成!")
        logger.info("="*80)

    except Exception as e:
        logger.error(f"❌ 实验失败: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
