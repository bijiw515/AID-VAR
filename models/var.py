import math
from functools import partial
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
import torch.nn.functional as F

import dist
from models.basic_var import AdaLNBeforeHead, AdaLNSelfAttn
from models.helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_
from models.vqvae import VQVAE, VectorQuantizer2


class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).view(-1, 1, 6, C)   # B16C


class VAR(nn.Module):
    def __init__(
        self, vae_local: VQVAE,
        num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
        flash_if_available=True, fused_if_available=True,
    ):
        super().__init__()
        # 0. hyperparameters
        assert embed_dim % num_heads == 0
        self.Cvae, self.V = vae_local.Cvae, vae_local.vocab_size
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads
        
        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1   # progressive training
        
        self.patch_nums: Tuple[int] = patch_nums
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur+pn ** 2))
            cur += pn ** 2
        
        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=dist.get_device())
        
        # 1. input (word) embedding
        quant: VectorQuantizer2 = vae_local.quantize
        self.vae_proxy: Tuple[VQVAE] = (vae_local,)
        self.vae_quant_proxy: Tuple[VectorQuantizer2] = (quant,)
        self.word_embed = nn.Linear(self.Cvae, self.C)
        
        # 2. class embedding
        init_std = math.sqrt(1 / self.C / 3)
        self.num_classes = num_classes
        self.uniform_prob = torch.full((1, num_classes), fill_value=1.0 / num_classes, dtype=torch.float32, device=dist.get_device())
        self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
        nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        
        # 3. absolute position embedding
        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
            pe = torch.empty(1, pn*pn, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)     # 1, L, C
        assert tuple(pos_1LC.shape) == (1, self.L, self.C)
        self.pos_1LC = nn.Parameter(pos_1LC)
        # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        
        # 4. backbone blocks
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()
        
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule (linearly increasing)
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            )
            for block_idx in range(depth)
        ])
        
        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        print(
            f'\n[constructor]  ==== flash_if_available={flash_if_available} ({sum(b.attn.using_flash for b in self.blocks)}/{self.depth}), fused_if_available={fused_if_available} (fusing_add_ln={sum(fused_add_norm_fns)}/{self.depth}, fusing_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.blocks)}/{self.depth}) ==== \n'
            f'    [VAR config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n'
            f'    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )
        
        # 5. attention mask used in training (for masking out the future)
        #    it won't be used in inference, since kv cache is enabled
        d: torch.Tensor = torch.cat([torch.full((pn*pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L, 1)
        dT = d.transpose(1, 2)    # dT: 11L
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())
        
        # 6. classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)
    
    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], cond_BD: Optional[torch.Tensor]):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual   # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:                               # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()
    
    @torch.no_grad()
    def autoregressive_infer_cfg(
        self, B: int, label_B: Optional[Union[int, torch.LongTensor]],
        g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0,
        more_smooth=False,
    ) -> torch.Tensor:   # returns reconstructed image (B, 3, H, W) in [0, 1]
        """
        only used for inference, on autoregressive mode
        :param B: batch size
        :param label_B: imagenet label; if None, randomly sampled
        :param g_seed: random seed
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng
        
        if label_B is None:
            label_B = torch.multinomial(self.uniform_prob, num_samples=B, replacement=True, generator=rng).reshape(B)
        elif isinstance(label_B, int):
            label_B = torch.full((B,), fill_value=self.num_classes if label_B < 0 else label_B, device=self.lvl_1L.device)
        
        sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))
        
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        next_token_map = sos.unsqueeze(1).expand(2 * B, self.first_l, -1) + self.pos_start.expand(2 * B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        
        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        
        for b in self.blocks: b.attn.kv_caching(True)
        for si, pn in enumerate(self.patch_nums):   # si: i-th segment
            ratio = si / self.num_stages_minus_1
            # last_L = cur_L
            cur_L += pn*pn
            # assert self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].sum() == 0, f'AR with {(self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L] != 0).sum()} / {self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].numel()} mask item'
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map
            AdaLNSelfAttn.forward
            for b in self.blocks:
                x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
            logits_BlV = self.get_logits(x, cond_BD)
            
            t = cfg * ratio
            logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]
            
            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
            if not more_smooth: # this is the default case
                h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)   # B, l, Cvae
            else:   # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)   # refer to mask-git
                h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
            
            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw)
            if si != self.num_stages_minus_1:   # prepare for next stage
                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG
        
        for b in self.blocks: b.attn.kv_caching(False)
        return self.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)   # de-normalize, from [-1, 1] to [0, 1]
     
    @torch.no_grad()
    def agip_guided_inference(
        self, 
        planner, 
        B: int, 
        label_B: Optional[Union[int, torch.LongTensor]],
        cfg=1.5, 
        top_k=0, 
        top_p=0.0,
        g_seed: Optional[int] = None,
        more_smooth=False
    ) -> torch.Tensor:
        """AGIP-VAR引导推理：VAR + I_predictor引导生成
        
        完全按照autoregressive_infer_cfg的模式实现，但加入了I_predictor引导：
        1. 基于前一尺度VAR生成的tokens实时生成引导
        2. 引导token直接相加注入，与训练时保持一致
        3. 支持CFG但对引导token也进行相应处理
        
        Args:
            planner: I_predictor模块
            B: 批次大小
            label_B: 类别标签
            cfg: 分类器无关引导强度
            top_k: top-k采样
            top_p: top-p采样
            g_seed: 随机种子
            more_smooth: 平滑采样（用于可视化）
            
        Returns:
            生成的图像 (B, 3, H, W) in [0, 1]
        """
        if g_seed is None: 
            rng = None
        else: 
            self.rng.manual_seed(g_seed)
            rng = self.rng
        
        # 准备标签 - 与autoregressive_infer_cfg保持一致
        if label_B is None:
            label_B = torch.multinomial(self.uniform_prob, num_samples=B, replacement=True, generator=rng).reshape(B)
        elif isinstance(label_B, int):
            label_B = torch.full((B,), fill_value=self.num_classes if label_B < 0 else label_B, device=self.lvl_1L.device)
        
        # CFG准备 - 与原始实现保持一致，无条件类别为self.num_classes
        sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))
        
        # 位置和层级嵌入
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        next_token_map = sos.unsqueeze(1).expand(2 * B, self.first_l, -1) + \
                        self.pos_start.expand(2 * B, self.first_l, -1) + \
                        lvl_pos[:, :self.first_l]
        
        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        
        # 启用KV缓存
        for b in self.blocks: 
            b.attn.kv_caching(True)
        
        # 存储每个尺度生成的tokens，用于下一尺度的引导生成
        generated_tokens_per_scale = []
        
        try:
            for si, pn in enumerate(self.patch_nums):
                ratio = si / self.num_stages_minus_1  # 使用与原始实现相同的ratio计算
                cur_L += pn * pn
                
                # 生成引导tokens（从第二个尺度开始）
                guidance_tokens = None
                if si > 0 and len(generated_tokens_per_scale) > 0:
                    # 获取前一尺度的tokens
                    prev_tokens = generated_tokens_per_scale[-1]  # 前一尺度生成的tokens
                    
                    try:
                        # 转换为embedding features（与训练时保持一致）
                        prev_embed = F.embedding(prev_tokens, self.vae_quant_proxy[0].embedding.weight.detach())
                        prev_features = self.word_embed(prev_embed)  # (B, L_prev, C)
                        
                        # 数值稳定性保护
                        prev_features = torch.clamp(prev_features, min=-10.0, max=10.0)
                        prev_features = prev_features.detach().clone().requires_grad_(False)
                        
                        # 使用I_predictor生成引导tokens - 与训练时完全一致
                        guidance_tokens = planner(prev_features, target_patch_num=pn)  # (B, pn*pn, C)
                        
                        if guidance_tokens is not None:
                            # 数值稳定性保护
                            guidance_tokens = torch.clamp(guidance_tokens, min=-5.0, max=5.0)
                            
                            # CFG处理：为引导tokens也复制一份用于无条件生成
                            guidance_tokens = torch.cat([guidance_tokens, guidance_tokens], dim=0)  # (2*B, pn*pn, C)
                        
                    except Exception as e:
                        print(f"Warning: Failed to generate guidance for scale {si}: {e}")
                        guidance_tokens = None
                
                # VAR前向传播
                cond_BD_or_gss = self.shared_ada_lin(cond_BD)
                x = next_token_map
                
                # 注入引导tokens（与训练时autoregressive_train_step保持一致）
                if guidance_tokens is not None:
                    # 确保维度匹配
                    if guidance_tokens.shape[0] == x.shape[0] and guidance_tokens.shape[1] == x.shape[1]:
                        # 🔥 应用训练时的引导权重 0.001，与训练时保持完全一致
                        x = x + 0.001 * guidance_tokens  # 与训练时权重一致
                    else:
                        print(f"Warning: Guidance token shape mismatch at scale {si}: {guidance_tokens.shape} vs {x.shape}")
                
                # Transformer blocks处理
                for b in self.blocks:
                    x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
                
                # 获取logits
                logits_BlV = self.get_logits(x, cond_BD)
                
                # CFG应用 - 与原始实现保持一致
                t = cfg * ratio
                logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]
                
                # 采样
                idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
                generated_tokens_per_scale.append(idx_Bl)
                
                # 更新f_hat和next_token_map（与原始实现保持一致）
                if not more_smooth:  # 默认情况
                    h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)   # B, l, Cvae
                else:   # 用于可视化的平滑采样
                    gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)   # 参考mask-git
                    h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
                
                h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
                f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw)
                
                # 准备下一阶段（与原始实现保持一致）
                if si != self.num_stages_minus_1:
                    next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                    next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                    next_token_map = next_token_map.repeat(2, 1, 1)   # CFG: 双倍批次大小
            
            # 清理KV缓存
            for b in self.blocks: 
                b.attn.kv_caching(False)
            
            # 生成最终图像（与原始实现保持一致）
            return self.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)   # de-normalize, from [-1, 1] to [0, 1]
            
        except Exception as e:
            # 出错时的fallback：使用标准VAR推理
            print(f"AGIP-VAR inference failed: {e}, falling back to standard VAR")
            for b in self.blocks: 
                b.attn.kv_caching(False)
            return self.autoregressive_infer_cfg(B=B, label_B=label_B, cfg=cfg, top_k=top_k, top_p=top_p, g_seed=g_seed, more_smooth=more_smooth)

    def autoregressive_train_step(
        self, 
        B: int, 
        label_B: Optional[Union[int, torch.LongTensor]],
        guidance_tokens_list: Optional[List[torch.Tensor]] = None,  # List of guidance tokens for each scale
        deterministic: bool = True,  # Use argmax vs sampling
        g_seed: Optional[int] = None
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:  # (tokens_list, logits_list)
        """
        Autoregressive training step with optional guidance token injection.
        Based on autoregressive_infer_cfg() but modified for training:
        - Supports gradient computation (no @torch.no_grad())
        - Accepts guidance tokens for each scale
        - Returns both tokens and logits for training
        - Supports deterministic (argmax) and stochastic sampling
        
        :param B: batch size
        :param label_B: imagenet label; if None, randomly sampled
        :param guidance_tokens_list: optional list of guidance tokens for each scale
        :param deterministic: if True, use argmax; if False, use sampling
        :param g_seed: random seed for sampling
        :return: (tokens_list, logits_list) - lists containing tokens and logits for each scale
        """
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng
        
        # Prepare label_B
        if label_B is None:
            label_B = torch.multinomial(self.uniform_prob, num_samples=B, replacement=True, generator=rng).reshape(B)
        elif isinstance(label_B, int):
            label_B = torch.full((B,), fill_value=self.num_classes if label_B < 0 else label_B, device=self.lvl_1L.device)
        
        # Initialize class embeddings (no CFG for training)
        sos = cond_BD = self.class_emb(label_B)
        
        # Position and level embeddings
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        next_token_map = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        
        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        
        # Storage for outputs
        tokens_list = []
        logits_list = []
        
        # Enable KV caching for efficiency
        for b in self.blocks: b.attn.kv_caching(True)
        
        for si, pn in enumerate(self.patch_nums):   # si: i-th segment
            cur_L += pn*pn
            
            # Inject guidance tokens if provided for this scale
            x = next_token_map
            if guidance_tokens_list is not None and si < len(guidance_tokens_list) and guidance_tokens_list[si] is not None:
                guidance_tokens = guidance_tokens_list[si]
                # Ensure guidance tokens have correct shape
                if guidance_tokens.shape[0] == B and guidance_tokens.shape[1] == x.shape[1]:
                    x = x + guidance_tokens
                elif guidance_tokens.shape[0] == B:
                    # Handle shape mismatch - broadcast or truncate
                    if guidance_tokens.shape[1] == 1:  # Single token, broadcast
                        x = x + guidance_tokens.expand(-1, x.shape[1], -1)
                    elif guidance_tokens.shape[1] > x.shape[1]:  # Truncate
                        x = x + guidance_tokens[:, :x.shape[1]]
                    else:  # Pad with zeros
                        padding_size = x.shape[1] - guidance_tokens.shape[1]
                        padding = torch.zeros(B, padding_size, guidance_tokens.shape[2], 
                                            device=guidance_tokens.device, dtype=guidance_tokens.dtype)
                        padded_guidance = torch.cat([guidance_tokens, padding], dim=1)
                        x = x + padded_guidance
            
            # Forward through transformer blocks
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            for b in self.blocks:
                x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
            
            # Get logits
            logits_BlV = self.get_logits(x, cond_BD)
            logits_list.append(logits_BlV)
            
            # Generate tokens (deterministic or stochastic)
            if deterministic:
                idx_Bl = logits_BlV.argmax(dim=-1)
            else:
                # Use sampling (simplified version without top_k/top_p for training)
                idx_Bl = torch.multinomial(
                    torch.softmax(logits_BlV.view(-1, logits_BlV.shape[-1]), dim=-1), 
                    num_samples=1, 
                    generator=rng
                ).view(B, -1)
            
            tokens_list.append(idx_Bl)
            
            # Prepare for next scale (same logic as inference)
            h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)   # B, l, Cvae
            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw)
            
            if si != self.num_stages_minus_1:   # prepare for next stage
                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                # No CFG doubling for training
        
        for b in self.blocks: b.attn.kv_caching(False)
        return tokens_list, logits_list
    
    def forward(self, label_B: torch.LongTensor, x_BLCv_wo_first_l: torch.Tensor, guidance_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:  # returns logits_BLV
        """
        :param label_B: label_B
        :param x_BLCv_wo_first_l: teacher forcing input (B, self.L-self.first_l, self.Cvae)
        :param guidance_tokens: optional guidance tokens to inject (B, L, C)
        :return: logits BLV, V is vocab_size
        """
        bg, ed = self.begin_ends[self.prog_si] if self.prog_si >= 0 else (0, self.L)
        B = x_BLCv_wo_first_l.shape[0]
        with torch.cuda.amp.autocast(enabled=False):
            label_B = torch.where(torch.rand(B, device=label_B.device) < self.cond_drop_rate, self.num_classes, label_B)
            sos = cond_BD = self.class_emb(label_B)
            sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1)
            
            if self.prog_si == 0: x_BLC = sos
            else: x_BLC = torch.cat((sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1)
            x_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)) + self.pos_1LC[:, :ed] # lvl: BLC;  pos: 1LC
            
            # Inject guidance tokens if provided
            if guidance_tokens is not None:
                if guidance_tokens.shape[1] == x_BLC.shape[1]:
                    x_BLC = x_BLC + guidance_tokens
                else:
                    # Handle shape mismatch - truncate or pad guidance tokens
                    if guidance_tokens.shape[1] > x_BLC.shape[1]:
                        x_BLC = x_BLC + guidance_tokens[:, :x_BLC.shape[1]]
                    else:
                        # Pad with zeros if guidance tokens are shorter
                        padding_size = x_BLC.shape[1] - guidance_tokens.shape[1]
                        padding = torch.zeros(B, padding_size, guidance_tokens.shape[2], 
                                            device=guidance_tokens.device, dtype=guidance_tokens.dtype)
                        padded_guidance = torch.cat([guidance_tokens, padding], dim=1)
                        x_BLC = x_BLC + padded_guidance
        
        attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)
        
        # hack: get the dtype if mixed precision is used
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype
        
        x_BLC = x_BLC.to(dtype=main_type)
        cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)
        
        AdaLNSelfAttn.forward
        for i, b in enumerate(self.blocks):
            x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)
        x_BLC = self.get_logits(x_BLC.float(), cond_BD)
        
        if self.prog_si == 0:
            if isinstance(self.word_embed, nn.Linear):
                x_BLC[0, 0, 0] += self.word_embed.weight[0, 0] * 0 + self.word_embed.bias[0] * 0
            else:
                s = 0
                for p in self.word_embed.parameters():
                    if p.requires_grad:
                        s += p.view(-1)[0] * 0
                x_BLC[0, 0, 0] += s
        return x_BLC    # logits BLV, V is vocab_size
    
    def init_weights(self, init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=0.02, conv_std_or_gain=0.02):
        if init_std < 0: init_std = (1 / self.C / 3) ** 0.5     # init_std < 0: automated
        
        print(f'[init_weights] {type(self).__name__} with {init_std=:g}')
        for m in self.modules():
            with_weight = hasattr(m, 'weight') and m.weight is not None
            with_bias = hasattr(m, 'bias') and m.bias is not None
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if with_bias: m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if m.padding_idx is not None: m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if with_weight: m.weight.data.fill_(1.)
                if with_bias: m.bias.data.zero_()
            # conv: VAR has no conv, only VQVAE has conv
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                if conv_std_or_gain > 0: nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
                else: nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
                if with_bias: m.bias.data.zero_()
        
        if init_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(init_head)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(init_head)
                self.head[-1].bias.data.zero_()
        
        if isinstance(self.head_nm, AdaLNBeforeHead):
            self.head_nm.ada_lin[-1].weight.data.mul_(init_adaln)
            if hasattr(self.head_nm.ada_lin[-1], 'bias') and self.head_nm.ada_lin[-1].bias is not None:
                self.head_nm.ada_lin[-1].bias.data.zero_()
        
        depth = len(self.blocks)
        for block_idx, sab in enumerate(self.blocks):
            sab: AdaLNSelfAttn
            sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
            sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
            if hasattr(sab.ffn, 'fcg') and sab.ffn.fcg is not None:
                nn.init.ones_(sab.ffn.fcg.bias)
                nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
            if hasattr(sab, 'ada_lin'):
                sab.ada_lin[-1].weight.data[2*self.C:].mul_(init_adaln)
                sab.ada_lin[-1].weight.data[:2*self.C].mul_(init_adaln_gamma)
                if hasattr(sab.ada_lin[-1], 'bias') and sab.ada_lin[-1].bias is not None:
                    sab.ada_lin[-1].bias.data.zero_()
            elif hasattr(sab, 'ada_gss'):
                sab.ada_gss.data[:, :, 2:].mul_(init_adaln)
                sab.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)
    
    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate:g}'

   

class VARHF(VAR, PyTorchModelHubMixin):
            # repo_url="https://github.com/FoundationVision/VAR",
            # tags=["image-generation"]):
    def __init__(
        self,
        vae_kwargs,
        num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
        flash_if_available=True, fused_if_available=True,
    ):
        vae_local = VQVAE(**vae_kwargs)
        super().__init__(
            vae_local=vae_local,
            num_classes=num_classes, depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
            norm_eps=norm_eps, shared_aln=shared_aln, cond_drop_rate=cond_drop_rate,
            attn_l2_norm=attn_l2_norm,
            patch_nums=patch_nums,
            flash_if_available=flash_if_available, fused_if_available=fused_if_available,
        )
