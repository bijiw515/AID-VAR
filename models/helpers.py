import torch
from torch import nn as nn
from torch.nn import functional as F
from typing import Tuple, Union, Optional


def sample_with_top_k_top_p_(logits_BlV: torch.Tensor, top_k: int = 0, top_p: float = 0.0, rng=None, num_samples=1) -> torch.Tensor:  # return idx, shaped (B, l)
    B, l, V = logits_BlV.shape
    if top_k > 0:
        idx_to_remove = logits_BlV < logits_BlV.topk(top_k, largest=True, sorted=False, dim=-1)[0].amin(dim=-1, keepdim=True)
        logits_BlV.masked_fill_(idx_to_remove, -torch.inf)
    if top_p > 0:
        sorted_logits, sorted_idx = logits_BlV.sort(dim=-1, descending=False)
        sorted_idx_to_remove = sorted_logits.softmax(dim=-1).cumsum_(dim=-1) <= (1 - top_p)
        sorted_idx_to_remove[..., -1:] = False
        logits_BlV.masked_fill_(sorted_idx_to_remove.scatter(sorted_idx.ndim - 1, sorted_idx, sorted_idx_to_remove), -torch.inf)
    # sample (have to squeeze cuz torch.multinomial can only be used for 2D tensor)
    replacement = num_samples >= 0
    num_samples = abs(num_samples)
    return torch.multinomial(logits_BlV.softmax(dim=-1).view(-1, V), num_samples=num_samples, replacement=replacement, generator=rng).view(B, l, num_samples)


def differentiable_sample_with_top_k_top_p(
    logits_BlV: torch.Tensor, 
    top_k: int = 0, 
    top_p: float = 0.0, 
    tau: float = 1.0, 
    hard: bool = True,
    rng: Optional[torch.Generator] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    可导的采样函数，使用Gumbel Softmax + Straight-Through Estimator
    
    Args:
        logits_BlV: 输入logits (B, L, V)
        top_k: top-k过滤
        top_p: nucleus采样过滤  
        tau: Gumbel Softmax温度参数
        hard: 是否使用hard Gumbel Softmax (STE)
        rng: 随机数生成器
    
    Returns:
        discrete_tokens: 离散tokens (B, L) - 用于判别器
        soft_embeddings: 软embedding (B, L, V) - 保持梯度流
    """
    B, L, V = logits_BlV.shape
    
    # 复制logits避免in-place操作
    filtered_logits = logits_BlV.clone()
    
    # Top-k过滤
    if top_k > 0:
        idx_to_remove = filtered_logits < filtered_logits.topk(
            top_k, largest=True, sorted=False, dim=-1
        )[0].amin(dim=-1, keepdim=True)
        filtered_logits.masked_fill_(idx_to_remove, -torch.inf)
    
    # Top-p (nucleus)过滤  
    if top_p > 0:
        sorted_logits, sorted_idx = filtered_logits.sort(dim=-1, descending=False)
        sorted_idx_to_remove = sorted_logits.softmax(dim=-1).cumsum_(dim=-1) <= (1 - top_p)
        sorted_idx_to_remove[..., -1:] = False
        filtered_logits.masked_fill_(
            sorted_idx_to_remove.scatter(sorted_idx.ndim - 1, sorted_idx, sorted_idx_to_remove), 
            -torch.inf
        )
    
    # 使用Gumbel Softmax进行可导采样
    soft_samples = gumbel_softmax_with_rng(
        logits=filtered_logits, 
        tau=tau, 
        hard=hard, 
        dim=-1, 
        rng=rng
    )  # (B, L, V)
    
    # 获取离散tokens (用于判别器)
    discrete_tokens = soft_samples.argmax(dim=-1)  # (B, L)
    
    # 如果使用hard模式，soft_samples已经是STE形式，保持梯度流
    # 如果使用soft模式，返回的是soft probabilities
    
    return discrete_tokens, soft_samples


def straight_through_estimator_sample(
    logits_BlV: torch.Tensor,
    top_k: int = 0,
    top_p: float = 0.0,
    rng: Optional[torch.Generator] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    使用Straight-Through Estimator的可导采样
    
    Args:
        logits_BlV: 输入logits (B, L, V)
        top_k: top-k过滤
        top_p: nucleus采样过滤
        rng: 随机数生成器
        
    Returns:
        discrete_tokens: 离散tokens (B, L) - 用于判别器
        ste_embeddings: STE embedding (B, L, V) - 保持梯度流
    """
    B, L, V = logits_BlV.shape
    
    # 应用top-k和top-p过滤
    filtered_logits = logits_BlV.clone()
    
    if top_k > 0:
        idx_to_remove = filtered_logits < filtered_logits.topk(
            top_k, largest=True, sorted=False, dim=-1
        )[0].amin(dim=-1, keepdim=True)
        filtered_logits.masked_fill_(idx_to_remove, -torch.inf)
    
    if top_p > 0:
        sorted_logits, sorted_idx = filtered_logits.sort(dim=-1, descending=False)
        sorted_idx_to_remove = sorted_logits.softmax(dim=-1).cumsum_(dim=-1) <= (1 - top_p)
        sorted_idx_to_remove[..., -1:] = False
        filtered_logits.masked_fill_(
            sorted_idx_to_remove.scatter(sorted_idx.ndim - 1, sorted_idx, sorted_idx_to_remove), 
            -torch.inf
        )
    
    # 计算概率分布
    probs = F.softmax(filtered_logits, dim=-1)  # (B, L, V)
    
    # 不可导采样（前向传播）
    with torch.no_grad():
        discrete_tokens = torch.multinomial(
            probs.view(-1, V), 
            num_samples=1, 
            replacement=True, 
            generator=rng
        ).view(B, L)  # (B, L)
    
    # 创建one-hot向量
    one_hot = F.one_hot(discrete_tokens, num_classes=V).float()  # (B, L, V)
    
    # Straight-Through Estimator: 前向使用one-hot，反向使用soft probabilities
    ste_embeddings = one_hot - probs.detach() + probs  # (B, L, V)
    
    return discrete_tokens, ste_embeddings


def gumbel_softmax_with_rng(logits: torch.Tensor, tau: float = 1, hard: bool = False, eps: float = 1e-10, dim: int = -1, rng: torch.Generator = None) -> torch.Tensor:
    if rng is None:
        return F.gumbel_softmax(logits=logits, tau=tau, hard=hard, eps=eps, dim=dim)
    
    gumbels = (-torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_(generator=rng).log())
    gumbels = (logits + gumbels) / tau
    y_soft = gumbels.softmax(dim)
    
    if hard:
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):    # taken from timm
    if drop_prob == 0. or not training: return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):  # taken from timm
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
    
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
    
    def extra_repr(self):
        return f'(drop_prob=...)'
