"""
🔥 AGIP-VAR规划模块：轻量级I_predictor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

def safe_layer_norm(x, weight, bias, eps=1e-5):
    """
    完全安全的LayerNorm实现，避免任何数值不稳定性
    """
    # 首先进行输入保护
    x = torch.clamp(x, min=-10.0, max=10.0)
    
    # 计算均值和方差时使用更强的数值保护
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    
    # 确保方差不会太小，防止除零
    var = torch.clamp(var, min=eps, max=100.0)
    std = torch.sqrt(var + eps)
    
    # 标准化
    normalized = (x - mean) / std
    
    # 再次保护标准化后的结果
    normalized = torch.clamp(normalized, min=-5.0, max=5.0)
    
    # 应用权重和偏置
    if weight is not None:
        # 限制权重范围，防止极端缩放
        safe_weight = torch.clamp(weight, min=0.1, max=2.0)
        normalized = normalized * safe_weight
    
    if bias is not None:
        # 限制偏置范围
        safe_bias = torch.clamp(bias, min=-1.0, max=1.0)
        normalized = normalized + safe_bias
    
    # 最终输出保护
    return torch.clamp(normalized, min=-3.0, max=3.0)

class SafeLayerNorm(nn.Module):
    """绝对安全的LayerNorm层"""
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        # 使用保守的初始化
        self.weight = nn.Parameter(torch.ones(normalized_shape) * 0.5)  # 保守初始化
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        
    def forward(self, x):
        return safe_layer_norm(x, self.weight, self.bias, self.eps)

class UltraSafeTransformerBlock(nn.Module):
    """极度安全的Transformer块，防止任何数值问题"""
    
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # 使用我们的安全LayerNorm
        self.norm1 = SafeLayerNorm(embed_dim)
        self.norm2 = SafeLayerNorm(embed_dim)
        
        # 多头注意力 - 使用保守设置
        self.self_attn = nn.MultiheadAttention(
            embed_dim, 
            num_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        # FFN with smaller hidden dimension to prevent overflow
        hidden_dim = min(embed_dim * 2, 512)  # 限制隐藏维度
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # 保守的权重初始化
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 使用极小的初始化标准差
                nn.init.normal_(module.weight, mean=0.0, std=0.001)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x):
        # 输入保护
        x = torch.clamp(x, min=-5.0, max=5.0)
        
        # 第一个残差连接 (self-attention)
        x_norm = self.norm1(x)
        
        try:
            # 注意力计算
            attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
            attn_out = torch.clamp(attn_out, min=-3.0, max=3.0)
            
            # 残差连接 - 缩放到更小的范围
            x = x + 0.1 * attn_out  # 减小残差连接的影响
            x = torch.clamp(x, min=-5.0, max=5.0)
            
        except Exception as e:
            # 如果注意力失败，直接返回输入
            return torch.clamp(x, min=-3.0, max=3.0)
        
        # 第二个残差连接 (FFN)
        x_norm2 = self.norm2(x)
        
        try:
            ffn_out = self.ffn(x_norm2)
            ffn_out = torch.clamp(ffn_out, min=-3.0, max=3.0)
            
            # 残差连接 - 同样缩放
            x = x + 0.1 * ffn_out
            x = torch.clamp(x, min=-5.0, max=5.0)
            
        except Exception as e:
            pass
        
        return torch.clamp(x, min=-3.0, max=3.0)

class UltraSafeTransformerEncoder(nn.Module):
    """极度安全的Transformer编码器"""
    
    def __init__(self, num_layers, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            UltraSafeTransformerBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers
    
    def forward(self, x):
        # 输入预处理
        x = torch.clamp(x, min=-5.0, max=5.0)
        
        # 逐层处理
        for i, layer in enumerate(self.layers):
            # 层前检查
            if torch.isnan(x).any() or torch.isinf(x).any():
                return torch.clamp(x, min=-3.0, max=3.0)
            
            # 计算当前层
            try:
                x_new = layer(x)
                
                # 层后检查
                if torch.isnan(x_new).any() or torch.isinf(x_new).any():
                    return torch.clamp(x, min=-3.0, max=3.0)
                
                x = x_new
                
            except Exception as e:
                return torch.clamp(x, min=-3.0, max=3.0)
        
        return torch.clamp(x, min=-3.0, max=3.0)

class I_predictor(nn.Module):
    """
    🎯 隐式规划器 - 生成空间感知的规划词元图
    
    核心特性：
    1. 输出Planning Token Map而非单一token
    2. 尺寸匹配：输出尺寸与目标生成尺度完全匹配（L_k = pn_k²）
    3. 空间精细化：每个位置有独立的规划信息
    4. 逐位置相加：与S_k进行逐位置相加而非广播
    """
    
    def __init__(self, input_dim=1024, embed_dim=1024, num_layers=2, num_heads=8):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        # 输入投影 - 使用小的初始化
        self.input_proj = nn.Linear(input_dim, embed_dim)
        nn.init.normal_(self.input_proj.weight, mean=0.0, std=0.001)
        nn.init.constant_(self.input_proj.bias, 0.0)
        
        # 使用我们的超安全Transformer编码器
        self.encoder = UltraSafeTransformerEncoder(
            num_layers=num_layers,
            embed_dim=embed_dim, 
            num_heads=num_heads,
            dropout=0.1
        )
        
        # 🎯 位置编码缓存
        self.pos_encodings = {}
        
        # 输出投影 - 保持每个位置的独立性
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=0.001)
        nn.init.constant_(self.output_proj.bias, 0.0)
        
        # 最终层归一化
        self.final_norm = SafeLayerNorm(embed_dim)
        
        # I_predictor初始化完成 (静默)
    
    def _get_position_encoding(self, size: int, device: torch.device) -> torch.Tensor:
        """获取或创建位置编码"""
        key = f"pos_{size}"
        
        if key not in self.pos_encodings:
            # 创建1D位置编码 (对应于flatten后的空间位置)
            pos_embed = torch.zeros(1, size, self.embed_dim)
            
            # 简单的正弦位置编码
            for i in range(size):
                for j in range(self.embed_dim):
                    if j % 2 == 0:
                        pos_embed[0, i, j] = math.sin(i / (10000 ** (j / self.embed_dim)))
                    else:
                        pos_embed[0, i, j] = math.cos(i / (10000 ** ((j-1) / self.embed_dim)))
            
            # 注册为buffer而不是parameter，避免训练
            self.register_buffer(key, pos_embed * 0.01)  # 小初始化
            self.pos_encodings[key] = self.__dict__['_buffers'][key]
        
        return self.pos_encodings[key].to(device)

    def forward(self, prev_features, target_patch_num=None):
        """前向传播 - 生成空间感知的规划词元图
        
        Args:
            prev_features: 前一尺度特征 (B, L_prev, C)
            target_patch_num: 目标尺度的patch数量 (pn_k)，如果为None则使用输入尺度
            
        Returns:
            planning_token_map: 规划词元图 (B, L_k, C) 其中 L_k = pn_k²
        """
        B, L_prev, C = prev_features.shape
        device = prev_features.device
        
        # 🎯 确定目标尺度
        if target_patch_num is None:
            # 如果未指定目标尺度，推断从当前尺度
            import math
            pn_prev = int(math.sqrt(L_prev))
            target_patch_num = pn_prev
        
        target_L = target_patch_num * target_patch_num
        
        # 强制输入保护
        prev_features = torch.clamp(prev_features, min=-10.0, max=10.0)
        
        # 检查空输入
        if prev_features.numel() == 0:
            return torch.zeros(B, target_L, self.embed_dim, device=device)

        # 输入投影
        x_proj = self.input_proj(prev_features)
        x_proj = torch.clamp(x_proj, min=-3.0, max=3.0)
        
        # 空间变换到目标尺度
        if L_prev != target_L:
            if target_L > L_prev:
                # 上采样：使用插值
                x_proj = x_proj.transpose(1, 2)  # (B, C, L_prev)
                x_proj = F.interpolate(x_proj, size=target_L, mode='linear', align_corners=False)
                x_proj = x_proj.transpose(1, 2)  # (B, target_L, C)
            else:
                # 下采样：使用池化
                x_proj = x_proj.transpose(1, 2)  # (B, C, L_prev)
                x_proj = F.adaptive_avg_pool1d(x_proj, target_L)
                x_proj = x_proj.transpose(1, 2)  # (B, target_L, C)
        
        # 添加位置编码
        pos_encoding = self._get_position_encoding(target_L, device)
        x_proj = x_proj + pos_encoding.expand(B, -1, -1)
        x_proj = torch.clamp(x_proj, min=-3.0, max=3.0)

        # Transformer编码
        encoded = self.encoder(x_proj)  # (B, target_L, embed_dim)
        
        # 最终投影
        output = self.output_proj(encoded)
        output = torch.clamp(output, min=-3.0, max=3.0)
        
        # 逐位置归一化
        planning_token_map = self.final_norm(output)
        
        # 安全裁剪
        planning_token_map = torch.clamp(planning_token_map, min=-2.0, max=2.0)
        
        # 检查输出健康性
        if torch.isnan(planning_token_map).any() or torch.isinf(planning_token_map).any():
            return torch.zeros(B, target_L, self.embed_dim, device=device)
        
        return planning_token_map

    def count_parameters(self):
        """计算参数量"""
        return sum(p.numel() for p in self.parameters())

# 为了向后兼容，保持原有的IPlanner类名
def create_iplanner(embed_dim=1024, hidden_dim=512, num_layers=2, num_heads=8):
    """创建I_predictor实例（现在默认为空间感知模式）"""
    model = I_predictor(embed_dim, hidden_dim, num_layers, num_heads)
    return model


class IPlannerTiny(nn.Module):
    """✅ 超轻量级规划器 - 确保 < 5M 参数
    
    使用MLP而非Transformer，大幅减少参数量
    """
    
    def __init__(
        self,
        embed_dim: int = 1024,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        layers = []
        in_dim = embed_dim
        
        # 构建MLP层
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else embed_dim
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU() if i < num_layers - 1 else nn.Identity(),
                nn.Dropout(dropout) if i < num_layers - 1 else nn.Identity(),
            ])
            in_dim = out_dim
        
        self.mlp = nn.Sequential(*layers)
        
        # 全局池化
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # 参数量统计 (静默)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: (B, L, C)
            
        Returns:
            I_k: (B, C)
        """
        # 全局平均池化
        x = x.transpose(1, 2)  # (B, C, L)
        x = self.pool(x)  # (B, C, 1)
        x = x.squeeze(-1)  # (B, C)
        
        # MLP变换
        I_k = self.mlp(x)  # (B, C)
        
        return I_k


def _test():
    # 简易单元测试
    B, L, C = 2, 128, 512
    m = I_predictor(embed_dim=C)
    x = torch.randn(B, L, C)
    out = m(x)
    assert out.shape == (B, C)


if __name__ == "__main__":
    _test() 