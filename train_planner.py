#!/usr/bin/env python3
"""
🎯 AID-VAR完整ImageNet多卡分布式训练脚本
基于train_single_class.py的成功架构，扩展到完整ImageNet和多卡训练
包括：分阶段训练、空间感知规划词元图、判别器准确率监控等

🔄 检查点恢复功能:
- 支持从指定检查点恢复训练
- 自动恢复模型状态、优化器状态和训练进度
- 支持分布式训练的检查点同步

使用示例:
  # 从头开始训练（8卡）
  torchrun --nproc_per_node=8 train_planner.py --data_path /path/to/imagenet --num_epochs 100
  
  # 从检查点恢复训练
  torchrun --nproc_per_node=8 train_planner.py --data_path /path/to/imagenet --num_epochs 100 \
    --resume_checkpoint experiments/aid_var_full_imagenet_staged_20241219_143021/checkpoints/checkpoint_epoch_10.pth
"""

import os
import sys
import argparse
import logging
import json
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import time
from datetime import datetime
from dataclasses import dataclass

# 添加stylegan_t路径
stylegan_path = os.path.join(os.path.dirname(__file__), 'stylegan_t')
if stylegan_path not in sys.path:
    sys.path.append(stylegan_path)

# 导入VAR项目模块
import dist
from utils import arg_util, misc
from utils.data import build_dataset
from utils.data_sampler import DistInfiniteBatchSampler, EvalDistributedSampler
from models import build_vae_var
from models.guidance_injector import GuidanceInjector
from models.discriminator_adapter import StyleGANDiscriminatorAdapter
from trainer_planner import PlannerTrainer
from utils.amp_sc import AmpOptimizer
from utils.misc import MetricLogger, TensorboardLogger, DistLogger

# 配置日志 - 将在setup_experiment_dir后重新配置到正确路径
logger = logging.getLogger(__name__)

@dataclass
class AIDVARTrainingConfig:
    """AID-VAR完整ImageNet训练配置"""
    
    # 基础配置
    DEVICE: str = 'cuda'
    OUTPUT_DIR: str = "experiments"
    
    # 模型配置
    VAR_CKPT: str = "checkpoints/var_d24.pth"  # VAR权重路径
    VQVAE_CKPT: str = "vae_ch160v4096z32.pth"  # VQVAE权重路径
    VAR_DEPTH: int = 24  # VAR模型深度（16, 20, 24等）
    
    # 训练配置 
    EPOCHS: int = 2
    GLOBAL_BATCH_SIZE: int = 16 * 32  # 全局批次大小（所有GPU总和）
    VAL_BATCH_SIZE: int = 32  # 验证批次大小（每GPU）
    NUM_WORKERS: int = 8
    
    # 🔥 关键修复：统一学习率，避免训练动态失衡
    LEARNING_RATE_PLANNER: float = 1e-06       # GuidanceInjector学习率
    LEARNING_RATE_DISCRIMINATOR: float = 1e-06  # 判别器学习率（与planner统一）
    
    # 梯度裁剪
    CLIP_PLANNER: float = 0.5   # 保守的梯度裁剪
    CLIP_DISCRIMINATOR: float = 0.5  # 保守的梯度裁剪
    
    # AID-VAR超参数
    LAMBDA_REC: float = 0  # 重构损失权重
    GUIDANCE_WEIGHT: float = 0.005  # 保守的初始引导权重
    GUIDANCE_TARGET_WEIGHT: float = 0.001  # 固定引导权重
    GUIDANCE_RAMP_EPOCHS: int = 15  # 渐进增加期
    
    # 🔥 新增：判别器稳定性参数
    R1_GAMMA: float = 0.2  # R1梯度惩罚权重
    PROGRESSIVE_GUIDANCE: bool = True  # 渐进式引导权重
    COLLAPSE_DETECTION: bool = True  # 崩溃检测
    
    # 分阶段训练配置
    ENABLE_STAGED_TRAINING: bool = True
    WARMUP_STEPS: int = 0  # 判别器预热期
    
    # 采样参数
    TOP_K: int = 0
    TOP_P: float = 0
    CFG: float = 1.5
    
    # VAR相关
    PATCH_NUMS: tuple = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)  # VAR的多尺度
    
    # GuidanceInjector架构 - 动态根据VAR深度调整
    PLANNER_LAYERS: int = 2
    PLANNER_DIM: int = VAR_DEPTH * 64  # 将根据VAR_DEPTH自动计算：depth * 64
    PLANNER_HEADS: int = 8
    
    # 日志和保存
    LOG_INTERVAL: int = 10  # 每50个batch打印一次
    SAVE_INTERVAL: int = 1  # 每10个epoch保存一次检查点
    VALIDATION_INTERVAL: int = 1  # 每5个epoch验证一次
    
    # 检查点恢复
    RESUME_CHECKPOINT: Optional[str] = None  # 恢复训练的检查点路径
    LOAD_OPTIMIZER_STATE: bool = True  # 是否加载优化器状态（包括学习率）
    
    @classmethod
    def to_dict(cls) -> Dict:
        """转换为字典格式"""
        return {
            key: value for key, value in cls.__dict__.items() 
            if not key.startswith('_') and not callable(value) and not isinstance(value, classmethod)
        }
    
    def update_for_var_depth(self, depth: int):
        """根据VAR深度更新配置参数
        
        Args:
            depth: VAR模型深度（16, 20, 24等）
        """
        self.VAR_DEPTH = depth
        self.PLANNER_DIM = depth * 64  # VAR嵌入维度：depth * 64
        
        # 自动检测VAR检查点路径
        var_ckpt_path = f"checkpoints/var_d{depth}.pth"
        if os.path.exists(var_ckpt_path):
            self.VAR_CKPT = var_ckpt_path
        else:
            logger.warning(f"⚠️ VAR权重文件不存在: {var_ckpt_path}")
        
        logger.info(f"🔧 配置已更新为d{depth}模型:")
        logger.info(f"   VAR深度: {self.VAR_DEPTH}")
        logger.info(f"   GuidanceInjector维度: {self.PLANNER_DIM}")
        logger.info(f"   VAR权重路径: {self.VAR_CKPT}")

class NullDDP(torch.nn.Module):
    """单GPU训练时的DDP替代类"""
    def __init__(self, module, *args, **kwargs):
        super(NullDDP, self).__init__()
        self.module = module
        self.require_backward_grad_sync = False
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

def setup_experiment_dir(config: AIDVARTrainingConfig, args) -> str:
    """设置实验目录并配置日志（仅master进程执行）"""
    if not dist.is_master():
        return ""  # 非master进程返回空字符串
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    staged_suffix = "staged" if config.ENABLE_STAGED_TRAINING else "joint"
    exp_name = f"aid_var_full_imagenet_{staged_suffix}_{timestamp}"
    exp_dir = os.path.join(config.OUTPUT_DIR, exp_name)
    
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "validation"), exist_ok=True)
    
    # 配置日志到实验目录
    log_file = os.path.join(exp_dir, "logs", "full_imagenet_training.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ],
        force=True  # 强制重新配置
    )
    
    # 保存配置
    config_dict = config.to_dict()
    config_dict.update({
        'world_size': dist.get_world_size(),
        'global_batch_size': args.glb_batch_size,
        'batch_size_per_gpu': args.batch_size,
        'data_path': args.data_path,
        'patch_nums': args.patch_nums,
        'depth': args.depth
    })
    
    with open(os.path.join(exp_dir, "config.json"), 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    logger.info(f"🗂️ 实验目录创建: {exp_dir}")
    logger.info(f"📝 日志文件: {log_file}")
    logger.info(f"🎯 训练模式: {'分阶段训练' if config.ENABLE_STAGED_TRAINING else '联合训练'}")
    logger.info(f"🌐 分布式配置: {dist.get_world_size()}卡, rank={dist.get_rank()}")
    return exp_dir

def load_models(config: AIDVARTrainingConfig, args) -> Tuple[nn.Module, nn.Module, nn.Module, nn.Module]:
    """加载所有模型 - 适配AID-VAR架构"""
    if dist.is_master():
        logger.info("📦 加载AID-VAR模型架构...")
    
    # 🔥 重要修复：使用正确的VQ-VAE配置参数
    # 根据vae_ch160v4096z32.pth的实际配置
    vqvae, var = build_vae_var(
        V=4096,      # 词汇表大小，与文件名中的4096匹配
        Cvae=32,     # VQ-VAE的特征维度，与文件名中的32匹配  
        ch=160,      # 通道数，与文件名中的160匹配
        share_quant_resi=4,
        device=dist.get_device(), 
        patch_nums=args.patch_nums,
        num_classes=1000,  # 完整ImageNet 1000类
        depth=args.depth,
        shared_aln=args.saln,
        attn_l2_norm=args.anorm,
        flash_if_available=args.fuse,
        fused_if_available=args.fuse,
        init_adaln=args.aln,
        init_adaln_gamma=args.alng,
        init_head=args.hd,
        init_std=args.ini
    )
    
    # 下载和加载预训练权重
    vae_ckpt = config.VQVAE_CKPT
    if dist.is_local_master():
        if not os.path.exists(vae_ckpt):
            os.system(f'wget https://huggingface.co/FoundationVision/var/resolve/main/{vae_ckpt}')
    dist.barrier()
    vqvae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
    
    # 加载VAR权重
    if os.path.exists(config.VAR_CKPT):
        var_state = torch.load(config.VAR_CKPT, map_location='cpu')
        var.load_state_dict(var_state, strict=True)
        if dist.is_master():
            logger.info(f"✅ VAR模型权重加载: {config.VAR_CKPT}")
    else:
        raise FileNotFoundError(f"VAR权重文件不存在: {config.VAR_CKPT}")
    
    # 冻结预训练模型
    for param in var.parameters():
        param.requires_grad = False
    for param in vqvae.parameters():
        param.requires_grad = False
    
    var.eval()
    vqvae.eval()
    
    # 统计冻结参数
    frozen_vae_params = sum(p.numel() for p in vqvae.parameters())
    frozen_var_params = sum(p.numel() for p in var.parameters())
    if dist.is_master():
        logger.info(f"🧊 冻结VAE参数: {frozen_vae_params:,}")
        logger.info(f"🧊 冻结VAR参数: {frozen_var_params:,}")
    
    # 🔥 验证VAR和VQ-VAE的配置匹配
    if dist.is_master():
        logger.info(f"📊 配置验证:")
        logger.info(f"   VQ-VAE embedding 形状: {vqvae.quantize.embedding.weight.shape}")  # 应该是[4096, 32]
        logger.info(f"   VAR word_embed输入维度: {var.word_embed.in_features}")  # 应该是32
        logger.info(f"   VAR word_embed输出维度: {var.word_embed.out_features}")  # 应该是1024
    
    # 创建AID-VAR组件 - 🔥 使用VAR的实际嵌入维度
    planner = GuidanceInjector(
        input_dim=var.C,  # VAR的特征维度 (动态：16*64=1024 或 20*64=1280)
        embed_dim=var.C,  # 规划器嵌入维度与VAR保持一致
        num_layers=config.PLANNER_LAYERS,
        num_heads=config.PLANNER_HEADS
    ).to(dist.get_device())
    
    # 🔥 修复：使用新的像素级判别器适配器
    discriminator = StyleGANDiscriminatorAdapter(
        vae=vqvae,  # VQ-VAE实例用于Token→RGB解码
        patch_nums=args.patch_nums,  # 多尺度patch配置
        img_resolution=256,  # 图像分辨率
        c_dim=0  # 类别条件维度
    ).to(dist.get_device())
    
    # 统计可训练参数
    planner_params = sum(p.numel() for p in planner.parameters() if p.requires_grad)
    disc_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    total_params = planner_params + disc_params
    
    if dist.is_master():
        logger.info(f"📊 AID-VAR可训练参数统计:")
        logger.info(f"   GuidanceInjector: {planner_params:,} 参数 ({planner_params/1e6:.2f}M)")
        logger.info(f"   Discriminator: {disc_params:,} 参数 ({disc_params/1e6:.2f}M)")
        logger.info(f"   总计: {total_params:,} 参数 ({total_params/1e6:.2f}M)")
        logger.info(f"🎯 模式: 空间感知规划词元图（逐位置相加）")
    
    return vqvae, var, planner, discriminator

def create_optimizers(planner: nn.Module, discriminator: nn.Module, config: AIDVARTrainingConfig) -> Tuple[AmpOptimizer, AmpOptimizer]:
    """创建优化器"""
    if dist.is_master():
        logger.info("⚙️ 创建AID-VAR优化器...")
    
    # 获取可训练参数
    planner_params = list(planner.parameters())
    
    # 🔥 修复：使用新的像素级判别器的可训练参数获取方式
    try:
        disc_trainable_params = discriminator.get_trainable_params()
        if dist.is_master():
            logger.info("✅ 使用像素级判别器的可训练参数")
    except AttributeError:
        # 兼容fallback: 获取所有可训练参数
        disc_trainable_params = [p for p in discriminator.parameters() if p.requires_grad]
        if dist.is_master():
            logger.warning("⚠️ 回退到所有可训练参数获取方式")
    
    if not disc_trainable_params:
        # 创建dummy参数以避免优化器错误
        dummy_param = torch.nn.Parameter(torch.tensor(0.0, device=dist.get_device()))
        disc_trainable_params = [dummy_param]
        if dist.is_master():
            logger.warning("⚠️ 判别器head参数为空，使用dummy参数")
    
    # 统计参数
    planner_param_count = sum(p.numel() for p in planner_params) / 1e6
    disc_param_count = sum(p.numel() for p in disc_trainable_params) / 1e6
    if dist.is_master():
        logger.info(f"📊 可训练参数统计:")
        logger.info(f"   GuidanceInjector: {planner_param_count:.2f}M 参数")
        logger.info(f"   StyleGAN-T Discriminator (heads only): {disc_param_count:.2f}M 参数")
        logger.info(f"   🎯 ADD风格: 冻结DINO backbone + 训练判别器heads")
    
    # 创建AmpOptimizer - GuidanceInjector
    amp_planner_opt = AmpOptimizer(
        mixed_precision=0,  # 使用float32
        optimizer=torch.optim.AdamW(
            planner_params,
            lr=config.LEARNING_RATE_PLANNER,
            weight_decay=1e-5,
            betas=(0.9, 0.999)
        ),
        names=['planner'],
        paras=planner_params,
        grad_clip=config.CLIP_PLANNER,
        n_gradient_accumulation=1
    )
    
    # 创建AmpOptimizer - 判别器
    amp_disc_opt = AmpOptimizer(
        mixed_precision=0,  # 使用float32
        optimizer=torch.optim.AdamW(
            disc_trainable_params,
            lr=config.LEARNING_RATE_DISCRIMINATOR,
            weight_decay=1e-5,
            betas=(0.9, 0.999)
        ),
        names=['discriminator'],
        paras=disc_trainable_params,
        grad_clip=config.CLIP_DISCRIMINATOR,
        n_gradient_accumulation=1
    )
    
    if dist.is_master():
        logger.info(f"✅ AmpOptimizer创建完成")
        logger.info(f"   GuidanceInjector学习率: {config.LEARNING_RATE_PLANNER}")
        logger.info(f"   Discriminator学习率: {config.LEARNING_RATE_DISCRIMINATOR}")
    
    return amp_planner_opt, amp_disc_opt

def create_data_loaders(args) -> Tuple[DataLoader, DataLoader, int]:
    """创建分布式数据加载器"""
    if dist.is_master():
        logger.info("📊 创建完整ImageNet分布式数据加载器...")
    
    # 构建ImageNet数据集
    num_classes, dataset_train, dataset_val = build_dataset(
        args.data_path, 
        final_reso=args.data_load_reso, 
        hflip=args.hflip, 
        mid_reso=args.mid_reso
    )
    
    # 创建分布式验证数据加载器
    val_loader = DataLoader(
        dataset_val, 
        num_workers=0, 
        pin_memory=True,
        batch_size=round(args.batch_size * 1.5),  # 验证时使用稍大的batch
        sampler=EvalDistributedSampler(
            dataset_val, 
            num_replicas=dist.get_world_size(), 
            rank=dist.get_rank()
        ),
        shuffle=False, 
        drop_last=False,
    )
    
    # 创建分布式训练数据加载器
    train_loader = DataLoader(
        dataset=dataset_train, 
        num_workers=args.workers, 
        pin_memory=True,
        generator=args.get_different_generator_for_each_rank(),
        batch_sampler=DistInfiniteBatchSampler(
            dataset_len=len(dataset_train), 
            glb_batch_size=args.glb_batch_size, 
            same_seed_for_all_ranks=args.same_seed_for_all_ranks,
            shuffle=True, 
            fill_last=True, 
            rank=dist.get_rank(), 
            world_size=dist.get_world_size(),
        ),
    )
    
    iters_train = len(train_loader)
    
    if dist.is_master():
        logger.info(f"✅ 分布式数据加载器创建成功")
        logger.info(f"   训练样本总数: {len(dataset_train):,}")
        logger.info(f"   验证样本总数: {len(dataset_val):,}")
        logger.info(f"   世界大小: {dist.get_world_size()}")
        logger.info(f"   全局批次大小: {args.glb_batch_size}")
        logger.info(f"   每GPU批次大小: {args.batch_size}")
        logger.info(f"   训练迭代数: {iters_train}")
    
    return train_loader, val_loader, iters_train

def train_full_imagenet(config: AIDVARTrainingConfig, args) -> str:
    """
    执行AID-VAR完整ImageNet分布式训练
    
    Returns:
        exp_dir: 实验目录路径（仅master进程有效）
    """
    if dist.is_master():
        logger.info("🚀 开始AID-VAR完整ImageNet分布式训练...")
        logger.info(f"   数据路径: {args.data_path}")
        logger.info(f"   设备: {dist.get_device()}")
        logger.info(f"   全局批次大小: {args.glb_batch_size}")
        logger.info(f"   训练轮数: {config.EPOCHS}")
        logger.info(f"   世界大小: {dist.get_world_size()}")
        if config.ENABLE_STAGED_TRAINING:
            logger.info(f"🎯 分阶段训练: 预热步数={config.WARMUP_STEPS}")
        else:
            logger.info(f"🎯 联合训练模式")
    
    # 设置实验目录（仅master进程）
    exp_dir = setup_experiment_dir(config, args)
    
    # 同步所有进程
    dist.barrier()
    
    # 加载模型
    vqvae, var, planner, discriminator = load_models(config, args)
    
    # 包装为DDP
    planner = (DDP if dist.initialized() else NullDDP)(
        planner, 
        device_ids=[dist.get_local_rank()], 
        find_unused_parameters=False, 
        broadcast_buffers=False
    )
    discriminator = (DDP if dist.initialized() else NullDDP)(
        discriminator, 
        device_ids=[dist.get_local_rank()], 
        find_unused_parameters=False, 
        broadcast_buffers=False
    )
    
    # 创建优化器
    amp_planner_opt, amp_disc_opt = create_optimizers(
        planner.module if isinstance(planner, DDP) else planner, 
        discriminator.module if isinstance(discriminator, DDP) else discriminator, 
        config
    )
    
    # 创建数据加载器
    train_loader, val_loader, iters_train = create_data_loaders(args)
    
    # 创建AID-VAR训练器
    trainer = PlannerTrainer(
        device=dist.get_device(),
        vae=vqvae,
        var=var,
        planner=planner.module if isinstance(planner, DDP) else planner,
        disc=discriminator.module if isinstance(discriminator, DDP) else discriminator,
        opt_planner=amp_planner_opt,
        opt_disc=amp_disc_opt,
        lambda_rec=config.LAMBDA_REC,
        patch_nums=var.patch_nums,
        top_k=config.TOP_K,
        top_p=config.TOP_P,
        cfg=config.CFG,
        # 🎯 分阶段训练参数
        warmup_steps=config.WARMUP_STEPS,
        enable_staged_training=config.ENABLE_STAGED_TRAINING,
        alternating_strategy="adaptive",
        guidance_weight_max=config.GUIDANCE_TARGET_WEIGHT,
    )
    
    # 创建度量记录器
    metric_logger = MetricLogger(delimiter='  ')
    
    # 设置TensorBoard日志（仅master进程）
    if dist.is_master() and exp_dir:
        tb_log_dir = os.path.join(exp_dir, "logs", "tensorboard")
        os.makedirs(tb_log_dir, exist_ok=True)
        tb_logger = DistLogger(
            TensorboardLogger(
                log_dir=tb_log_dir, 
                filename_suffix='aid_var_full_imagenet'
            ), 
            verbose=True
        )
    else:
        tb_logger = DistLogger(None, verbose=False)
    
    # 检查是否需要从checkpoint恢复
    start_epoch = 0
    best_val_loss = float('inf')
    
    if config.RESUME_CHECKPOINT and os.path.exists(config.RESUME_CHECKPOINT):
        if dist.is_master():
            logger.info(f"🔄 检测到恢复检查点: {config.RESUME_CHECKPOINT}")
        try:
            start_epoch, resume_train_metrics, resume_val_metrics = load_checkpoint(
                config.RESUME_CHECKPOINT,
                planner.module if isinstance(planner, DDP) else planner,
                discriminator.module if isinstance(discriminator, DDP) else discriminator,
                amp_planner_opt.optimizer,
                amp_disc_opt.optimizer,
                dist.get_device(),
                trainer,  # 传入trainer以恢复其内部状态
                load_optimizer=config.LOAD_OPTIMIZER_STATE  # 🔥 控制是否加载优化器状态
            )
            if dist.is_master():
                logger.info(f"✅ 成功恢复训练，将从epoch {start_epoch + 1}开始")
            # 从恢复的验证指标中获取最佳损失
            best_val_loss = resume_val_metrics.get('loss', float('inf'))
            
            # 🔥 重置学习率（如果加载了优化器状态但需要使用不同的学习率）
            if config.LOAD_OPTIMIZER_STATE:
                for param_group in amp_planner_opt.optimizer.param_groups:
                    old_lr = param_group['lr']
                    param_group['lr'] = config.LEARNING_RATE_PLANNER
                    if dist.is_master() and old_lr != config.LEARNING_RATE_PLANNER:
                        logger.info(f"🔄 GuidanceInjector学习率重置: {old_lr} → {config.LEARNING_RATE_PLANNER}")
                
                for param_group in amp_disc_opt.optimizer.param_groups:
                    old_lr = param_group['lr']
                    param_group['lr'] = config.LEARNING_RATE_DISCRIMINATOR
                    if dist.is_master() and old_lr != config.LEARNING_RATE_DISCRIMINATOR:
                        logger.info(f"🔄 判别器学习率重置: {old_lr} → {config.LEARNING_RATE_DISCRIMINATOR}")
            else:
                if dist.is_master():
                    logger.info(f"✅ 使用新学习率: GuidanceInjector={config.LEARNING_RATE_PLANNER}, 判别器={config.LEARNING_RATE_DISCRIMINATOR}")
        except Exception as e:
            if dist.is_master():
                logger.error(f"❌ 检查点加载失败: {e}")
                logger.info("🔄 将从头开始训练")
            start_epoch = 0
    elif config.RESUME_CHECKPOINT:
        if dist.is_master():
            logger.warning(f"⚠️ 指定的检查点文件不存在: {config.RESUME_CHECKPOINT}")
            logger.info("🔄 将从头开始训练")
    
    # 同步所有进程
    dist.barrier()
    
    # 训练循环
    if dist.is_master():
        logger.info("🔄 开始训练循环...")
    
    train_loader_iter = iter(train_loader)
    
    for epoch in range(start_epoch, config.EPOCHS):
        if dist.is_master():
            logger.info(f"📅 Epoch {epoch+1}/{config.EPOCHS}")
        
        # 训练一个epoch
        train_metrics = train_one_epoch(
            trainer, train_loader_iter, iters_train, epoch, config, metric_logger, tb_logger
        )
        
        # 验证（如果需要）
        val_metrics = {}
        if (epoch + 1) % config.VALIDATION_INTERVAL == 0:
            val_metrics = trainer.validate_one_epoch(
                val_loader, epoch, exp_dir if dist.is_master() else ""
            )
            
            # 记录验证指标
            for key, value in val_metrics.items():
                if isinstance(value, (int, float)):  # 只记录数值类型
                    tb_logger.update(head='val', step=epoch, **{key: value})
        
        # 保存检查点（仅master进程）
        if dist.is_master() and (epoch + 1) % config.SAVE_INTERVAL == 0:
            is_best = val_metrics.get('loss', float('inf')) < best_val_loss
            if is_best:
                best_val_loss = val_metrics.get('loss', float('inf'))
                
            save_checkpoint(
                exp_dir, epoch, 
                planner.module if isinstance(planner, DDP) else planner,
                discriminator.module if isinstance(discriminator, DDP) else discriminator,
                amp_planner_opt.optimizer, amp_disc_opt.optimizer,
                train_metrics, val_metrics, is_best, trainer
            )
        
        # 记录训练指标
        for key, value in train_metrics.items():
            tb_logger.update(head='train', step=epoch, **{key: value})
        
        # 同步所有进程
        dist.barrier()
    
    if dist.is_master():
        logger.info(f"🎉 AID-VAR完整ImageNet分布式训练完成！")
        logger.info(f"📁 实验结果保存在: {exp_dir}")
    
    return exp_dir if dist.is_master() else ""

def train_one_epoch(
    trainer: PlannerTrainer, 
    train_loader_iter, 
    iters_train: int,
    epoch: int,
    config: AIDVARTrainingConfig,
    metric_logger: MetricLogger,
    tb_logger
) -> Dict[str, float]:
    """训练一个epoch"""
    
    trainer.planner.train()
    trainer.disc.train()
    
    epoch_metrics = {
        'loss_planner': 0.0,
        'loss_discriminator': 0.0,
        'loss_adversarial': 0.0,
        'loss_reconstruction': 0.0,
        'discriminator_accuracy': 0.0,
        'discriminator_acc_real': 0.0,
        'discriminator_acc_fake': 0.0,
        'valid_scales': 0.0,
        'warmup_phase': False,
        'collapse_count': 0,
        'guidance_weight': 0.0,
        'r1_penalty': 0.0,
    }
    
    for batch_idx in range(iters_train):
        images, labels = next(train_loader_iter)
        
        # 执行AID-VAR逐尺度训练步骤
        step_metrics = trainer.train_step(images, labels, metric_logger)
        
        # 🔥 修复：直接使用trainer返回的正确指标名称
        if 'loss_D' in step_metrics:
            epoch_metrics['loss_discriminator'] += step_metrics['loss_D']
        if 'loss_P' in step_metrics:
            epoch_metrics['loss_planner'] += step_metrics['loss_P']
        if 'loss_adversarial' in step_metrics:
            epoch_metrics['loss_adversarial'] += step_metrics['loss_adversarial']
        if 'loss_reconstruction' in step_metrics:
            epoch_metrics['loss_reconstruction'] += step_metrics['loss_reconstruction']
        if 'acc_D' in step_metrics:
            epoch_metrics['discriminator_accuracy'] += step_metrics['acc_D']
        if 'acc_real' in step_metrics:
            epoch_metrics['discriminator_acc_real'] += step_metrics['acc_real']
        if 'acc_fake' in step_metrics:
            epoch_metrics['discriminator_acc_fake'] += step_metrics['acc_fake']
        if 'valid_scales' in step_metrics:
            epoch_metrics['valid_scales'] += step_metrics['valid_scales']
        if 'warmup_phase' in step_metrics:
            epoch_metrics['warmup_phase'] = step_metrics['warmup_phase']
            
        # 🔥 新增：记录崩溃检测和稳定性指标
        if hasattr(trainer, 'collapse_history') and len(trainer.collapse_history) > 0:
            epoch_metrics['collapse_count'] = len(trainer.collapse_history)
        
        # 🔥 修复：从trainer的alternating_manager获取当前引导权重
        current_guidance_weight = step_metrics.get('guidance_weight', 0.0)
        epoch_metrics['guidance_weight'] = current_guidance_weight
        
        # 记录当前步骤的指标（仅master进程）- 简化版本
        if dist.is_master() and batch_idx % config.LOG_INTERVAL == 0:
            warmup_phase = step_metrics.get('warmup_phase', False)
            phase_str = "预热" if warmup_phase else "联合"
            
            # 🎯 精简日志：只保留最关键的监控信息，包含正负样本准确率
            logger.info(f"Epoch {epoch+1} Batch {batch_idx}/{iters_train} [{phase_str}] "
                       f"D_loss={step_metrics.get('loss_D', 0):.4f} "
                       f"P_loss={step_metrics.get('loss_P', 0):.4f} "
                       f"D_acc={step_metrics.get('acc_D', 0):.3f} "
                       f"real_acc={step_metrics.get('acc_real', 0):.3f} "
                       f"fake_acc={step_metrics.get('acc_fake', 0):.3f} "
                       f"guidance={current_guidance_weight:.4f}")
                       
            # 只在有崩溃时警告
            if hasattr(trainer, 'collapse_history') and len(trainer.collapse_history) > 0:
                recent_collapses = [c for c in trainer.collapse_history if c['collapsed']]
                if recent_collapses:
                    logger.warning(f"⚠️ Discriminator collapses: {len(recent_collapses)}")
    
    # 计算平均值
    if iters_train > 0:
        for key in epoch_metrics:
            if key not in ['warmup_phase', 'collapse_count']:
                epoch_metrics[key] /= iters_train
    
    return epoch_metrics

def save_checkpoint(
    exp_dir: str,
    epoch: int,
    planner: nn.Module,
    discriminator: nn.Module,
    planner_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    is_best: bool = False,
    trainer: Optional['PlannerTrainer'] = None
):
    """保存检查点（仅master进程执行）"""
    if not exp_dir:  # 非master进程
        return
    
    checkpoint = {
        'epoch': epoch,
        'planner_state_dict': planner.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'planner_optimizer_state_dict': planner_optimizer.state_dict(),
        'discriminator_optimizer_state_dict': discriminator_optimizer.state_dict(),
        'train_metrics': train_metrics,
        'val_metrics': val_metrics
    }
    
    # 保存trainer的内部状态（如果提供）
    if trainer is not None:
        trainer_state = {
            'current_step': getattr(trainer, 'current_step', 0),
            'is_warmup_phase': getattr(trainer, 'is_warmup_phase', True),
            'collapse_history': getattr(trainer, 'collapse_history', []),
            'recovery_mode': getattr(trainer, 'recovery_mode', False),
        }
        
        # 保存交替训练管理器状态
        if hasattr(trainer, 'alternating_manager'):
            alternating_state = {
                'step_count': trainer.alternating_manager.step_count,
                'disc_updates': trainer.alternating_manager.disc_updates,
                'planner_updates': trainer.alternating_manager.planner_updates,
                'last_metrics': trainer.alternating_manager.last_metrics,
            }
            trainer_state['alternating_manager'] = alternating_state
        
        checkpoint['trainer_state'] = trainer_state
        logger.info(f"💾 保存trainer状态: step={trainer_state['current_step']}, warmup={trainer_state['is_warmup_phase']}")
    
    # 保存常规检查点
    checkpoint_path = os.path.join(exp_dir, "checkpoints", f"checkpoint_epoch_{epoch+1}.pth")
    torch.save(checkpoint, checkpoint_path)
    
    # 保存最佳检查点
    if is_best:
        best_path = os.path.join(exp_dir, "checkpoints", "best_checkpoint.pth")
        torch.save(checkpoint, best_path)
        logger.info(f"💾 最佳检查点保存: {best_path}")
    
    logger.info(f"💾 检查点保存: {checkpoint_path}")

def load_checkpoint(
    checkpoint_path: str,
    planner: nn.Module,
    discriminator: nn.Module,
    planner_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    device: str = 'cuda',
    trainer: Optional['PlannerTrainer'] = None,
    load_optimizer: bool = True  # 🔥 新增参数：是否加载优化器状态
) -> Tuple[int, Dict[str, float], Dict[str, float]]:
    """从检查点加载训练状态"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
    
    if dist.is_master():
        logger.info(f"🔄 加载检查点: {checkpoint_path}")
    
    # 加载检查点数据
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 恢复模型状态
    try:
        missing_keys, unexpected_keys = planner.load_state_dict(checkpoint['planner_state_dict'], strict=False)
        if unexpected_keys:
            pos_keys = [k for k in unexpected_keys if k.startswith('pos_')]
            other_keys = [k for k in unexpected_keys if not k.startswith('pos_')]
            if pos_keys and dist.is_master():
                logger.info(f"🔄 忽略位置编码buffer: {pos_keys}")
            if other_keys and dist.is_master():
                logger.warning(f"⚠️ 其他未预期的键: {other_keys}")
        if missing_keys and dist.is_master():
            logger.warning(f"⚠️ 缺失的键: {missing_keys}")
        if dist.is_master():
            logger.info("✅ GuidanceInjector模型状态加载成功")
    except Exception as e:
        if dist.is_master():
            logger.error(f"❌ GuidanceInjector模型状态加载失败: {e}")
        raise
    
    try:
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'], strict=True)
        if dist.is_master():
            logger.info("✅ 判别器模型状态加载成功")
    except Exception as e:
        if dist.is_master():
            logger.error(f"❌ 判别器模型状态加载失败: {e}")
        raise
    
    # 恢复优化器状态（可选）
    if load_optimizer:
        try:
            planner_optimizer.load_state_dict(checkpoint['planner_optimizer_state_dict'])
            if dist.is_master():
                logger.info("✅ GuidanceInjector优化器状态加载成功")
        except Exception as e:
            if dist.is_master():
                logger.warning(f"⚠️ GuidanceInjector优化器状态加载失败，将使用默认状态: {e}")
        
        try:
            discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])
            if dist.is_master():
                logger.info("✅ 判别器优化器状态加载成功")
        except Exception as e:
            if dist.is_master():
                logger.warning(f"⚠️ 判别器优化器状态加载失败，将使用默认状态: {e}")
    else:
        if dist.is_master():
            logger.info("⏩ 跳过优化器状态加载，将使用新的学习率设置")
    
    # 获取训练进度和指标
    # 🔥 修复：checkpoint中保存的是刚完成的epoch，下次训练应该从下一个epoch开始
    completed_epoch = checkpoint.get('epoch', -1)  # 已完成的epoch
    start_epoch = completed_epoch + 1  # 下一个要开始的epoch
    train_metrics = checkpoint.get('train_metrics', {})
    val_metrics = checkpoint.get('val_metrics', {})
    
    # 恢复trainer的内部状态（如果提供）
    if trainer is not None and 'trainer_state' in checkpoint:
        trainer_state = checkpoint['trainer_state']
        
        # 恢复基础状态
        if hasattr(trainer, 'current_step'):
            trainer.current_step = trainer_state.get('current_step', 0)
        if hasattr(trainer, 'is_warmup_phase'):
            trainer.is_warmup_phase = trainer_state.get('is_warmup_phase', True)
        if hasattr(trainer, 'collapse_history'):
            trainer.collapse_history = trainer_state.get('collapse_history', [])
        if hasattr(trainer, 'recovery_mode'):
            trainer.recovery_mode = trainer_state.get('recovery_mode', False)
        
        # 恢复交替训练管理器状态
        if hasattr(trainer, 'alternating_manager') and 'alternating_manager' in trainer_state:
            alt_state = trainer_state['alternating_manager']
            trainer.alternating_manager.step_count = alt_state.get('step_count', 0)
            trainer.alternating_manager.disc_updates = alt_state.get('disc_updates', 0)
            trainer.alternating_manager.planner_updates = alt_state.get('planner_updates', 0)
            trainer.alternating_manager.last_metrics = alt_state.get('last_metrics', {})
        
        if dist.is_master():
            logger.info(f"✅ Trainer状态恢复: step={trainer.current_step}, warmup={trainer.is_warmup_phase}")
    elif trainer is not None and dist.is_master():
        logger.warning("⚠️ 检查点中未找到trainer状态，trainer将使用默认状态")
    
    if dist.is_master():
        logger.info(f"📊 检查点信息:")
        logger.info(f"   已完成epoch: {completed_epoch + 1}")
        logger.info(f"   下次开始epoch: {start_epoch + 1}")
        logger.info(f"   训练指标: {train_metrics}")
        logger.info(f"   验证指标: {val_metrics}")
    
    return start_epoch, train_metrics, val_metrics

def main_training():
    """主训练函数"""
    # 初始化分布式训练和参数解析
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    
    # 检查数据路径
    if not args.local_debug and not os.path.exists(args.data_path):
        raise ValueError(f"数据路径不存在: {args.data_path}")
    
    # 如果是local_debug模式，快速测试
    if args.local_debug:
        if dist.is_master():
            print("[local debug] AID-VAR调试模式 - 快速测试")
        # 这里可以添加简单的功能测试
        return
    
    # 创建配置对象并从args更新
    config = AIDVARTrainingConfig()
    
    # 从命令行参数更新配置
    if hasattr(args, 'ep'):
        config.EPOCHS = args.ep
    if hasattr(args, 'resume_checkpoint') and args.resume_checkpoint:
        config.RESUME_CHECKPOINT = args.resume_checkpoint
    
    # 🔥 根据depth参数自动调整配置
    if hasattr(args, 'depth'):
        config.update_for_var_depth(args.depth)
    
    # 🔥 添加学习率命令行参数支持
    if hasattr(args, 'lr_planner') and args.lr_planner is not None:
        config.LEARNING_RATE_PLANNER = args.lr_planner
        if dist.is_master():
            logger.info(f"🔄 从命令行覆盖GuidanceInjector学习率: {args.lr_planner}")
    
    if hasattr(args, 'lr_discriminator') and args.lr_discriminator is not None:
        config.LEARNING_RATE_DISCRIMINATOR = args.lr_discriminator  
        if dist.is_master():
            logger.info(f"🔄 从命令行覆盖判别器学习率: {args.lr_discriminator}")
    
    if hasattr(args, 'load_optimizer') and args.load_optimizer is not None:
        config.LOAD_OPTIMIZER_STATE = args.load_optimizer
        if dist.is_master():
            logger.info(f"🔄 从命令行设置优化器状态加载: {args.load_optimizer}")
        
    try:
        exp_dir = train_full_imagenet(config, args)
        if dist.is_master():
            print(f"\n🎉 分布式训练成功完成！")
            print(f"📁 实验结果: {exp_dir}")
            if exp_dir:
                print(f"📊 查看日志: tail -f {exp_dir}/logs/*.log")
        
    except Exception as e:
        if dist.is_master():
            logger.error(f"❌ 训练失败: {e}")
            import traceback
            traceback.print_exc()
        raise
    finally:
        dist.finalize()

if __name__ == "__main__":
    try:
        main_training()
    finally:
        if dist.initialized():
            dist.finalize() 