#!/usr/bin/env python3
"""
🎯 AGIP-VAR单分类训练脚本 - 适配最新架构
专门用于在ImageNet单个分类上验证AGIP-VAR框架有效性
包括：分阶段训练、空间感知规划词元图、判别器准确率监控等

🔄 检查点恢复功能:
- 支持从指定检查点恢复训练
- 自动恢复模型状态、优化器状态和训练进度
- 命令行参数: --resume_checkpoint /path/to/checkpoint.pth

使用示例:
  # 从头开始训练
  python train_single_class.py --class_id 207 --num_epochs 100
  
  # 从检查点恢复训练
  python train_single_class.py --class_id 207 --num_epochs 100 \
    --resume_checkpoint single_class_experiments/agip_var_class_207_staged_20241219_143021/checkpoints/checkpoint_epoch_10.pth
  
  # 从最佳检查点恢复训练
  python train_single_class.py --class_id 207 --num_epochs 100 \
    --resume_checkpoint single_class_experiments/agip_var_class_207_staged_20241219_143021/checkpoints/best_checkpoint.pth
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
import time
from datetime import datetime
from dataclasses import dataclass

# 添加stylegan_t路径
stylegan_path = os.path.join(os.path.dirname(__file__), 'stylegan_t')
if stylegan_path not in sys.path:
    sys.path.append(stylegan_path)

# 导入项目模块
from models import build_vae_var
from models.planning_module import I_predictor
from models.discriminator_adapter import StyleGANDiscriminatorAdapter
from trainer_planner import PlannerTrainer
from utils.single_class_data import create_single_class_dataloaders
from utils.amp_sc import AmpOptimizer
from utils.misc import MetricLogger, TensorboardLogger
import agip_helpers

# 配置日志 - 将在setup_experiment_dir后重新配置到正确路径
logger = logging.getLogger(__name__)

@dataclass
class SingleClassTrainingConfig:
    """AGIP-VAR单类训练配置"""
    
    # 基础配置
    DEVICE: str = 'cuda'
    CLASS_ID: int = 207  # Golden Retriever
    DATA_DIR: str = "data"
    DATA_ROOT: str = "/home/intern/zkx/Dataset/imagenet-1k/datasets--ILSVRC--imagenet-1k/snapshots/4603483700ee984ea9debe3ddbfdeae86f6489eb/data"  # 🔥 添加缺失的DATA_ROOT
    OUTPUT_DIR: str = "single_class_experiments"
    
    # 模型配置
    VAR_CKPT: str = "checkpoints/var_d16.pth"  # 🔥 修复：正确的VAR权重路径
    VQVAE_CKPT: str = "vae_ch160v4096z32.pth"  # 🔥 修复：正确的VQVAE权重路径
    VAR_DEPTH: int = 20  # VAR模型深度（16, 20, 24等）
    
    # 训练配置 
    EPOCHS: int = 100
    BATCH_SIZE: int = 8
    VAL_BATCH_SIZE: int = 4
    NUM_WORKERS: int = 4
    
    # 🔥 关键修复：统一学习率，避免训练动态失衡
    LEARNING_RATE_PLANNER: float = 5e-06       # I_predictor学习率
    LEARNING_RATE_DISCRIMINATOR: float = 1e-06  # 🔥 修复：统一学习率，消除4倍差异导致的动态失衡
    
    # 梯度裁剪
    CLIP_PLANNER: float = 0.5   # 🔥 修复：更保守的梯度裁剪
    CLIP_DISCRIMINATOR: float = 0.5  # 🔥 修复：更保守的梯度裁剪
    
    # AGIP-VAR超参数
    LAMBDA_REC: float = 0.01  # 重构损失权重
    GUIDANCE_WEIGHT: float = 0.005  # 🔥 修复：更保守的初始引导权重
    GUIDANCE_TARGET_WEIGHT: float = 0.001  # 🔥 修复：固定引导权重为0.001
    GUIDANCE_RAMP_EPOCHS: int = 15  # 🔥 修复：更长的渐进增加期
    
    # 🔥 新增：判别器稳定性参数
    R1_GAMMA: float = 0.2  # 🔥 修复：更强的R1梯度惩罚权重
    PROGRESSIVE_GUIDANCE: bool = True  # 渐进式引导权重
    COLLAPSE_DETECTION: bool = True  # 崩溃检测
    
    # 分阶段训练配置
    ENABLE_STAGED_TRAINING: bool = True
    WARMUP_STEPS: int = 0  # 🔥 修复：延长判别器预热期
    
    # 采样参数
    TOP_K: int = 0
    TOP_P: float = 0
    CFG: float = 1.5
    
    # VAR相关
    PATCH_NUMS: tuple = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)  # VAR的多尺度
    
    # I_predictor架构 - 动态根据VAR深度调整
    PLANNER_LAYERS: int = 2
    PLANNER_DIM: int = 1024  # 将根据VAR_DEPTH自动计算：depth * 64
    PLANNER_HEADS: int = 8
    
    # 日志和保存
    LOG_INTERVAL: int = 10
    SAVE_INTERVAL: int = 5  # 每5个epoch保存一次检查点
    MAX_SAMPLES_PER_CLASS: int = 1000  # 每类最大样本数
    VALIDATION_INTERVAL: int = 1  # 每2个epoch验证一次
    
    # 检查点恢复
    RESUME_CHECKPOINT: Optional[str] = None  # 恢复训练的检查点路径
    
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
        
        logger.info(f"🔧 单分类训练配置已更新为d{depth}模型:")
        logger.info(f"   VAR深度: {self.VAR_DEPTH}")
        logger.info(f"   I_predictor维度: {self.PLANNER_DIM}")
        logger.info(f"   VAR权重路径: {self.VAR_CKPT}")

def setup_experiment_dir(config: SingleClassTrainingConfig) -> str:
    """设置实验目录并配置日志"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    staged_suffix = "staged" if config.ENABLE_STAGED_TRAINING else "joint"
    exp_name = f"agip_var_class_{config.CLASS_ID}_{staged_suffix}_{timestamp}"
    exp_dir = os.path.join(config.OUTPUT_DIR, exp_name)
    
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "validation"), exist_ok=True)
    
    # 配置日志到实验目录
    log_file = os.path.join(exp_dir, "logs", "single_class_training.log")
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
    with open(os.path.join(exp_dir, "config.json"), 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    
    logger.info(f"🗂️ 实验目录创建: {exp_dir}")
    logger.info(f"📝 日志文件: {log_file}")
    logger.info(f"🎯 训练模式: {'分阶段训练' if config.ENABLE_STAGED_TRAINING else '联合训练'}")
    return exp_dir

def load_models(config: SingleClassTrainingConfig) -> Tuple[nn.Module, nn.Module, nn.Module, nn.Module]:
    """加载所有模型 - 适配AGIP-VAR架构"""
    logger.info("📦 加载AGIP-VAR模型架构...")
    
    # 🔥 重要修复：使用正确的VQ-VAE配置参数
    # 根据vae_ch160v4096z32.pth的实际配置
    vqvae, var = build_vae_var(
        V=4096,      # 词汇表大小，与文件名中的4096匹配
        Cvae=32,     # VQ-VAE的特征维度，与文件名中的32匹配  
        ch=160,      # 通道数，与文件名中的160匹配
        share_quant_resi=4,
        device=config.DEVICE, 
        patch_nums=config.PATCH_NUMS,
        num_classes=1000,
        depth=config.VAR_DEPTH,  # 🔥 使用配置中的VAR深度
        shared_aln=False,
        attn_l2_norm=True,
        flash_if_available=False,
        fused_if_available=False,
        init_adaln=0.5,
        init_adaln_gamma=1e-5,
        init_head=0.02,
        init_std=-1
    )
    
    # 加载预训练权重
    if os.path.exists(config.VAR_CKPT):
        var_state = torch.load(config.VAR_CKPT, map_location=config.DEVICE)
        var.load_state_dict(var_state, strict=True)
        logger.info(f"✅ VAR模型权重加载: {config.VAR_CKPT}")
    else:
        raise FileNotFoundError(f"VAR权重文件不存在: {config.VAR_CKPT}")
    
    if os.path.exists(config.VQVAE_CKPT):
        vqvae_state = torch.load(config.VQVAE_CKPT, map_location=config.DEVICE)
        vqvae.load_state_dict(vqvae_state, strict=True)
        logger.info(f"✅ VQVAE模型权重加载: {config.VQVAE_CKPT}")
    else:
        raise FileNotFoundError(f"VQVAE权重文件不存在: {config.VQVAE_CKPT}")
    
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
    logger.info(f"🧊 冻结VAE参数: {frozen_vae_params:,}")
    logger.info(f"🧊 冻结VAR参数: {frozen_var_params:,}")
    
    # 🔥 验证VAR和VQ-VAE的配置匹配
    logger.info(f"📊 配置验证:")
    logger.info(f"   VQ-VAE embedding 形状: {vqvae.quantize.embedding.weight.shape}")  # 应该是[4096, 32]
    logger.info(f"   VAR word_embed输入维度: {var.word_embed.in_features}")  # 应该是32
    logger.info(f"   VAR word_embed输出维度: {var.word_embed.out_features}")  # 应该是1024
    
    # 创建AGIP-VAR组件 - 🔥 使用VAR的实际嵌入维度
    planner = I_predictor(
        input_dim=var.C,  # VAR的特征维度 (动态：16*64=1024 或 20*64=1280)
        embed_dim=var.C,  # 规划器嵌入维度与VAR保持一致
        num_layers=config.PLANNER_LAYERS,
        num_heads=config.PLANNER_HEADS
    ).to(config.DEVICE)
    
    # 🔥 修复：使用新的像素级判别器适配器
    discriminator = StyleGANDiscriminatorAdapter(
        vae=vqvae,  # VQ-VAE实例用于Token→RGB解码
        patch_nums=config.PATCH_NUMS,  # 多尺度patch配置
        img_resolution=256,  # 图像分辨率
        c_dim=0  # 类别条件维度
    ).to(config.DEVICE)
    
    # 统计可训练参数
    planner_params = sum(p.numel() for p in planner.parameters() if p.requires_grad)
    disc_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    total_params = planner_params + disc_params
    
    logger.info(f"📊 AGIP-VAR可训练参数统计:")
    logger.info(f"   I_predictor: {planner_params:,} 参数 ({planner_params/1e6:.2f}M)")
    logger.info(f"   Discriminator: {disc_params:,} 参数 ({disc_params/1e6:.2f}M)")
    logger.info(f"   总计: {total_params:,} 参数 ({total_params/1e6:.2f}M)")
    logger.info(f"🎯 模式: 空间感知规划词元图（逐位置相加）")
    
    return vqvae, var, planner, discriminator

def create_optimizers(planner: nn.Module, discriminator: nn.Module, config: SingleClassTrainingConfig) -> Tuple[AmpOptimizer, AmpOptimizer]:
    """创建优化器"""
    logger.info("⚙️ 创建AGIP-VAR优化器...")
    
    # 获取可训练参数
    planner_params = list(planner.parameters())
    
    # 🔥 修复：使用新的像素级判别器的可训练参数获取方式
    try:
        disc_trainable_params = discriminator.get_trainable_params()
        logger.info("✅ 使用像素级判别器的可训练参数")
    except AttributeError:
        # 兼容fallback: 获取所有可训练参数
        disc_trainable_params = [p for p in discriminator.parameters() if p.requires_grad]
        logger.warning("⚠️ 回退到所有可训练参数获取方式")
    
    if not disc_trainable_params:
        # 创建dummy参数以避免优化器错误
        dummy_param = torch.nn.Parameter(torch.tensor(0.0, device=config.DEVICE))
        disc_trainable_params = [dummy_param]
        logger.warning("⚠️ 判别器head参数为空，使用dummy参数")
    
    # 统计参数
    planner_param_count = sum(p.numel() for p in planner_params) / 1e6
    disc_param_count = sum(p.numel() for p in disc_trainable_params) / 1e6
    logger.info(f"📊 可训练参数统计:")
    logger.info(f"   I_predictor: {planner_param_count:.2f}M 参数")
    logger.info(f"   StyleGAN-T Discriminator (heads only): {disc_param_count:.2f}M 参数")
    logger.info(f"   🎯 ADD风格: 冻结DINO backbone + 训练判别器heads")
    
    # 创建AmpOptimizer - I_predictor
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
    
    logger.info(f"✅ AmpOptimizer创建完成")
    logger.info(f"   I_predictor学习率: {config.LEARNING_RATE_PLANNER}")
    logger.info(f"   Discriminator学习率: {config.LEARNING_RATE_DISCRIMINATOR}")
    
    return amp_planner_opt, amp_disc_opt

def create_data_loaders(config: SingleClassTrainingConfig) -> Tuple[DataLoader, DataLoader]:
    """创建数据加载器"""
    logger.info("📊 创建单分类数据加载器...")
    
    try:
        train_loader, val_loader = create_single_class_dataloaders(
            data_root=config.DATA_ROOT,
            class_id=config.CLASS_ID,
            batch_size=config.BATCH_SIZE,
            num_workers=0,  # 🔧 修复：禁用多进程避免/tmp空间不足
            max_train_samples=config.MAX_SAMPLES_PER_CLASS,
            max_val_samples=config.MAX_SAMPLES_PER_CLASS // 2 # 验证集通常是训练集的一半
        )
        
        logger.info(f"✅ 数据加载器创建成功")
        logger.info(f"   训练批次数: {len(train_loader)}")
        logger.info(f"   验证批次数: {len(val_loader)}")
        
        return train_loader, val_loader
        
    except Exception as e:
        logger.error(f"❌ 数据加载器创建失败: {e}")
        raise

def train_single_class(config: SingleClassTrainingConfig) -> str:
    """
    执行AGIP-VAR单分类训练
    
    Returns:
        exp_dir: 实验目录路径
    """
    logger.info("🚀 开始AGIP-VAR单分类训练...")
    logger.info(f"   类别ID: {config.CLASS_ID}")
    logger.info(f"   设备: {config.DEVICE}")
    logger.info(f"   批次大小: {config.BATCH_SIZE}")
    logger.info(f"   训练轮数: {config.EPOCHS}")
    if config.ENABLE_STAGED_TRAINING:
        logger.info(f"🎯 分阶段训练: 预热步数={config.WARMUP_STEPS}")
    else:
        logger.info(f"🎯 联合训练模式")
    
    # 设置实验目录
    exp_dir = setup_experiment_dir(config)
    
    # 加载模型
    vqvae, var, planner, discriminator = load_models(config)
    
    # 创建优化器
    amp_planner_opt, amp_disc_opt = create_optimizers(planner, discriminator, config)
    
    # 创建数据加载器
    train_loader, val_loader = create_data_loaders(config)
    
    # 创建AGIP-VAR训练器
    trainer = PlannerTrainer(
        device=config.DEVICE,
        vae=vqvae,
        var=var,
        planner=planner,
        disc=discriminator,
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
        # 🔥 修复：使用正确的参数名称
        alternating_strategy="adaptive",
        guidance_weight_max=config.GUIDANCE_TARGET_WEIGHT,
    )
    
    # 创建度量记录器
    metric_logger = MetricLogger(delimiter='  ')
    
    # 设置TensorBoard日志
    tb_log_dir = os.path.join(exp_dir, "logs", "tensorboard")
    os.makedirs(tb_log_dir, exist_ok=True)
    tb_logger = TensorboardLogger(log_dir=tb_log_dir, filename_suffix='agip_var_single_class')
    
    # 检查是否需要从checkpoint恢复
    start_epoch = 0
    if config.RESUME_CHECKPOINT and os.path.exists(config.RESUME_CHECKPOINT):
        logger.info(f"🔄 检测到恢复检查点: {config.RESUME_CHECKPOINT}")
        try:
            start_epoch, resume_train_metrics, resume_val_metrics = load_checkpoint(
                config.RESUME_CHECKPOINT,
                planner,
                discriminator,
                amp_planner_opt.optimizer,
                amp_disc_opt.optimizer,
                config.DEVICE,
                trainer  # 传入trainer以恢复其内部状态
            )
            logger.info(f"✅ 成功从epoch {start_epoch}恢复训练")
            # 从恢复的验证指标中获取最佳损失
            best_val_loss = resume_val_metrics.get('loss', float('inf'))
        except Exception as e:
            logger.error(f"❌ 检查点加载失败: {e}")
            logger.info("🔄 将从头开始训练")
            start_epoch = 0
    elif config.RESUME_CHECKPOINT:
        logger.warning(f"⚠️ 指定的检查点文件不存在: {config.RESUME_CHECKPOINT}")
        logger.info("🔄 将从头开始训练")
    
    # 训练循环
    logger.info("🔄 开始训练循环...")
    best_val_loss = float('inf') if start_epoch == 0 else best_val_loss
    
    for epoch in range(start_epoch, config.EPOCHS):
        logger.info(f"📅 Epoch {epoch+1}/{config.EPOCHS}")
        
        # 训练一个epoch
        train_metrics = train_one_epoch(
            trainer, train_loader, epoch, config, metric_logger, tb_logger
        )
        
        # 验证（如果需要）
        val_metrics = {}
        if (epoch + 1) % config.VALIDATION_INTERVAL == 0: # 改为按保存间隔验证
            val_metrics = trainer.validate_one_epoch(
                val_loader, epoch, exp_dir
            )
            
            # 记录验证指标
            for key, value in val_metrics.items():
                if isinstance(value, (int, float)):  # 只记录数值类型
                    tb_logger.update(head='val', step=epoch, **{key: value})
        
        # 保存检查点
        if (epoch + 1) % config.SAVE_INTERVAL == 0: # 改为按保存间隔保存
            is_best = val_metrics.get('loss', float('inf')) < best_val_loss
            if is_best:
                best_val_loss = val_metrics.get('loss', float('inf'))
                
            save_checkpoint(
                exp_dir, epoch, planner, discriminator,
                amp_planner_opt.optimizer, amp_disc_opt.optimizer,
                train_metrics, val_metrics, is_best, trainer  # 传入trainer以保存其内部状态
            )
        
        # 记录训练指标
        for key, value in train_metrics.items():
            tb_logger.update(head='train', step=epoch, **{key: value})
    
    logger.info(f"🎉 AGIP-VAR单分类训练完成！")
    logger.info(f"📁 实验结果保存在: {exp_dir}")
    
    return exp_dir

def train_one_epoch(
    trainer: PlannerTrainer, 
    train_loader: DataLoader, 
    epoch: int,
    config: SingleClassTrainingConfig,
    metric_logger: MetricLogger,
    tb_logger: TensorboardLogger
) -> Dict[str, float]:
    """训练一个epoch"""
    
    # 🔥 修复：PlannerTrainer不需要设置current_epoch，它内部管理步骤计数
    # trainer.current_epoch = epoch  # 注释掉，因为PlannerTrainer使用current_step内部管理
    
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
        # 🔥 新增：崩溃检测指标
        'collapse_count': 0,
        'guidance_weight': 0.0,
        'r1_penalty': 0.0,
    }
    
    num_batches = len(train_loader)
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        # 执行AGIP-VAR逐尺度训练步骤
        step_metrics = trainer.train_step(images, labels, metric_logger)
        
        # 🔥 修复：直接使用trainer返回的正确指标名称
        # trainer.train_step 返回的字典包含：
        # 'loss_D', 'loss_P', 'loss_adversarial', 'loss_reconstruction', 
        # 'acc_D', 'acc_real', 'acc_fake', 'valid_scales', 'warmup_phase', 'current_step'
        
        # 直接映射到epoch指标
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
        
        # 记录当前步骤的指标
        if batch_idx % config.LOG_INTERVAL == 0:  # 每10个batch打印一次
            warmup_phase = step_metrics.get('warmup_phase', False)
            phase_str = "预热" if warmup_phase else "联合"
            
            logger.info(f"  Batch {batch_idx}/{num_batches}: [{phase_str}] "
                       f"Loss_D={step_metrics.get('loss_D', 0):.4f}, "
                       f"Loss_P={step_metrics.get('loss_P', 0):.4f}, "
                       f"Acc_D={step_metrics.get('acc_D', 0):.3f}, "
                       f"real_acc={step_metrics.get('acc_real', 0):.3f}, "
                       f"fake_acc={step_metrics.get('acc_fake', 0):.3f}, "
                       f"引导权重={current_guidance_weight:.4f}")
                       
            # 崩溃警告
            if hasattr(trainer, 'collapse_history') and len(trainer.collapse_history) > 0:
                recent_collapses = [c for c in trainer.collapse_history if c['collapsed']]
                if recent_collapses:
                    logger.warning(f"  🚨 检测到 {len(recent_collapses)} 次判别器崩溃")
    
    # 计算平均值
    if num_batches > 0:
        for key in epoch_metrics:
            if key not in ['warmup_phase', 'collapse_count']:
                epoch_metrics[key] /= num_batches
    
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
    trainer: Optional['PlannerTrainer'] = None  # 添加trainer参数以保存其内部状态
):
    """保存检查点"""
    
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
    trainer: Optional['PlannerTrainer'] = None  # 添加trainer参数以恢复其内部状态
) -> Tuple[int, Dict[str, float], Dict[str, float]]:
    """
    从检查点加载训练状态
    
    Args:
        checkpoint_path: 检查点文件路径
        planner: I_predictor模型
        discriminator: 判别器模型
        planner_optimizer: I_predictor优化器
        discriminator_optimizer: 判别器优化器
        device: 设备
        trainer: PlannerTrainer实例（可选，用于恢复内部状态）
        
    Returns:
        Tuple[起始epoch, 训练指标, 验证指标]
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
    
    logger.info(f"🔄 加载检查点: {checkpoint_path}")
    
    # 加载检查点数据
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 恢复模型状态
    try:
        # 对于I_predictor，使用strict=False来允许位置编码buffer的差异
        # 这些buffer会在首次使用时自动创建
        missing_keys, unexpected_keys = planner.load_state_dict(checkpoint['planner_state_dict'], strict=False)
        if unexpected_keys:
            # 过滤出位置编码相关的keys
            pos_keys = [k for k in unexpected_keys if k.startswith('pos_')]
            other_keys = [k for k in unexpected_keys if not k.startswith('pos_')]
            if pos_keys:
                logger.info(f"🔄 忽略位置编码buffer: {pos_keys}")
            if other_keys:
                logger.warning(f"⚠️ 其他未预期的键: {other_keys}")
        if missing_keys:
            logger.warning(f"⚠️ 缺失的键: {missing_keys}")
        logger.info("✅ I_predictor模型状态加载成功")
    except Exception as e:
        logger.error(f"❌ I_predictor模型状态加载失败: {e}")
        raise
    
    try:
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'], strict=True)
        logger.info("✅ 判别器模型状态加载成功")
    except Exception as e:
        logger.error(f"❌ 判别器模型状态加载失败: {e}")
        raise
    
    # 恢复优化器状态
    try:
        planner_optimizer.load_state_dict(checkpoint['planner_optimizer_state_dict'])
        logger.info("✅ I_predictor优化器状态加载成功")
    except Exception as e:
        logger.warning(f"⚠️ I_predictor优化器状态加载失败，将使用默认状态: {e}")
    
    try:
        discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])
        logger.info("✅ 判别器优化器状态加载成功")
    except Exception as e:
        logger.warning(f"⚠️ 判别器优化器状态加载失败，将使用默认状态: {e}")
    
    # 获取训练进度和指标
    start_epoch = checkpoint.get('epoch', 0)
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
        
        logger.info(f"✅ Trainer状态恢复: step={trainer.current_step}, warmup={trainer.is_warmup_phase}")
    elif trainer is not None:
        logger.warning("⚠️ 检查点中未找到trainer状态，trainer将使用默认状态")
    
    logger.info(f"📊 检查点信息:")
    logger.info(f"   起始epoch: {start_epoch + 1}")
    logger.info(f"   训练指标: {train_metrics}")
    logger.info(f"   验证指标: {val_metrics}")
    
    return start_epoch, train_metrics, val_metrics

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='AGIP-VAR单分类训练')
    
    # 数据配置
    parser.add_argument('--data_root', type=str, default=SingleClassTrainingConfig.DATA_ROOT,
                       help='ImageNet数据根目录')
    parser.add_argument('--class_id', type=int, default=SingleClassTrainingConfig.CLASS_ID,
                       help='训练的类别ID')
    parser.add_argument('--batch_size', type=int, default=SingleClassTrainingConfig.BATCH_SIZE,
                       help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=SingleClassTrainingConfig.EPOCHS,
                       help='训练轮数')
    parser.add_argument('--max_train_samples', type=int, default=SingleClassTrainingConfig.MAX_SAMPLES_PER_CLASS,
                       help='最大训练样本数')
    parser.add_argument('--device', type=str, default=SingleClassTrainingConfig.DEVICE,
                       help='训练设备')
    
    # AGIP-VAR配置
    parser.add_argument('--var_depth', type=int, default=SingleClassTrainingConfig.VAR_DEPTH,
                       help='VAR模型深度: 16/20/24')
    parser.add_argument('--warmup_steps', type=int, default=SingleClassTrainingConfig.WARMUP_STEPS,
                       help='判别器预热步数')
    parser.add_argument('--enable_staged_training', type=bool, default=SingleClassTrainingConfig.ENABLE_STAGED_TRAINING,
                       help='是否启用分阶段训练')
    parser.add_argument('--lambda_rec', type=float, default=SingleClassTrainingConfig.LAMBDA_REC,
                       help='重建损失权重')
    parser.add_argument('--lr_discriminator', type=float, default=SingleClassTrainingConfig.LEARNING_RATE_DISCRIMINATOR,
                       help='判别器学习率')
    
    # 检查点恢复
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                       help='从指定检查点恢复训练的文件路径')
    
    args = parser.parse_args()
    
    # 更新配置
    config = SingleClassTrainingConfig()
    
    # 🔥 首先根据depth参数更新VAR相关配置
    if hasattr(args, 'var_depth'):
        config.update_for_var_depth(args.var_depth)
    
    config.DATA_ROOT = args.data_root
    config.CLASS_ID = args.class_id
    config.BATCH_SIZE = args.batch_size
    config.EPOCHS = args.num_epochs
    config.MAX_SAMPLES_PER_CLASS = args.max_train_samples
    config.DEVICE = args.device
    config.WARMUP_STEPS = args.warmup_steps
    config.ENABLE_STAGED_TRAINING = args.enable_staged_training
    config.LAMBDA_REC = args.lambda_rec
    config.LEARNING_RATE_DISCRIMINATOR = args.lr_discriminator
    config.RESUME_CHECKPOINT = args.resume_checkpoint
    
    try:
        exp_dir = train_single_class(config)
        print(f"\n🎉 训练成功完成！")
        print(f"📁 实验结果: {exp_dir}")
        print(f"📊 查看日志: tail -f {exp_dir}/logs/*.log")
        
    except Exception as e:
        logger.error(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 