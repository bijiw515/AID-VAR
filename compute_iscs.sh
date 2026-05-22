#!/bin/bash
# -*- coding: utf-8 -*-
#
# 🎯 ISCS (Inter-Scale Consistency Score) 实验运行脚本
#
# 此脚本用于运行完整的ISCS评估实验，比较基础VAR和AID-VAR的性能。
# ISCS是专门为量化GuidanceInjector解决的"误差"而设计的新指标。
#
# 使用方法:
#   bash run_iscs_experiment.sh [配置选项]
#
# 作者：Expert AI Developer
# 日期：2025-09-23
#

set -e  # 遇到错误时退出

# 🎯 实验配置
echo "🚀 开始ISCS评估实验"
echo "=================================================="

# 默认配置
VAE_CKPT="vae_ch160v4096z32.pth"
VAR_CKPT="checkpoints/var_d20.pth"
PLANNER_CKPT="/home/intern/Ligong/VAR/exp_ckpt/checkpoints/GuidanceInjector_d20.pth"
MODEL_DEPTH=20

# 数据配置
REAL_IMAGES_PATH="VIRTUAL_imagenet256_labeled.npz"  # 真实图像数据
NUM_SAMPLES=50000  # 评估样本数量（用于ISCS计算）

# 生成配置（与generate_aid_fid_samples.py保持一致）
CFG=1.5
TOP_K=900
TOP_P=0.96
MORE_SMOOTH=""  # 设置为"--more_smooth"以启用，默认不启用
DTYPE="float16"

# 系统配置
DEVICE="cuda"
BATCH_SIZE=50  # 固定为50，与generate_aid_fid_samples.py一致（1类=50样本=1批次）
OUTPUT_DIR="./iscs_results_$(date +%Y%m%d_%H%M%S)"

# DINOv2配置
DINOV2_MODEL="dinov2_vits14"  # 可选: dinov2_vitb14, dinov2_vitl14

# 检查必要文件
echo "🔍 检查必要文件..."

if [ ! -f "$VAE_CKPT" ]; then
    echo "❌ VAE检查点不存在: $VAE_CKPT"
    echo "请确保VAE模型文件存在，或更新VAE_CKPT路径"
    exit 1
fi

if [ ! -f "$VAR_CKPT" ]; then
    echo "❌ VAR检查点不存在: $VAR_CKPT"
    echo "请确保VAR模型文件存在，或更新VAR_CKPT路径"
    exit 1
fi

if [ ! -f "$REAL_IMAGES_PATH" ]; then
    echo "❌ 真实图像数据不存在: $REAL_IMAGES_PATH"
    echo "请准备真实图像数据（目录或npz文件），或更新REAL_IMAGES_PATH路径"
    exit 1
fi

echo "✅ 必要文件检查完成"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
echo "📁 输出目录: $OUTPUT_DIR"

# 记录实验配置
cat > "$OUTPUT_DIR/experiment_config.txt" << EOF
ISCS评估实验配置
================

模型配置:
- VAE检查点: $VAE_CKPT
- VAR检查点: $VAR_CKPT
- GuidanceInjector检查点: $PLANNER_CKPT
- 模型深度: $MODEL_DEPTH

数据配置:
- 真实图像路径: $REAL_IMAGES_PATH
- 评估样本数量: $NUM_SAMPLES

生成配置（与generate_aid_fid_samples.py一致）:
- CFG引导强度: $CFG
- Top-k采样: $TOP_K
- Top-p采样: $TOP_P
- 数据类型: $DTYPE
- More smooth: $([ -n "$MORE_SMOOTH" ] && echo "启用" || echo "禁用")

特征配置:
- DINOv2模型: $DINOV2_MODEL

系统配置:
- 设备: $DEVICE
- 批次大小: $BATCH_SIZE（固定，与generate_aid_fid_samples.py一致）
- 生成策略: 1000类 × 50样本/类，以类别ID为随机种子

实验时间: $(date)
EOF

echo "📋 实验配置已保存到: $OUTPUT_DIR/experiment_config.txt"

# 构建Python命令（与generate_aid_fid_samples.py参数保持一致）
PYTHON_CMD="python compute_ISCS.py \
    --vae_ckpt $VAE_CKPT \
    --var_ckpt $VAR_CKPT \
    --model_depth $MODEL_DEPTH \
    --real_images_path $REAL_IMAGES_PATH \
    --num_samples $NUM_SAMPLES \
    --cfg $CFG \
    --top_k $TOP_K \
    --top_p $TOP_P \
    --dtype $DTYPE \
    --dinov2_model $DINOV2_MODEL \
    --device $DEVICE \
    --batch_size $BATCH_SIZE \
    --output_dir $OUTPUT_DIR"

# 添加more_smooth参数（如果启用）
if [ -n "$MORE_SMOOTH" ]; then
    PYTHON_CMD="$PYTHON_CMD $MORE_SMOOTH"
fi

# 如果GuidanceInjector检查点存在，添加到命令中
if [ -f "$PLANNER_CKPT" ]; then
    PYTHON_CMD="$PYTHON_CMD --planner_ckpt $PLANNER_CKPT"
    echo "✅ 将评估AID-VAR (with GuidanceInjector)"
else
    echo "⚠️ GuidanceInjector检查点不存在，仅评估基础VAR"
fi

echo "🔄 执行ISCS计算..."
echo "⚠️ 注意：将生成 $NUM_SAMPLES 张样本用于ISCS计算（样本不会保存到磁盘）"
echo "📋 生成策略：1000类 × 50样本/类，批次大小=50，以类别ID为随机种子"
echo "⏱️ 预计运行时间：2-3小时（取决于硬件配置）"
echo "💾 内存使用：流式处理，GPU内存占用较低"
echo "命令: $PYTHON_CMD"
echo "=================================================="

# 记录开始时间
START_TIME=$(date +%s)

# 执行Python脚本
$PYTHON_CMD 2>&1 | tee "$OUTPUT_DIR/experiment.log"

# 记录结束时间
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "=================================================="
echo "✅ ISCS评估实验完成!"
echo "⏱️ 总耗时: ${DURATION}秒 ($(($DURATION / 60))分钟)"
echo "📁 结果保存在: $OUTPUT_DIR"
echo ""

# 显示结果摘要
if [ -f "$OUTPUT_DIR/iscs_results.json" ]; then
    echo "📊 ISCS评估结果摘要:"
    echo "===================="
    
    # 使用Python快速解析JSON结果
    python3 -c "
import json
import sys

try:
    with open('$OUTPUT_DIR/iscs_results.json', 'r') as f:
        results = json.load(f)
    
    print(f\"📈 基础VAR ISCS: {results['base_var']['total_iscs']:.4f}\")
    
    if 'aid_var' in results:
        aid_score = results['aid_var']['total_iscs']
        base_score = results['base_var']['total_iscs']
        improvement = base_score - aid_score  # 距离减少为正改善
        improvement_pct = (improvement / base_score) * 100
        
        print(f\"📈 AID-VAR ISCS: {aid_score:.4f}\")
        print(f\"📊 ISCS改善: {improvement:.4f} ({improvement_pct:.2f}%)\")
        
        if improvement > 0:
            print(\"✅ GuidanceInjector有效提升了尺度间一致性（距离减少）!\")
        else:
            print(\"⚠️ GuidanceInjector未显著提升尺度间一致性（距离未减少）\")
    
    print(f\"\\n📋 详细的各尺度分数请查看: $OUTPUT_DIR/iscs_results.json\")
    print(f\"📊 可视化结果请查看: $OUTPUT_DIR/iscs_comparison.png\")
    
except Exception as e:
    print(f\"❌ 解析结果文件失败: {e}\")
    sys.exit(1)
"
else
    echo "❌ 未找到结果文件，实验可能失败"
fi

echo ""
echo "🎯 实验完成! 感谢使用ISCS评估工具。" 