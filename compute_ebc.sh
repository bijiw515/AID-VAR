#!/bin/bash
# -*- coding: utf-8 -*-
#
# 🎯 EB-C (Exposure Bias - Consistency) 实验运行脚本
#
# 此脚本用于运行完整的EB-C评估实验，比较基础VAR和AID-VAR的暴露偏差。
# EB-C是用于量化暴露偏差影响的度量指标。
#
# 使用方法:
#   bash compute_ebc.sh [配置选项]
#
# 作者：Expert AI Developer
# 日期：2025-11-19
#

set -e  # 遇到错误时退出

# 🎯 实验配置
echo "🚀 开始EB-C评估实验"
echo "=================================================="

# 默认配置
VAE_CKPT="vae_ch160v4096z32.pth"
VAR_CKPT="checkpoints/var_d20.pth"
PLANNER_CKPT="/home/intern/Ligong/VAR/exp_ckpt/checkpoints/GuidanceInjector_d20.pth"
MODEL_DEPTH=20

# 数据配置
REAL_IMAGES_PATH="VIRTUAL_imagenet256_labeled.npz"  # 真实图像数据
NUM_REAL_SAMPLES=10000  # 分母CGD(M|D)使用的真实样本数（与VIRTUAL_imagenet256_labeled.npz的样本数一致）
NUM_SAMPLES=50000  # 分子CGD(M|M)生成的样本数量（1000类×50样本/类）
BATCH_SIZE=50  # 批处理大小（与compute_ISCS.py一致）

# 生成配置（与generate_aid_fid_samples.py和compute_ISCS.py完全一致）
CFG=1.5
TOP_K=900
TOP_P=0.96
MORE_SMOOTH=""  # 设置为"--more_smooth"以启用，默认不启用

# 系统配置
DEVICE="cuda"
OUTPUT_DIR="./ebc_results_$(date +%Y%m%d_%H%M%S)"

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
    echo "请准备真实图像数据（npz文件），或更新REAL_IMAGES_PATH路径"
    exit 1
fi

echo "✅ 必要文件检查完成"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
echo "📁 输出目录: $OUTPUT_DIR"

# 记录实验配置
cat > "$OUTPUT_DIR/experiment_config.txt" << EOF
EB-C评估实验配置
================

模型配置:
- VAE检查点: $VAE_CKPT
- VAR检查点: $VAR_CKPT
- GuidanceInjector检查点: $PLANNER_CKPT
- 模型深度: $MODEL_DEPTH

数据配置:
- 真实图像路径: $REAL_IMAGES_PATH
- 分母CGD(M|D)样本数: $NUM_REAL_SAMPLES (使用真实图像)
- 分子CGD(M|M)样本数: $NUM_SAMPLES (模型自回归生成)
- 批处理大小: $BATCH_SIZE

生成配置（与generate_aid_fid_samples.py和compute_ISCS.py完全一致）:
- CFG引导强度: $CFG
- Top-k采样: $TOP_K
- Top-p采样: $TOP_P
- More smooth: $([ -n "$MORE_SMOOTH" ] && echo "启用" || echo "禁用")

系统配置:
- 设备: $DEVICE

实验时间: $(date)
EOF

echo "📋 实验配置已保存到: $OUTPUT_DIR/experiment_config.txt"

# 构建Python命令（与compute_ISCS.py参数保持一致）
export CUDA_VISIBLE_DEVICES=2 && PYTHON_CMD="python compute_ebc.py \
    --vae_ckpt $VAE_CKPT \
    --var_ckpt $VAR_CKPT \
    --model_depth $MODEL_DEPTH \
    --real_images_path $REAL_IMAGES_PATH \
    --num_real_samples $NUM_REAL_SAMPLES \
    --num_samples $NUM_SAMPLES \
    --batch_size $BATCH_SIZE \
    --cfg $CFG \
    --top_k $TOP_K \
    --top_p $TOP_P \
    --device $DEVICE \
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

echo "🔄 执行EB-C计算..."
echo "⚠️ 注意："
echo "   - 分母CGD(M|D): 使用 $NUM_REAL_SAMPLES 个真实样本（真实前缀+forward）"
echo "   - 分子CGD(M|M): 生成 $NUM_SAMPLES 个样本（模型自回归inference）"
echo "📋 生成策略：1000类循环分配，批大小=$BATCH_SIZE，以class_id为随机种子"
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
echo "✅ EB-C评估实验完成!"
echo "⏱️ 总耗时: ${DURATION}秒 ($(($DURATION / 60))分钟)"
echo "📁 结果保存在: $OUTPUT_DIR"
echo ""

# 显示结果摘要
if [ -f "$OUTPUT_DIR/ebc_results.json" ]; then
    echo "📊 EB-C评估结果摘要:"
    echo "===================="

    # 使用Python快速解析JSON结果
    python3 -c "
import json
import sys

try:
    with open('$OUTPUT_DIR/ebc_results.json', 'r') as f:
        results = json.load(f)

    print(f\"📈 基础VAR EB-C: {results['var']['mean_ebc']:.4f} ± {results['var']['std_ebc']:.4f}\")

    if 'aid_var' in results:
        aid_ebc = results['aid_var']['mean_ebc']
        base_ebc = results['var']['mean_ebc']
        improvement = ((base_ebc - aid_ebc) / base_ebc) * 100

        print(f\"📈 AID-VAR EB-C: {aid_ebc:.4f} ± {results['aid_var']['std_ebc']:.4f}\")
        print(f\"📊 EB-C改善: {improvement:.2f}%\")

        if aid_ebc < base_ebc:
            print(\"✅ GuidanceInjector有效减少了暴露偏差（EB-C降低）!\")
        else:
            print(\"⚠️ GuidanceInjector未显著减少暴露偏差\")

    print(f\"\\n📋 详细的各尺度分数请查看: $OUTPUT_DIR/ebc_results.json\")
    print(f\"📊 可视化结果请查看: $OUTPUT_DIR/ebc_comparison.png\")

except Exception as e:
    print(f\"❌ 解析结果文件失败: {e}\")
    sys.exit(1)
"
else
    echo "❌ 未找到结果文件，实验可能失败"
fi

echo ""
echo "🎯 实验完成! 感谢使用EB-C评估工具。"
