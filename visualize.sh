#!/bin/bash

# AID-VAR Guidance可视化运行脚本
# 生成50K张图片的guidance可视化，不保存原图

echo "🎯 启动AID-VAR Guidance可视化生成..."
echo "📊 只保存guidance可视化结果，节省存储空间"

# 基础配置
MODEL_DEPTH=16
CFG=5.0
TOP_P=0.96
TOP_K=900
GUIDANCE_ALPHA=0.4

# 检查点路径 - 请根据实际路径修改
PLANNER_CKPT="/home/intern/Ligong/VAR/experiments/aid_var_full_imagenet_staged_20250901_043215/checkpoints/checkpoint_epoch_9.pth"

# 检查GuidanceInjector检查点是否存在
if [ ! -f "$PLANNER_CKPT" ]; then
    echo "❌ 错误: GuidanceInjector检查点不存在: $PLANNER_CKPT"
    echo "请修改PLANNER_CKPT变量为正确的检查点路径"
    exit 1
fi

echo "✅ GuidanceInjector检查点: $PLANNER_CKPT"

# 运行guidance可视化生成
export CUDA_VISIBLE_DEVICES=7 && python visualize_guidance.py \
    --planner_ckpt "$PLANNER_CKPT" \
    --model_depth $MODEL_DEPTH \
    --cfg $CFG \
    --top_p $TOP_P \
    --top_k $TOP_K \
    --guidance_alpha $GUIDANCE_ALPHA \
    --save_format viz_only \
    --save_guidance_viz \
    --output_dir guidance_visualization \
    --device cuda \
