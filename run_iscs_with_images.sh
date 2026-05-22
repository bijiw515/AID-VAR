#!/bin/bash

# 🎯 运行增强版ISCS计算脚本 - 生成样本并保存带ISCS分数标注的图片
# 
# 这个脚本运行修改后的compute_ISCS.py，它会：
# 1. 生成样本用于ISCS计算
# 2. 保存前20张生成的图片
# 3. 在每张图片上标注其ISCS分数
#
# 作者：Expert AI Developer
# 日期：2025-09-25

echo "🚀 开始运行增强版ISCS计算实验..."
echo "📸 该版本会保存前20张生成图片并标注ISCS分数"

# 设置基本参数
MODEL_DEPTH=16  # VAR模型深度
NUM_SAMPLES=1000  # 评估样本数量（设置较小的值用于测试）
BATCH_SIZE=8     # 批次大小
CFG=1.5          # 分类器无关引导强度
TOP_K=900        # Top-k采样参数
TOP_P=0.96       # Top-p采样参数

# 设置路径
VAE_CKPT="vae_ch160v4096z32.pth"
VAR_CKPT="checkpoints/var_d${MODEL_DEPTH}.pth"
PLANNER_CKPT=""  # GuidanceInjector检查点路径（如果有的话）
REAL_IMAGES_PATH="VIRTUAL_imagenet256_labeled.npz"  # 使用现有的虚拟ImageNet数据

# 设置输出目录
OUTPUT_DIR="./iscs_results_with_images_$(date +%Y%m%d_%H%M%S)"

echo "📋 实验配置："
echo "  🎯 模型深度: ${MODEL_DEPTH}"
echo "  📊 样本数量: ${NUM_SAMPLES}"
echo "  📦 批次大小: ${BATCH_SIZE}"
echo "  🎮 CFG引导强度: ${CFG}"
echo "  🎲 Top-k采样: ${TOP_K}"
echo "  🎲 Top-p采样: ${TOP_P}"
echo "  📁 输出目录: ${OUTPUT_DIR}"
echo ""

# 检查必要文件是否存在
echo "🔍 检查必要文件..."

if [ ! -f "${VAE_CKPT}" ]; then
    echo "❌ VAE检查点文件不存在: ${VAE_CKPT}"
    echo "💡 请确保该文件存在或修改脚本中的路径"
    exit 1
fi

if [ ! -f "${VAR_CKPT}" ]; then
    echo "❌ VAR检查点文件不存在: ${VAR_CKPT}"
    echo "💡 请确保该文件存在或修改脚本中的路径"
    exit 1
fi

if [ ! -f "${REAL_IMAGES_PATH}" ]; then
    echo "❌ 真实图像数据文件不存在: ${REAL_IMAGES_PATH}"
    echo "💡 请确保该文件存在或修改脚本中的路径"
    exit 1
fi

echo "✅ 所有必要文件检查完成"
echo ""

# 构建命令行参数
CMD_ARGS=""
CMD_ARGS="${CMD_ARGS} --vae_ckpt ${VAE_CKPT}"
CMD_ARGS="${CMD_ARGS} --var_ckpt ${VAR_CKPT}"
CMD_ARGS="${CMD_ARGS} --model_depth ${MODEL_DEPTH}"
CMD_ARGS="${CMD_ARGS} --real_images_path ${REAL_IMAGES_PATH}"
CMD_ARGS="${CMD_ARGS} --num_samples ${NUM_SAMPLES}"
CMD_ARGS="${CMD_ARGS} --batch_size ${BATCH_SIZE}"
CMD_ARGS="${CMD_ARGS} --output_dir ${OUTPUT_DIR}"
CMD_ARGS="${CMD_ARGS} --cfg ${CFG}"
CMD_ARGS="${CMD_ARGS} --top_k ${TOP_K}"
CMD_ARGS="${CMD_ARGS} --top_p ${TOP_P}"
CMD_ARGS="${CMD_ARGS} --dtype float16"
CMD_ARGS="${CMD_ARGS} --device cuda"

# 如果有GuidanceInjector检查点，添加到参数中
if [ ! -z "${PLANNER_CKPT}" ] && [ -f "${PLANNER_CKPT}" ]; then
    CMD_ARGS="${CMD_ARGS} --planner_ckpt ${PLANNER_CKPT}"
    echo "🎯 将使用GuidanceInjector: ${PLANNER_CKPT}"
else
    echo "📊 将仅评估基础VAR模型（未指定GuidanceInjector）"
fi

echo "🚀 启动ISCS计算实验..."
echo "💬 完整命令："
echo "python compute_ISCS.py ${CMD_ARGS}"
echo ""

# 运行实验
python compute_ISCS.py ${CMD_ARGS}

# 检查结果
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ ISCS计算实验成功完成！"
    echo ""
    echo "📁 结果文件位置："
    echo "  📊 ISCS分数结果: ${OUTPUT_DIR}/iscs_results.json"
    echo "  📈 可视化图表: ${OUTPUT_DIR}/iscs_comparison.png"
    echo "  📸 带分数标注的样本图片: ${OUTPUT_DIR}/sample_images_with_scores/"
    echo ""
    echo "🔍 查看样本图片："
    echo "  ls -la ${OUTPUT_DIR}/sample_images_with_scores/"
    echo ""
    echo "📖 查看ISCS结果："
    echo "  cat ${OUTPUT_DIR}/iscs_results.json"
else
    echo ""
    echo "❌ ISCS计算实验失败"
    echo "💡 请检查上述错误信息并重试"
    exit 1
fi 