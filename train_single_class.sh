#!/bin/bash

# ===============================================
# 🎯 AGIP-VAR 单分类训练脚本
# 用于在ImageNet单个类别上训练AGIP-VAR模型
# ===============================================

# 设置严格模式
set -e
set -u

# --------- 颜色定义 ---------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# --------- 日志函数 ---------
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_blue() {
    echo -e "${BLUE}[AGIP-VAR]${NC} $1"
}

# --------- VAR模型快速切换函数 ---------
switch_var_model() {
    local depth=$1
    case $depth in
        16)
            VAR_DEPTH=16
            log_info "🔄 切换到VAR-d16 (嵌入维度: 1024)"
            ;;
        20)
            VAR_DEPTH=20  
            log_info "🔄 切换到VAR-d20 (嵌入维度: 1280)"
            ;;
        24)
            VAR_DEPTH=24
            log_info "🔄 切换到VAR-d24 (嵌入维度: 1536)"
            ;;
        *)
            log_error "不支持的VAR深度: $depth，支持的深度: 16, 20, 24"
            exit 1
            ;;
    esac
    
    # 更新相关配置
    VAR_CKPT="./checkpoints/var_d${VAR_DEPTH}.pth"
}

# --------- 配置区 ---------
# 数据配置
DATA_ROOT="/home/intern/datasets/imagenet-1k/data"
CLASS_ID=207                               # Golden Retriever (默认类别)
BATCH_SIZE=8                                # 更小批次避免CUDA内存不足
NUM_EPOCHS=10                              # 训练轮数
MAX_SAMPLES=1000                            # 每类最大样本数

# 🎯 模型配置 - 快速切换不同VAR模型
# 支持的VAR模型: d16 (1024维), d20 (1280维), d24 (1536维)
VAR_DEPTH=20                              # 🔥 VAR模型深度：16, 20, 24等
VAR_CKPT="./checkpoints/var_d${VAR_DEPTH}.pth"    # VAR模型权重（自动匹配深度）
VQVAE_CKPT="./vae_ch160v4096z32.pth"       # VQVAE权重
DEVICE="cuda:0"                            # 训练设备

# 🔧 根据VAR深度自动计算参数（depth * 64）
# d16: 1024维, d20: 1280维, d24: 1536维
log_info "🎯 当前选择: VAR-d${VAR_DEPTH} (嵌入维度: $((VAR_DEPTH * 64)))"

# AGIP-VAR特定配置
WARMUP_STEPS=63                          # 判别器预热步数
ENABLE_STAGED_TRAINING=true                # 启用分阶段训练
LAMBDA_REC=0.01                             # 重建损失权重

# 优化器配置
LR_PLANNER=5e-6                            # I_predictor学习率
LR_DISCRIMINATOR=1e-6                      # 判别器学习率（提高以防止崩溃）

# 输出配置
OUTPUT_DIR="./single_class_experiments"     # 输出目录
NUM_WORKERS=2                              # 数据加载线程数
AUTO_YES=false                             # 自动确认

# 检查点恢复配置
RESUME_CHECKPOINT=""  # 恢复训练的检查点路径

# --------- 环境检查 ---------
check_environment() {
    log_info "🔍 检查环境和依赖..."
    
    # 检查CUDA
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "nvidia-smi未找到，请确保CUDA已安装"
        exit 1
    fi
    
    # 检查Python环境
    if ! command -v python &> /dev/null; then
        log_error "Python未找到"
        exit 1
    fi
    
    # 检查数据路径
    if [ ! -d "$DATA_ROOT" ]; then
        log_error "数据路径不存在: $DATA_ROOT"
        log_info "请更新DATA_ROOT变量或创建数据目录"
        exit 1
    fi
    
    # 检查模型权重
    if [ ! -f "$VAR_CKPT" ]; then
        log_warn "VAR权重文件不存在: $VAR_CKPT"
        log_info "请下载VAR模型权重到指定路径"
    fi
    
    if [ ! -f "$VQVAE_CKPT" ]; then
        log_warn "VQVAE权重文件不存在: $VQVAE_CKPT"
        log_info "正在尝试下载VQVAE权重..."
        wget -O "$VQVAE_CKPT" "https://huggingface.co/FoundationVision/var/resolve/main/vae_ch160v4096z32.pth"
    fi
    
    # 检查checkpoint文件（如果指定）
    if [ -n "$RESUME_CHECKPOINT" ]; then
        if [ -f "$RESUME_CHECKPOINT" ]; then
            log_info "✅ 找到恢复检查点: $RESUME_CHECKPOINT"
        else
            log_error "❌ 指定的检查点文件不存在: $RESUME_CHECKPOINT"
            log_info "请检查路径是否正确或移除--resume参数从头开始训练"
            exit 1
        fi
    fi
    
    log_info "✅ 环境检查完成"
}

# --------- 环境变量设置 ---------
setup_environment() {
    log_info "🔧 设置环境变量..."
    
    # 激活conda环境
    source /home/intern/miniconda3/etc/profile.d/conda.sh
    conda activate agip-var
    
    export TMPDIR=/home/intern/tmp

    # 🚀 CUDA基础设置
    export CUDA_VISIBLE_DEVICES=0
    export CUDA_LAUNCH_BLOCKING=1          # 调试用，同步CUDA操作
    
    # 🔥 CUDNN兼容性设置 - 解决"GET was unable to find an engine"错误
    log_info "🛠️  设置CUDNN兼容性环境变量..."
    export CUDNN_DETERMINISTIC=1           # 强制使用确定性算法
    export CUDNN_BENCHMARK=0               # 禁用CUDNN自动调优，避免算法选择问题
    export TORCH_CUDNN_V8_API_DISABLED=1   # 禁用CUDNN V8 API，使用更稳定的旧版API
    export CUBLAS_WORKSPACE_CONFIG=:4096:8 # 配置CUBLAS工作空间，避免内存问题          # 禁用TF32，使用更精确的FP32计算
    
    # 🔥 PyTorch CUDA内存管理 - 避免内存分配导致的CUDNN问题
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,roundup_power2_divisions:16
    export TORCH_SHOW_CPP_STACKTRACES=1    # 显示C++堆栈跟踪，便于调试
    
    # 🔧 数值稳定性设置
    export PYTHONHASHSEED=0                # 确保Python哈希的确定性
    export TORCH_DETERMINISTIC=1           # PyTorch确定性设置
    
    # PyTorch线程设置
    export OMP_NUM_THREADS=4
    export MKL_NUM_THREADS=4
    export OPENBLAS_NUM_THREADS=1          # 限制OpenBLAS线程，避免冲突
    
    # 添加stylegan_t路径
    export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/stylegan_t"
    
    # 创建输出目录
    mkdir -p "$OUTPUT_DIR"
    mkdir -p logs
    
    log_info "✅ 环境变量设置完成，包含CUDNN兼容性修复"
    log_info "   🔧 CUDNN_DETERMINISTIC=1 (强制确定性)"
    log_info "   🔧 CUDNN_BENCHMARK=0 (禁用自动调优)"  
    log_info "   🔧 TORCH_CUDNN_V8_API_DISABLED=1 (使用稳定API)"
    log_info "   🔧 NVIDIA_TF32_OVERRIDE=0 (禁用TF32)"
}

# --------- 显示配置信息 ---------
show_configuration() {
    log_blue "==================== AGIP-VAR 单分类训练配置 ===================="
    log_blue "📊 数据配置:"
    log_blue "   数据根目录: $DATA_ROOT"
    log_blue "   训练类别ID: $CLASS_ID"
    log_blue "   批次大小: $BATCH_SIZE"
    log_blue "   最大样本数: $MAX_SAMPLES"
    log_blue ""
    log_blue "🚀 模型配置:"
    log_blue "   VAR权重: $VAR_CKPT"
    log_blue "   VQVAE权重: $VQVAE_CKPT"
    log_blue "   训练设备: $DEVICE"
    log_blue ""
    log_blue "🎯 AGIP-VAR配置:"
    log_blue "   预热步数: $WARMUP_STEPS"
    log_blue "   分阶段训练: $ENABLE_STAGED_TRAINING"
    log_blue "   重建损失权重: $LAMBDA_REC"
    log_blue "   I_predictor学习率: $LR_PLANNER"
    log_blue "   判别器学习率: $LR_DISCRIMINATOR"
    log_blue ""
    log_blue "📈 训练配置:"
    log_blue "   训练轮数: $NUM_EPOCHS"
    log_blue "   工作线程: $NUM_WORKERS"
    log_blue "   输出目录: $OUTPUT_DIR"
    if [ -n "$RESUME_CHECKPOINT" ]; then
        log_blue "   恢复检查点: $RESUME_CHECKPOINT"
    else
        log_blue "   恢复检查点: 无 (从头开始训练)"
    fi
    log_blue "================================================================="
}

# --------- 启动训练 ---------
start_training() {
    log_info "🚀 启动AGIP-VAR单分类训练..."
    
    # 生成时间戳
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="logs/single_class_train_${CLASS_ID}_${TIMESTAMP}.log"
    
    # 构建训练命令
    PYTHON_CMD="python train_single_class.py \
        --data_root=\"$DATA_ROOT\" \
        --class_id=$CLASS_ID \
        --batch_size=$BATCH_SIZE \
        --num_epochs=$NUM_EPOCHS \
        --max_train_samples=$MAX_SAMPLES \
        --device=\"$DEVICE\" \
        --var_depth=$VAR_DEPTH \
        --warmup_steps=$WARMUP_STEPS \
        --enable_staged_training=$ENABLE_STAGED_TRAINING \
        --lambda_rec=$LAMBDA_REC \
        --lr_discriminator=$LR_DISCRIMINATOR"
    
    # 添加checkpoint恢复参数（如果指定）
    if [ -n "$RESUME_CHECKPOINT" ]; then
        PYTHON_CMD="$PYTHON_CMD --resume_checkpoint=\"$RESUME_CHECKPOINT\""
        log_info "🔄 将从检查点恢复训练: $RESUME_CHECKPOINT"
    fi
    
    log_blue "执行命令: $PYTHON_CMD"
    log_blue "日志文件: $LOG_FILE"
    
    # 启动训练并记录日志
    echo "训练开始时间: $(date)" > "$LOG_FILE"
    echo "配置信息:" >> "$LOG_FILE"
    echo "  数据路径: $DATA_ROOT" >> "$LOG_FILE"
    echo "  类别ID: $CLASS_ID" >> "$LOG_FILE"
    echo "  VAR深度: $VAR_DEPTH (嵌入维度: $((VAR_DEPTH * 64)))" >> "$LOG_FILE"
    echo "  批次大小: $BATCH_SIZE" >> "$LOG_FILE"
    echo "  训练轮数: $NUM_EPOCHS" >> "$LOG_FILE"
    echo "  分阶段训练: $ENABLE_STAGED_TRAINING" >> "$LOG_FILE"
    echo "  预热步数: $WARMUP_STEPS" >> "$LOG_FILE"
    echo "================================" >> "$LOG_FILE"
    
    # 执行训练
    eval "$PYTHON_CMD" 2>&1 | tee -a "$LOG_FILE"
    
    # 检查训练结果
    if [ $? -eq 0 ]; then
        log_info "✅ 训练成功完成！"
        log_info "📊 查看训练日志: tail -f $LOG_FILE"
        log_info "📁 实验结果保存在: $OUTPUT_DIR"
    else
        log_error "❌ 训练失败，请检查日志"
        log_error "📊 错误日志: tail -f $LOG_FILE"
        exit 1
    fi
}

# --------- 清理函数 ---------
cleanup() {
    log_info "🧹 清理临时文件..."
    # 这里可以添加清理逻辑
    log_info "✅ 清理完成"
}

# --------- 中断处理 ---------
handle_interrupt() {
    log_warn "🛑 接收到中断信号"
    cleanup
    exit 1
}

# 设置中断处理
trap handle_interrupt SIGINT SIGTERM

# --------- 主函数 ---------
main() {
    log_blue "🔥 AGIP-VAR单分类训练启动脚本"
    log_blue "================================================"
    
    # 检查环境
    check_environment
    
    # 设置环境
    setup_environment
    
    # 显示配置
    show_configuration
    
    # 确认开始（如果不是非交互式或自动确认）
    if [[ "$AUTO_YES" = false ]] && [[ -t 0 ]]; then
        echo ""
        read -p "是否开始训练? (y/N): " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "训练已取消"
            exit 0
        fi
    else
        log_info "🤖 自动开始训练"
    fi
    
    # 开始训练
    start_training
    
    # 完成
    log_blue "🎉 AGIP-VAR单分类训练脚本执行完成！"
}

# --------- 帮助信息 ---------
show_help() {
    cat << EOF
🎯 AGIP-VAR单分类训练脚本使用说明

用法: $0 [选项]

选项:
  -h, --help          显示此帮助信息
  -c, --class-id      设置训练类别ID (默认: $CLASS_ID)
  -b, --batch-size    设置批次大小 (默认: $BATCH_SIZE)
  -e, --epochs        设置训练轮数 (默认: $NUM_EPOCHS)
  -d, --data-root     设置数据根目录 (默认: $DATA_ROOT)
  -o, --output-dir    设置输出目录 (默认: $OUTPUT_DIR)
  -w, --warmup-steps  设置预热步数 (默认: $WARMUP_STEPS)
  -r, --resume        从指定的检查点文件恢复训练
  --no-staged         禁用分阶段训练
  --var-depth         设置VAR模型深度: 16/20/24 (默认: $VAR_DEPTH)
  --device            设置训练设备 (默认: $DEVICE)
  -y, --yes           自动确认，不询问

示例:
  $0 -c 207 -b 8 -e 100           # 类别207，批次8，训练100轮
  $0 --var-depth 20 -c 207        # 使用VAR-d20模型训练类别207
  $0 --class-id 281 --no-staged   # 类别281，禁用分阶段训练
  $0 --data-root /path/to/data     # 自定义数据路径
  $0 -r /path/to/checkpoint.pth    # 从检查点恢复训练

环境要求:
  - CUDA支持的GPU
  - Python 3.8+
  - PyTorch 2.0+
  - 所需的Python依赖包

更多信息请参考: https://github.com/FoundationVision/VAR
EOF
}

# --------- 参数解析 ---------
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -c|--class-id)
                CLASS_ID="$2"
                shift 2
                ;;
            -b|--batch-size)
                BATCH_SIZE="$2"
                shift 2
                ;;
            -e|--epochs)
                NUM_EPOCHS="$2"
                shift 2
                ;;
            -d|--data-root)
                DATA_ROOT="$2"
                shift 2
                ;;
            -o|--output-dir)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            -w|--warmup-steps)
                WARMUP_STEPS="$2"
                shift 2
                ;;
            -r|--resume)
                RESUME_CHECKPOINT="$2"
                shift 2
                ;;
            --no-staged)
                ENABLE_STAGED_TRAINING=false
                shift
                ;;
            --var-depth)
                switch_var_model "$2"
                shift 2
                ;;
            --device)
                DEVICE="$2"
                shift 2
                ;;
            -y|--yes)
                AUTO_YES=true
                shift
                ;;
            *)
                log_error "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# --------- 脚本入口 ---------
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    parse_arguments "$@"
    main
fi