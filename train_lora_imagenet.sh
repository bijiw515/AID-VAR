#!/bin/bash

# ===============================================
# 🎯 LoRA-VAR 完整ImageNet多卡分布式训练脚本
# 基于AID-VAR架构，使用LoRA替代guidance injector进行消融实验
# ===============================================

# 设置严格模式
set -e
set -u

# --------- 颜色定义 ---------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# --------- 日志函数 ---------
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
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
    DEPTH=$VAR_DEPTH
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_blue() {
    echo -e "${BLUE}[LoRA-VAR]${NC} $1"
}

log_cyan() {
    echo -e "${CYAN}[MULTI-NODE]${NC} $1"
}

# --------- 配置区 ---------
# 🌐 分布式配置（支持单机/多机切换）
NNODES=${NNODES:-1}                       # 节点总数（1=单机，>1=多机）
NODE_RANK=${NODE_RANK:-${SLURM_PROCID:-0}}  # 节点rank（从环境变量自动获取）
NPROC_PER_NODE=${NPROC_PER_NODE:-8}       # 每节点GPU数量（单机8卡）
MASTER_ADDR=${MASTER_ADDR:-"localhost"}   # 主节点IP（单机用localhost）
MASTER_PORT=${MASTER_PORT:-29600}         # 主节点端口

# 计算总体配置
TOTAL_GPUS=$((NNODES * NPROC_PER_NODE))   # 总GPU数量
PER_GPU_BATCH_SIZE=32                     # 每GPU批次大小（受GPU内存限制）
GLOBAL_BATCH_SIZE=$((PER_GPU_BATCH_SIZE * TOTAL_GPUS))  # 全局批次大小

# 数据配置
DATA_ROOT="/home/intern/datasets/imagenet-1k/data"
NUM_EPOCHS=100                            # 训练轮数

# 兼容性配置（保持向后兼容）
NUM_GPUS=$NPROC_PER_NODE                  # 向后兼容：每节点GPU数
CUDA_VISIBLE_DEVICES_LIST="0,1,2,3,4,5,6,7"  # 每节点的可见GPU

# 🎯 模型配置 - 快速切换不同VAR模型
VAR_DEPTH=16                              # 🔥 VAR模型深度：16, 20, 24等
VAR_CKPT="./checkpoints/var_d${VAR_DEPTH}.pth"    # VAR模型权重（自动匹配深度）
VQVAE_CKPT="./vae_ch160v4096z32.pth"       # VQVAE权重

# 🔧 根据VAR深度自动计算参数（depth * 64）
log_info "🎯 当前选择: VAR-d${VAR_DEPTH} (嵌入维度: $((VAR_DEPTH * 64)))"

# LoRA特定配置
LORA_RANK_ATTENTION=64                    # 注意力层LoRA秩
LORA_RANK_FFN=32                          # FFN层LoRA秩
LORA_ALPHA_ATTENTION=16.0                 # 注意力层LoRA alpha
LORA_ALPHA_FFN=8.0                        # FFN层LoRA alpha
LORA_DROPOUT=0.05                         # LoRA dropout率

# 训练配置 - 根据总GPU数动态计算
STEPS_PER_EPOCH=$((1281167 / GLOBAL_BATCH_SIZE))  # 每epoch步数
WARMUP_STEPS=$STEPS_PER_EPOCH             # 判别器预热步数 (1个epoch)
ENABLE_STAGED_TRAINING=true               # 启用分阶段训练
LAMBDA_REC=0.1                            # 重建损失权重
LAMBDA_ADV=1.0                            # 对抗损失权重

# 优化器配置
LR_VAR=1e-6                               # LoRA参数学习率
LR_DISCRIMINATOR=1e-6                     # 判别器学习率
CLIP_VAR=0.5                              # LoRA梯度裁剪
CLIP_DISCRIMINATOR=1.0                    # 判别器梯度裁剪
WEIGHT_DECAY=0.01                         # 权重衰减

# 引导权重配置（repurposed as LoRA scaling）
GUIDANCE_WEIGHT_MAX=0.01                  # 最大引导权重

# 输出配置
OUTPUT_DIR="./experiments/lora_var"       # 输出目录
NUM_WORKERS=8                             # 每GPU数据加载线程数
AUTO_YES=false                            # 自动确认

# 检查点恢复配置
RESUME_CHECKPOINT=""                      # 恢复训练的检查点路径
LOAD_OPTIMIZER_STATE="true"               # 是否加载优化器状态

# 判别器配置
DISC_BACKBONE="dino"                      # 判别器骨干网络
DISC_FREEZE_BACKBONE=true                 # 冻结判别器骨干

# 日志和保存配置
LOG_INTERVAL=10                           # 每10个batch打印一次
SAVE_INTERVAL=5                           # 每5个epoch保存一次检查点
VALIDATION_INTERVAL=5                     # 每5个epoch验证一次

# 可视化调试
ENABLE_VISUAL_DEBUG=true                  # 启用可视化调试
VISUAL_DEBUG_INTERVAL=50                  # 可视化调试间隔

# --------- 自动环境检测 ---------
detect_cluster_environment() {
    log_info "🔍 检测集群调度环境..."

    # 检测Slurm环境
    if [ -n "${SLURM_JOB_ID:-}" ]; then
        log_info "✅ 检测到Slurm集群环境"
        export NNODES=${SLURM_NNODES:-$NNODES}
        export NODE_RANK=${SLURM_PROCID:-$NODE_RANK}
        export NPROC_PER_NODE=${SLURM_GPUS_PER_NODE:-$NPROC_PER_NODE}
        export MASTER_ADDR=${SLURM_LAUNCH_NODE_IPADDR:-$MASTER_ADDR}
        log_info "   SLURM_JOB_ID: ${SLURM_JOB_ID}"
    fi

    # 检测Kubernetes环境
    if [ -n "${KUBERNETES_SERVICE_HOST:-}" ]; then
        log_info "✅ 检测到Kubernetes集群环境"
        export NODE_RANK=${POD_INDEX:-${RANK:-$NODE_RANK}}
        export NNODES=${WORLD_SIZE:-$NNODES}
        export MASTER_ADDR=${MASTER_ADDR:-${POD_IP:-$MASTER_ADDR}}
    fi

    # 检测torchrun环境变量
    if [ -n "${LOCAL_RANK:-}" ]; then
        log_info "✅ 检测到torchrun分布式环境"
        export NODE_RANK=${RANK:-$NODE_RANK}
        export NNODES=${WORLD_SIZE:-$NNODES}
        export NPROC_PER_NODE=${LOCAL_WORLD_SIZE:-$NPROC_PER_NODE}
    fi

    # 更新计算配置
    TOTAL_GPUS=$((NNODES * NPROC_PER_NODE))
    GLOBAL_BATCH_SIZE=$((PER_GPU_BATCH_SIZE * TOTAL_GPUS))
    STEPS_PER_EPOCH=$((1281167 / GLOBAL_BATCH_SIZE))
    WARMUP_STEPS=$STEPS_PER_EPOCH

    log_info "🔧 最终配置:"
    log_info "   节点总数: $NNODES"
    log_info "   当前节点rank: $NODE_RANK"
    log_info "   每节点GPU数: $NPROC_PER_NODE"
    log_info "   主节点地址: $MASTER_ADDR:$MASTER_PORT"
    log_info "   总GPU数: $TOTAL_GPUS"
    log_info "   全局批次大小: $GLOBAL_BATCH_SIZE"
}

# --------- 环境检查 ---------
check_environment() {
    log_info "🔍 检查分布式训练环境和依赖..."

    # 检查CUDA
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "nvidia-smi未找到，请确保CUDA已安装"
        exit 1
    fi

    # 检查可用GPU数量
    AVAILABLE_GPUS=$(nvidia-smi --list-gpus | wc -l)
    if [ "$AVAILABLE_GPUS" -lt "$NUM_GPUS" ]; then
        log_error "可用GPU数量($AVAILABLE_GPUS)少于所需数量($NUM_GPUS)"
        exit 1
    fi

    # 检查Python环境
    if ! command -v python &> /dev/null; then
        log_error "Python未找到"
        exit 1
    fi

    # 检查torchrun
    if ! command -v torchrun &> /dev/null; then
        log_error "torchrun未找到，请确保PyTorch >= 1.9.0已安装"
        exit 1
    fi

    # 检查VAR模型权重
    if [ ! -f "$VAR_CKPT" ]; then
        log_error "VAR模型权重文件不存在: $VAR_CKPT"
        log_info "请确保checkpoints/var_d${VAR_DEPTH}.pth文件存在"
        exit 1
    fi

    # 检查数据路径
    if [ ! -d "$DATA_ROOT" ]; then
        log_error "ImageNet数据路径不存在: $DATA_ROOT"
        exit 1
    fi

    # 检查ImageNet目录结构
    if [ ! -d "$DATA_ROOT/train" ] || [ ! -d "$DATA_ROOT/val" ]; then
        log_error "ImageNet目录结构不正确，需要包含train/和val/子目录"
        exit 1
    fi

    # 检查checkpoint文件（如果指定）
    if [ -n "$RESUME_CHECKPOINT" ]; then
        if [ -f "$RESUME_CHECKPOINT" ]; then
            log_info "✅ 找到恢复检查点: $RESUME_CHECKPOINT"
        else
            log_error "❌ 指定的检查点文件不存在: $RESUME_CHECKPOINT"
            exit 1
        fi
    fi

    log_info "✅ 分布式训练环境检查完成"
}

# --------- 环境变量设置 ---------
setup_environment() {
    log_info "🔧 设置分布式训练环境变量..."

    # 激活conda环境
    source /home/intern/miniconda3/etc/profile.d/conda.sh
    conda activate aid-var

    export TMPDIR=/home/intern/tmp

    # CUDA设置
    export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES_LIST

    # PyTorch CUDA内存管理
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,roundup_power2_divisions:16
    export TORCH_SHOW_CPP_STACKTRACES=1

    # 数值稳定性设置
    export PYTHONHASHSEED=0
    export TORCH_DETERMINISTIC=1

    # 分布式训练环境变量
    export MASTER_ADDR=$MASTER_ADDR
    export MASTER_PORT=$MASTER_PORT
    export NNODES=$NNODES
    export NODE_RANK=$NODE_RANK
    export NPROC_PER_NODE=$NPROC_PER_NODE
    export WORLD_SIZE=$TOTAL_GPUS

    # NCCL多机通信优化
    export NCCL_DEBUG=INFO
    export NCCL_TREE_THRESHOLD=0
    export NCCL_IB_DISABLE=1
    export NCCL_SOCKET_IFNAME=eth0

    # 多机NCCL性能优化
    if [ "$NNODES" -gt 1 ]; then
        export NCCL_P2P_DISABLE=1
        export NCCL_SOCKET_NTHREADS=16
        export NCCL_NSOCKS_PERTHREAD=8
        export NCCL_BUFFSIZE=2097152
        export NCCL_NET_GDR_LEVEL=0
        log_info "   🌐 多机NCCL优化配置已启用"
    fi

    # PyTorch线程设置
    export OMP_NUM_THREADS=4
    export MKL_NUM_THREADS=4
    export OPENBLAS_NUM_THREADS=1

    # 添加stylegan_t路径
    export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/stylegan_t"

    # 创建输出目录
    mkdir -p "$OUTPUT_DIR"
    mkdir -p logs

    log_info "✅ 分布式训练环境变量设置完成"
}

# --------- 显示配置信息 ---------
show_configuration() {
    if [ "$NNODES" -gt 1 ]; then
        log_blue "==================== LoRA-VAR 多机多卡分布式训练配置 ===================="
    else
        log_blue "==================== LoRA-VAR 单机多卡分布式训练配置 ===================="
    fi
    log_blue "🌐 分布式配置:"
    log_blue "   节点总数: $NNODES"
    log_blue "   当前节点rank: $NODE_RANK"
    log_blue "   每节点GPU数: $NPROC_PER_NODE"
    log_blue "   总GPU数: $TOTAL_GPUS"
    log_blue "   主节点地址: $MASTER_ADDR:$MASTER_PORT"
    log_blue ""
    log_blue "📊 数据配置:"
    log_blue "   ImageNet数据根目录: $DATA_ROOT"
    log_blue "   全局批次大小: $GLOBAL_BATCH_SIZE"
    log_blue "   每GPU批次大小: $PER_GPU_BATCH_SIZE"
    log_blue "   每epoch步数: $STEPS_PER_EPOCH"
    log_blue ""
    log_blue "🚀 模型配置:"
    log_blue "   VAR权重: $VAR_CKPT"
    log_blue "   VQVAE权重: $VQVAE_CKPT"
    log_blue "   VAR深度: $VAR_DEPTH"
    log_blue ""
    log_blue "🎯 LoRA配置:"
    log_blue "   注意力层LoRA秩: $LORA_RANK_ATTENTION"
    log_blue "   FFN层LoRA秩: $LORA_RANK_FFN"
    log_blue "   注意力层LoRA alpha: $LORA_ALPHA_ATTENTION"
    log_blue "   FFN层LoRA alpha: $LORA_ALPHA_FFN"
    log_blue "   LoRA dropout: $LORA_DROPOUT"
    log_blue ""
    log_blue "🎯 训练配置:"
    log_blue "   预热步数: $WARMUP_STEPS (1个epoch)"
    log_blue "   分阶段训练: $ENABLE_STAGED_TRAINING"
    log_blue "   重建损失权重: $LAMBDA_REC"
    log_blue "   对抗损失权重: $LAMBDA_ADV"
    log_blue "   LoRA学习率: $LR_VAR"
    log_blue "   判别器学习率: $LR_DISCRIMINATOR"
    log_blue "   最大引导权重: $GUIDANCE_WEIGHT_MAX"
    log_blue ""
    log_blue "📈 训练配置:"
    log_blue "   训练轮数: $NUM_EPOCHS"
    log_blue "   保存间隔: $SAVE_INTERVAL epochs"
    log_blue "   验证间隔: $VALIDATION_INTERVAL epochs"
    log_blue "   输出目录: $OUTPUT_DIR"
    if [ -n "$RESUME_CHECKPOINT" ]; then
        log_blue "   恢复检查点: $RESUME_CHECKPOINT"
    else
        log_blue "   恢复检查点: 无 (从头开始训练)"
    fi
    log_blue "================================================================="
}

# --------- 启动分布式训练 ---------
start_training() {
    log_info "🚀 启动LoRA-VAR完整ImageNet分布式训练..."

    # 生成时间戳
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="logs/lora_train_${TOTAL_GPUS}gpus_${NNODES}nodes_node${NODE_RANK}_${TIMESTAMP}.log"

    # 构建训练命令
    PYTHON_CMD="torchrun \
        --nnodes=$NNODES \
        --node_rank=$NODE_RANK \
        --nproc_per_node=$NPROC_PER_NODE \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        train_lora.py \
        --data_path=\"$DATA_ROOT\" \
        --ep=$NUM_EPOCHS \
        --bs=$PER_GPU_BATCH_SIZE \
        --num_workers=$NUM_WORKERS \
        --depth=$VAR_DEPTH \
        --var_ckpt=\"$VAR_CKPT\" \
        --lora_rank_attention=$LORA_RANK_ATTENTION \
        --lora_rank_ffn=$LORA_RANK_FFN \
        --lora_alpha_attention=$LORA_ALPHA_ATTENTION \
        --lora_alpha_ffn=$LORA_ALPHA_FFN \
        --lora_dropout=$LORA_DROPOUT \
        --lr_var=$LR_VAR \
        --lr_discriminator=$LR_DISCRIMINATOR \
        --wd=$WEIGHT_DECAY \
        --clip_var=$CLIP_VAR \
        --clip_discriminator=$CLIP_DISCRIMINATOR \
        --lambda_rec=$LAMBDA_REC \
        --lambda_adv=$LAMBDA_ADV \
        --enable_staged_training=$ENABLE_STAGED_TRAINING \
        --warmup_steps=$WARMUP_STEPS \
        --guidance_weight_max=$GUIDANCE_WEIGHT_MAX \
        --disc_backbone=$DISC_BACKBONE \
        --disc_freeze_backbone=$DISC_FREEZE_BACKBONE \
        --output_dir=\"$OUTPUT_DIR\" \
        --save_interval=$SAVE_INTERVAL \
        --val_interval=$VALIDATION_INTERVAL \
        --enable_visual_debug=$ENABLE_VISUAL_DEBUG \
        --visual_debug_interval=$VISUAL_DEBUG_INTERVAL \
        --seed=42"

    # 添加checkpoint恢复参数（如果指定）
    if [ -n "$RESUME_CHECKPOINT" ]; then
        PYTHON_CMD="$PYTHON_CMD --resume=\"$RESUME_CHECKPOINT\""
        PYTHON_CMD="$PYTHON_CMD --load_optimizer=$LOAD_OPTIMIZER_STATE"
        log_info "🔄 将从检查点恢复训练: $RESUME_CHECKPOINT"
    fi

    log_blue "执行命令: $PYTHON_CMD"
    log_blue "日志文件: $LOG_FILE"

    # 启动训练并记录日志
    echo "LoRA-VAR分布式训练开始时间: $(date)" > "$LOG_FILE"
    echo "分布式配置:" >> "$LOG_FILE"
    echo "  节点总数: $NNODES" >> "$LOG_FILE"
    echo "  当前节点rank: $NODE_RANK" >> "$LOG_FILE"
    echo "  每节点GPU数: $NPROC_PER_NODE" >> "$LOG_FILE"
    echo "  总GPU数: $TOTAL_GPUS" >> "$LOG_FILE"
    echo "================================" >> "$LOG_FILE"

    # 执行分布式训练
    eval "$PYTHON_CMD" 2>&1 | tee -a "$LOG_FILE"

    # 检查训练结果
    if [ $? -eq 0 ]; then
        log_info "✅ LoRA-VAR分布式训练成功完成！"
        log_info "📊 查看训练日志: tail -f $LOG_FILE"
        if [ "$NODE_RANK" -eq 0 ]; then
            log_info "📁 实验结果保存在: $OUTPUT_DIR"
        fi
    else
        log_error "❌ LoRA-VAR分布式训练失败，请检查日志"
        log_error "📊 错误日志: tail -f $LOG_FILE"
        exit 1
    fi
}

# --------- 清理函数 ---------
cleanup() {
    log_info "🧹 清理分布式训练资源..."
    pkill -f "train_lora.py" 2>/dev/null || true
    pkill -f "torchrun" 2>/dev/null || true
    rm -rf /tmp/nccl* 2>/dev/null || true
    log_info "✅ 清理完成"
}

# --------- 中断处理 ---------
handle_interrupt() {
    log_warn "🛑 接收到中断信号，正在停止分布式训练..."
    cleanup
    exit 1
}

# 设置中断处理
trap handle_interrupt SIGINT SIGTERM

# --------- 主函数 ---------
main() {
    # 自动检测集群环境
    detect_cluster_environment

    if [ "$NNODES" -gt 1 ]; then
        log_cyan "🌐 LoRA-VAR多机多卡分布式训练启动脚本"
        log_cyan "   节点${NODE_RANK}/${NNODES} | 总GPU数: ${TOTAL_GPUS}"
    else
        log_blue "🔥 LoRA-VAR单机多卡分布式训练启动脚本"
    fi
    log_cyan "================================================================"

    # 检查环境
    check_environment

    # 设置环境
    setup_environment

    # 显示配置
    show_configuration

    # 确认开始
    if [[ "$AUTO_YES" = false ]] && [[ -t 0 ]]; then
        echo ""
        read -p "是否开始${TOTAL_GPUS}卡LoRA-VAR分布式训练? (y/N): " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "训练已取消"
            exit 0
        fi
    else
        log_info "🤖 自动开始${TOTAL_GPUS}卡LoRA-VAR分布式训练"
    fi

    # 开始训练
    start_training

    # 完成
    log_blue "🎉 LoRA-VAR分布式训练脚本执行完成！"
}

# --------- 帮助信息 ---------
show_help() {
    cat << EOF
🚀 LoRA-VAR完整ImageNet分布式训练脚本使用说明

用法: $0 [选项]

选项:
  -h, --help              显示此帮助信息
  -e, --epochs            设置训练轮数 (默认: $NUM_EPOCHS)
  -d, --data-root         设置ImageNet数据根目录 (默认: $DATA_ROOT)
  -o, --output-dir        设置输出目录 (默认: $OUTPUT_DIR)
  -r, --resume            从指定的检查点文件恢复训练
  --var-depth             设置VAR模型深度: 16/20/24 (默认: $VAR_DEPTH)
  --lr-var                LoRA参数学习率 (默认: $LR_VAR)
  --lr-discriminator      判别器学习率 (默认: $LR_DISCRIMINATOR)
  --lora-rank-attention   注意力层LoRA秩 (默认: $LORA_RANK_ATTENTION)
  --lora-rank-ffn         FFN层LoRA秩 (默认: $LORA_RANK_FFN)
  -y, --yes               自动确认，不询问

示例:
  $0 -y                                     # 使用默认配置训练
  $0 --var-depth 20 -y                     # 使用VAR-d20模型训练
  $0 -r /path/to/checkpoint.pth            # 从检查点恢复训练
  $0 --lr-var 2e-6 --lr-discriminator 1e-6 # 自定义学习率

更多信息请参考: CLAUDE.md
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
            -r|--resume)
                RESUME_CHECKPOINT="$2"
                shift 2
                ;;
            --var-depth)
                switch_var_model "$2"
                shift 2
                ;;
            --lr-var)
                LR_VAR="$2"
                shift 2
                ;;
            --lr-discriminator)
                LR_DISCRIMINATOR="$2"
                shift 2
                ;;
            --lora-rank-attention)
                LORA_RANK_ATTENTION="$2"
                shift 2
                ;;
            --lora-rank-ffn)
                LORA_RANK_FFN="$2"
                shift 2
                ;;
            --load-optimizer)
                LOAD_OPTIMIZER_STATE="$2"
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
