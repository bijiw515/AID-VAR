#!/bin/bash

# ===============================================
# 🎯 AGIP-VAR 完整ImageNet多卡分布式训练脚本
# 基于train_single_class.sh的成功架构，扩展到完整ImageNet和多卡训练
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
    echo -e "${BLUE}[AGIP-VAR]${NC} $1"
}

log_cyan() {
    echo -e "${CYAN}[MULTI-NODE]${NC} $1"
}

# --------- 配置区 ---------
# 🌐 分布式配置（支持单机/多机切换）
# 自动检测集群环境变量，如果不存在则使用默认值
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
NUM_EPOCHS=2                            # 训练轮数 - 大规模多机训练

# 兼容性配置（保持向后兼容）
NUM_GPUS=$NPROC_PER_NODE                  # 向后兼容：每节点GPU数
CUDA_VISIBLE_DEVICES_LIST="0,1,2,3,4,5,6,7"  # 每节点的可见GPU

# 🎯 模型配置 - 快速切换不同VAR模型
# 支持的VAR模型: d16 (1024维), d20 (1280维), d24 (1536维)
VAR_DEPTH=16                              # 🔥 VAR模型深度：16, 20, 24等
VAR_CKPT="./checkpoints/var_d${VAR_DEPTH}.pth"    # VAR模型权重（自动匹配深度）
VQVAE_CKPT="./vae_ch160v4096z32.pth"       # VQVAE权重

# 🔧 根据VAR深度自动计算参数（depth * 64）
# d16: 1024维, d20: 1280维, d24: 1536维
log_info "🎯 当前选择: VAR-d${VAR_DEPTH} (嵌入维度: $((VAR_DEPTH * 64)))"

# AGIP-VAR特定配置 - 根据总GPU数动态计算
# 🔥 关键配置：设置WARMUP_STEPS为一个epoch的步数
STEPS_PER_EPOCH=$((1281167 / GLOBAL_BATCH_SIZE))  # 每epoch步数：ImageNet训练样本数 / 全局批次大小
WARMUP_STEPS=0              # 判别器预热步数 (1个epoch)
ENABLE_STAGED_TRAINING=true                # 启用分阶段训练
LAMBDA_REC=0.01                               # 重建损失权重

# 优化器配置 - 🔥 与单分类训练完全相同的参数
LR_PLANNER=1e-6                            # I_predictor学习率
LR_DISCRIMINATOR=1e-6                      # 判别器学习率（统一学习率）

# 🔥 关键修复：统一学习率，避免训练动态失衡
GUIDANCE_WEIGHT=0.005                      # 保守的初始引导权重
GUIDANCE_TARGET_WEIGHT=0.001               # 固定引导权重
GUIDANCE_RAMP_EPOCHS=15                    # 渐进增加期
R1_GAMMA=0.2                              # R1梯度惩罚权重

# 输出配置
OUTPUT_DIR="./experiments"                 # 输出目录
NUM_WORKERS=8                             # 每GPU数据加载线程数（8卡优化）
AUTO_YES=false                            # 自动确认

# 检查点恢复配置
RESUME_CHECKPOINT=""  # 恢复训练的检查点路径
LOAD_OPTIMIZER_STATE=""  # 是否加载优化器状态（true/false，默认为空使用程序默认值）

# VAR模型配置 - 与单分类训练完全相同
DEPTH=$VAR_DEPTH                         # VAR深度（使用上面定义的VAR_DEPTH）
PATCH_NUMS="1_2_3_4_5_6_8_10_13_16"      # 多尺度patch配置
SHARED_ADALN=false                        # 共享AdaLN
ATTN_L2_NORM=true                         # 注意力L2归一化
FUSE=false                               # 融合优化
INIT_ADALN=0.5                           # AdaLN初始化
INIT_ADALN_GAMMA=1e-5                    # AdaLN gamma初始化
INIT_HEAD=0.02                           # 头部初始化
INIT_STD=-1                              # 标准初始化

# 日志和保存配置 - 8卡分布式优化
LOG_INTERVAL=10                          # 每10个batch打印一次
SAVE_INTERVAL=1                         # 每10个epoch保存一次检查点
VALIDATION_INTERVAL=1                    # 每5个epoch验证一次

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
        log_info "   SLURM_NNODES: ${SLURM_NNODES:-N/A}"
        log_info "   SLURM_PROCID: ${SLURM_PROCID:-N/A}"
    fi
    
    # 检测Kubernetes环境
    if [ -n "${KUBERNETES_SERVICE_HOST:-}" ]; then
        log_info "✅ 检测到Kubernetes集群环境"
        # 从环境变量中获取节点信息
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
        log_error "请调整NUM_GPUS参数或增加可用GPU"
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
    
    # 检查VQVAE权重
    if [ ! -f "$VQVAE_CKPT" ]; then
        log_error "VQVAE权重文件不存在: $VQVAE_CKPT"
        log_info "请确保vae_ch160v4096z32.pth文件存在"
        exit 1
    fi
    
    # 检查数据路径
    if [ ! -d "$DATA_ROOT" ]; then
        log_error "ImageNet数据路径不存在: $DATA_ROOT"
        log_info "请更新DATA_ROOT变量或创建数据目录"
        exit 1
    fi
    
    # 检查ImageNet目录结构
    if [ ! -d "$DATA_ROOT/train" ] || [ ! -d "$DATA_ROOT/val" ]; then
        log_error "ImageNet目录结构不正确，需要包含train/和val/子目录"
        log_info "正确结构: $DATA_ROOT/train/ 和 $DATA_ROOT/val/"
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
    
    # 检查端口可用性
    if netstat -tuln | grep ":$MASTER_PORT " > /dev/null; then
        log_warn "端口 $MASTER_PORT 已被占用，将尝试自动选择端口"
        MASTER_PORT=$((MASTER_PORT + $RANDOM % 1000))
        log_info "使用新端口: $MASTER_PORT"
    fi
    
    log_info "✅ 分布式训练环境检查完成"
}

# --------- 环境变量设置 ---------
setup_environment() {
    log_info "🔧 设置分布式训练环境变量..."
    
    # 激活conda环境
    source /home/intern/miniconda3/etc/profile.d/conda.sh
    conda activate agip-var
    
    export TMPDIR=/home/intern/tmp

    # 🚀 多卡CUDA基础设置
    export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES_LIST
    
    # 🔥 CUDNN兼容性设置 - 解决"GET was unable to find an engine"错误
    log_info "🛠️  设置多卡CUDNN兼容性环境变量..."
    #export CUDNN_DETERMINISTIC=1           # 强制使用确定性算法
    #export CUDNN_BENCHMARK=0               # 禁用CUDNN自动调优，避免算法选择问题
    #export TORCH_CUDNN_V8_API_DISABLED=1   # 禁用CUDNN V8 API，使用更稳定的旧版API
    #export CUBLAS_WORKSPACE_CONFIG=:4096:8 # 配置CUBLAS工作空间，避免内存问题
    
    # 🔥 PyTorch CUDA内存管理 - 避免内存分配导致的CUDNN问题
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,roundup_power2_divisions:16
    export TORCH_SHOW_CPP_STACKTRACES=1    # 显示C++堆栈跟踪，便于调试
    
    # 🔧 数值稳定性设置
    export PYTHONHASHSEED=0                # 确保Python哈希的确定性
    export TORCH_DETERMINISTIC=1           # PyTorch确定性设置
    
    # 🌐 多机分布式训练环境变量
    export MASTER_ADDR=$MASTER_ADDR
    export MASTER_PORT=$MASTER_PORT
    export NNODES=$NNODES
    export NODE_RANK=$NODE_RANK
    export NPROC_PER_NODE=$NPROC_PER_NODE
    export WORLD_SIZE=$TOTAL_GPUS
    
    # 🌐 NCCL多机通信优化
    export NCCL_DEBUG=INFO                 # NCCL调试信息
    export NCCL_TREE_THRESHOLD=0           # 优化小消息传递
    export NCCL_IB_DISABLE=1               # 如果没有InfiniBand，禁用IB
    export NCCL_SOCKET_IFNAME=eth0         # 网络接口（根据实际情况调整）
    
    # 🔥 多机NCCL性能优化
    if [ "$NNODES" -gt 1 ]; then
        export NCCL_P2P_DISABLE=1          # 多机时禁用P2P，强制使用网络通信
        export NCCL_SOCKET_NTHREADS=16     # 增加socket线程数
        export NCCL_NSOCKS_PERTHREAD=8     # 每线程socket数
        export NCCL_BUFFSIZE=2097152       # 增加buffer大小（2MB）
        export NCCL_NET_GDR_LEVEL=0        # 禁用GPU Direct RDMA（兼容性）
        log_info "   🌐 多机NCCL优化配置已启用"
    fi
    
    # PyTorch线程设置 - 多卡训练优化
    export OMP_NUM_THREADS=4
    export MKL_NUM_THREADS=4
    export OPENBLAS_NUM_THREADS=1          # 限制OpenBLAS线程，避免冲突
    
    # 添加stylegan_t路径
    export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/stylegan_t"
    
    # 创建输出目录
    mkdir -p "$OUTPUT_DIR"
    mkdir -p logs
    
    log_info "✅ 分布式训练环境变量设置完成"
    log_info "   🔧 CUDNN_DETERMINISTIC=1 (强制确定性)"
    log_info "   🔧 CUDNN_BENCHMARK=0 (禁用自动调优)"  
    log_info "   🔧 TORCH_CUDNN_V8_API_DISABLED=1 (使用稳定API)"
    log_info "   🔧 NCCL_DEBUG=INFO (分布式通信调试)"
    log_info "   🌐 CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES_LIST"
}

# --------- 显示配置信息 ---------
show_configuration() {
    if [ "$NNODES" -gt 1 ]; then
        log_blue "==================== AGIP-VAR 多机多卡分布式训练配置 ===================="
    else
        log_blue "==================== AGIP-VAR 单机多卡分布式训练配置 ===================="
    fi
    log_blue "🌐 分布式配置:"
    log_blue "   节点总数: $NNODES"
    log_blue "   当前节点rank: $NODE_RANK"
    log_blue "   每节点GPU数: $NPROC_PER_NODE"
    log_blue "   总GPU数: $TOTAL_GPUS"
    log_blue "   主节点地址: $MASTER_ADDR:$MASTER_PORT"
    if [ "$NNODES" -gt 1 ]; then
        log_blue "   ⚠️ 多机模式：请确保所有节点网络连通"
    fi
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
    log_blue "   VAR深度: $DEPTH"
    log_blue "   多尺度配置: $PATCH_NUMS"
    log_blue ""
    log_blue "🎯 AGIP-VAR配置:"
    log_blue "   预热步数: $WARMUP_STEPS (1个epoch)"
    log_blue "   分阶段训练: $ENABLE_STAGED_TRAINING"
    log_blue "   重建损失权重: $LAMBDA_REC"
    log_blue "   I_predictor学习率: $LR_PLANNER"
    log_blue "   判别器学习率: $LR_DISCRIMINATOR"
    log_blue "   引导权重: $GUIDANCE_WEIGHT"
    log_blue "   目标引导权重: $GUIDANCE_TARGET_WEIGHT"
    log_blue "   R1惩罚权重: $R1_GAMMA"
    log_blue ""
    log_blue "📈 训练配置:"
    log_blue "   训练轮数: $NUM_EPOCHS"
    log_blue "   保存间隔: $SAVE_INTERVAL epochs"
    log_blue "   验证间隔: $VALIDATION_INTERVAL epochs"
    log_blue "   日志间隔: $LOG_INTERVAL batches (精简模式)"
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
    log_info "🚀 启动AGIP-VAR完整ImageNet 8卡分布式训练..."
    
    # 生成时间戳
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="logs/full_imagenet_train_${TOTAL_GPUS}gpus_${NNODES}nodes_node${NODE_RANK}_${TIMESTAMP}.log"
    
    # 构建训练命令 - 使用torchrun进行多机多卡分布式训练
    PYTHON_CMD="torchrun \
        --nnodes=$NNODES \
        --node_rank=$NODE_RANK \
        --nproc_per_node=$NPROC_PER_NODE \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        train_planner.py \
        --data_path=\"$DATA_ROOT\" \
        --ep=$NUM_EPOCHS \
        --bs=$GLOBAL_BATCH_SIZE \
        --workers=$NUM_WORKERS \
        --depth=$DEPTH \
        --patch_nums=$PATCH_NUMS \
        --saln=$SHARED_ADALN \
        --anorm=$ATTN_L2_NORM \
        --fuse=$FUSE \
        --aln=$INIT_ADALN \
        --alng=$INIT_ADALN_GAMMA \
        --hd=$INIT_HEAD \
        --ini=$INIT_STD"
    
    # 添加checkpoint恢复参数（如果指定）
    if [ -n "$RESUME_CHECKPOINT" ]; then
        PYTHON_CMD="$PYTHON_CMD --resume_checkpoint=\"$RESUME_CHECKPOINT\""
        log_info "🔄 将从检查点恢复训练: $RESUME_CHECKPOINT"
    fi
    
    # 🔥 添加AGIP-VAR新参数支持
    if [ -n "$LR_PLANNER" ]; then
        PYTHON_CMD="$PYTHON_CMD --lr_planner=$LR_PLANNER"
    fi
    
    if [ -n "$LR_DISCRIMINATOR" ]; then
        PYTHON_CMD="$PYTHON_CMD --lr_discriminator=$LR_DISCRIMINATOR"
    fi
    
    if [ -n "$LOAD_OPTIMIZER_STATE" ]; then
        PYTHON_CMD="$PYTHON_CMD --load_optimizer=$LOAD_OPTIMIZER_STATE"
    fi
    
    log_blue "执行命令: $PYTHON_CMD"
    log_blue "日志文件: $LOG_FILE"
    
    # 启动训练并记录日志
    echo "多机多卡分布式训练开始时间: $(date)" > "$LOG_FILE"
    echo "分布式配置:" >> "$LOG_FILE"
    echo "  节点总数: $NNODES" >> "$LOG_FILE"
    echo "  当前节点rank: $NODE_RANK" >> "$LOG_FILE"
    echo "  每节点GPU数: $NPROC_PER_NODE" >> "$LOG_FILE"
    echo "  总GPU数: $TOTAL_GPUS" >> "$LOG_FILE"
    echo "  主节点地址: $MASTER_ADDR:$MASTER_PORT" >> "$LOG_FILE"
    echo "数据配置:" >> "$LOG_FILE"
    echo "  ImageNet数据路径: $DATA_ROOT" >> "$LOG_FILE"
    echo "  全局批次大小: $GLOBAL_BATCH_SIZE" >> "$LOG_FILE"
    echo "  每GPU批次大小: $PER_GPU_BATCH_SIZE" >> "$LOG_FILE"
    echo "  每epoch步数: $STEPS_PER_EPOCH" >> "$LOG_FILE"
    echo "训练配置:" >> "$LOG_FILE"
    echo "  训练轮数: $NUM_EPOCHS" >> "$LOG_FILE"
    echo "  分阶段训练: $ENABLE_STAGED_TRAINING" >> "$LOG_FILE"
    echo "  预热步数: $WARMUP_STEPS (1个epoch)" >> "$LOG_FILE"
    echo "  I_predictor学习率: $LR_PLANNER" >> "$LOG_FILE"
    echo "  判别器学习率: $LR_DISCRIMINATOR" >> "$LOG_FILE"
    echo "================================" >> "$LOG_FILE"
    
    # 执行8卡分布式训练
    eval "$PYTHON_CMD" 2>&1 | tee -a "$LOG_FILE"
    
    # 检查训练结果
    if [ $? -eq 0 ]; then
        if [ "$NNODES" -gt 1 ]; then
            log_info "✅ 节点${NODE_RANK}/${NNODES} 多机分布式训练成功完成！"
        else
            log_info "✅ 单机多卡分布式训练成功完成！"
        fi
        log_info "📊 查看训练日志: tail -f $LOG_FILE"
        if [ "$NODE_RANK" -eq 0 ]; then
        log_info "📁 实验结果保存在: $OUTPUT_DIR"
        fi
    else
        if [ "$NNODES" -gt 1 ]; then
            log_error "❌ 节点${NODE_RANK}/${NNODES} 多机分布式训练失败，请检查日志"
    else
            log_error "❌ 单机多卡分布式训练失败，请检查日志"
        fi
        log_error "📊 错误日志: tail -f $LOG_FILE"
        
        # 提供故障排除建议
        if [ "$NNODES" -gt 1 ]; then
            log_error "多机分布式训练故障排除建议:"
            log_error "  1. 检查所有节点的网络连通性"
            log_error "  2. 检查主节点是否首先启动"
            log_error "  3. 检查NCCL配置和网络接口设置"
            log_error "  4. 检查各节点的端口是否被占用"
            log_error "  5. 查看各节点GPU状态: nvidia-smi"
            log_error "  6. 检查节点间通信: NCCL_DEBUG=INFO"
            log_error "  7. 确保所有节点有相同的数据路径"
        else
            log_error "单机分布式训练故障排除建议:"
        log_error "  1. 检查所有GPU内存是否足够"
        log_error "  2. 检查NCCL网络配置和带宽"
        log_error "  3. 检查端口是否被占用"
        log_error "  4. 查看各GPU进程状态: nvidia-smi"
        log_error "  5. 检查节点间通信: NCCL_DEBUG=INFO"
        fi
        exit 1
    fi
}

# --------- 清理函数 ---------
cleanup() {
    log_info "🧹 清理分布式训练资源..."
    
    # 杀死可能残留的进程
    pkill -f "train_planner.py" 2>/dev/null || true
    pkill -f "torchrun" 2>/dev/null || true
    
    # 清理NCCL临时文件
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
        log_cyan "🌐 AGIP-VAR多机多卡分布式训练启动脚本"
        log_cyan "   节点${NODE_RANK}/${NNODES} | 总GPU数: ${TOTAL_GPUS}"
    else
        log_blue "🔥 AGIP-VAR单机多卡分布式训练启动脚本"
    fi
    log_cyan "================================================================"
    
    # 检查环境
    check_environment
    
    # 设置环境
    setup_environment
    
    # 显示配置
    show_configuration
    
    # 确认开始（如果不是非交互式或自动确认）
    if [[ "$AUTO_YES" = false ]] && [[ -t 0 ]]; then
        echo ""
        if [ "$NNODES" -gt 1 ]; then
            read -p "是否在节点${NODE_RANK}/${NNODES}开始多机分布式训练? (y/N): " -n 1 -r
        else
            read -p "是否开始${TOTAL_GPUS}卡分布式训练? (y/N): " -n 1 -r
        fi
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "分布式训练已取消"
            exit 0
        fi
    else
        if [ "$NNODES" -gt 1 ]; then
            log_info "🤖 节点${NODE_RANK}自动开始多机分布式训练"
        else
            log_info "🤖 自动开始${TOTAL_GPUS}卡分布式训练"
        fi
    fi
    
    # 开始训练
    start_training
    
    # 完成
    if [ "$NNODES" -gt 1 ]; then
        log_blue "🎉 节点${NODE_RANK} AGIP-VAR多机分布式训练脚本执行完成！"
    else
        log_blue "🎉 AGIP-VAR单机多卡分布式训练脚本执行完成！"
    fi
}

# --------- 帮助信息 ---------
show_help() {
    cat << EOF
�� AGIP-VAR完整ImageNet 8卡分布式训练脚本使用说明

用法: $0 [选项]

选项:
  -h, --help          显示此帮助信息
  -g, --num-gpus      设置GPU数量 (默认: $NUM_GPUS)
  -b, --batch-size    设置全局批次大小 (默认: $GLOBAL_BATCH_SIZE)
  -e, --epochs        设置训练轮数 (默认: $NUM_EPOCHS)
  -d, --data-root     设置ImageNet数据根目录 (默认: $DATA_ROOT)
  -o, --output-dir    设置输出目录 (默认: $OUTPUT_DIR)
  -w, --warmup-steps  设置预热步数 (默认: $WARMUP_STEPS)
  -r, --resume        从指定的检查点文件恢复训练
  --no-staged         禁用分阶段训练
  --var-depth         设置VAR模型深度: 16/20/24 (默认: $VAR_DEPTH)
  --gpus              设置可见GPU列表 (默认: $CUDA_VISIBLE_DEVICES_LIST)
  --port              设置主节点端口 (默认: $MASTER_PORT)
  -y, --yes           自动确认，不询问

AGIP-VAR特定选项:
  --lr-planner        I_predictor学习率 (默认: $LR_PLANNER)
  --lr-discriminator  判别器学习率 (默认: $LR_DISCRIMINATOR)
  --guidance-weight   引导权重 (默认: $GUIDANCE_WEIGHT)
  --r1-gamma          R1梯度惩罚权重 (默认: $R1_GAMMA)
  --load-optimizer    是否加载优化器状态 true/false (默认: 使用程序默认值)

示例:
  $0 -y                                     # 8卡分布式训练，默认配置
  $0 --var-depth 20 -y                     # 使用VAR-d20模型训练
  $0 --var-depth 16 -y                     # 使用VAR-d16模型训练
  $0 -b 1024 -e 200                        # 自定义批次大小和训练轮数
  $0 --num-gpus 4 --no-staged              # 4卡，禁用分阶段训练
  $0 --data-root /path/to/imagenet          # 自定义ImageNet路径
  $0 -r /path/to/checkpoint.pth             # 从检查点恢复训练
  $0 --port 29700                           # 自定义端口避免冲突
  
  # 使用不同学习率继续训练的示例:
  $0 -r /path/to/checkpoint.pth --lr-planner 1e-6 --lr-discriminator 5e-7  # 恢复训练并使用新学习率
  $0 -r /path/to/checkpoint.pth --load-optimizer false                      # 恢复模型但不加载优化器状态
  $0 -r /path/to/checkpoint.pth --lr-planner 2e-6 --load-optimizer false   # 最佳方案：新学习率+重新开始优化器

8卡分布式配置说明:
  - 全局批次大小: $GLOBAL_BATCH_SIZE (每GPU: $((GLOBAL_BATCH_SIZE / NUM_GPUS)))
  - 使用全部8张A100 GPU (每GPU最大支持BS=32)
  - 优化的NCCL通信设置
  - 数据并行训练策略
  - 一个epoch步数: ~5,005步

环境要求:
  - 8张CUDA支持的GPU (推荐A100)
  - Python 3.8+
  - PyTorch 2.0+ (支持torchrun)
  - NCCL (分布式通信)
  - 完整的ImageNet数据集
  - 充足的系统内存和存储空间

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
            -g|--num-gpus)
                NUM_GPUS="$2"
                shift 2
                ;;
            -b|--batch-size)
                GLOBAL_BATCH_SIZE="$2"
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
            --gpus)
                CUDA_VISIBLE_DEVICES_LIST="$2"
                shift 2
                ;;
            --port|--master-port)
                MASTER_PORT="$2"
                shift 2
                ;;
            --nnodes)
                NNODES="$2"
                TOTAL_GPUS=$((NNODES * NPROC_PER_NODE))
                GLOBAL_BATCH_SIZE=$((PER_GPU_BATCH_SIZE * TOTAL_GPUS))
                STEPS_PER_EPOCH=$((1281167 / GLOBAL_BATCH_SIZE))
                WARMUP_STEPS=$STEPS_PER_EPOCH
                shift 2
                ;;
            --node-rank)
                NODE_RANK="$2"
                shift 2
                ;;
            --nproc-per-node)
                NPROC_PER_NODE="$2"
                TOTAL_GPUS=$((NNODES * NPROC_PER_NODE))
                GLOBAL_BATCH_SIZE=$((PER_GPU_BATCH_SIZE * TOTAL_GPUS))
                STEPS_PER_EPOCH=$((1281167 / GLOBAL_BATCH_SIZE))
                WARMUP_STEPS=$STEPS_PER_EPOCH
                shift 2
                ;;
            --master-addr)
                MASTER_ADDR="$2"
                shift 2
                ;;
            --per-gpu-batch-size)
                PER_GPU_BATCH_SIZE="$2"
                GLOBAL_BATCH_SIZE=$((PER_GPU_BATCH_SIZE * TOTAL_GPUS))
                STEPS_PER_EPOCH=$((1281167 / GLOBAL_BATCH_SIZE))
                WARMUP_STEPS=$STEPS_PER_EPOCH
                shift 2
                ;;
            --lr-planner)
                LR_PLANNER="$2"
                shift 2
                ;;
            --lr-discriminator)
                LR_DISCRIMINATOR="$2"
                shift 2
                ;;
            --guidance-weight)
                GUIDANCE_WEIGHT="$2"
                shift 2
                ;;
            --r1-gamma)
                R1_GAMMA="$2"
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
    
    # 验证多机分布式配置
    if [ "$NODE_RANK" -ge "$NNODES" ]; then
        log_error "节点rank($NODE_RANK)必须小于节点总数($NNODES)"
        exit 1
    fi
    
    # 验证批次大小的合理性
    if [ $((GLOBAL_BATCH_SIZE % TOTAL_GPUS)) -ne 0 ]; then
        log_warn "全局批次大小($GLOBAL_BATCH_SIZE)不能被总GPU数量($TOTAL_GPUS)整除"
        # 自动调整到最接近的可整除值
        PER_GPU_BATCH_SIZE=$((GLOBAL_BATCH_SIZE / TOTAL_GPUS))
        GLOBAL_BATCH_SIZE=$((PER_GPU_BATCH_SIZE * TOTAL_GPUS))
        STEPS_PER_EPOCH=$((1281167 / GLOBAL_BATCH_SIZE))
        WARMUP_STEPS=$STEPS_PER_EPOCH
        log_info "调整后的全局批次大小: $GLOBAL_BATCH_SIZE (每GPU: $PER_GPU_BATCH_SIZE)"
    fi
    
    # 警告多机配置
    if [ "$NNODES" -gt 1 ]; then
        log_warn "⚠️ 多机分布式模式已启用"
        log_warn "   请确保:"
        log_warn "   1. 主节点($MASTER_ADDR)首先启动"
        log_warn "   2. 所有节点网络连通，端口$MASTER_PORT可用"
        log_warn "   3. 所有节点有相同的数据路径和模型权重"
        log_warn "   4. 启动命令参数在所有节点上一致"
    fi
}

# --------- 脚本入口 ---------
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    parse_arguments "$@"
    main
fi 