# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **AID-VAR (Adversarially Injected Diagnosis for Visual Autoregressive Generation)** research implementation. It extends the VAR (Visual AutoRegressive) model with an adversarially-guided planning module for improved image generation quality.

**Core Concept**: The system uses a lightweight GuidanceInjector (planning module) that generates planning token maps which are added element-wise to VAR's internal states. A StyleGAN-T discriminator provides adversarial feedback to guide the planning process.

## Architecture Overview

### Key Components

1. **VAR (Visual AutoRegressive Model)** - Frozen pretrained model
   - Located in `models/var.py` and `models/basic_var.py`
   - Generates images autoregressively at multiple scales
   - Uses patch_nums configuration: `(1, 2, 3, 4, 5, 6, 8, 10, 13, 16)` for multi-scale generation

2. **VQ-VAE** - Frozen pretrained model
   - Located in `models/vqvae.py`
   - Tokenizes images into discrete tokens
   - Configuration: ch=160, vocab_size=4096, z_channels=32

3. **GuidanceInjector (Planning Module)** - Trainable
   - Located in `models/guidance_injector.py`
   - Lightweight transformer that generates planning token maps
   - Output is element-wise added to VAR's internal states at each scale
   - Architecture: UltraSafeTransformerBlock with extensive numerical stability protections

4. **StyleGANDiscriminatorAdapter** - Trainable (heads only)
   - Located in `models/discriminator_adapter.py`
   - Uses frozen DINO backbone with trainable discriminator heads
   - Provides pixel-level adversarial feedback
   - Multi-scale discrimination matching VAR's patch_nums

### Training Architecture

- **PlannerTrainer** (`trainer_planner.py`): Core training logic
  - Implements alternating training between discriminator and planner
  - Includes collapse detection and recovery mechanisms
  - Progressive guidance weight ramping
  - Staged training: warmup phase → joint training phase

- **AlternatingTrainingManager** (`aid_helpers.py`): Controls training dynamics
  - Manages when to update discriminator vs planner
  - Implements progressive guidance weight scheduling
  - Monitors discriminator health to prevent mode collapse

### Model Depth Variants

The codebase supports multiple VAR model depths (affects embedding dimensions):
- **d16**: 1024 dimensions (16 * 64)
- **d20**: 1280 dimensions (20 * 64)
- **d24**: 1536 dimensions (24 * 64)

The GuidanceInjector automatically adjusts its dimensions based on VAR depth.

## Common Commands

### Environment Setup

```bash
# Activate conda environment
source /home/intern/miniconda3/etc/profile.d/conda.sh
conda activate aid-var

# Set required environment variables
export TMPDIR=/home/intern/tmp
export PYTHONPATH="${PYTHONPATH}:$(pwd)/stylegan_t"
```

### Training

**Single-node multi-GPU training (8 GPUs):**
```bash
# From scratch
./train_full_imagenet.sh -y

# With specific VAR depth
./train_full_imagenet.sh --var-depth 16 -y

# Resume from checkpoint
./train_full_imagenet.sh -r /path/to/checkpoint.pth

# Resume with new learning rates (without loading optimizer state)
./train_full_imagenet.sh -r /path/to/checkpoint.pth \
  --lr-planner 1e-6 \
  --lr-discriminator 5e-7 \
  --load-optimizer false
```

**Direct Python invocation:**
```bash
torchrun --nproc_per_node=8 train_planner.py \
  --data_path /path/to/imagenet \
  --ep 100 \
  --depth 16 \
  --lr_planner 1e-6 \
  --lr_discriminator 1e-6
```

**Multi-node distributed training:**
```bash
# Node 0 (master)
./train_full_imagenet.sh --nnodes 2 --node-rank 0 --master-addr <master-ip>

# Node 1
./train_full_imagenet.sh --nnodes 2 --node-rank 1 --master-addr <master-ip>
```

### Sample Generation

**Generate samples for FID/IS evaluation:**
```bash
CUDA_VISIBLE_DEVICES=0 python generate_aid_fid_samples.py \
  --model_depth 16 \
  --cfg 1.5 \
  --top_p 0.96 \
  --top_k 900 \
  --planner_ckpt /path/to/checkpoint.pth \
  --create_npz \
  --dtype float16
```

### Evaluation

**Compute FID:**
```bash
python evaluate_fid_unified.py \
  --generated_samples /path/to/samples.npz \
  --reference_stats /path/to/VIRTUAL_imagenet256_labeled.npz
```

**Compute ISCS (Inception Score):**
```bash
./compute_iscs.sh  # Configure paths inside the script
```

### Visualization

**Visualize guidance effects:**
```bash
./visualize.sh
# Or directly:
python visualize_guidance.py \
  --model_depth 16 \
  --planner_ckpt /path/to/checkpoint.pth
```

## Development Notes

### Critical Configuration Parameters

1. **Learning Rates**: Both planner and discriminator use the same learning rate (1e-6 by default) to maintain training balance
2. **Guidance Weight**: Progressively ramped from 0.005 to 0.001 over training
3. **Warmup Steps**: Typically set to one epoch worth of steps
4. **Batch Size**: Global batch size = per_GPU_batch_size * num_GPUs (e.g., 32 * 8 = 256)

### Key Architectural Decisions

1. **Spatial-Aware Planning**: GuidanceInjector outputs are added element-wise to VAR states (not concatenated)
2. **Frozen Backbones**: Both VAR and VQ-VAE are frozen; only GuidanceInjector and discriminator heads are trained
3. **Alternating Training**: Discriminator and planner are updated in separate backward passes to avoid computation graph conflicts
4. **Multi-Scale Processing**: All components operate at VAR's 10 scale levels defined by patch_nums

### Checkpoint Structure

Checkpoints contain:
- `planner_state_dict`: GuidanceInjector weights
- `discriminator_state_dict`: Discriminator weights
- `planner_optimizer_state_dict`: Optimizer state for planner
- `discriminator_optimizer_state_dict`: Optimizer state for discriminator
- `trainer_state`: Training progress (step count, warmup phase, collapse history, etc.)
- `train_metrics`, `val_metrics`: Performance metrics
- `epoch`: Completed epoch number

### File Organization

- `models/`: Model architectures (VAR, VQ-VAE, GuidanceInjector, discriminators)
- `utils/`: Training utilities (data loading, distributed training, optimization)
- `experiments/`: Training outputs (checkpoints, logs, validation samples)
- `checkpoints/`: Pretrained VAR weights (var_d16.pth, var_d20.pth, var_d24.pth)
- `logs/`: Training logs
- Shell scripts (`.sh`): High-level training/evaluation workflows
- Python scripts: Core implementation

### Distributed Training

The codebase uses PyTorch's `torchrun` for distributed training:
- Supports both single-node multi-GPU and multi-node configurations
- NCCL backend for GPU communication
- Automatic environment detection for Slurm/Kubernetes clusters
- Environment variables: `NNODES`, `NODE_RANK`, `NPROC_PER_NODE`, `MASTER_ADDR`, `MASTER_PORT`

### Numerical Stability

The GuidanceInjector includes extensive numerical stability measures:
- SafeLayerNorm with clamping to prevent NaN/Inf values
- Conservative weight initialization (std=0.001)
- Gradient clipping (default: 0.5)
- Input/output clamping throughout the network

### Common Gotchas

1. **Data Path**: ImageNet must be in standard format with `train/` and `val/` subdirectories
2. **VAE Checkpoint**: The system auto-downloads `vae_ch160v4096z32.pth` if missing
3. **VAR Weights**: Must manually place `var_d{depth}.pth` in `checkpoints/` directory
4. **CUDA Environment**: Requires CUDA with proper cuDNN setup (version 8.9.7+ recommended)
5. **Resume Training**: When resuming with different learning rates, set `--load-optimizer false`

### Metrics to Monitor

During training, watch these key metrics:
- `D_acc`: Discriminator accuracy (should stay around 0.6-0.8, not too high or low)
- `real_acc`, `fake_acc`: Per-class discriminator accuracy (should be balanced)
- `P_loss`: Planner loss (adversarial + reconstruction)
- `D_loss`: Discriminator loss
- `guidance_weight`: Current guidance weight (progressively increases)
- Collapse warnings: If discriminator collapses (acc_D too high/low), training auto-adjusts

### Testing Single Class

For faster iteration during development:
```bash
./train_single_class.sh  # Trains on a single ImageNet class
python train_single_class.py --class_id 0 --ep 50
```
