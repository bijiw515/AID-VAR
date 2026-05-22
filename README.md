# Adversarially Error Correction for Visual Autoregressive Generation

Code for paper Adversarially Error Correction for Visual Autoregressive Generation
<img width="1620" height="831" alt="2bb37033e79da37186eb37c2e769a2c5" src="https://github.com/user-attachments/assets/1e07accb-9173-4585-b8cd-928ce8a4fb35" />

## Highlights

- 💪 **Cross‑scale error correction** – A lightweight injector identifies and corrects cumulated errors in VAR models.
- 💡 **Stable adversarial training** – RGB discriminator + soft‑decode route enable effective gradient flow; dynamic training loop keeps fake samples fresh.
- 🚀 **High efficiency** – Improves various VAR backbones with negligible extra cost.

## Installation

```bash
conda env create -n aid-var python=3.10
conda activate aid-var
pip install -r requirements.txt
```


## Preparation

### Dataset

Download ImageNet and organize it as:

```
datasets/imagenet-1k/data/
  train/
  val/
```

## Training

**Single-node 8-GPU training:**

```bash
torchrun \
    --nnodes=1 --node_rank=0 --nproc_per_node=8 \
    --master_addr=localhost --master_port=29600 \
    train.py \
    --data_path datasets/imagenet-1k/data \
    --ep 100 --bs 32 --workers 8 \
    --depth 16 --var_ckpt ./checkpoints/var_d16.pth \
    --lr_planner 1e-6 --lr_discriminator 1e-6 \
    --warmup_steps 0 --lambda_rec 0 \
    --guidance_weight 0.005 --guidance_target_weight 0.001 --guidance_ramp_epochs 15 \
    --r1_gamma 0.2 \
    --output_dir ./experiments --save_interval 1 --val_interval 1
```

Or use the provided script (edit paths inside first):

```bash
bash train.sh
```

**Resume from checkpoint:**

```bash
torchrun ... train.py ... \
    --resume_checkpoint experiments/checkpoint_epoch_10.pth
```

To resume with a new learning rate (without loading optimizer state):

```bash
torchrun ... train.py ... \
    --resume_checkpoint experiments/checkpoint_epoch_10.pth \
    --load_optimizer false \
    --lr_planner 5e-7
```

**Multi-node training** (run on each node):

```bash
# Node 0
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=8 \
    --master_addr=<node0-ip> --master_port=29600 train.py ...

# Node 1
torchrun --nnodes=2 --node_rank=1 --nproc_per_node=8 \
    --master_addr=<node0-ip> --master_port=29600 train.py ...
```

## Sampling

Generate 50K samples for evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 python generate_aid_var_samples.py \
    --model_depth 16 \
    --cfg 1.5 --top_p 0.96 --top_k 900 \
    --planner_ckpt /path/to/checkpoint.pth \
    --create_npz \
    --dtype float16
```

## Models


| model       | reso. | FID  | rel. cost | #params | HF weights🤗 |
| ----------- | ----- | ---- | --------- | ------- | ------------ |
| AID-VAR-d16 | 256   | 3.24 | 0.4       | 321M    | coming soon  |
| VAR-d20     | 256   | 2.54 | 0.5       | 619M    | coming soon  |
| VAR-d24     | 256   | 2.08 | 0.6       | 1.02B   | coming soon  |


## Checkpoint Format

Saved checkpoints contain:

```
planner_state_dict          # GuidanceInjector weights
discriminator_state_dict    # Discriminator head weights
planner_optimizer_state_dict
discriminator_optimizer_state_dict
trainer_state               # Step count, warmup phase, collapse history
epoch
train_metrics / val_metrics
```

## Acknowledgements

Our code builds on [VAR](https://github.com/FoundationVision/VAR) and [StyleGAN-T](https://github.com/autonomousvision/stylegan-t). Thanks for their excellent work.
