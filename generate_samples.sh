export CUDA_VISIBLE_DEVICES=1 && python generate_agip_fid_samples.py \
    --model_depth 16 \
    --cfg 1.5 \
    --top_p 0.96 \
    --top_k 900 \
    --planner_ckpt /home/intern/Ligong/VAR/experiments/agip_var_full_imagenet_staged_20250829_105140/checkpoints/checkpoint_epoch_1.pth
    --save_format both \
    --create_npz \
    --dtype float16 \
    --device cuda