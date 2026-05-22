export CUDA_VISIBLE_DEVICES=6 && python generate_aid_fid_samples.py \
    --model_depth 24 \
    --cfg 5.0 \
    --top_p 0.96 \
    --top_k 900 \
    --planner_ckpt /home/intern/Ligong/VAR/exp_ckpt/checkpoints/GuidanceInjector_d24.pth
    --create_npz \
    --dtype float16 \
    --device cuda