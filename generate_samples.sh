python generate_aid_fid_samples.py \
    --model_depth 24 \
    --cfg 1.5 \
    --top_p 0.96 \
    --top_k 900 \
    --planner_ckpt GuidanceInjector_d24.pth
    --dtype float16 \
    --device cuda