#!/bin/bash

# Quick test script for local noise injection experiment
# This demonstrates different local noise configurations

# Activate environment
source /home/intern/miniconda3/etc/profile.d/conda.sh
conda activate aid-var
export TMPDIR=/home/intern/tmp
export PYTHONPATH="${PYTHONPATH}:$(pwd)/stylegan_t"

echo "================================================"
echo "Testing Local Noise Injection"
echo "================================================"

# Test 2: Local noise at top-left corner
echo ""
echo "Test 2: Local noise at top-left corner"
python experiment_scale_noise.py \
  --noise_type local \
  --noise_scale_idx 4 \
  --noise_magnitude 1.0 \
  --noise_region 2 3 2 3 \
  --num_samples 4

echo ""
echo "================================================"
echo "All tests complete!"
echo "Check experiments/scale_noise_visualization/"
echo "================================================"
