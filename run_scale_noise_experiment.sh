#!/bin/bash

# Experiment: Add random noise at second scale and visualize all scales
# This script runs the scale noise injection experiment
#
# Usage:
#   ./run_scale_noise_experiment.sh [OPTIONS]
#
# Examples:
#   ./run_scale_noise_experiment.sh --noise_scale_idx 1 --noise_magnitude 0.3
#   ./run_scale_noise_experiment.sh --noise_scale_idx 2 --noise_magnitude 0.5 --model_depth 20

# Activate conda environment
source /home/intern/miniconda3/etc/profile.d/conda.sh
conda activate aid-var

# Set environment variables
export TMPDIR=/home/intern/tmp
export PYTHONPATH="${PYTHONPATH}:$(pwd)/stylegan_t"

echo "================================================"
echo "VAR Scale Noise Injection Experiment"
echo "================================================"

# Run the experiment with all arguments passed through
python experiment_scale_noise.py "$@"

echo ""
echo "================================================"
echo "Experiment complete!"
echo "Check experiments/scale_noise_visualization/ for results"
echo "================================================"
