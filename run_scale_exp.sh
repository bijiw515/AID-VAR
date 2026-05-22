#!/bin/bash

# Enhanced script for VAR scale noise injection experiments
# Supports batch experiments, parameter sweeps, and comprehensive analysis
#
# Usage:
#   ./run_scale_exp.sh [MODE] [OPTIONS]
#
# Modes:
#   single    - Run a single experiment with specified parameters
#   sweep     - Sweep across multiple scales or magnitudes
#   compare   - Compare different noise types (global vs local)
#   batch     - Run multiple predefined experiments
#
# Examples:
#   ./run_scale_exp.sh single --noise_scale_idx 2 --noise_magnitude 0.5
#   ./run_scale_exp.sh sweep --sweep_type scale
#   ./run_scale_exp.sh sweep --sweep_type magnitude
#   ./run_scale_exp.sh compare --noise_scale_idx 3
#   ./run_scale_exp.sh batch

set -e  # Exit on error

# Activate conda environment
source /home/intern/miniconda3/etc/profile.d/conda.sh
conda activate aid-var

# Set environment variables
export TMPDIR=/home/intern/tmp
export PYTHONPATH="${PYTHONPATH}:$(pwd)/stylegan_t"

# Default parameters
MODEL_DEPTH=16
NUM_SAMPLES=4
CLASS_LABELS="207 283 22 284"  # goldfish, tiger_cat, boar, siamese_cat
CFG=4.0
TOP_K=900
TOP_P=0.95
SEED=42
LOSS_TYPE="l2"
OUTPUT_DIR="experiments/scale_noise_visualization"

# Parse mode
MODE=${1:-single}
shift || true  # Remove mode from arguments

# Function to print header
print_header() {
    echo ""
    echo "========================================================================"
    echo "$1"
    echo "========================================================================"
}

# Function to print step
print_step() {
    echo ""
    echo ">>> $1"
}

# Function: Run single experiment
run_single() {
    print_header "Running Single Experiment"

    python experiment_scale_noise.py \
        --model_depth $MODEL_DEPTH \
        --num_samples $NUM_SAMPLES \
        --class_labels $CLASS_LABELS \
        --cfg $CFG \
        --top_k $TOP_K \
        --top_p $TOP_P \
        --seed $SEED \
        --loss_type $LOSS_TYPE \
        --output_dir $OUTPUT_DIR \
        "$@"

    echo ""
    echo "Results saved to: $OUTPUT_DIR"
}

# Function: Sweep across different scales
sweep_scales() {
    print_header "Sweeping Across All Scales"

    local magnitude=${1:-0.5}
    local noise_type=${2:-local}

    print_step "Configuration:"
    echo "  - Noise magnitude: $magnitude"
    echo "  - Noise type: $noise_type"
    echo "  - Scales to test: 0-9 (all 10 scales)"

    # Create sweep output directory
    local sweep_dir="${OUTPUT_DIR}/sweep_scales_mag${magnitude}_${noise_type}"
    mkdir -p "$sweep_dir"

    # Run experiments for each scale
    for scale_idx in {0..9}; do
        print_step "Testing scale $scale_idx..."

        python experiment_scale_noise.py \
            --model_depth $MODEL_DEPTH \
            --num_samples $NUM_SAMPLES \
            --class_labels $CLASS_LABELS \
            --noise_scale_idx $scale_idx \
            --noise_magnitude $magnitude \
            --noise_type $noise_type \
            --cfg $CFG \
            --top_k $TOP_K \
            --top_p $TOP_P \
            --seed $SEED \
            --loss_type $LOSS_TYPE \
            --output_dir "$sweep_dir/scale${scale_idx}"
    done

    print_header "Scale Sweep Complete!"
    echo "All results saved to: $sweep_dir"
    echo ""
    echo "To analyze results, check the loss_curve.png in each subdirectory"
}

# Function: Sweep across different magnitudes
sweep_magnitudes() {
    print_header "Sweeping Across Different Noise Magnitudes"

    local scale_idx=${1:-2}
    local noise_type=${2:-local}

    # Magnitudes to test
    local magnitudes=(0.1 0.2 0.5 1.0 2.0 5.0)

    print_step "Configuration:"
    echo "  - Testing scale: $scale_idx"
    echo "  - Noise type: $noise_type"
    echo "  - Magnitudes: ${magnitudes[@]}"

    # Create sweep output directory
    local sweep_dir="${OUTPUT_DIR}/sweep_magnitudes_scale${scale_idx}_${noise_type}"
    mkdir -p "$sweep_dir"

    # Run experiments for each magnitude
    for mag in "${magnitudes[@]}"; do
        print_step "Testing magnitude $mag..."

        python experiment_scale_noise.py \
            --model_depth $MODEL_DEPTH \
            --num_samples $NUM_SAMPLES \
            --class_labels $CLASS_LABELS \
            --noise_scale_idx $scale_idx \
            --noise_magnitude $mag \
            --noise_type $noise_type \
            --cfg $CFG \
            --top_k $TOP_K \
            --top_p $TOP_P \
            --seed $SEED \
            --loss_type $LOSS_TYPE \
            --output_dir "$sweep_dir/mag${mag}"
    done

    print_header "Magnitude Sweep Complete!"
    echo "All results saved to: $sweep_dir"
}

# Function: Compare global vs local noise
compare_noise_types() {
    print_header "Comparing Global vs Local Noise"

    local scale_idx=${1:-2}
    local magnitude=${2:-0.5}

    print_step "Configuration:"
    echo "  - Testing scale: $scale_idx"
    echo "  - Noise magnitude: $magnitude"
    echo "  - Comparing: GLOBAL vs LOCAL"

    # Create comparison output directory
    local compare_dir="${OUTPUT_DIR}/compare_types_scale${scale_idx}_mag${magnitude}"
    mkdir -p "$compare_dir"

    # Test GLOBAL noise
    print_step "Testing GLOBAL noise..."
    python experiment_scale_noise.py \
        --model_depth $MODEL_DEPTH \
        --num_samples $NUM_SAMPLES \
        --class_labels $CLASS_LABELS \
        --noise_scale_idx $scale_idx \
        --noise_magnitude $magnitude \
        --noise_type global \
        --cfg $CFG \
        --top_k $TOP_K \
        --top_p $TOP_P \
        --seed $SEED \
        --loss_type $LOSS_TYPE \
        --output_dir "$compare_dir/global"

    # Test LOCAL noise
    print_step "Testing LOCAL noise..."
    python experiment_scale_noise.py \
        --model_depth $MODEL_DEPTH \
        --num_samples $NUM_SAMPLES \
        --class_labels $CLASS_LABELS \
        --noise_scale_idx $scale_idx \
        --noise_magnitude $magnitude \
        --noise_type local \
        --cfg $CFG \
        --top_k $TOP_K \
        --top_p $TOP_P \
        --seed $SEED \
        --loss_type $LOSS_TYPE \
        --output_dir "$compare_dir/local"

    print_header "Comparison Complete!"
    echo "Global results: $compare_dir/global"
    echo "Local results:  $compare_dir/local"
}

# Function: Run batch of predefined experiments
run_batch() {
    print_header "Running Batch of Predefined Experiments"

    # Create batch output directory
    local batch_dir="${OUTPUT_DIR}/batch_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$batch_dir"

    echo "Batch output directory: $batch_dir"

    # Experiment 1: Early scale, small noise
    print_step "Experiment 1: Early scale (1), small noise (0.2)"
    python experiment_scale_noise.py \
        --noise_scale_idx 1 --noise_magnitude 0.2 --noise_type local \
        --output_dir "$batch_dir/exp1_scale1_mag0.2_local" \
        --model_depth $MODEL_DEPTH --num_samples $NUM_SAMPLES \
        --class_labels $CLASS_LABELS --cfg $CFG --seed $SEED

    # Experiment 2: Middle scale, medium noise
    print_step "Experiment 2: Middle scale (5), medium noise (0.5)"
    python experiment_scale_noise.py \
        --noise_scale_idx 5 --noise_magnitude 0.5 --noise_type local \
        --output_dir "$batch_dir/exp2_scale5_mag0.5_local" \
        --model_depth $MODEL_DEPTH --num_samples $NUM_SAMPLES \
        --class_labels $CLASS_LABELS --cfg $CFG --seed $SEED

    # Experiment 3: Late scale, large noise
    print_step "Experiment 3: Late scale (8), large noise (1.0)"
    python experiment_scale_noise.py \
        --noise_scale_idx 8 --noise_magnitude 1.0 --noise_type local \
        --output_dir "$batch_dir/exp3_scale8_mag1.0_local" \
        --model_depth $MODEL_DEPTH --num_samples $NUM_SAMPLES \
        --class_labels $CLASS_LABELS --cfg $CFG --seed $SEED

    # Experiment 4: Middle scale, global noise
    print_step "Experiment 4: Middle scale (5), global noise (0.5)"
    python experiment_scale_noise.py \
        --noise_scale_idx 5 --noise_magnitude 0.5 --noise_type global \
        --output_dir "$batch_dir/exp4_scale5_mag0.5_global" \
        --model_depth $MODEL_DEPTH --num_samples $NUM_SAMPLES \
        --class_labels $CLASS_LABELS --cfg $CFG --seed $SEED

    print_header "Batch Experiments Complete!"
    echo "All results saved to: $batch_dir"
    echo ""
    echo "Summary:"
    echo "  - Exp 1: Early scale (1), small noise (0.2), local"
    echo "  - Exp 2: Middle scale (5), medium noise (0.5), local"
    echo "  - Exp 3: Late scale (8), large noise (1.0), local"
    echo "  - Exp 4: Middle scale (5), global noise (0.5)"
}

# Function: Show usage
show_usage() {
    cat << EOF
Usage: $0 [MODE] [OPTIONS]

MODES:
  single              Run a single experiment (default)
  sweep               Sweep across parameters
  compare             Compare different noise types
  batch               Run multiple predefined experiments
  help                Show this help message

OPTIONS FOR 'single' MODE:
  All options from experiment_scale_noise.py are supported, including:
    --noise_scale_idx N       Scale to inject noise (0-9, default: 2)
    --noise_magnitude M       Noise standard deviation (default: 0.2)
    --noise_type TYPE         'global' or 'local' (default: local)
    --noise_region H1 H2 W1 W2  Local noise region coordinates
    --model_depth D           VAR model depth: 16, 20, 24, or 30 (default: 16)
    --num_samples N           Number of samples (default: 4)
    --loss_type TYPE          Loss type: l1, l2, or cosine (default: l2)
    --cfg C                   CFG strength (default: 4.0)
    --seed S                  Random seed (default: 42)

OPTIONS FOR 'sweep' MODE:
  --sweep_type TYPE         'scale' or 'magnitude' (default: scale)
  --noise_magnitude M       For scale sweep (default: 0.5)
  --noise_scale_idx N       For magnitude sweep (default: 2)
  --noise_type TYPE         'global' or 'local' (default: local)

OPTIONS FOR 'compare' MODE:
  --noise_scale_idx N       Scale to test (default: 2)
  --noise_magnitude M       Noise magnitude (default: 0.5)

EXAMPLES:
  # Single experiment with custom parameters
  $0 single --noise_scale_idx 3 --noise_magnitude 0.8 --noise_type global

  # Sweep across all scales with magnitude 0.5
  $0 sweep --sweep_type scale --noise_magnitude 0.5

  # Sweep magnitudes at scale 2
  $0 sweep --sweep_type magnitude --noise_scale_idx 2

  # Compare global vs local at scale 5
  $0 compare --noise_scale_idx 5 --noise_magnitude 1.0

  # Run batch of experiments
  $0 batch

EOF
}

# Main execution logic
print_header "VAR Scale Noise Injection Experiment Suite"
echo "Mode: $MODE"
echo "Time: $(date)"

case $MODE in
    single)
        run_single "$@"
        ;;

    sweep)
        # Parse sweep options
        SWEEP_TYPE="scale"
        while [[ $# -gt 0 ]]; do
            case $1 in
                --sweep_type)
                    SWEEP_TYPE="$2"
                    shift 2
                    ;;
                --noise_magnitude)
                    NOISE_MAG="$2"
                    shift 2
                    ;;
                --noise_scale_idx)
                    SCALE_IDX="$2"
                    shift 2
                    ;;
                --noise_type)
                    NOISE_TYPE="$2"
                    shift 2
                    ;;
                *)
                    shift
                    ;;
            esac
        done

        if [ "$SWEEP_TYPE" = "scale" ]; then
            sweep_scales "${NOISE_MAG:-0.5}" "${NOISE_TYPE:-local}"
        elif [ "$SWEEP_TYPE" = "magnitude" ]; then
            sweep_magnitudes "${SCALE_IDX:-2}" "${NOISE_TYPE:-local}"
        else
            echo "Error: Unknown sweep type '$SWEEP_TYPE'"
            echo "Use 'scale' or 'magnitude'"
            exit 1
        fi
        ;;

    compare)
        # Parse compare options
        SCALE_IDX=2
        NOISE_MAG=0.5
        while [[ $# -gt 0 ]]; do
            case $1 in
                --noise_scale_idx)
                    SCALE_IDX="$2"
                    shift 2
                    ;;
                --noise_magnitude)
                    NOISE_MAG="$2"
                    shift 2
                    ;;
                *)
                    shift
                    ;;
            esac
        done

        compare_noise_types $SCALE_IDX $NOISE_MAG
        ;;

    batch)
        run_batch
        ;;

    help|--help|-h)
        show_usage
        exit 0
        ;;

    *)
        echo "Error: Unknown mode '$MODE'"
        echo ""
        show_usage
        exit 1
        ;;
esac

print_header "All Tasks Complete!"
echo "Check the output directory for results: $OUTPUT_DIR"
echo ""
