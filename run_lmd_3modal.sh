#!/bin/bash

# Script to run LMD for 3-modality (Camera+Radar+LiDAR) model

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Override these via environment variables if needed.
DATA_ROOT="${DATA_ROOT:-$SCRIPT_DIR/data}"
CKPT="${CKPT:-$SCRIPT_DIR/checkpoints/4x5_3e-4s_cam_rad_lid_18:06:48}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/lmd_results/3modal}"

echo "========================================"
echo "Running LMD for 3-Modality Model"
echo "========================================"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run 3-modal LMD
echo ""
echo "Running Camera+Radar+LiDAR LMD..."
echo "-----------------------------------"
python3 "$SCRIPT_DIR/visualize_lmd_3modal.py" \
    --checkpoint "$CKPT" \
    --output_dir "$OUTPUT_DIR" \
    --data_root "$DATA_ROOT" \
    --start_step 1 \
    --end_step 6019 \
    --step_interval 50 \
    --visualize

echo ""
echo "========================================"
echo "3-Modal LMD processing complete!"
echo "Results saved in $OUTPUT_DIR"
echo ""
echo "Directory structure:"
echo "  lmd_results/"
echo "    └── 3modal/"
echo "        └── data/"
echo "            ├── (LMD pickle and numpy files)"
echo "            └── visualize_3/ (PNG visualizations)"
echo ""
echo "Pickle files contain:"
echo "  - Camera predictions"
echo "  - Radar predictions"
echo "  - LiDAR predictions"
echo "  - Bias predictions"
echo "========================================"
