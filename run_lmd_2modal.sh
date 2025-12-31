#!/bin/bash

# Script to run LMD for both camera+radar and camera+lidar modalities

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Override these via environment variables if needed.
DATA_ROOT="${DATA_ROOT:-$SCRIPT_DIR/data}"
RADAR_CKPT="${RADAR_CKPT:-$SCRIPT_DIR/checkpoints/camera_radar}"
LIDAR_CKPT="${LIDAR_CKPT:-$SCRIPT_DIR/checkpoints/camera_lidar}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/lmd_results}"

echo "========================================"
echo "Running LMD for Two-Modality Models"
echo "========================================"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run camera+radar LMD
echo ""
echo "Running Camera+Radar LMD..."
echo "-----------------------------------"
python3 "$SCRIPT_DIR/visualize_lmd_2modal.py" \
    --checkpoint "$RADAR_CKPT" \
    --model_type camera_radar \
    --start_step 1 \
    --end_step 6019 \
    --step_interval 50 \
    --output_dir "$OUTPUT_DIR" \
    --data_root "$DATA_ROOT" \
    --visualize

# Run camera+lidar LMD  
echo ""
echo "Running Camera+LiDAR LMD..."
echo "-----------------------------------"
python3 "$SCRIPT_DIR/visualize_lmd_2modal.py" \
    --checkpoint "$LIDAR_CKPT" \
    --model_type camera_lidar \
    --start_step 1 \
    --end_step 6019 \
    --step_interval 50 \
    --output_dir "$OUTPUT_DIR" \
    --data_root "$DATA_ROOT" \
    --visualize

echo ""
echo "========================================"
echo "LMD processing complete!"
echo "Results saved in $OUTPUT_DIR"
echo ""
echo "Directory structure:"
echo "  lmd_results/"
echo "    ├── camera_radar/"
echo "    │   ├── data/          (LMD pickle files)"
echo "    │   └── visualize_2/   (PNG visualizations)"
echo "    └── camera_lidar/"
echo "        ├── data/          (LMD pickle files)"
echo "        └── visualize_2/   (PNG visualizations)"
echo "========================================"
