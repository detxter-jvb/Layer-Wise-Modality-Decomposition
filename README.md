<h1 align="center">Layer-Wise Modality Decomposition for Interpretable Multimodal Sensor Fusion</h1>

> **Official implementation** of **LMD** (NeurIPS 2025, Poster)

<p align="center">
  <img src="https://img.shields.io/badge/venue-NeurIPS%202025-7a77ff" />
  <img src="https://img.shields.io/badge/license-MIT-green" />
</p>

LMD is a **post-hoc interpretability** method that **decomposes a pretrained fusion model's prediction into modality-wise components at every layer**—without changing the model architecture. In autonomous driving (e.g., camera-radar or camera-LiDAR), LMD reveals how each sensor contributes to the final decision, and comes with **structured perturbation metrics** to validate separation.

---

## 📣 News

* **2025-9**: Paper accepted to **NeurIPS 2025** 🎉
* **2025-12**: Codebase will be cleaned up and released here.

---

## 🚀 Key Features

* **Model-agnostic**: Works with existing fusion models (e.g., SimpleBEV variants) without retraining.
* **Layer-wise decomposition**: Obtain `camera`, `radar`/`lidar`, and `bias` predictions at **every layer**.
* **Faithful linearization**: Cache **activation ratios** and **normalization statistics**; uses **identity** & **ratio** rules for BatchNorm and Layer/InstanceNorm.
* **Validation metrics**: Pearson/MSE-based **modality replacement** test to check separation.
* **Fast & scalable**: Only **two forward passes**, no backprop; near-inference latency.

---

## 📁 Repository Layout

```
LMD_two_three_modal_github/
  project/                        # Core model + utils (SimpleBEV variant)
  data/                           # LMD evaluation data (pt files)
  checkpoints/                    # Pretrained checkpoints
  run_lmd_2modal.sh
  run_lmd_3modal.sh
  visualize_lmd_2modal.py
  visualize_lmd_3modal.py
  eval_lmd_cam_rad.py
  eval_lmd_cam_lid.py
  eval_lmd_cam_rad_lid.py
  eval_nuscenes_save_data.py
  eval_analysis_2modal_identity_ratio.ipynb
```

---

## 🧰 Installation

Python 3.10+ recommended.

Create and activate the conda env:
```bash
conda create -n bev python=3.10 -y
conda activate bev
```

Install PyTorch (matching the version used for this codebase):
```bash
pip install torch==2.7.1+cu126 torchvision==0.22.1+cu126 --index-url https://download.pytorch.org/whl/cu126
```

Minimal runtime (LMD visualization/eval):
```
pip install numpy matplotlib pandas fire efficientnet_pytorch
```

For dataset export (nuScenes) you also need:
```
pip install nuscenes-devkit pyquaternion opencv-python Pillow lyft-dataset-sdk
```

Optional (notebook KDE plot):
```
pip install scipy
```

---

## 📦 Checkpoints

Checkpoints will be uploaded to Google Drive (TBD).
Expected structure:

```
checkpoints/
  camera_radar/
  camera_lidar/
  4x5_3e-4s_cam_rad_lid_18:06:48/
```

You can override paths with environment variables in the run scripts.

---

## 📚 Dataset (nuScenes)

Download nuScenes from the official site and set the root directory as:

```
/path/to/nuscenes/
  samples/
  sweeps/
  v1.0-trainval/
  v1.0-test/
  maps/
```

---

## 🧪 Create Evaluation Data (pt files)

The evaluation scripts use **pre-extracted** pt files:

```
LMD_two_three_modal_github/data/
  batch_1_validation_data/         # camera+radar
  batch_1_validation_data_lidar/   # camera+lidar
```

Generate them with:

```
python eval_nuscenes_save_data.py \
  --data_dir /path/to/nuscenes \
  --save_mode both \
  --radar_out_dir data/batch_1_validation_data \
  --lidar_out_dir data/batch_1_validation_data_lidar \
  --device_ids 0
```

If you want only one modality:
```
python eval_nuscenes_save_data.py --save_mode radar --radar_out_dir data/batch_1_validation_data
python eval_nuscenes_save_data.py --save_mode lidar --lidar_out_dir data/batch_1_validation_data_lidar
```

---

## ▶️ Run LMD (2-Modal)

Use the script (edit env vars or the script itself):

```
DATA_ROOT=/path/to/data \
RADAR_CKPT=/path/to/checkpoints/camera_radar \
LIDAR_CKPT=/path/to/checkpoints/camera_lidar \
OUTPUT_DIR=/path/to/output \
  bash run_lmd_2modal.sh
```

This runs:
- `visualize_lmd_2modal.py` for camera+radar and camera+lidar
- Outputs in `OUTPUT_DIR/camera_radar/...` and `OUTPUT_DIR/camera_lidar/...`

---

## ▶️ Run LMD (3-Modal)

```
DATA_ROOT=/path/to/data \
CKPT=/path/to/checkpoints/4x5_3e-4s_cam_rad_lid_18:06:48 \
OUTPUT_DIR=/path/to/output \
  bash run_lmd_3modal.sh
```

---

## 📊 Run Evaluation (Perturbation Metrics)

2-modality:
```
LMD_DATA_ROOT=/path/to/data \
LMD_CKPT_DIR=/path/to/checkpoints/camera_radar \
python eval_lmd_cam_rad.py

LMD_DATA_ROOT=/path/to/data \
LMD_CKPT_DIR=/path/to/checkpoints/camera_lidar \
python eval_lmd_cam_lid.py
```
Note: These scripts currently write `.pkl` to the **current working directory**.

3-modality:
```
LMD_DATA_ROOT=/path/to/data \
LMD_CKPT_DIR=/path/to/checkpoints/4x5_3e-4s_cam_rad_lid_18:06:48 \
LMD_OUTPUT_DIR=/path/to/output_eval_pkl_variants \
python eval_lmd_cam_rad_lid.py
```

---

## 📈 Analysis Notebook (IPYNB)

Use:
```
eval_analysis_2modal_identity_ratio.ipynb
```

- Update `FILE_NAMES` to point to your saved `.pkl` files.
- Requires `pandas` and `matplotlib` (and `scipy` for KDE plots).

---

## ✅ Notes on Portability

All scripts resolve paths relative to this repo by default. You can relocate the repo and simply update:

- `run_lmd_2modal.sh` / `run_lmd_3modal.sh` (DATA_ROOT, CKPT, OUTPUT_DIR)
- or environment variables (`LMD_DATA_ROOT`, `LMD_CKPT_DIR`, `LMD_OUTPUT_DIR`)
