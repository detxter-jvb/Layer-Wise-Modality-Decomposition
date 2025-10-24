<h1 align="center">Layer-Wise Modality Decomposition for Interpretable Multimodal Sensor Fusion</h1>

> **Official implementation** of **LMD** (NeurIPS 2025, Poster)

<p align="center">
  <img src="https://img.shields.io/badge/venue-NeurIPS%202025-7a77ff" />
  <img src="https://img.shields.io/badge/license-MIT-green" />
</p>

LMD is a **post‑hoc interpretability** method that **decomposes a pretrained fusion model’s prediction into modality‑wise components at every layer**—without changing the model architecture. In autonomous driving (e.g., camera–radar or camera–LiDAR), LMD reveals how each sensor contributes to the final decision, and comes with **structured perturbation metrics** to validate separation.

---

## 📣 News

* **2025‑9**: Paper accepted to **NeurIPS 2025** 🎉
* **2025‑12**: Codebase will be cleaned up and released here.

---

## 🚀 Key Features

* **Model‑agnostic**: Works with existing fusion models (e.g., SimpleBEV variants) without retraining.
* **Layer‑wise decomposition**: Obtain `camera`, `radar`/`lidar`, and `bias` predictions at **every layer**.
* **Faithful linearization**: Cache **activation ratios** and **normalization statistics**; uses **identity** & **ratio** rules for BatchNorm and Layer/InstanceNorm.
* **Validation metrics**: Pearson/MSE‑based **modality replacement** test to check separation.
* **Fast & scalable**: Only **two forward passes**, no backprop; near‑inference latency.

---
