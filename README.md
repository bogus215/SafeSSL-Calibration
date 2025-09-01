# CaliMatch: Adaptive Calibration for Improving Safe Semi-supervised Learning

[![Paper](https://img.shields.io/badge/ICCV-2025-blue)](https://arxiv.org/abs/2508.00922)

Official PyTorch implementation of **CaliMatch**, published at **ICCV 2025**.  
CaliMatch addresses **overconfidence in Safe Semi-supervised Learning (Safe SSL)** by calibrating both the classifier and the out-of-distribution (OOD) detector.

---

## 🔍 Overview
Semi-supervised learning (SSL) boosts performance by leveraging unlabeled data, but real-world SSL often suffers from **label distribution mismatch** (unlabeled data may contain unseen classes).  
Previous Safe SSL methods mitigate this via OOD detection, but they suffer from **overconfident neural networks**, leading to:

- Incorrect high-confidence pseudo-labels  
- Misclassified unseen data as in-distribution  

**CaliMatch** introduces:
- **Adaptive Label Smoothing**: dynamically adjusts smoothing strength using validation accuracy distribution  
- **Learnable Temperature Scaling**: calibrates both classifier and OOD detector during training  

This yields **better pseudo-label quality**, **stronger OOD rejection**, and overall **robust Safe SSL performance**.

---

## ✨ Key Contributions
- First to jointly calibrate both the **classifier** and the **OOD detector** in Safe SSL.  
- Adaptive calibration removes the need for manual tuning of smoothing parameters.  
- Outperforms prior Safe SSL methods (FixMatch, OpenMatch, SafeStudent, IOMatch, SCOMatch, ADELLO) on benchmarks:  
  - **CIFAR-10, CIFAR-100, SVHN, Tiny-ImageNet, ImageNet**  
- Provides both **higher accuracy** and **lower calibration error (ECE)** compared to baselines.

---
