# Self-Learned Representation-Guided Latent Diffusion Model for Breast Cancer Classification in Deep Ultraviolet Whole Surface Images

Official implementation of our paper on improving breast cancer classification in Deep Ultraviolet Fluorescence Scanning Microscopy (DUV-FSM) images using self-supervised representation-guided latent diffusion models.

## Authors
Pouya Afshin, David Helminiak, Tianling Niu, Julie M. Jorns, Tina Yen, Bing Yu, Dong Hye Ye

Georgia State University, Marquette University, Medical College of Wisconsin

---

## Overview

Breast-Conserving Surgery (BCS) requires accurate intraoperative margin assessment to ensure complete tumor removal while preserving healthy tissue.

Deep Ultraviolet Fluorescence Scanning Microscopy (DUV-FSM) provides rapid high-resolution surface imaging. However, training deep learning models is challenging due to limited annotated data.

We propose a **Self-Supervised Learning (SSL)-guided Latent Diffusion Model (LDM)** that generates realistic synthetic DUV patches using semantic representations extracted from DINO.

The generated synthetic data improves Vision Transformer (ViT) classification performance.

---

## Method Overview

![Pipeline](figures/pipeline.png)

Pipeline:

1. Extract self-supervised features using DINO
2. Guide latent diffusion model with semantic embeddings
3. Generate synthetic DUV patches
4. Train Vision Transformer classifier using real + synthetic data
5. Aggregate patch predictions for WSI-level classification

---

## Installation

Clone the repository:

```bash
git clone https://github.com/pouya12/ssl-guided-ldm-duv-breast-cancer.git
cd ssl-guided-ldm-duv-breast-cancer
