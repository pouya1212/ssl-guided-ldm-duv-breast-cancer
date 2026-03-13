# Self-Learned Representation-Guided Latent Diffusion Model for Breast Cancer Classification in Deep Ultraviolet Whole Surface Images

Official implementation of our paper:

**"Self-Learned Representation-Guided Latent Diffusion Model for Breast Cancer Classification in Deep Ultraviolet Whole Surface Images"**

📄 Paper: [Link to Paper](https://www.researchgate.net/publication/399875627_Self-learned_representation-guided_latent_diffusion_model_for_breast_cancer_classification_in_deep_ultraviolet_whole_surface_images)

## Authors
Pouya Afshin, David Helminiak, Tianling Niu, Julie M. Jorns, Tina Yen, Bing Yu, Dong Hye Ye

Georgia State University, Marquette University, Medical College of Wisconsin

---

## Overview

Breast-Conserving Surgery (BCS) requires accurate intraoperative margin assessment to ensure complete tumor removal while preserving healthy tissue.

Deep Ultraviolet Fluorescence Scanning Microscopy (DUV-FSM) provides rapid, high-resolution surface imaging. However, training deep learning models is challenging due to the limited availability of annotated data.

We propose a **Self-Supervised Learning (SSL)-guided Latent Diffusion Model (LDM)** that generates realistic synthetic DUV patches using semantic representations extracted from DINO.

The generated synthetic data improves Vision Transformer (ViT) classification performance.

---

## Method Overview

![Pipeline](figures/system-model.png)

Pipeline:

1. Extract self-supervised features using DINO
2. Guide latent diffusion model with semantic embeddings
3. Generate synthetic DUV patches
4. Train Vision Transformer classifier using real + synthetic data
5. Evaluate the finetuned model on the patches of the DUV WSI Test sample
6. Aggregate patch predictions to classify the test DUV WSI 
---




## Installation & Requirements

Clone the repository:

git clone https://github.com/pouya12/ssl-guided-ldm-duv-breast-cancer.git
cd ssl-guided-ldm-duv-breast-cancer

Install required dependencies:

pip install -r requirements.txt

---
## Dataset
Due to patient privacy and medical data regulations, the DUV-FSM dataset used in this work cannot be publicly released.

Researchers interested in the dataset may contact the Medical College of Wisconsin for potential access.
---
## Acknowledgements

The Vision Transformer (ViT) implementation used in this repository is adapted from the following open-source project:

https://github.com/jeonsworld/ViT-pytorch

The original implementation was modified to support loading pretrained models trained on large-scale public datasets and integrated into our training pipeline for DUV-FSM breast cancer classification.
---
## Citation

If you find this work useful, please cite:

```bibtex
@misc{afshin2026selflearnedrepresentationguidedlatentdiffusion,
      title={Self-learned representation-guided latent diffusion model for breast cancer classification in deep ultraviolet whole surface images}, 
      author={Pouya Afshin and David Helminiak and Tianling Niu and Julie M. Jorns and Tina Yen and Bing Yu and Dong Hye Ye},
      year={2026},
      eprint={2601.10917},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2601.10917}, 
}
