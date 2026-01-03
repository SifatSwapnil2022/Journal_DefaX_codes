 # 📌 DeFaX: A Cross-Attention Fusion Framework for Robust and Explainable Deepfake Detection

This repository contains the official implementation of DeFaX, a hybrid deep learning framework proposed in our IEEE Access paper:

DeFaX: A Cross-Attention Fusion Framework for Robust and Explainable Deepfake Detection
IEEE Access, 2025
DOI: 10.1109/ACCESS.2025.3645769
📄 Paper link: https://ieeexplore.ieee.org/document/11303744

--- 
# 🧠 Overview

The rapid advancement of GAN-based face generation has made deepfake detection increasingly challenging. Existing approaches often struggle to jointly capture:

Global semantic inconsistencies (best handled by Transformers)

Fine-grained local artifacts (best handled by CNNs)

DeFaX addresses this limitation through a cross-attention fusion mechanism that unifies both perspectives while maintaining model interpretability.

---

# 🚀 Key Contributions

🔹 SwinEffAttn Module: A novel cross-attention fusion of

Swin Transformer (global hierarchical reasoning)

EfficientNet (local texture-sensitive feature extraction)

🔹 Explainable AI (XAI) integration using:

Grad-CAM

LIME

🔹 State-of-the-art performance

99.8% Accuracy

AUC = 1.000

Evaluated on a 140K real/fake face dataset

🔹 Deployment-ready

Lightweight classification head

Flask-based web interface for real-time inference

---

# 🏗️ Architecture

DeFaX Workflow

Input Image → Data Preprocessing → Feature Extraction → Cross-Attention Fusion → Classification → XAI Visualization

Core Components
1. Data Preprocessing Pipeline
Normalization: RGB images standardized to 224×224 pixels

Augmentation: RandomResizedCrop, HorizontalFlip, Rotation, ColorJitter

Validation: Consistent pixel intensity distributions across splits

2. Dual-Branch Feature Extraction
Swin Transformer Branch: Captures global semantic dependencies via hierarchical self-attention

EfficientNet Branch: Extracts fine-grained local texture features through convolutional operations

3. Cross-Attention Fusion (SwinEffAttn)
Algorithm 1: DeFaX – Cross Attention Fusion

Input: Image batch x ∈ ℝ^{B×3×224×224}

S ← SwinBackbone(x)            # Global tokens [B, N, C_s]
E ← EfficientNetBranch(x)      # Local tokens [B, M, C_e]

Q ← S W_Q
K ← E W_K
V ← E W_V                      # Linear projections

A ← MultiHeadAttention(Q, K, V)  # Cross-attention
F ← A W_O                       # Fused tokens

z ← MeanPool(F)                 # Global feature
y ← Classifier(z)               # MLP head → [B, 2]

return y


4. Classification Head
Normalized MLP with dropout and ReLU activation

Binary classification output (real vs. fake)

5. Explainability Modules
Grad-CAM: Highlights influential regions in final convolutional layers

LIME: Approximates local decision boundaries via perturbed samples


---

📊 Experimental Results
Dataset
140K Real and Fake Faces: 70K real (Flickr), 70K fake (StyleGAN)

Split: 70% training, 20% validation, 10% testing

Additional Datasets: StyleGAN-StyleGAN2 Combined, Fake-vs-Real-Faces (Hard)

Performance Comparison
| Model              | Accuracy (%) | ROC AUC | Log Loss | Brier Score | Cohen’s Kappa |
|--------------------|--------------|---------|----------|-------------|---------------|
| **DeFaX (Ours)**   | **99.80**    | **1.0000** | **0.0063** | **0.0016** | **0.9960** |
| FaceViT            | 99.80        | 0.9979  | 0.0739   | 0.0021      | 0.9959        |
| EfficientNetV2     | 99.73        | 0.9998  | 0.0094   | 0.0027      | 0.9947        |
| GSF-TFN            | 99.13        | 0.9997  | 0.3136   | 0.0087      | 0.9826        |
| GSU-TFR            | 99.09        | 0.9997  | 0.3298   | 0.0092      | 0.9817        |
| FakeViT            | 98.86        | 0.9996  | 0.4109   | 0.0114      | 0.9772        |
| VGG16              | 80.18        | 0.9980  | 0.4528   | 0.1982      | 0.6036        |


Benchmark Against State-of-the-Art
| Reference | Method            | Dataset                     | Accuracy (%) |
|----------|-------------------|-----------------------------|--------------|
| [6]      | D-CNN             | GDWCT, CelebA, FFHQ         | 99.33        |
| [11]     | VFDNET            | 140K Real/Fake Faces        | 99.13        |
| [20]     | Ensemble Learning | CelebA                     | 97.04        |
| [21]     | ResNet18 + KNN    | Real/Fake Faces             | 89.50        |
| [7]      | VGG16 + DFT       | 140K Real/Fake Faces        | 99.00        |
| **Our Work** | **DeFaX**     | **140K Real/Fake Faces**    | **99.80**    |


Cross-Dataset Generalization
| Dataset                         | Real Accuracy (%) | Fake Accuracy (%) | Overall Accuracy (%) |
|---------------------------------|-------------------|-------------------|----------------------|
| Dataset A (StyleGAN Combined)   | 100.0             | 100.0             | 100.0                |
| Dataset C (140K Faces)          | 100.0             | 100.0             | 100.0                |
| Dataset B (Hard Set)            | 100.0             | 65.0              | 82.5                 |


---

# 🔍 Explainability Results
Grad-CAM Visualizations
Focus Areas: Eyes, nose, mouth, and facial boundaries

Real Images: Uniform attention across natural facial features

Fake Images: Concentrated attention on artifact-prone regions

LIME Analysis
Correct Predictions: Localized clusters around manipulation artifacts

Misclassifications: Diffuse attention patterns lacking discriminative focus

Domain Adaptation: Reveals sensitivity to generator-specific artifacts

--- 
# 🛠️ Technical Implementation
Training Configuration
CNN Models (TensorFlow/Keras)

Input Size: 224×224×3
Batch Size: 32
Epochs: 10
Optimizer: Adam/Adamax/AdamW (LR=1e-5)
Loss: Categorical Crossentropy
Hardware: NVIDIA Tesla P100 (16GB VRAM)

Attention Models (PyTorch)

Input Size: 224×224
Batch Size: 32
Epochs: 10 (early stopping patience=2-4)
Optimizer: AdamW (LR=1e-4, WD=1e-5 to 1e-4)
Scheduler: CosineAnnealingLR (η_min=1e-6)
Framework: PyTorch with torchvision/timm

Evaluation Metrics
Primary: Accuracy, Precision, Recall, F1-Score

Advanced: ROC AUC, Log Loss, Brier Score, Cohen's Kappa

Calibration: Reliability diagrams via sklearn

---

# 🌐 Web Application
Flask-Based Deployment
Real-time Inference: Batch processing of multiple images

Supported Formats: JPG, JPEG, PNG, BMP, TIF, TIFF, WEBP

Maximum Upload: 20MB per request

Output: Classification label with probability distribution

Interface Features
Drag-and-drop image upload

Batch prediction results display

Visual confidence indicators

Export functionality for results

# 📈 Key Findings
Superior Fusion Effectiveness: Cross-attention mechanism outperforms standalone CNN or Transformer models

Artifact Sensitivity: DeFaX excels at detecting both high-level semantic inconsistencies and low-level texture anomalies

Calibration Excellence: Near-perfect probability calibration (Brier Score: 0.0016)

Practical Robustness: Maintains >80% accuracy on challenging cross-dataset evaluations

Interpretability Value: XAI visualizations align with known deepfake artifact patterns

---

# 📚 Citation

@article{defax2025,
  title={DeFaX: A Cross-Attention Fusion Framework for Robust and Explainable Deepfake Detection},
  author={Al-Imran, MD and Sheikh, MD Sifatullah and Kirtonia, Urmi and Arthi, Nuzath Tabassum and Ripon, Shamim},
  journal={IEEE Access},
  year={2025},
  volume={13},
  pages={213964--213979},
  doi={10.1109/ACCESS.2025.3645769}
}

---

# 🔗 Dataset References
140K Real and Fake Faces: Kaggle Dataset

StyleGAN-StyleGAN2 Combined: Kaggle Dataset

Fake-vs-Real-Faces (Hard): Kaggle Dataset

---

# 👥 Authors
MD Al-Imran – East West University (Senior Lecturer, Chancellor's Gold Medal Award)

MD Sifatullah Sheikh – East West University (Undergraduate Researcher)

Urmi Kirtonia – East West University (Undergraduate Researcher)

Nuzath Tabassum Arthi – East West University (Undergraduate Researcher)

Shamim Ripon – East West University (Professor, PhD University of Southampton)

Corresponding Author: Shamim Ripon (dshr@ewubd.edu)

---

# 📄 License
© 2025 The Authors. This work is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 License.

---
# 🙏 Acknowledgments
The authors thank Paperpal for assisting in grammar refinement and language clarity. This assistance was limited to linguistic improvements and did not influence the scientific content, analysis, or conclusions.
---
# ⭐ Star this repository if you find this research valuable!




