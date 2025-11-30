# MRI_Slice-SuperResolution (MISR)

This Project implements **Multi-Image Super-Resolution (MISR)** for medical MRI.  
Given two neighboring slices *(i−1 and i+1)*, the objective is to reconstruct the **missing slice i** to enhance through-plane MRI resolution.

Three deep learning models are implemented and compared:

- **Residual Interpolation CNN (Baseline)**
- **SRGAN (Adversarial Network)**
- **Fast-DDPM (10-step Diffusion Model)**

This project follows the methods and concepts taught in **Lecture 16: Multi-Image Super Resolution**.

---

# 📂 Project Folder Structure

MRI_SLICE-SUPERRESOLUTION/
│
├── .ipynb_checkpoints/ # Auto-generated notebook checkpoints
│
├── models/ # Saved model weights and checkpoints
│ ├── cnn_best.pth
│ ├── srgan_g_best.pth
│ ├── test_dataset_cache.pt
│ └── diffusion_best.pth
│
├── Results/ # All outputs, metrics & final reconstructions
│ ├── axial_comparison.png
│ ├── sagittal_comparison.png
│ ├── cnn_outputs/
│ ├── srgan_outputs/
│ └── diffusion_outputs/
│
├── Evaluation & Visualizations.ipynb # Full evaluation pipeline (PSNR, SSIM, plots)
├── Training_ImageSuperResolution.ipynb # Training pipeline for all models
├── models.py # CNN, SRGAN, and Fast-DDPM architectures
│
└── README.md # Project documentation


---

# Problem Statement

MRI scanners produce **anisotropic voxel spacing**, meaning:

- In-plane resolution ≈ **0.55 mm**
- Through-plane resolution ≈ **1.5 mm**

This causes **blurry sagittal and coronal views**.

### Goal: Multi-Image Super-Resolution (MISR)
Reconstruct the missing slice using its neighbors:
Input: Slice(i-1) and Slice(i+1)
Output: Predicted Slice(i)


This improves overall volume quality without rescanning the patient.

---

#  Rigor of Approach

To solve this problem, we implemented three progressively stronger models using a unified training pipeline, patch-based sampling, and consistent evaluation metrics.

## **1️⃣ Residual Interpolation CNN**
- Entry: Conv(2→64)
- **5 residual blocks**
- Output: Conv(64→1)
- Loss: L1  
- Batch Size: 4  
- LR: 1e−4  
- Strength: stable, smooth predictions  
- Weakness: slight blurring  

## **2️⃣ SRGAN**
- Generator: 8 residual blocks  
- Discriminator: PatchGAN  
- Loss: **L1 + Adversarial BCE**  
- Strength: sharp edges, best SSIM  
- Weakness: potential artifacts  

## **3️⃣ Fast-DDPM (Diffusion Model)**
- 10-step diffusion  
- Non-uniform β schedule (40% early noise / 60% late refinement)  
- UNet2D with timestep embeddings  
- Loss: noise prediction MSE  
- Strength: best anatomical continuity  
- Weakness: lower PSNR/SSIM  

---

#  Training Details

Shared hyperparameters:
- Patches: **128×128**
- Batch size: **4**
- Optimizer: **Adam (lr = 1e−4)**
- Normalization: `(x − mean) / std`
- Train / Validation / Test split: **patient-wise**

Training notebook:
Training_ImageSuperResolution.ipynb

# Evaluation

Evaluation notebook:
Evaluation & Visualizations.ipynb


### Metrics Used:
- **PSNR** — Pixel fidelity
- **SSIM** — Structural similarity
- Axial reconstruction comparisons  
- Sagittal reformat comparisons  

---

# 📈 Final Quantitative Results

| Model               | MSE       | MAE       | PSNR (dB) | SSIM  |
|---------------------|-----------|-----------|-----------|--------|
| ResidualInterpCNN   | 0.006338  | 0.055743  | 29.027    | 0.843 |
| SRGAN               | 0.006300  | 0.055723  | **29.045**| **0.850** |
| Fast-DDPM           | 0.009355  | 0.069518  | 27.019    | 0.801 |

### ✔ Interpretation
- **SRGAN** achieves the highest numerical performance.  
- ** FAST Diffusion model** produces smoothest anatomical continuity in sagittal view.  
- **CNN** provides good baseline performance.  

---

#  How to Run

### **1. Train Models**
Open and run:
Training_ImageSuperResolution.ipynb


### **2. Evaluate Models**
Then Open and run:
Evaluation & Visualizations.ipynb




### **3. Load Models**
```python
from models import ResidualInterpCNN, SRGAN_G, SRGAN_D, FastDDPM
```
# Key Findings
- MISR significantly improves through-plane MRI resolution
- SRGAN achieves best PSNR/SSIM
- Diffusion model gives best visual anatomical consistency
- CNN is lightweight and stable



