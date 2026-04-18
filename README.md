# Self-Supervised 3D Voxel Masked Autoencoder Representations for CO₂ Working Capacity Prediction in Metal–Organic Frameworks

**IIIT-MOF** is an end-to-end deep learning pipeline for representation learning and property prediction in Metal–Organic Frameworks (MOFs). The framework converts raw crystallographic data (CIF) into multi-channel 3D voxel grids, trains 3D Masked Autoencoders for self-supervised and supervised learning, and provides post-hoc structural and statistical analysis tools for interpreting structure–property relationships.

This repository is designed for researchers working at the intersection of materials science and deep learning, particularly in adsorption, gas storage, and structure–property modeling of porous materials.

---

## Repository Structure

This repository is organized into three main functional components: voxelization, training, and analysis.

```text
.
├── analysis
│   ├── analysis.py        # Prediction analysis and structural comparison workflows
│   └── stat_analysis.py   # Statistical metrics and hypothesis testing
├── attribution
│   └── attribute_analysis.py # Post-hoc feature attribution for model explainability
├── graph_benchmarking
│   └── exp1/              # GNN benchmarking results and checkpoints
├── pretraining_contri
│   ├── cgcnn/             # CGCNN ablation experiments (01-20%)
│   ├── cnn/               # 3D CNN baseline experiments
│   ├── mae/               # MAE pre-training experiments
│   └── mae_ft_only/       # Supervised MAE baseline (no pre-training)
├── voxel_ablation
│   ├── 32/                # 32³ resolution model results
│   ├── 64/                # 64³ resolution model results
│   ├── 96/                # 96³ resolution model results
│   └── an2.py             # Resolution fidelity and signal analysis script
├── voxelization
│   └── voxel.py           # CIF → multi-channel 3D voxelization pipeline
├── training
│   └── train.py           # Training and fine-tuning entry point (MAE + regression)
├── splits                 # Experimental data splits (01pct, 05pct, 10pct, 20pct)
├── mae_best.pt            # Pre-trained / fine-tuned model checkpoint
└── README.md
```

---

## Features

* **CIF → 3D Voxel Pipeline**

  * Fractional and Cartesian lattice-aware mapping
  * Gaussian splatting and trilinear interpolation
  * Element-specific channels and optional partial charge support

* **3D Masked Autoencoder**

  * Vision Transformer (ViT)-based architecture for volumetric data
  * Self-supervised pre-training and supervised fine-tuning
  * Support for checkpoint loading and resuming

* **Post-hoc Analysis & Interpretability**

  * Regression metrics (RMSE, MAE, R²)
  * Best vs. worst MOF structural comparison
  * Statistical testing (Cohen’s d, Mann–Whitney U, bootstrapped confidence intervals)

---

## Installation

### Requirements

* Python **3.8+**
* PyTorch **1.12+** (CUDA optional but recommended for training)
* Linux/macOS recommended for large-scale experiments

### Setup

Clone the repository:

```bash
git clone https://github.com/your-username/IIIT-MOF.git
cd IIIT-MOF
```

Install dependencies:

```bash
# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Scientific stack
pip install numpy pandas scipy matplotlib scikit-learn pymatgen tqdm
```

---

## Usage Guide

The workflow is divided into three stages:

1. **Voxelization** – Convert CIF files into multi-channel 3D voxel grids
2. **Training** – Pre-train or fine-tune a 3D Masked Autoencoder
3. **Analysis** – Evaluate predictions and interpret structural features

---

## 1. Voxelization

Convert `.cif` files into multi-channel 3D voxel grids stored as compressed `.npz` tensors.

### Basic Usage

```bash
python voxelization/voxel.py \
    --cif-dir ./data/cifs \
    --out-dir ./data/voxels \
    --grid 64 \
    --elem-channels "C,O,N,H"
```

### Advanced Usage

Enable partial charges and per-atom Gaussian density splatting:

```bash
python voxelization/voxel.py \
    --cif-dir ./data/cifs \
    --out-dir ./data/voxels \
    --include-charge \
    --per-atom-gauss \
    --overwrite
```

### Output

Each MOF is saved as a compressed `.npz` file containing:

* Total atomic density channel
* Metal–organic distinction channel
* Element-specific channels
* (Optional) Partial charge channel

---

## 2. Training

Train a 3D Masked Autoencoder for representation learning or downstream property prediction.

### Self-Supervised Pre-Training

```bash
python training/train.py \
    --vox-dir ./data/voxels \
    --out-dir ./checkpoints \
    --patch 8 \
    --mask-ratio 0.75 \
    --batch-size 8 \
    --epochs 200
```

### Supervised Fine-Tuning (Property Prediction)

```bash
python training/train.py \
    --vox-dir ./data/voxels \
    --out-dir ./checkpoints_ft \
    --patch 8 \
    --finetune \
    --targets-csv ./data/properties.csv \
    --target-col "wc_mmolg" \
    --resume ./mae_best.pt
```

### Notes

* Supports **checkpoint loading and resuming**
* Compatible with **Automatic Mixed Precision (AMP)** if enabled in the training script
* Saves **test predictions to CSV** for downstream analysis

---

## 3. Analysis

Evaluate model performance and interpret structural differences between high- and low-performing MOFs.

### Analyze Predictions

Generates regression metrics, residual plots, and bin-wise statistics:

```bash
python analysis/analysis.py \
    --csv ./checkpoints_ft/test_predictions.csv \
    --out-dir ./results/prediction_analysis
```

### Structural Analysis (Best vs Worst MOFs)

Extracts and compares voxel-level descriptors for top and bottom performers:

```bash
python analysis/stat_analysis.py \
    --csv ./checkpoints_ft/test_predictions.csv \
    --cif-root ./data/cifs \
    --voxel-script ./voxelization/voxel.py \
    --out-dir ./results/structural_analysis \
    --top-k 50
```

---

## Module Overview

### `voxelization/`

Handles translation from crystallographic structures to tensor representations:

* Fractional and Cartesian coordinate mapping
* Periodic boundary handling
* Gaussian splatting and trilinear interpolation
* Element and charge-aware channel construction

### `training/`

PyTorch implementation of a 3D Masked Autoencoder:

* 3D patch embedding and volumetric attention
* Self-supervised reconstruction objective
* Supervised regression head for property prediction

### `analysis/`

Interpretability and evaluation toolkit:

* **Statistics:** RMSE, MAE, R², Cohen’s d, Mann–Whitney U, bootstrapped confidence intervals
* **Visualization:** Regression plots, residual histograms, feature importance bars
* **Structure:** Voxel-derived geometric and chemical descriptors

### `attribution/`

* **Explainability:** Post-hoc feature attribution using Integrated Gradients or similar techniques to identify which voxel regions contribute most to property predictions.

### `graph_benchmarking/`

* **Benchmarks:** Directory containing experimental results (`exp1`) for comparing the 3D Voxel MAE performance against state-of-the-art graph-based models.

### `pretraining_contri/`

* **Ablation Studies:** Multi-model assessment (MAE, CNN, CGCNN) across different data regimes to quantify the impact of self-supervised pre-training on downstream regression performance.

### `voxel_ablation/`

* **Resolution Study:** Systematic evaluation of model performance and signal fidelity across different voxel grid sizes (32³, 64³, 96³) using the `an2.py` analysis script.

### `splits/`

* **Low-Data Regime:** Standardized training data splits (01pct, 05pct, 10pct, and 20pct) used to evaluate model robustness and learning efficiency in data-scarce scenarios.

---

## Citation

If you use this code in your research, please cite this work. 
