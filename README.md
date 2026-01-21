# Self-Supervised 3D Voxel Masked Autoencoder Representations for CO₂ Working Capacity Prediction in Metal–Organic Frameworks

**IIIT-MOF** is an end-to-end deep learning pipeline for representation learning and property prediction in Metal–Organic Frameworks (MOFs). The framework converts raw crystallographic data (CIF) into multi-channel 3D voxel grids, trains 3D Masked Autoencoders for self-supervised and supervised learning, and provides a comprehensive suite of post-hoc structural and statistical analysis tools.

This repository is designed for researchers working at the intersection of materials science and deep learning, particularly in adsorption, gas storage, and structure–property modeling of porous materials.

---

## Features

* **CIF → 3D Voxel Pipeline**

  * Fractional and Cartesian lattice-aware mapping
  * Gaussian splatting and trilinear interpolation
  * Element-specific channels and partial charge support

* **3D Masked Autoencoder**

  * Vision Transformer (ViT)-based architecture for volumetric data
  * Self-supervised pre-training and supervised fine-tuning
  * Distributed Data Parallel (DDP) and Automatic Mixed Precision (AMP)

* **Post-hoc Analysis & Interpretability**

  * Regression metrics and residual analysis
  * Structural feature extraction (40+ voxel-derived descriptors)
  * Statistical testing (Cohen’s d, Mann–Whitney U, bootstrapped CIs)

---

## Project Structure

```text
.
├── analysis
│   ├── main.py              # CLI for analysis subcommands
│   └── mof_analysis
│       ├── __init__.py
│       ├── plotting.py    # Visualization and plotting utilities
│       ├── stats.py       # Statistical tests and metrics
│       ├── utils.py      # Shared helpers
│       ├── voxels.py    # Voxel feature extraction
│       └── workflows.py # End-to-end analysis workflows
├── mae_best.pt           # Pre-trained model checkpoint
├── README.md
├── training
│   ├── data.py          # Dataset and dataloader logic
│   ├── engine.py       # Training/validation loops
│   ├── model.py        # Masked Autoencoder / ViT architecture
│   ├── train.py        # Training and fine-tuning entry point
│   └── utils.py       # Training utilities
└── voxelization
    ├── main.py         # Voxelization CLI
    └── voxelizer
        ├── chemistry.py  # Element and charge handling
        ├── constants.py  # Physical and numerical constants
        ├── core.py       # Core voxelization logic
        ├── grid.py       # Lattice/grid construction
        ├── __init__.py
        ├── io_utils.py   # CIF and file I/O
        ├── pipeline.py  # End-to-end voxelization pipeline
        └── utils.py     # Shared helpers
```

---

## Installation

### Requirements

* Python **3.8+**
* PyTorch **1.12+** (CUDA optional but recommended)
* Linux/macOS recommended for large-scale training

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

Convert `.cif` files into multi-channel 3D voxel grids stored as `.npz` tensors.

### Basic Usage

```bash
python voxelization/main.py \
    --cif-dir ./data/cifs \
    --out-dir ./data/voxels \
    --grid 64 \
    --elem-channels "C,O,N,H"
```

### Advanced Usage

Enable partial charges and per-atom Gaussian density splatting:

```bash
python voxelization/main.py \
    --cif-dir ./data/cifs \
    --out-dir ./data/voxels \
    --include-charge \
    --per-atom-gauss \
    --overwrite
```

### Output

Each MOF is saved as a compressed `.npz` file containing:

* Atomic density channel
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

* Supports **multi-GPU training** via PyTorch DDP
* Uses **Automatic Mixed Precision (AMP)** for faster training
* Saves **per-epoch checkpoints** and test predictions to CSV

---

## 3. Analysis

Evaluate model performance and interpret structural differences between high- and low-performing MOFs.

### Analyze Predictions

Generates regression metrics, residual plots, and bin-wise statistics:

```bash
python analysis/main.py analyze-preds \
    --csv ./checkpoints_ft/test_predictions.csv \
    --out-dir ./results/prediction_analysis
```

### Structural Analysis (Best vs Worst MOFs)

Extracts and compares voxel-level descriptors for top and bottom performers:

```bash
python analysis/main.py analyze-struct \
    --csv ./checkpoints_ft/test_predictions.csv \
    --cif-root ./data/cifs \
    --voxel-script ./voxelization/main.py \
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

PyTorch implementation of a 3D Vision Transformer with Masked Autoencoder logic:

* 3D patch embedding and volumetric attention
* Self-supervised reconstruction objective
* Supervised regression head for property prediction
* DDP and AMP support

### `analysis/`

Interpretability and evaluation toolkit:

* **Statistics:** RMSE, MAE, R², Cohen’s d, Mann–Whitney U, bootstrapped CIs
* **Visualization:** Regression plots, residual histograms, feature importance bars
* **Structure:** 40+ voxel-derived geometric and chemical descriptors

---

## Citation

If you use this code in your research, please cite this work.