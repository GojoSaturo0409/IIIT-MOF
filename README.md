# Yet to do : clean up the codes

## MAE3D — Masked Autoencoder for 3D Voxelized MOFs

This repository contains tools to voxelize CIFs (Metal–Organic Frameworks) and pretrain / fine‑tune a 3D masked autoencoder (MAE) on multi‑channel voxel grids. The codebase focuses on reproducible training, DDP robustness, and convenient test-time CSV output.

---

## Repository structure

* `train.py` — Main training & fine‑tuning script (DDP‑aware). Produces per‑epoch checkpoints (`mae_epoch{EPOCH}.pt`) and a best model (`mae_best.pt`). Also saves test predictions to CSV (`test_predictions.csv`).

* `voxel.py` — Robust publication‑quality voxelizer: CIF → multi‑channel 3D voxel grids (`*_vox.npz`) + metadata (`*_meta.json`). Supports lattice‑aware fractional mapping, trilinear splatting, per‑atom gaussian splatting, optional charge channel, and multiple normalization modes.

* `mae_best.pt` — (example or user‑provided) trained checkpoint. Place in an `out_dir` or specify path via `--resume` when resuming / evaluating.

* `README.md` — (this file)

---

## Quick start

### 1 Install dependencies

A minimal working environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install torch torchvision torchaudio  # install the correct CUDA build for your machine
pip install numpy scipy pandas tqdm pymatgen
```

> `pymatgen` is required for CIF parsing in `voxel.py`. Use your system CUDA‑compatible `torch` build.

### 2 Voxelize CIFs

Example: create voxel files (default grid 64)

```bash
python voxel.py --cif-dir repeat_cifs --out-dir voxels --grid 64 --Lmin 35.0 --include-charge
```

Options of note:

* `--grid` grid size `G` (voxel shape = `(C,G,G,G)`)
* `--map-mode` `fractional` (lattice aware, default) or `cartesian`
* `--per-atom-gauss` higher fidelity, slower
* `--normalize` normalization mode (`none`, `per_channel_max`, `global_max`, `sum_normalize`)
* `--save-torch` also save `.pt` torch tensors

Output files follow the naming convention: `<cif_stem>_vox.npz` and `<cif_stem>_meta.json`.

### 3 Train / Fine‑tune

**Pretraining (MAE objective):**

```bash
python train.py --vox-dir voxels --out-dir mae_runs --patch 4 --batch 8 --epochs 200 --lr 3e-4
```

**Fine‑tuning (regression head):**

```bash
python train.py --vox-dir voxels --out-dir mae_runs_ft --patch 4 --batch 8 --epochs 100 \
  --finetune --targets-csv targets.csv --normalize-target --ft-hidden 256 --reg-loss l1
```

Important CLI flags (see `train.py` for full list):

* `--patch` patch size (must divide grid G)
* `--finetune` enables regression head and supervised training
* `--targets-csv` CSV with `filename` and target column (default `wc_mmolg`) for regression
* `--normalize-target` normalize targets using training set mean/std
* `--distributed` enable DDP (NCCL) — a `dist_timeout_seconds` argument is available
* `--resume` path to a checkpoint to resume from (or `mae_best.pt`)

Checkpoints saved to `--out-dir`:

* `mae_epoch{E}.pt` — per epoch
* `mae_best.pt` — best model (by validation loss)

**Test predictions:** After training (if a test split exists), predictions are written to
`<out-dir>/test_predictions.csv` with columns: `filename`, `prediction`, `target`, `is_valid`.

---

## File / format conventions

* Voxel file: `.npz` with `vox` (float32 ndarray of shape `(C,G,G,G)`) and `channels` stored alongside.
* Metadata: `*_meta.json` contains `channels`, `grid`, `box_size_ang`, `pymatgen` version, timestamp, etc.
* Targets CSV: must contain columns matching `--filename-col` (default `filename`) and `--target-col` (default `wc_mmolg`). Filenames will be normalized to end with `.cif` when matching.

---

## Tips & Troubleshooting

* Ensure `--patch` divides `G`. Example: `G=64`, `--patch 4` ⇒ `(64/4)^3` patches.
* If using DDP / NCCL, set appropriate environment vars and use `torch.distributed.launch`/`torchrun` with `LOCAL_RANK` set. The training script accepts `--distributed` and increases NCCL timeout by default.
* If all targets are `NaN` in a batch during fine‑tuning, the script will raise; prefilter your CSV or use `is_valid` handling.
* To resume finetuning but reset early stopping patience, use `--resume` together with `--reset-patience` and `--finetune`.

---

## Reproducibility & Seeds

Use `--seed` to set RNG seeds used across numpy, python `random`, and torch. The script sets `torch.backends.cudnn.benchmark = True` for performance; adjust for strict determinism if needed.

---

## License & Citation

This repository is provided under the MIT License. If you use these tools in publications, please cite the authors and mention the voxelization and MAE training pipeline in your methods.

---
