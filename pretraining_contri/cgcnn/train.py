#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import MOFCIFDataset, collate_pool
from model import CGCNNRegressor


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def ensure_cif_ext(name: str) -> str:
    if not isinstance(name, str):
        name = str(name)
    return name if name.lower().endswith(".cif") else name + ".cif"


def prepare_sorted_csv(csv_path: str, cif_dir: str, out_csv: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)

    if "filename" not in df.columns or "wc_mmolg" not in df.columns:
        raise RuntimeError("CSV must contain columns: filename and wc_mmolg")

    df = df.copy()
    df["filename"] = df["filename"].astype(str).map(ensure_cif_ext)

    cif_names = {p.name for p in Path(cif_dir).glob("*.cif")}
    if len(cif_names) == 0:
        raise RuntimeError(f"No CIF files found in {cif_dir}")

    before = len(df)
    df = df[df["filename"].isin(cif_names)].copy()
    after = len(df)
    if after == 0:
        raise RuntimeError("No CSV rows matched CIF files in cif_dir")

    df = df.sort_values("filename").reset_index(drop=True)
    df.to_csv(out_csv, index=False)

    if after != before:
        print(f"[Data] Filtered CSV rows: {before} -> {after} to match CIF folder")
    print(f"[Data] Saved sorted CSV to {out_csv}")

    return df


def make_exact_split(n: int, seed: int):
    rng = np.random.RandomState(seed)
    idxs = np.arange(n)
    rng.shuffle(idxs)

    n_train = int(math.floor(0.8 * n))
    n_val = int(math.floor(0.1 * n))
    n_test = n - n_train - n_val

    train_idx = idxs[:n_train]
    val_idx = idxs[n_train : n_train + n_val]
    test_idx = idxs[n_train + n_val :]

    assert len(train_idx) + len(val_idx) + len(test_idx) == n
    assert len(train_idx) == n_train
    assert len(val_idx) == n_val
    assert len(test_idx) == n_test

    return train_idx, val_idx, test_idx


@torch.no_grad()
def evaluate(model, loader, device, y_scaler):
    model.eval()
    preds = []
    trues = []

    for atom_z, nbr_fea, nbr_idx, crystal_atom_idx, y in loader:
        atom_z = atom_z.to(device)
        nbr_fea = nbr_fea.to(device)
        nbr_idx = nbr_idx.to(device)
        crystal_atom_idx = crystal_atom_idx.to(device)
        y = y.to(device)

        out = model(atom_z, nbr_fea, nbr_idx, crystal_atom_idx)
        preds.append(out.cpu().numpy())
        trues.append(y.cpu().numpy())

    preds = np.vstack(preds)
    trues = np.vstack(trues)

    preds_orig = y_scaler.inverse_transform(preds)
    trues_orig = y_scaler.inverse_transform(trues)

    mae = np.mean(np.abs(preds_orig - trues_orig))
    rmse = np.sqrt(np.mean((preds_orig - trues_orig) ** 2))
    return mae, rmse


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    os.makedirs(args.outdir, exist_ok=True)

    sorted_csv = os.path.join(args.outdir, "sorted_input.csv")
    df = prepare_sorted_csv(args.csv, args.cif_dir, sorted_csv)

    full_dataset = MOFCIFDataset(
        csv_file=sorted_csv,
        cif_dir=args.cif_dir,
        filename_col="filename",
        target_col="wc_mmolg",
        cutoff=args.cutoff,
        max_num_nbr=args.max_num_nbr,
        num_gaussians=args.num_gaussians,
    )

    train_idx, val_idx, test_idx = make_exact_split(len(df), args.seed)

    np.save(os.path.join(args.outdir, "train_idx.npy"), train_idx)
    np.save(os.path.join(args.outdir, "val_idx.npy"), val_idx)
    np.save(os.path.join(args.outdir, "test_idx.npy"), test_idx)

    with open(os.path.join(args.outdir, "split_files.json"), "w") as f:
        json.dump(
            {
                "train_files": df.iloc[train_idx]["filename"].tolist(),
                "val_files": df.iloc[val_idx]["filename"].tolist(),
                "test_files": df.iloc[test_idx]["filename"].tolist(),
            },
            f,
            indent=2,
        )

    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(full_dataset, val_idx)
    test_subset = Subset(full_dataset, test_idx)

    y_train = np.array([[float(df.iloc[i]["wc_mmolg"])] for i in train_idx], dtype=np.float32)
    y_scaler = StandardScaler()
    y_scaler.fit(y_train)
    joblib.dump(y_scaler, os.path.join(args.outdir, "y_scaler.pkl"))

    class ScaledSubset(torch.utils.data.Dataset):
        def __init__(self, subset, scaler):
            self.subset = subset
            self.scaler = scaler

        def __len__(self):
            return len(self.subset)

        def __getitem__(self, idx):
            atom_z, nbr_fea, nbr_idx, y = self.subset[idx]
            y_scaled = self.scaler.transform(y.numpy().reshape(1, -1)).astype(np.float32)
            y_scaled = torch.tensor(y_scaled.squeeze(0), dtype=torch.float32)
            return atom_z, nbr_fea, nbr_idx, y_scaled

    train_ds = ScaledSubset(train_subset, y_scaler)
    val_ds = ScaledSubset(val_subset, y_scaler)
    test_ds = ScaledSubset(test_subset, y_scaler)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_pool,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_pool,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_pool,
        pin_memory=True,
        drop_last=False,
    )

    model = CGCNNRegressor(
        atom_fea_len=args.atom_fea_len,
        nbr_fea_len=args.num_gaussians,
        n_conv=args.n_conv,
        h_fea_len=args.h_fea_len,
        dropout=args.dropout,
        max_z=args.max_z,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_mae = float("inf")
    patience_counter = 0

    log_path = os.path.join(args.outdir, "train_log.csv")
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,val_mae,val_rmse\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for atom_z, nbr_fea, nbr_idx, crystal_atom_idx, y in tqdm(
            train_loader, desc=f"Epoch {epoch}", leave=False
        ):
            atom_z = atom_z.to(device)
            nbr_fea = nbr_fea.to(device)
            nbr_idx = nbr_idx.to(device)
            crystal_atom_idx = crystal_atom_idx.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(atom_z, nbr_fea, nbr_idx, crystal_atom_idx)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y.size(0)

        train_loss = running_loss / max(1, len(train_loader.dataset))
        val_mae, val_rmse = evaluate(model, val_loader, device, y_scaler)

        with open(log_path, "a") as f:
            f.write(f"{epoch},{train_loss:.6f},{val_mae:.6f},{val_rmse:.6f}\n")

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
            f"val_MAE={val_mae:.4f} | val_RMSE={val_rmse:.4f}"
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                },
                os.path.join(args.outdir, "best_model.pt"),
            )
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print("Early stopping triggered.")
            break

    best_ckpt = torch.load(os.path.join(args.outdir, "best_model.pt"), map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_mae, test_rmse = evaluate(model, test_loader, device, y_scaler)

    print(f"\nTest MAE:  {test_mae:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--cif-dir", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="runs/exp1")

    parser.add_argument("--cutoff", type=float, default=8.0)
    parser.add_argument("--max-num-nbr", type=int, default=12)
    parser.add_argument("--num-gaussians", type=int, default=50)

    parser.add_argument("--atom-fea-len", type=int, default=64)
    parser.add_argument("--n-conv", type=int, default=3)
    parser.add_argument("--h-fea-len", type=int, default=128)
    parser.add_argument("--max-z", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()
    train(args)
