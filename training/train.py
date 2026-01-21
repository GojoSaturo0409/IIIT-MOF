#!/usr/bin/env python3
import argparse
import os
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
import math

from utils import (setup_distributed, cleanup_distributed, set_seed, 
                   load_targets_csv, save_checkpoint, is_main_process, _clean_state_dict)
from data import create_dataloaders
from model import MAE3D
from engine import train_one_epoch, evaluate

def main(args):
    is_distributed, device = setup_distributed(args)
    set_seed(args.seed)
    is_main = is_main_process()

    # Targets & Data
    targets = None
    if args.targets_csv:
        targets = load_targets_csv(args.targets_csv, args.filename_col, args.target_col)
        
    # Calculate target stats for normalization if needed
    if args.finetune and args.normalize_target and targets:
        vals = [v for v in targets.values() if not math.isnan(v)]
        args._target_mean = float(np.mean(vals))
        args._target_std = float(np.std(vals))
        if is_main: print(f"Target Norm: mean={args._target_mean:.3f}, std={args._target_std:.3f}")

    train_loader, val_loader, test_loader, train_sampler = create_dataloaders(args, targets, is_main)

    # Model Setup
    # Calculate patch_dim based on first batch
    sample_vox, _, _ = train_loader.dataset[0]
    C, G, _, _ = sample_vox.shape
    patch_dim = C * (args.patch ** 3)
    num_patches = (G // args.patch) ** 3
    
    model = MAE3D(
        patch_dim=patch_dim, enc_embed=args.enc_embed, enc_depth=args.enc_depth,
        enc_heads=args.enc_heads, dec_embed=args.dec_embed, dec_depth=args.dec_depth,
        dec_heads=args.dec_heads, mask_ratio=args.mask_ratio
    ).to(device)
    
    reg_head = None
    if args.finetune:
        reg_head = nn.Sequential(
            nn.Linear(args.enc_embed, args.ft_hidden), nn.ReLU(),
            nn.Linear(args.ft_hidden, 1)
        ).to(device)

    # Checkpoint Loading (Logic Simplified)
    start_epoch = 1
    best_loss = float('inf')
    patience_cnt = 0
    bad_batches = []
    
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        
        # Load Model
        state = ckpt.get('model_state', ckpt)
        try: model.load_state_dict(state)
        except: model.load_state_dict(_clean_state_dict(state))
        
        # Load Head if exists
        if reg_head and 'reg_head_state' in ckpt:
            try: reg_head.load_state_dict(ckpt['reg_head_state'])
            except: reg_head.load_state_dict(_clean_state_dict(ckpt['reg_head_state']))

        if not (args.finetune and args.reset_patience):
            start_epoch = ckpt.get('epoch', 0) + 1
            best_loss = ckpt.get('best_loss', float('inf'))
            patience_cnt = ckpt.get('patience_cnt', 0)

    # DDP Wrapping
    if is_distributed:
        model = DDP(model, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=False)
        if reg_head:
            reg_head = DDP(reg_head, device_ids=[args.gpu], output_device=args.gpu)

    # Optimizer
    params = list(model.parameters()) + (list(reg_head.parameters()) if reg_head else [])
    opt = torch.optim.AdamW([p for p in params if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # Training Loop
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            if is_distributed: train_sampler.set_epoch(epoch)
            
            # Freeze/Unfreeze Logic
            if args.finetune and args.freeze_encoder_epochs > 0:
                m = model.module if is_distributed else model
                requires_grad = (epoch > args.freeze_encoder_epochs)
                for p in m.encoder.parameters(): p.requires_grad = requires_grad
            
            train_loss = train_one_epoch(model, train_loader, opt, scaler, device, epoch, args, reg_head)
            if is_main: print(f"Epoch {epoch} Train Loss: {train_loss:.4f}")
            
            if val_loader:
                val_loss = evaluate(model, val_loader, device, args, reg_head)
                if is_main: print(f"Epoch {epoch} Val Loss: {val_loss:.4f}")
                
                metric = val_loss
                is_best = metric < best_loss
                if is_best: 
                    best_loss = metric
                    patience_cnt = 0
                else:
                    patience_cnt += 1
                
                save_checkpoint(args, epoch, model, opt, sched, scaler, best_loss, 
                                patience_cnt, bad_batches, reg_head, is_best)
                
                if patience_cnt >= args.patience:
                    if is_main: print("Early stopping triggered")
                    break
            
            sched.step()

        # Final Test
        if test_loader:
            best_path = os.path.join(args.out_dir, "mae_best.pt")
            if os.path.exists(best_path):
                ckpt = torch.load(best_path, map_location='cpu')
                m = model.module if is_distributed else model
                m.load_state_dict(_clean_state_dict(ckpt['model_state']))
                if reg_head:
                    r = reg_head.module if is_distributed else reg_head
                    r.load_state_dict(_clean_state_dict(ckpt.get('reg_head_state')))
            
            test_loss = evaluate(model, test_loader, device, args, reg_head, 
                                 output_csv=os.path.join(args.out_dir, "test_predictions.csv"))
            if is_main: print(f"Final Test Loss: {test_loss:.4f}")

    finally:
        cleanup_distributed()

if __name__ == "__main__":
    import numpy as np # Needed for main shim
    p = argparse.ArgumentParser()
    p.add_argument("--vox-dir", required=True)
    p.add_argument("--out-dir", default="mae_runs")
    p.add_argument("--patch", type=int, required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--mask-ratio", type=float, default=0.75)
    
    # Model dims
    p.add_argument("--enc-embed", type=int, default=512)
    p.add_argument("--enc-depth", type=int, default=8)
    p.add_argument("--enc-heads", type=int, default=8)
    p.add_argument("--dec-embed", type=int, default=256)
    p.add_argument("--dec-depth", type=int, default=4)
    p.add_argument("--dec-heads", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)

    # Finetune
    p.add_argument("--finetune", action="store_true")
    p.add_argument("--targets-csv", type=str, default=None)
    p.add_argument("--filename-col", type=str, default='filename')
    p.add_argument("--target-col", type=str, default='wc_mmolg')
    p.add_argument("--normalize-target", action='store_true')
    p.add_argument("--reg-loss", choices=['l1','mse'], default='l1')
    p.add_argument("--ft-hidden", type=int, default=256)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--freeze-encoder-epochs", type=int, default=5)

    # DDP
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--distributed", action="store_true")
    p.add_argument("--dist-timeout-seconds", type=int, default=3600)
    p.add_argument("--reset-patience", action="store_true")
    p.add_argument("--resume", type=str, default=None)

    args = p.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    main(args)
