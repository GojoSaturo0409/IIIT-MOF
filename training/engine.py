import torch
import torch.nn as nn
import torch.distributed as dist
import pandas as pd
from tqdm import tqdm
from model import patchify_batch, random_masking, mae_loss_on_masked

def train_one_epoch(model, loader, opt, scaler, device, epoch, args, reg_head=None):
    model.train()
    if reg_head: reg_head.train()
    
    loss_fn = nn.L1Loss() if args.reg_loss == 'l1' else nn.MSELoss()
    running_loss = 0.0
    steps = 0
    
    is_main = (not dist.is_initialized()) or (dist.get_rank() == 0)
    pbar = tqdm(loader, desc=f"Epoch {epoch}", ncols=100) if is_main else loader

    for batch in pbar:
        vox, target, _ = batch
        vox = vox.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).float()
        patches = patchify_batch(vox, args.patch)

        opt.zero_grad()
        with torch.amp.autocast("cuda", enabled=True):
            if args.finetune:
                feats = model.module.encode(patches) if args.distributed else model.encode(patches)
                preds = reg_head(feats).view(-1)
                
                # Normalize target logic
                if args.normalize_target:
                    target_norm = (target - args._target_mean) / args._target_std
                else:
                    target_norm = target
                
                valid_mask = ~torch.isnan(target_norm)
                if valid_mask.sum() == 0: continue 
                loss = loss_fn(preds[valid_mask], target_norm[valid_mask])
            else:
                mask_idx, keep_idx = random_masking(patches.shape[1], args.mask_ratio, device)
                pred = model(patches, mask_idx, keep_idx)
                loss = mae_loss_on_masked(patches, pred, mask_idx)

        scaler.scale(loss).backward()
        if args.max_grad_norm > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        
        scaler.step(opt)
        scaler.update()

        running_loss += loss.item()
        steps += 1
        if is_main and isinstance(pbar, tqdm):
            pbar.set_postfix({'loss': running_loss/steps})

    return running_loss / max(1, steps)

def evaluate(model, loader, device, args, reg_head=None, output_csv=None):
    model.eval()
    if reg_head: reg_head.eval()
    
    loss_fn = nn.L1Loss(reduction='mean') if args.reg_loss == 'l1' else nn.MSELoss(reduction='mean')
    local_loss, local_count = 0.0, 0
    
    results = {'filename': [], 'prediction': [], 'target': [], 'is_valid': []}

    with torch.no_grad():
        for vox, target, names in loader:
            vox = vox.to(device)
            target = target.to(device).float()
            patches = patchify_batch(vox, args.patch)

            if reg_head:
                feats = model.module.encode(patches) if args.distributed else model.encode(patches)
                preds = reg_head(feats).view(-1)
                
                if args.normalize_target:
                    target_norm = (target - args._target_mean) / args._target_std
                    preds_denorm = preds * args._target_std + args._target_mean
                else:
                    target_norm = target
                    preds_denorm = preds

                valid_mask = ~torch.isnan(target_norm)
                n_valid = valid_mask.sum().item()
                if n_valid > 0:
                    local_loss += loss_fn(preds[valid_mask], target_norm[valid_mask]).item() * n_valid
                    local_count += n_valid

                results['filename'].extend(names)
                results['prediction'].extend(preds_denorm.cpu().numpy().tolist())
                results['target'].extend(target.cpu().numpy().tolist())
                results['is_valid'].extend(valid_mask.cpu().numpy().tolist())
            else:
                mask_idx, keep_idx = random_masking(patches.shape[1], args.mask_ratio, device)
                pred = model(patches, mask_idx, keep_idx)
                n_masked = mask_idx.numel()
                local_loss += mae_loss_on_masked(patches, pred, mask_idx).item() * n_masked
                local_count += n_masked

    # Aggregate loss
    if args.distributed:
        tensor = torch.tensor([local_loss, local_count], device=device, dtype=torch.float64)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        total_loss, total_count = tensor[0].item(), tensor[1].item()
    else:
        total_loss, total_count = local_loss, local_count

    # Save CSV (Rank 0 only)
    if output_csv and (not args.distributed or dist.get_rank() == 0) and results['filename']:
        pd.DataFrame(results).to_csv(output_csv, index=False)
        print(f"[CSV] Saved to {output_csv}")

    return total_loss / total_count if total_count > 0 else float('nan')
