#!/usr/bin/env python3
# flowmatch_train.py
"""
Flow Matching training (single-node multi-GPU) for generating ee-pos trajectories.

Usage (example):
  # 2 GPUs
  torchrun --nproc_per_node=2 flowmatch_train.py \
      --data_root /path/to/A \
      --ee_folder ee_pos \
      --segments_folder segments_fine \
      --out_dir /path/to/out \
      --mode train \
      --batch_size 64 \
      --gpus 2

Modes:
  train   - train then checkpoint
  sample  - sample a few trajectories from checkpoint and save visualizations
  eval    - compute validation loss (optional)
"""

import os
import json
import math
import random
import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm
from src.flowmatch_dataset import FlowMatchDataset, collate_fn, prepare_dataloaders
from src.flowmatching import FlowMatching, sample_trajectory
from src.utils import *
from src.simple_diffusion import simpleDiffusion
from src.models.diffusion import UNet



# ------------------------------
# Model and optimizer
# ------------------------------
def build_model_and_optimizer(cfg, device):
    normalizer = Normalizer14(cfg.data_root).to(device)
    # flow = FlowMatching(cfg, device)
    DiffusionUNet = UNet().to(device)
    flow = simpleDiffusion(DiffusionUNet).to(device)
    flow.model = torch.nn.parallel.DistributedDataParallel(
        flow.model, device_ids=[device.index], output_device=device.index
    )
    flow.optim = torch.optim.Adam(flow.model.parameters(), lr=cfg.lr)
    return flow


# ------------------------------
# Validation
# ------------------------------
def run_validation(flow, val_loader, args, device):
    flow.model.eval()
    total_loss, count = 0, 0
    vis_id, max_vis = 0, args.val_samples

    with torch.no_grad():
        for batch in val_loader:
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device)

            loss, loss_dict = flow.train_step(batch)
            total_loss += loss.item()
            count += 1

            # Visualization
            if vis_id < max_vis:
                model = flow.model.module
                x_pred, mask = flow.infer_step(batch)
                # x_pred, mask = sample_trajectory(flow, batch, steps=args.sample_steps, device=device)
                x_gt = batch["traj"]

                visualize_sample(
                    x_pred.cpu().numpy(),
                    x_gt.cpu().numpy(),
                    mask.cpu().numpy(),
                    os.path.join(args.out_dir, "val_vis", f"val_{vis_id:03d}.png"),
                )
                vis_id += 1

    return total_loss / max(1, count)


# ------------------------------
# Training loop
# ------------------------------
def train(flow, train_loader, val_loader, train_sampler, args, device, is_main):
    global_step = 0
    ckpt_path = os.path.join(args.out_dir, "flow_ckpt.pth")

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        epoch_loss_sum, epoch_loss_count = 0, 0

        for batch in train_loader:
            flow.model.train()
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device)

            flow.optim.zero_grad()
            loss, loss_dict = flow.train_step(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.model.parameters(), 1.0)
            flow.optim.step()

            loss_val = loss.item()
            epoch_loss_sum += loss_val
            epoch_loss_count += 1

            if is_main and (global_step % args.log_every == 0):
                print(f"[Train] Step {global_step}  Loss={loss_val:.6f}  "
                    #   f"FlowLoss: {loss_dict['loss_flow']:.6f}  "
                    #   f"SmoothLoss: {loss_dict['loss_smooth']:.6f}  "
                    #   f"TrajLoss: {loss_dict['loss_traj']:.6f}"
                      )

            if is_main and (global_step % args.save_every == 0 and global_step > 0):
                flow.save(ckpt_path)
                print(f"[Checkpoint] Saved to {ckpt_path}")
                val_loss = run_validation(flow, val_loader, args, device)
                print(f"[Validation] Step {global_step}  ValLoss={val_loss:.6f}")

            global_step += 1

        if is_main:
            avg_loss = epoch_loss_sum / max(1, epoch_loss_count)

    if is_main:
        flow.save(ckpt_path)
        print("[Training Complete] Final checkpoint saved.")

# --------------------------------------------------------------------
# main
# --------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', type=str, required=True)
    p.add_argument('--ee_folder', type=str, default='ee_pos')
    p.add_argument('--segments_folder', type=str, default='segments_fine')
    p.add_argument('--out_dir', type=str, default='./output')
    p.add_argument('--mode', type=str, default='train', choices=['train','sample','eval'])
    p.add_argument('--history_len', type=int, default=1)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--epochs', type=int, default=200000)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--hidden_dim', type=int, default=256)
    p.add_argument('--n_layers', type=int, default=4)
    p.add_argument('--n_heads', type=int, default=4)
    p.add_argument('--prior_sigma', type=float, default=1.0)
    p.add_argument('--save_every', type=int, default=1000)
    p.add_argument('--log_every', type=int, default=50)
    p.add_argument('--seed', type=int, default=12580)
    p.add_argument('--num_samples', type=int, default=8)
    p.add_argument('--sample_steps', type=int, default=20)
    p.add_argument('--val_samples', type=int, default=10)
    return p.parse_args()
# ------------------------------
# Main entrypoint
# ------------------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    device, rank, world_size = setup_distributed(args)
    is_main = (rank == 0)

    if is_main:
        make_dirs(args.out_dir, os.path.join(args.out_dir, "val_vis"))

    train_loader, val_loader, train_sampler = prepare_dataloaders(args, is_main)

    # Config object
    class Cfg: pass
    cfg = Cfg()
    cfg.history_len = args.history_len
    cfg.hidden_dim = args.hidden_dim
    cfg.n_layers = args.n_layers
    cfg.n_heads = args.n_heads
    cfg.lr = args.lr
    cfg.prior_sigma = args.prior_sigma
    cfg.data_root = args.data_root

    flow = build_model_and_optimizer(cfg, device)

    if args.mode == "train":
        train(flow, train_loader, val_loader, train_sampler, args, device, is_main)
    elif args.mode in ("sample", "eval"):
        # TODO: implement sampling / evaluation
        pass
    else:
        raise RuntimeError(f"Unknown mode {args.mode}")


if __name__ == "__main__":
    main()