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

from .utils import *

# --------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------
class FlowMatchDataset(Dataset):
    """
    Build training samples:
      For each json file (segments list) and the matching npz file:
        - find stable segments that have moving before and after (as required)
        - For each such stable segment:
            source_pose = end pose of previous moving (16,)
            target_pose = start pose of next moving (16,)
            history = T frames BEFORE source_pose (if insufficient pad with source_pose)
            traj = full stable frames (m,16)  <-- this is x1
            We return: history (T,16), source_pose (16,), target_pose (16,), traj (m,16), length m
    """
    def __init__(self, data_root, ee_folder='ee_pos', segments_folder='segments_fine', history_len=16, pad_value=0.0, min_m=4):
        super().__init__()
        self.data_root = Path(data_root)
        self.ee_folder = self.data_root / ee_folder
        self.segments_folder = self.data_root / segments_folder
        # print(self.ee_folder, self.segments_folder)
        assert self.ee_folder.exists() and self.segments_folder.exists()
        self.history_len = history_len
        self.pad_value = pad_value
        self.min_m = min_m

        # Build index of valid samples
        self.samples = []  # list of dicts: {record_id, npz_path, stable_range (s,e), src_idx, tgt_idx, history_indices(list)}
        self._build_index()
        self.data_dim = 16
        mean, std = self._compute_stats()
        self.mean_np = mean.astype(np.float32)
        self.std_np = std.astype(np.float32)
        self.mean = torch.tensor(mean, dtype=torch.float32)    # (16,)
        self.std  = torch.tensor(std, dtype=torch.float32)     # (16,)
        stats_path = os.path.join(data_root, "norm_stats.npz")
        np.savez(stats_path, mean=mean, std=std)

    def _compute_stats(self):
        """
        Scan all `.npz` files under self.ee_folder, collect (16,) or (L,16) arrays,
        concatenate to (N,16), then compute mean/std.
        """

        all_vecs = []  # list of (k,16) arrays

        # 1. Walk all npz files
        for fname in os.listdir(self.ee_folder):
            if not fname.endswith(".npz"):
                continue

            path = os.path.join(self.ee_folder, fname)
            data = np.load(path)

            # 2. Walk all keys in the npz
            for key in data.files:
                arr = data[key]   # could be (16,) or (L,16) or irrelevant

                arr = np.asarray(arr)

                # 3. Keep only 16-D vectors or trajectories
                if arr.ndim == 1 and arr.shape[0] == self.data_dim:
                    all_vecs.append(arr[None, :])   # reshape → (1,16)

                elif arr.ndim == 2 and arr.shape[1] == self.data_dim:
                    all_vecs.append(arr)            # (L,16)

                # Skip non-16D arrays (history, mask, index, etc.)
                else:
                    continue

        if len(all_vecs) == 0:
            raise RuntimeError("No (16-dim) data found in npz files.")

        # 4. Stack to (N,16)
        cat = np.concatenate(all_vecs, axis=0)

        # 5. mean/std
        mean = cat.mean(axis=0)
        std = cat.std(axis=0) + 1e-6

        return mean, std

    def _build_index(self):
        files = sorted(list(self.segments_folder.glob('*.json')))
        for json_path in files:
            data = read_json(str(json_path))
            segs = data.get('segments', [])
            if not segs:
                continue
            # load the corresponding npz
            record_id = json_path.stem
            npz_path = self.ee_folder / (record_id + '.npz')
            if not npz_path.exists():
                continue
            # find stable segments that have moving before and after
            # loop through segs, index i: if segs[i] is stable and i-1 and i+1 exist and are moving
            for i in range(0, len(segs)):
                if segs[i]['state'] == 'stable':
                    s = segs[i]['start']
                    e = segs[i]['end']
                    m = e - s + 1
                    if m < self.min_m:
                        continue
                    # source pose = end of previous moving (segs[i-1].end)
                    # src_idx = segs[i-1]['end']
                    # tgt_idx = segs[i+1]['start']
                    src_idx, tgt_idx = s, e
                    # history indices: take history_len frames before src_idx (inclusive? we'll choose last history_len frames ending at src_idx-1)
                    # we'll take frames [src_idx - history_len, src_idx-1] as history (if insufficient, pad using frame 0 of available region)
                    history_start = src_idx - self.history_len
                    # store
                    self.samples.append({
                        'record_id': record_id,
                        'npz_path': str(npz_path),
                        'stable_start': s,
                        'stable_end': e,
                        'src_idx': src_idx,
                        'tgt_idx': tgt_idx,
                        'history_start': history_start,
                    })

    def __len__(self):
        return len(self.samples)

    def normalize(self, x):
        """
        x: NumPy array or torch.Tensor, shape (..., 16)
        """
        # NumPy branch
        if isinstance(x, np.ndarray):
            return (x - self.mean_np) / self.std_np

        # Torch branch
        if isinstance(x, torch.Tensor):
            mean = self._to_device_dtype(self.mean_torch, x)
            std = self._to_device_dtype(self.std_torch, x)
            return (x - mean) / std

        raise TypeError("Input must be numpy array or torch tensor")

    def denormalize(self, x):
        """
        x: NumPy array or torch.Tensor, shape (..., 16)
        """
        # NumPy branch
        if isinstance(x, np.ndarray):
            return x * self.std_np + self.mean_np

        # Torch branch
        if isinstance(x, torch.Tensor):
            mean = self._to_device_dtype(self.mean_torch, x)
            std = self._to_device_dtype(self.std_torch, x)
            return x * std + mean

        raise TypeError("Input must be numpy array or torch tensor")

    def __getitem__(self, idx):
        srec = self.samples[idx]
        arr = np.load(srec['npz_path'])
        # choose the first array in npz (your earlier code used data.files[1] - but here pick first)
        # to be robust, pick the first array that has shape[1]==16
        chosen = None
        for k in arr.files:
            a = arr[k]
            if a.ndim == 2 and a.shape[1] == self.data_dim:
                chosen = a
                break
        if chosen is None:
            raise RuntimeError(f"no (n,16) array in {srec['npz_path']}")
        frames = chosen  # numpy array (n,16)

        s = srec['stable_start']
        e = srec['stable_end']
        m = e - s + 1
        # x1 = stable trajectory frames s..e inclusive
        x1 = frames[s:(e+1)].astype(np.float32)  # shape (m,16)

        # source and target poses
        src_idx = srec['src_idx']
        tgt_idx = srec['tgt_idx']
        src_pose = frames[src_idx].astype(np.float32)
        tgt_pose = frames[tgt_idx].astype(np.float32)

        # history: we take frames [history_start, src_idx-1], pad on the left if needed
        history_len = self.history_len
        hstart = max(0, srec['history_start'])
        hend = src_idx  # exclusive? let's take history as last history_len frames ending at src_idx (exclusive)
        # choose frames from max(0, src_idx-history_len) to src_idx-1 (inclusive)
        hist_from = max(0, src_idx - history_len)
        hist = frames[hist_from:src_idx].astype(np.float32)  # shape (h,16)
        h_actual = hist.shape[0]
        if h_actual < history_len:
            # pad in front with the earliest available frame (or src_pose) to reach history_len
            pad_n = history_len - h_actual
            pad_frame = frames[0].astype(np.float32) if hist.shape[0] > 0 else src_pose
            pad_arr = np.tile(pad_frame[None, :], (pad_n, 1)).astype(np.float32)
            hist = np.concatenate([pad_arr, hist], axis=0)
        # ensure length
        hist = hist[-history_len:].astype(np.float32)

        hist = self.normalize(hist)
        src_pose = self.normalize(src_pose)
        tgt_pose = self.normalize(tgt_pose)
        x1 = self.normalize(x1)

        return {
            'record_id': srec['record_id'],
            'history': hist,             # (T,16)
            'src_pose': src_pose,        # (16,)
            'tgt_pose': tgt_pose,        # (16,)
            'traj': x1,                  # (m,16)
            'len': m
        }

# collate with padding
def collate_fn(batch):
    # batch is list of dicts
    B = len(batch)
    T = batch[0]['history'].shape[0]
    # determine max traj len
    max_m = max([b['len'] for b in batch])
    feat = 16
    # allocate
    history = np.stack([b['history'] for b in batch], axis=0)  # (B,T,16)
    src = np.stack([b['src_pose'] for b in batch], axis=0)    # (B,16)
    tgt = np.stack([b['tgt_pose'] for b in batch], axis=0)    # (B,16)
    traj = np.zeros((B, max_m, feat), dtype=np.float32)
    mask = np.zeros((B, max_m), dtype=np.float32)
    lengths = np.zeros((B,), dtype=np.int64)
    record_ids = []
    for i,b in enumerate(batch):
        m = b['len']
        traj[i, :m, :] = b['traj']
        mask[i, :m] = 1.0
        lengths[i] = m
        record_ids.append(b['record_id'])
    # convert to tensors
    import torch
    return {
        'history': torch.from_numpy(history).float(),   # (B,T,16)
        'src': torch.from_numpy(src).float(),           # (B,16)
        'tgt': torch.from_numpy(tgt).float(),           # (B,16)
        'traj': torch.from_numpy(traj).float(),         # (B,max_m,16)
        'mask': torch.from_numpy(mask).float(),         # (B,max_m)
        'lengths': torch.from_numpy(lengths),
        'record_ids': record_ids
    }

# ------------------------------
# Dataset and Dataloader
# ------------------------------
def prepare_dataloaders(args, is_main):
    # Load full dataset
    full_dataset = FlowMatchDataset(
        args.data_root,
        ee_folder=args.ee_folder,
        segments_folder=args.segments_folder,
        history_len=args.history_len,
    )
    if len(full_dataset) == 0:
        raise RuntimeError("No samples found")

    # Split train/val
    val_size = max(1, int(len(full_dataset) * 0.05))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    if is_main:
        print(f"[Dataset] Train={train_size}, Val={val_size}")

    # Distributed samplers
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader, train_sampler