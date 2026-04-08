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

# --------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------
def makedir(path):
    os.makedirs(path, exist_ok=True)

def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_json(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)

def set_seed(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def make_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)

# --------------------------------------------------------------------
# Distributed helpers
# --------------------------------------------------------------------
def setup_distributed(args):
    # args.local_rank provided by torchrun
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    return device, rank, world_size

class Normalizer14:
    """
    A reusable normalization helper for 14D robot state vectors.

    It loads saved mean/std from a .npz file:
        {'mean': (14,), 'std': (14,)}

    Usage:
        norm = Normalizer14("norm_stats.npz")
        x_norm = norm.normalize(x)
        x_back = norm.denormalize(x_norm)
    """

    def __init__(self, data_root):
        stats_path = os.path.join(Path(data_root), "norm_stats.npz")
        if not os.path.exists(stats_path):
            raise FileNotFoundError(
                f"[Normalizer14] Stats file not found: {stats_path}\n"
                f"You must first compute and save mean/std into this file."
            )

        data = np.load(stats_path)
        if 'mean' not in data or 'std' not in data:
            raise ValueError(f"[Normalizer14] Invalid stats file: {stats_path}. "
                             "It must contain 'mean' and 'std' arrays.")
        
        self.mean_np = data['mean']   # shape (14,)
        self.std_np  = data['std']    # shape (14,)

        # convert to torch tensors for easy use on GPU
        self.mean = torch.tensor(self.mean_np, dtype=torch.float32)
        self.std  = torch.tensor(self.std_np, dtype=torch.float32)

        # avoid division instability
        self.std[self.std < 1e-6] = 1e-6

    def to(self, device):
        """ Move mean/std to a specific device """
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    # --------------------------
    # Normalization API
    # --------------------------
    def normalize(self, x):
        """
        x: NumPy array or torch.Tensor, shape (..., 14)
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
        x: NumPy array or torch.Tensor, shape (..., 14)
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

def visualize_sample(x_pred_np, x_gt_np, mask_np, save_path):
    """
    x_pred_np, x_gt_np: (L,14) or (B,L,14) -> handle first sample
    mask_np: (L,) or (B,L), 0 for invalid frames
    We'll plot all 14 dims over time, mask out invalid frames
    """
    if x_pred_np.ndim == 3:
        x_pred_np = x_pred_np[0]
        x_gt_np = x_gt_np[0]
        mask_np = mask_np[0]

    L, D = x_pred_np.shape
    t = np.arange(L)

    fig, axs = plt.subplots(2, 8, figsize=(21,6))
    labels_left = [f'left_{i}' for i in range(8)]
    labels_right = [f'right_{i}' for i in range(8)]

    for i in range(8):
        # Left arm
        mask = mask_np.astype(bool)
        axs[0,i].plot(t[mask], x_gt_np[mask, i], label='gt')
        axs[0,i].plot(t[mask], x_pred_np[mask, i], label='pred')
        axs[0,i].set_title(labels_left[i])
        axs[0,i].legend()

        # Right arm
        axs[1,i].plot(t[mask], x_gt_np[mask, 8+i], label='gt')
        axs[1,i].plot(t[mask], x_pred_np[mask, 8+i], label='pred')
        axs[1,i].set_title(labels_right[i])
        axs[1,i].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()