import os
import cv2
import math
import json
import shutil
import hashlib
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import models, transforms
from tqdm import tqdm
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

def saveJson(filename, data):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

def video_to_cache_path(video_path, cache_dir, sample_rate):
    key = f"{video_path}_{sample_rate}"
    h = hashlib.md5(key.encode()).hexdigest()
    return Path(cache_dir) / f"{h}.pt"


def load_video_frames(video_path, sample_rate):
    cap = cv2.VideoCapture(video_path)
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % sample_rate == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        idx += 1
    cap.release()
    return frames

class VideoBinaryDataset(Dataset):
    def __init__(
        self,
        true_dir,
        false_dir,
        sample_rate,
        cache_dir,
        clip_seconds: int = 8,
        pre_cache: bool = True,
        cache_threads: int = 8,
        fps: int = 30
    ):
        self.sample_rate = sample_rate
        self.clip_seconds = clip_seconds
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))

        # ---------- collect & balance ----------
        pos_paths = sorted([p.as_posix() for p in Path(true_dir).glob("*")])
        neg_paths = sorted([p.as_posix() for p in Path(false_dir).glob("*")])

        n = min(len(pos_paths), len(neg_paths))
        random.seed(42)
        pos_paths = random.sample(pos_paths, n)
        neg_paths = random.sample(neg_paths, n)

        self.samples = [(p, 1) for p in pos_paths] + [(p, 0) for p in neg_paths]
        random.shuffle(self.samples)

        if self.rank == 0:
            print(f"Dataset size: {len(self.samples)} (pos={n}, neg={n})")

        # ---------- transform ----------
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ])
        self.fps = fps

        # ---------- multi-threaded pre-cache ----------
        if pre_cache:
            self._prepare_cache_mt(cache_threads)

    # =========================================================
    #                  clip + pad
    # =========================================================
    def _clip_and_pad(self, frames: torch.Tensor, fps: float):
        """
        frames: [T, C, H, W]
        """
        target_len = int(self.clip_seconds * fps / self.sample_rate)
        T = frames.shape[0]

        if T >= target_len:
            frames = frames[-target_len:]
        else:
            pad_len = target_len - T
            pad = torch.zeros(
                pad_len,
                frames.shape[1],
                frames.shape[2],
                frames.shape[3],
                dtype=frames.dtype,
            )
            frames = torch.cat([pad, frames], dim=0)

        return frames

    # =========================================================
    #                  multi-thread cache
    # =========================================================
    def _cache_one(self, video_path):
        cache_path = video_to_cache_path(
            video_path,
            self.cache_dir,
            self.sample_rate,
        )
        if cache_path.exists():
            return

        # Prefer load_video_frames to return frames + fps
        raw_frames = load_video_frames(video_path, self.sample_rate)
        fps = self.fps

        frames = torch.stack([self.transform(f) for f in raw_frames])
        frames = self._clip_and_pad(frames, fps)

        torch.save(frames, cache_path)

    def _prepare_cache_mt(self, num_threads):
        if self.world_size > 1:
            dist.barrier()

        if self.rank == 0:
            print(
                f"[Cache] Start multi-thread caching "
                f"(threads={num_threads}, clip={self.clip_seconds}s)",
                flush=True,
            )

            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [
                    executor.submit(self._cache_one, video_path)
                    for video_path, _ in self.samples
                ]

                for _ in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Caching videos",
                ):
                    pass

            print("[Cache] All videos cached.", flush=True)

        if self.world_size > 1:
            dist.barrier()

    # =========================================================

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        cache_path = video_to_cache_path(
            video_path,
            self.cache_dir,
            self.sample_rate,
        )

        frames = torch.load(cache_path, map_location="cpu")
        return frames, torch.tensor(label), video_path


def collate_fn(batch):
    videos, labels, paths = zip(*batch)
    lengths = torch.tensor([v.size(0) for v in videos])

    max_len = max(lengths)
    padded = torch.zeros(len(videos), max_len, *videos[0].shape[1:])

    for i, v in enumerate(videos):
        padded[i, :v.size(0)] = v

    return padded, lengths, torch.stack(labels), paths


class VideoTransformer(nn.Module):
    def __init__(self, dim=512, depth=4, heads=8):
        super().__init__()
        backbone = models.resnet18(weights="IMAGENET1K_V1")
        self.cnn = nn.Sequential(*list(backbone.children())[:-1])
        self.proj = nn.Linear(512, dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, depth)
        self.cls = nn.Linear(dim, 1)

    def forward(self, x, lengths):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feat = self.cnn(x).squeeze(-1).squeeze(-1)
        feat = self.proj(feat)
        feat = feat.view(B, T, -1)

        mask = torch.arange(T, device=lengths.device)[None, :] >= lengths[:, None]
        feat = self.transformer(feat, src_key_padding_mask=mask)

        pooled = feat.mean(dim=1)
        return self.cls(pooled).squeeze(-1)

def train_ddp(
    model, train_loader, eval_loader,
    max_steps, eval_every, device,
    rank, world_size, save_dir
):
    opt = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    step = 0
    pbar = tqdm(total=max_steps, disable=(rank != 0))

    while step < max_steps:
        for videos, lengths, labels, _ in train_loader:
            if step >= max_steps:
                break

            videos = videos.to(device)
            lengths = lengths.to(device)
            labels = labels.float().to(device)

            logits = model(videos, lengths)
            loss = loss_fn(logits, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            step += 1
            if rank == 0:
                pbar.update(1)
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            if step % eval_every == 0:
                evaluate(model, eval_loader, device, rank, step, save_dir)

    if rank == 0:
        pbar.close()

def evaluate(model, loader, device, rank, step, save_dir):
    model.eval()
    correct = 0
    total = 0
    mistakes = []

    with torch.no_grad():
        for videos, lengths, labels, paths in loader:
            videos = videos.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)

            logits = model(videos, lengths)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()

            for p, gt, pr, prob in zip(paths, labels, preds, probs):
                if gt != pr:
                    mistakes.append(f"{p}, gt={gt.item()}, prob={prob.item():.4f}")

            correct += (preds == labels).sum().item()
            total += labels.numel()

    if rank == 0:
        acc = correct / total
        print(f"[Step {step}] Eval acc = {acc:.4f}")

        with open(Path(save_dir) / f"mistakes_step{step}.txt", "w") as f:
            f.write("\n".join(mistakes))

        torch.save(
            model.state_dict(),
            Path(save_dir) / f"model_step{step}.pt"
        )

    model.train()

def inference(model, root, sample_rate, device):
    model.eval()
    video_dir = os.path.join(root, "videos")
    print(video_dir)
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            ])
    output = {}
    suc = 0
    fail = 0
    err = 0

    num = 0
    for p in Path(video_dir).glob("*"):

        basename = os.path.splitext(os.path.basename(p))[0]
        
        try:
            frames = load_video_frames(p.as_posix(), sample_rate)
            frames = torch.stack([transform(f) for f in frames])
            frames = frames.unsqueeze(0).to(device)
            lengths = torch.tensor([frames.size(1)]).to(device)

            with torch.no_grad():
                prob = torch.sigmoid(model(frames, lengths)).item()
                pred = (prob > 0.5)

            if pred:
                suc += 1
                print(f"{basename} → SUCCESS/{suc}")
                output.update({basename:prob})
            else:
                fail += 1
                print(f"{basename} → FAILED/{fail}")
                output.update({basename:prob})
                # move_related_files(basename, root)
        except:
            err += 1
            print(f"{basename} → video error/{err}")
            output.update({basename:"error"})
            # move_related_files(basename, root)
    file_path = os.path.join(root, "preds.json")
    saveJson(file_path, output)

def move_related_files(basename, root):
    """Move mp4/usd/npz to *_failed folders."""
    subfolders = ["videos", "usd", "gen"]
    exts = {"videos": "mp4", "usd": "usd", "gen": "npz"}

    for sf in subfolders:
        src = os.path.join(root, sf, f"{basename}.{exts[sf]}")
        dst_folder = os.path.join(root, f"{sf}_failed")
        os.makedirs(dst_folder, exist_ok=True)
        dst = os.path.join(dst_folder, f"{basename}.{exts[sf]}")

        if os.path.exists(src):
            print(f"  Moving {src} -> {dst}")
            shutil.move(src, dst)
        else:
            print(f"[WARN] missing related file: {src}")

def main():
    parser = argparse.ArgumentParser()

    # ---------- mode ----------
    parser.add_argument("--mode", type=str, default="inference",
                        choices=["train", "inference"])
    parser.add_argument("--sample_rate", type=int, default=10)
    # ---------- inference ----------
    parser.add_argument("--root", type=str)
    parser.add_argument("--ckpt", type=str)

    args = parser.parse_args()

    # ---------- DDP init ----------
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if world_size > 1:
        dist.init_process_group(backend="nccl")

    # ---------- model ----------
    model = VideoTransformer()
    model.to(device)

    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank]
        )

    # ==========================================================
    #                          TRAIN
    # ==========================================================
    if args.mode == "train":
        pass
    # ==========================================================
    #                        INFERENCE
    # ==========================================================
    else:
        assert args.ckpt is not None, "Need --ckpt for inference"
        # device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        ck = torch.load(args.ckpt, map_location=device)

        new_ck = {}
        for k, v in ck.items():
            if k.startswith('module.'):
                k = k[7:]
            new_ck[k] = v

        model.load_state_dict(new_ck)
        model.to(device)

        inference(
            model,
            args.root,
            args.sample_rate,
            device,
        )

    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
