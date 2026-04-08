import os
import cv2
import math
import json
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

def video_to_cache_path(video_path, cache_dir, sample_rate):
    p = Path(video_path)
    fname = f"{p.stem}.pt"
    return cache_dir / fname


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
        use_first: bool = False,
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

        if pre_cache:
            self._prepare_cache_mt(cache_threads)

    def _clip_and_pad(self, frames: torch.Tensor, fps: float):
        """
        frames: [T, C, H, W]
        """
        target_len = int(self.clip_seconds * fps / self.sample_rate)
        T = frames.shape[0]
        # new version, use the first frame
        first = frames[0].unsqueeze(0)

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
        if self.use_first:
            frames = torch.cat([first, frames], dim=0)
        return frames


    def _cache_one(self, video_path):
        cache_path = video_to_cache_path(
            video_path,
            self.cache_dir,
            self.sample_rate,
        )
        if cache_path.exists():
            return
        
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

def inference(model, video_dir, sample_rate, device):
    model.eval()
    for p in Path(video_dir).glob("*"):
        frames = load_video_frames(p.as_posix(), sample_rate)
        frames = torch.stack([transforms.ToTensor()(f) for f in frames])
        frames = frames.unsqueeze(0).to(device)
        lengths = torch.tensor([frames.size(1)]).to(device)

        with torch.no_grad():
            prob = torch.sigmoid(model(frames, lengths)).item()

        print(f"{p.name}: prob_true={prob:.4f}")


def main():
    parser = argparse.ArgumentParser()

    # ---------- mode ----------
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "inference"])

    # ---------- data ----------
    parser.add_argument("--videos_true", type=str)
    parser.add_argument("--videos_false", type=str)
    parser.add_argument("--eval_videos_true", type=str)
    parser.add_argument("--eval_videos_false", type=str)
    parser.add_argument("--sample_rate", type=int, default=5)
    parser.add_argument("--clip_seconds", type=int, default=8)
    parser.add_argument("--use_first", type=bool, default=False)
    parser.add_argument("--cache_dir", type=str, default=".cache")

    # ---------- training ----------
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--eval_every", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="./ckpt")

    # ---------- inference ----------
    parser.add_argument("--inference_dir", type=str)
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
        train_dataset = VideoBinaryDataset(
            args.videos_true,
            args.videos_false,
            args.sample_rate,
            args.cache_dir,
            clip_seconds = args.clip_seconds,
            use_first = args.use_first
        )

        eval_dataset = VideoBinaryDataset(
            args.eval_videos_true,
            args.eval_videos_false,
            args.sample_rate,
            args.cache_dir,
            clip_seconds = args.clip_seconds,
            use_first = args.use_first
        )

        train_sampler = DistributedSampler(train_dataset) if world_size > 1 else None
        eval_sampler = DistributedSampler(eval_dataset, shuffle=False) if world_size > 1 else None

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        eval_loader = DataLoader(
            eval_dataset,
            batch_size=1,
            sampler=eval_sampler,
            shuffle=False,
            num_workers=1,
            collate_fn=collate_fn,
        )

        os.makedirs(args.save_dir, exist_ok=True)

        train_ddp(
            model=model,
            train_loader=train_loader,
            eval_loader=eval_loader,
            max_steps=args.max_steps,
            eval_every=args.eval_every,
            device=device,
            rank=rank,
            world_size=world_size,
            save_dir=args.save_dir,
        )

    # ==========================================================
    #                        INFERENCE
    # ==========================================================
    else:
        assert args.ckpt is not None, "Need --ckpt for inference"
        model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
        model.to(device)

        inference(
            model,
            args.inference_dir,
            args.sample_rate,
            device,
        )

    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
