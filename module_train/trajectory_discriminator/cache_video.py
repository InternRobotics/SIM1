#!/usr/bin/env python3
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import torch
from torchvision import transforms
import random
import cv2
import os

def load_video_frames(video_path, sample_rate=1):
    """
    return: frames_list, fps
    frames_list: list of HWC ndarray
    fps:  fps
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_rate == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        frame_idx += 1
    cap.release()
    return frames, fps

def video_to_cache_path(video_path: str, cache_dir: Path, sample_rate: int, clip_seconds: int):
    p = Path(video_path)
    fname = f"{p.stem}.pt"
    return cache_dir / fname


class VideoCache:
    def __init__(self, true_dir, false_dir, sample_rate, cache_dir, clip_seconds=8, num_threads=8, use_first=False):
        self.sample_rate = sample_rate
        self.clip_seconds = clip_seconds
        self.use_first = use_first
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.num_threads = num_threads

        # balance pos/neg
        pos_paths = sorted([p.as_posix() for p in Path(true_dir).glob("*")])
        neg_paths = sorted([p.as_posix() for p in Path(false_dir).glob("*")])
        n = min(len(pos_paths), len(neg_paths))
        random.seed(42)
        pos_paths = random.sample(pos_paths, n)
        neg_paths = random.sample(neg_paths, n)

        self.samples = [(p, 1) for p in pos_paths] + [(p, 0) for p in neg_paths]
        random.shuffle(self.samples)
        print(f"Found {len(self.samples)} videos to cache (pos={n}, neg={n})")

        # transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ])

    def _clip_and_pad(self, frames: torch.Tensor, fps: float):
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
        cache_path = video_to_cache_path(video_path, self.cache_dir, self.sample_rate, self.clip_seconds)
        if cache_path.exists():
            return
        raw_frames, fps = load_video_frames(video_path, self.sample_rate)
        if len(raw_frames) == 0:
            print(f"Warning: video {video_path} has no frames!")
            return
        frames = torch.stack([self.transform(f) for f in raw_frames])
        frames = self._clip_and_pad(frames, fps)
        torch.save(frames, cache_path)
    
    def test(self):
        for p, _ in self.samples:
            self._cache_one(p)

    def run(self):
        print(f"Start caching with {self.num_threads} threads ...")
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [executor.submit(self._cache_one, p) for p, _ in self.samples]
            for _ in tqdm(as_completed(futures), total=len(futures), desc="Caching videos"):
                pass
        print("All videos cached.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--true_dir", type=str, required=True)
    parser.add_argument("--false_dir", type=str, required=True)
    parser.add_argument("--sample_rate", type=int, default=1)
    parser.add_argument("--cache_dir", type=str, default=".cache")
    parser.add_argument("--clip_seconds", type=int, default=8)
    parser.add_argument("--num_threads", type=int, default=8)
    parser.add_argument("--use_first", type=bool, default=False)
    args = parser.parse_args()

    vc = VideoCache(
        args.true_dir,
        args.false_dir,
        args.sample_rate,
        args.cache_dir,
        clip_seconds=args.clip_seconds,
        num_threads=args.num_threads,
        use_first=args.use_first
    )

    vc.run()

if __name__ == "__main__":
    main()
