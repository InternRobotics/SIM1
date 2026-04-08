import os
import glob
import shutil
import argparse
import numpy as np
import torch
import cv2
from .video_filtter import run_on_video, move_related_files
from .ppo_value_from_videos import (
    ValueNet,
    OnlineValuePredictor,
    default_transform
)

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

class Discriminator:

    def __init__(self, ckpt, root, threshold=-0.2):
        self._load_model(ckpt)
        transform = default_transform()
        self.predictor = OnlineValuePredictor(self.model, device, mem_len=32, transform=transform)
        self.root = root
        self.threshold = threshold

        self._load_model(ckpt)
        
    def _load_model(self, ckpt):
        self.model = ValueNet(d_model=256, nhead=8, num_layers=3).to(device)

        ck = torch.load(ckpt, map_location=device)
        
        new_ck = {}
        for k, v in ck['state'].items():
            if k.startswith('module.'):
                k = k[7:]
            new_ck[k] = v

        
        if "state" in new_ck:
            self.model.load_state_dict(new_ck["state"])
        else:
            self.model.load_state_dict(new_ck)  # if pure state_dict
        self.model.eval()

    def forward(self):
        video_dir = os.path.join(self.root, "video")
        videos = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))

        print(f"Found {len(videos)} videos in {video_dir}")

        for vp in videos:
            basename = os.path.splitext(os.path.basename(vp))[0]
            print(f"\n=== Processing {basename} ===")

            preds = run_on_video(vp, self.predictor)
            if preds is None:
                continue

            final_score = preds[-1]
            print(f"  Final score = {final_score:.4f}")

            if final_score < self.threshold:
                print(f"  → FAILED  (score < {self.threshold})")
                move_related_files(basename, self.root)
            else:
                print(f"  → SUCCESS")