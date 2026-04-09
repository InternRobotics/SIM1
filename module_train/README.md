# module_train

This folder contains two training pipelines:

- `trajectory_generator`: flow-matching model for EE trajectory generation.
- `trajectory_discriminator`: video binary classifier (true/false trajectory quality).

All commands below use script default hyperparameters unless explicitly required by the script.

## 1) Trajectory Generator Training

Script: `trajectory_generator/train.py`

### Input

Provide `--data_root` that contains at least:

- `ee_pos/*.npz`
- `segments_fine/*.json`

These two subfolders are the default values of `--ee_folder` and `--segments_folder`.

### Output

Default output directory is `./output` (relative to where you run the command), including:

- `flow_ckpt.pth` (checkpoint)
- `val_vis/` (validation visualization images)
- `norm_stats.npz` (normalization stats, saved under `data_root`)

### Run (defaults)

Single GPU:

```bash
cd module_train/trajectory_generator
python train.py --data_root /path/to/data_root
```

Multi-GPU (DDP):

```bash
cd module_train/trajectory_generator
torchrun --nproc_per_node=8 train.py --data_root /path/to/data_root
```

## 2) Trajectory Discriminator Training

Script: `trajectory_discriminator/video_binary_ddp.py`

### Input

Training needs two classes of videos and evaluation videos:

- `--videos_true`
- `--videos_false`
- `--eval_videos_true`
- `--eval_videos_false`

Optional cache path:

- `--cache_dir` (default `.cache`)

### Output

Default output directory is `./ckpt`, including:

- `model_step*.pt` (saved checkpoints)
- `mistakes_step*.txt` (evaluation mistakes)

Cache tensors (`*.pt`) are written to `cache_dir`.

### Run (defaults)

```bash
cd module_train/trajectory_discriminator
torchrun --nproc_per_node=8 video_binary_ddp.py \
  --mode train \
  --videos_true /path/to/videos_true \
  --videos_false /path/to/videos_false \
  --eval_videos_true /path/to/eval_videos_true \
  --eval_videos_false /path/to/eval_videos_false
```

## Notes

- The generator script requires `--data_root`; all other generator args use defaults.
- The discriminator script requires video paths; training args (sample rate, steps, batch size, etc.) use defaults.
- If you want to pre-build video cache before training discriminator, use `trajectory_discriminator/cache_video.py`.
