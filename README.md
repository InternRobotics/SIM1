# SIM1: Physics-Aligned Simulator as Zero-Shot Data Scaler in Deformable Worlds

![SIM1 real2sim2real](doc/real2sim2real.gif)

A research project from [InternRobotics](https://github.com/InternRobotics).

[![Demo](https://img.shields.io/badge/Demo-SIM1-0366d6?style=flat&logo=googlechrome&logoColor=white)](https://sim1-demo.intern-robotics.com/) [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE) [![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-b31b1b.svg)](https://arxiv.org/) [![Project Page](https://img.shields.io/badge/Project%20Page-SIM1-0366d6?style=flat&logo=githubpages&logoColor=white)](https://internrobotics.github.io/sim1.github.io/) [![Hugging Face · Assets](https://img.shields.io/badge/🤗%20Sim1-Assets-yellow)](https://huggingface.co/InternRobotics/Sim1_Assets) [![Hugging Face · Dataset](https://img.shields.io/badge/🤗%20Sim1-Dataset-yellow)](https://huggingface.co/datasets/InternRobotics/Sim1_Dataset)

[YouTube 1](https://youtu.be/tsPLa-1Lygw) · [YouTube 2](https://youtu.be/LXStHGWHh18) · [YouTube 3](https://youtu.be/zesn7aK9sgQ)

**Sim1** is a physics-aligned simulator and data stack for dual-arm cloth manipulation in simulation: teleoperation, diffusion-based data generation, replay, filtering, and optional photorealistic rendering, built on [Newton](https://newton-physics.github.io/newton/) and NVIDIA [Warp](https://nvidia.github.io/warp/). This repository contains the full pipeline from interactive control and synthetic trajectory generation to rendering and LeRobot-style dataset export.

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start — Interactive Teleoperation](#quick-start--interactive-teleoperation)
3. [Quick Start — Data Generation](#quick-start--data-generation)
4. [Rendering Pipeline](#rendering-pipeline)
5. [Data Conversion](#data-conversion)
6. [Project Structure](#project-structure)
7. [TODO List](#todo-list)
8. [Citation](#citation)
9. [License](#license)

---

## Installation

### Prerequisites

Use Python 3.11 with conda (environment name `sim1`) and CUDA toolkit ≥ 11.8 if you want GPU acceleration.

<details>
<summary>Reference: Newton Installation Guide</summary>

[Newton — manual setup with pip (virtual environment)](https://newton-physics.github.io/newton/0.2.2/guide/installation.html#method-3-manual-setup-using-pip-in-a-virtual-environment)

</details>

---

### Step 1 — Create the conda environment

```bash
conda create -n sim1 python=3.11 -y
conda activate sim1
```

---

### Step 2 — Clone the repository

Clone with submodules so `components/render/MeisterRender` ([SIM1MeisterRender](https://github.com/InternRobotics/SIM1MeisterRender), `main` branch) is checked out automatically:

```bash
git clone --recurse-submodules https://github.com/InternRobotics/SIM1.git sim1
cd sim1
```

---

### Step 3 — Install dependencies

With `sim1` active, from the repository root:

```bash
conda activate sim1
bash setup.sh
```

All Python dependencies (simulation, DataGen, asset download helpers, optional full render stack, and post-install checks) are installed by [`setup.sh`](setup.sh) only. Open that file for the full list, optional environment variables (`SIM1_SKIP_RENDER`, `TORCH_INDEX_URL`), and the exact `pip` commands. If you want to install a separate environment (for example render-only) or see which package each render step uses, refer to [`components/render/README.md`](components/render/README.md).

---

### Step 4 — Download assets (required before data generation)

Simulation, `run_pipeline.sh`, and the render stack all read the same Hugging Face bundle root. By default that is `./assets/` at the repo root (what `download_assets.sh` uses). The canonical resolver is `sim1_asset_paths.py`; override the root with:

```bash
export SIM1_ASSETS_ROOT=/absolute/or/relative/path   # parent of acone/, cloth/, random/, model/, …
```

If you use `bash download_assets.sh /other/path`, set `SIM1_ASSETS_ROOT` to that same path (the script prints a suggested `export` line when it finishes).

Download the official bundle from Hugging Face (`InternRobotics/Sim1_Assets`) into `./assets/`:

```bash
# From the repository root (after setup.sh)
bash download_assets.sh
```

---

### Verify installation

```bash
conda activate sim1
python -c "import newton; print('Newton version:', newton.__version__)"
python -c "import warp as wp; print('Warp OK')"
python -c "import torch, torchvision; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

Newton smoke test (MuJoCo humanoid + `nv_humanoid.xml`; needs a display for the GL viewer):

```bash
cd newton
python newton/examples/robot/example_robot_humanoid.py
```

Equivalent: `python -m newton.examples robot_humanoid` (from the same `newton/` directory). The MJCF asset is `newton/examples/assets/nv_humanoid.xml`.

After `bash download_assets.sh`, you should see at least `assets/acone/acone.urdf`, `assets/cloth/short-shirt.usdc`, and `assets/model/flow_ckpt_three.pth` before running `run_pipeline.sh`.

---

## Quick Start — Interactive Teleoperation

Launch a real-time interactive simulation with keyboard-driven dual-arm control:

```bash
python apps/teleoperation_app.py --task lift_manip_shirt
```

### Keyboard Controls

与 [`apps/teleoperation_app.py`](apps/teleoperation_app.py) 启动时打印的说明一致（shields 样式与上方 [Demo](https://sim1-demo.intern-robotics.com/) 徽章一致）：

| Key | Action |
|---|---|
| ![W/S](https://img.shields.io/badge/W%2FS-0366d6?style=flat&logo=googlechrome&logoColor=white) | Left gripper: forward / back |
| ![A/D](https://img.shields.io/badge/A%2FD-0366d6?style=flat&logo=googlechrome&logoColor=white) | Left gripper: left / right |
| ![Q/E](https://img.shields.io/badge/Q%2FE-0366d6?style=flat&logo=googlechrome&logoColor=white) | Left gripper: down / up |
| ![X](https://img.shields.io/badge/X-0366d6?style=flat&logo=googlechrome&logoColor=white) | Toggle left gripper |
| ![I/K](https://img.shields.io/badge/I%2FK-0366d6?style=flat&logo=googlechrome&logoColor=white) | Right gripper: forward / back |
| ![J/L](https://img.shields.io/badge/J%2FL-0366d6?style=flat&logo=googlechrome&logoColor=white) | Right gripper: left / right |
| ![U/O](https://img.shields.io/badge/U%2FO-0366d6?style=flat&logo=googlechrome&logoColor=white) | Right gripper: down / up |
| ![M](https://img.shields.io/badge/M-0366d6?style=flat&logo=googlechrome&logoColor=white) | Toggle right gripper |
| ![Arrow](https://img.shields.io/badge/Arrow%20keys-0366d6?style=flat&logo=googlechrome&logoColor=white) | Move camera |
| ![Mouse](https://img.shields.io/badge/Mouse%20drag-0366d6?style=flat&logo=googlechrome&logoColor=white) | Look around (left button drag) |
| ![Scroll](https://img.shields.io/badge/Scroll-0366d6?style=flat&logo=googlechrome&logoColor=white) | Zoom |
| ![H](https://img.shields.io/badge/H-0366d6?style=flat&logo=googlechrome&logoColor=white) | Toggle UI |
| ![Space](https://img.shields.io/badge/Space-0366d6?style=flat&logo=googlechrome&logoColor=white) | Pause / resume |
| ![345](https://img.shields.io/badge/3%20%2F%204%20%2F%205-0366d6?style=flat&logo=googlechrome&logoColor=white) | Camera presets |
| ![R](https://img.shields.io/badge/R-0366d6?style=flat&logo=googlechrome&logoColor=white) | Full reset (restarts process) |
| ![ESC](https://img.shields.io/badge/ESC-0366d6?style=flat&logo=googlechrome&logoColor=white) | Quit |

### WebSocket Streaming (Remote Teleoperation)

For headless or remote machines, enable WebSocket streaming:

```bash
python apps/teleoperation_app.py --task lift_manip_shirt --stream --host 0.0.0.0 --ws-port 8765 --http-port 8080
```

Then open `http://<server-ip>:8080` in a browser to view and control the simulation remotely.

---

## Quick Start — Data Generation

`run_pipeline.sh` runs the full data path in one shot: generate → Kalman smooth → replay (NPZ + USD) → filter. Generation uses the diffusion-policy path; robot URDF and cloth USD come from the Hugging Face bundle (default `./assets/`, or `SIM1_ASSETS_ROOT`; see [Step 4 — Download assets](#step-4--download-assets)). The script prints `HF assets : …` on startup. For extra diversity at replay, add `--position-randomize`.

### Generate data

From the repository root (conda env, assets, and clone steps are in [Installation](#installation) and [Step 4 — Download assets](#step-4--download-assets)):

```bash
bash run_pipeline.sh --num 100
```

Optional: `bash run_pipeline.sh --num 100 --position-randomize`

You get trajectories under `./dataset/example/` and a replay session `replay/pipeline_output_XXXX/` (the script prints the path). Use that folder as `--root_dir` for [Rendering Pipeline](#rendering-pipeline).

<details>
<summary>All <code>run_pipeline.sh</code> options (advanced)</summary>

| Option | Description | Default |
|---|---|---|
| `--data_folder DIR` | Data root | `./dataset/example` |
| `--num N` | Trajectories to generate (DP pipeline only) | 10 |
| `--workers N` | Parallel workers (smooth + filters) | 8 |
| `--skip_smooth` / `--skip_replay` / `--skip_filter` | Skip a stage | off |
| `--folder_name NAME` | `replay/<NAME>_XXXX/` base name | `pipeline_output` |
| `--position-randomize` | Random cloth pose at replay; joint filter also runs EE reachability (FK). Omit → joint filter uses `--no-check-ee` (jump / mutation only) | off |
| `--ref_usd PATH` | Reference USD for aligned cloth filter (with randomization); auto-picked if omitted | auto |
| `--skip_asset_check` | Do not verify the HF bundle (`SIM1_ASSETS_ROOT`) before running | off |

</details>

### What runs internally

```
run_pipeline.sh
│
├─ 1. Generate    →  apps/datagen_app.py --use_dp --mode fine (DP only; fixed in script)
│                     → <data_folder>/gen/*.npz
├─ 2. Smooth      →  scripts/smooth_trajectory_multi_thread.py (Kalman; fixed variances in script)
│                     → <data_folder>/gen/kf/*.npz
├─ 3. Replay      →  apps/replay_app.py [--position-randomize]
│                     → replay/<folder_name>_NNNN/{npz,usd}/
└─ 4. Filter      →  filter_joint_unreachable.py (joint jump + first-5 mutation; + EE FK if --position-randomize)
                     → filter_cloth_quality.py (aligned + --ref-usd if randomize, else direct)
```

### Session layout

```
replay/
└── pipeline_output_0001/
    ├── npz/
    ├── usd/
    ├── npz_bad_cloth/
    ├── usd_bad_cloth/
    ├── npz_unreachable/          # joint / EE rejects (see filter_joint_unreachable.py logs)
    └── cloth_filter_summary.txt
```

<details>
<summary>Manual step-by-step (only if you are not using <code>run_pipeline.sh</code>)</summary>

1. Generate: `python apps/datagen_app.py --data_folder ./dataset/example --num 100 --use_dp --mode fine`  
2. Smooth: `python scripts/smooth_trajectory_multi_thread.py ./dataset/example/gen ./dataset/example/gen/kf --method kalman --workers 8`  
3. Replay: `python apps/replay_app.py ./dataset/example/gen/kf --folder_name my_replay`  
Optional cloth position randomization at replay: add `--position-randomize` (then use the matching manual filters as in Step 4 above).  
4a. Joint / EE filter: `python scripts/filter_joint_unreachable.py ./replay/my_replay_0001/npz --usd-dir ./replay/my_replay_0001/usd --workers 8` (add `--no-check-ee` to skip EE FK; joint checks always run)  
4b. Cloth quality: `python scripts/filter_cloth_quality.py ./replay/my_replay_0001` (add `--ref-usd ...` if you used randomization)

</details>

---

## Rendering Pipeline

Convert simulation USD output to photorealistic data: `main.py` runs Steps 1–3 by default (USD → blend → cameras → `blend_out/`). Step 4 (MeisterRender path tracing + LMDB) writes under `out_updated/<record_id>/`; run it via `batch_step4.sh`, or inline with `main.py --step4`.

MeisterRender lives in the git submodule `components/render/MeisterRender` ([InternRobotics/SIM1MeisterRender](https://github.com/InternRobotics/SIM1MeisterRender), `main`). Use `git clone --recurse-submodules` in [Step 2](#step-2--clone-the-repository) so it is checked out automatically.

Environment: use the same `sim1` env; the render stack is installed by `setup.sh` unless you set `SIM1_SKIP_RENDER=1` (see comments in `setup.sh`). For a separate install or per-step package notes, see [`components/render/README.md`](components/render/README.md).

```bash
conda activate sim1
python components/render/main.py --root_dir ./replay/my_run_0001
bash components/render/batch_step4.sh ./replay/my_run_0001

# Optional: Step 4 inside main after each Step 3
# python components/render/main.py --root_dir ./replay/my_run_0001 --step4
```

### Asset Configuration

Rendering resolves the HF bundle via `SIM1_ASSETS_ROOT` (default `<repo>/assets/`). HDRI / table / cloth glTF roots default to `assets/random/{bg,table,mat}/` inside that bundle (`sim1_asset_paths.py`); no extra `export` is required for the usual layout.

---

## Data Conversion

After Step 4 rendering, trajectories are stored under `replay/<session>/out_updated/<record_id>/` as LMDB + `meta_info.pkl`. To convert them into a LeRobot v2 dataset for training, use `components/lmdb2lerobot/`.

One-time environment (separate conda env `lerobot`, Python 3.12 — see full docs for details):

```bash
bash components/lmdb2lerobot/setup_conda_lerobot.sh
conda activate lerobot
```

Single session → LeRobot dataset:

```bash
bash components/lmdb2lerobot/run_local.sh \
  --src ./replay/my_session/out_updated \
  --out ./replay/my_session/lerobot_dataset
```

This runs LMDB→LeRobot, sim2real, then removes near-static frames by default (`--keep-static-frames` to skip).

Batch / multi-GPU (optional): `components/lmdb2lerobot/run_batch.sh` — see [`components/lmdb2lerobot/README.md`](components/lmdb2lerobot/README.md).

---

## Project Structure

```
sim1/
├── setup.sh                    # Dependency installation (setup.sh)
├── download_assets.sh          # Hugging Face → ./assets/ (InternRobotics/Sim1_Assets)
├── run_pipeline.sh             # Data generation pipeline (generate→smooth→replay→filter)
├── apps/
│   ├── teleoperation_app.py    # Interactive teleoperation entry point
│   ├── datagen_app.py          # SIM1-DataGen entry (diffusion-policy mode)
│   ├── datagen_fine_app.py     # Optional fine-grained DataGen entry
│   └── replay_app.py           # Trajectory replay (headless)
├── replay_batch.sh             # Batch replay script
│
├── newton/                     # Newton physics engine (local install)
│   ├── pyproject.toml
│   └── newton/                 # Newton source code
│
├── assets/                     # Robot URDFs, meshes, render assets
├── configs/                    # Task configuration files
├── envs/                       # Simulation environments
├── tasks/                      # Task definitions (cloth manipulation)
├── stream/                     # WebSocket streaming server + web UI
│
├── components/
│   ├── datagen/                # SIM1-DataGen core (splitter, selector, diffusion)
│   │   ├── datagen_core.py     # DataGenerator class
│   │   ├── splitter.py         # Trajectory splitter
│   │   ├── selector.py         # Segment selector
│   │   ├── traj_df/            # Diffusion model for trajectory generation
│   │   └── configs/            # Task split configurations
│   ├── function/               # Utility functions (FK, IK, video, analysis)
│   ├── randomization/          # Environment randomization
│   ├── recorder/               # Dual-arm data recorder
│   ├── render/                 # USD → Blender → MeisterRender (git submodule) pipeline
│   └── lmdb2lerobot/           # LMDB → LeRobot v2 (+ sim2real + remove_static_frames by default)
│
├── scripts/                    # Post-processing scripts
│   ├── smooth_trajectory_multi_thread.py   # Kalman smooth (used by run_pipeline.sh)
│   ├── filter_joint_unreachable.py         # Joint jump + optional EE reachability (see --no-check-ee)
│   ├── filter_cloth_quality.py             # Cloth-quality filter (used by run_pipeline.sh)
│   └── convert_ee_quat.py                  # EE pose conversion (used by datagen)
│
├── module_train/               # Training modules
│   ├── trajectory_discriminator/
│   └── trajectory_generator/
│
└── dataset/                    # Example datasets (npz, segments, etc.)
```

---

## TODO List

### Completed

- [x] Simulation assets — Robot URDFs, cloth meshes, render assets on Hugging Face ([Sim1 assets](https://huggingface.co/InternRobotics/Sim1_Assets)); see [Download assets](#step-4--download-assets) and `download_assets.sh`.
- [x] Public datasets — Pre-generated trajectories / rendered data ([Sim1 dataset](https://huggingface.co/datasets/InternRobotics/Sim1_Dataset)) and related releases.
- [x] Data generation pipeline — Generate → smooth → replay → filter with `run_pipeline.sh`; optional `--position-randomize` ([Quick Start — Data Generation](#quick-start--data-generation)).
- [x] Training utilities — Policy / trajectory code under `module_train/` (e.g. discriminator, generator).

### Planned

- [ ] Upgrade to latest Newton — Bump bundled `newton/` to upstream; adapt API changes in envs/tasks/components.
- [ ] Integrate libuipc solver — Optional [libuipc](https://github.com/libuipc/libuipc) cloth/deformable backend for richer contact and friction.

---

## Citation

If you use Sim1 (code, assets, or datasets) in research, please cite the paper below. Code: [github.com/InternRobotics/SIM1](https://github.com/InternRobotics/SIM1). Project page: [internrobotics.github.io/sim1.github.io](https://internrobotics.github.io/sim1.github.io/).

```bibtex
@article{sim1_2026,
  title   = {{SIM1}: Physics-Aligned Simulator as Zero-Shot Data Scaler in Deformable Worlds},
  author  = {Yunsong Zhou and Hangxu Liu and Xuekun Jiang and Xing Shen and Xingyi Liu and Yuanzhen Zhou and Hui Wang and Baole Fang and Yang Tian and Zihan Zhang and Ziqi Fan and Mulin Yu and Qiaojun Yu and Li Ma and Hengjie Li and Hanqing Wang and Jia Zeng and Jiangmiao Pang},
  year    = {2026},
  note    = {https://internrobotics.github.io/sim1.github.io/},
}
```

---

## License

Unless otherwise noted, all resources and code in this repository are licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). Language data is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). [Newton](https://github.com/newton-physics/newton) and other third-party components follow their respective distribution licenses; see e.g. [newton/LICENSE.md](newton/LICENSE.md).
