# SIM1
# Sim1

Dual-arm cloth manipulation in simulation: teleoperation, diffusion-based data generation, replay, filtering, and optional photorealistic rendering (built on [Newton](https://newton-physics.github.io/newton/) and NVIDIA [Warp](https://nvidia.github.io/warp/)).

**Resources:** [Code (GitHub)](https://github.com/InternRobotics/SIM1) · [Paper (arXiv)](https://arxiv.org/) · [Sim1 assets (Hugging Face)](https://huggingface.co/InternRobotics/Sim1_Assets) · [Sim1 dataset (Hugging Face)](https://huggingface.co/datasets/InternRobotics/Sim1_Dataset) · [Video (YouTube)](https://youtu.be/tsPLa-1Lygw)

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

Use **Python 3.11** with **conda** (environment name `sim1`) and **CUDA toolkit ≥ 11.8** if you want GPU acceleration.

> **Reference**: [Newton Installation Guide](https://newton-physics.github.io/newton/0.2.2/guide/installation.html#method-3-manual-setup-using-pip-in-a-virtual-environment)

---

### Step 1 — Create the conda environment

```bash
conda create -n sim1 python=3.11 -y
conda activate sim1
```

---

### Step 2 — Clone the repository

Clone **with submodules** so **`components/render/MeisterRender`** ([SIM1MeisterRender](https://github.com/InternRobotics/SIM1MeisterRender), `main` branch) is populated automatically:

```bash
git clone --recurse-submodules https://github.com/InternRobotics/SIM1.git sim1
cd sim1
```

If you already cloned without submodules, fetch them once:

```bash
cd sim1
git submodule update --init --recursive
```

---

### Step 3 — Install dependencies

With `sim1` active, from the repository root:

```bash
conda activate sim1
bash setup.sh
```

**All Python dependencies** (simulation, DataGen, asset download helpers, optional full render stack, and post-install checks) are installed by **[`setup.sh`](setup.sh)** only. Open that file for the full list, optional environment variables (**`SIM1_SKIP_RENDER`**, **`TORCH_INDEX_URL`**), and the exact `pip` commands. For **which render step uses which package**, see [`components/render/README.md`](components/render/README.md).

---

### Step 4 — Download assets (required before data generation)

Simulation and rendering expect a fixed layout under the **repository root** `./assets/`. The project does **not** ask you to configure asset paths for data generation: `envs/lift2_short_shirt.py` (and related code) resolves `assets/acone/`, `assets/cloth/`, etc. relative to the repo root.

Download the official bundle from Hugging Face (**`InternRobotics/Sim1_Assets`**) into `./assets/`:

```bash
# From the repository root (after setup.sh)
bash download_assets.sh
```

Use the **default** destination (`./assets` at the repo root) so it matches the hard-coded paths in `envs/`. A custom directory from `bash download_assets.sh /other/path` will **not** be picked up unless you symlink it to `./assets` or change the code.

> **Note:** With **`--position-randomize`**, the EE reachability step reads an extra URDF via `newton.examples.get_asset(...)` under `newton/newton/examples/assets/`. That tree is **not** filled by `download_assets.sh`. If Step 4 fails, install Newton example assets there, run without **`--position-randomize`**, or use **`--skip_filter`** (not recommended for training).

---

### Verify installation

```bash
conda activate sim1
python -c "import newton; print('Newton version:', newton.__version__)"
python -c "import warp as wp; print('Warp OK')"
python -c "import torch, torchvision; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

After **`bash download_assets.sh`**, you should see at least `assets/acone/acone.urdf` and `assets/cloth/short-shirt.usdc` before running **`run_pipeline.sh`**.

---

## Quick Start — Interactive Teleoperation

Launch a real-time interactive simulation with keyboard-driven dual-arm control:

```bash
python apps/teleoperation_app.py --task lift_manip_shirt
```

### Keyboard Controls

| Key | Action |
|---|---|
| W / S | Left arm forward / back |
| A / D | Left arm left / right |
| Q / E | Left arm down / up |
| X | Toggle left gripper |
| I / K | Right arm forward / back |
| J / L | Right arm left / right |
| U / O | Right arm down / up |
| M | Toggle right gripper |
| Arrow keys | Move camera |
| Mouse drag | Look around |
| Scroll | Zoom |
| Space | Pause / resume |
| R | Full reset (restarts process) |
| ESC | Quit |

### WebSocket Streaming (Remote Teleoperation)

For headless or remote machines, enable WebSocket streaming:

```bash
python apps/teleoperation_app.py --task lift_manip_shirt --stream --host 0.0.0.0 --ws-port 8765 --http-port 8080
```

Then open `http://<server-ip>:8080` in a browser to view and control the simulation remotely.

---

## Quick Start — Data Generation

`run_pipeline.sh` runs the full data path in one shot: **generate → Kalman smooth → replay (NPZ + USD) → filter**. Generation uses the diffusion-policy path; assets come from **`./assets/`** after **`bash download_assets.sh`** (see [Step 4 — Download assets](#step-4--download-assets)). For extra diversity at replay, add **`--position-randomize`**.

### Generate data

From the **repository root**, after [Installation](#installation):

```bash
conda activate sim1
cd /path/to/sim1
bash download_assets.sh    # once per machine / fresh clone
bash run_pipeline.sh --num 10
```

Optional: `bash run_pipeline.sh --num 10 --position-randomize`

You get trajectories under **`./dataset/example/`** and a replay session **`replay/pipeline_output_XXXX/`** (the script prints the path). Use that folder as **`--root_dir`** for [Rendering Pipeline](#rendering-pipeline).

<details>
<summary>All <code>run_pipeline.sh</code> options (advanced)</summary>

| Option | Description | Default |
|---|---|---|
| `--data_folder DIR` | Data root | `./dataset/example` |
| `--num N` | Trajectories to generate (DP pipeline only) | 10 |
| `--workers N` | Parallel workers (smooth + filters) | 8 |
| `--skip_smooth` / `--skip_replay` / `--skip_filter` | Skip a stage | off |
| `--folder_name NAME` | `replay/<NAME>_XXXX/` base name | `pipeline_output` |
| `--position-randomize` | Random cloth pose at replay; **auto-selects** EE + aligned cloth filters. Omit → **standard** cloth-quality filter only | off |
| `--ref_usd PATH` | Reference USD for aligned cloth filter (with randomization); auto-picked if omitted | auto |
| `--skip_asset_check` | Do not verify `./assets/` before running | off |

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
└─ 4. Filter      →  No flag: filter_cloth_quality.py (direct)
                     With --position-randomize: filter_joint_unreachable.py → filter_cloth_quality.py --ref-usd
```

### Session layout

```
replay/
└── pipeline_output_0001/
    ├── npz/
    ├── usd/
    ├── npz_bad_cloth/
    ├── usd_bad_cloth/
    ├── npz_unreachable/          # only with --position-randomize
    └── cloth_filter_summary.txt
```

<details>
<summary>Manual step-by-step (only if you are not using <code>run_pipeline.sh</code>)</summary>

**1. Generate:** `python apps/datagen_app.py --data_folder ./dataset/example --num 10 --use_dp --mode fine`  
**2. Smooth:** `python scripts/smooth_trajectory_multi_thread.py ./dataset/example/gen ./dataset/example/gen/kf --method kalman --workers 8`  
**3. Replay:** `python apps/replay_app.py ./dataset/example/gen/kf --folder_name my_replay`  
Optional cloth position randomization at replay: add `--position-randomize` (then use the matching manual filters as in Step 4 above).  
**4a. Reachability (only with randomization):** `python scripts/filter_joint_unreachable.py ./replay/my_replay_0001/npz --usd-dir ./replay/my_replay_0001/usd --workers 8`  
**4b. Cloth quality:** `python scripts/filter_cloth_quality.py ./replay/my_replay_0001` (add `--ref-usd ...` if you used randomization)

</details>

---

## Rendering Pipeline

Convert simulation USD output to photorealistic data: **`main.py` runs Steps 1–3 by default** (USD → blend → cameras → `blend_out/`). **Step 4** (MeisterRender path tracing + LMDB) writes under **`out_updated/<record_id>/`**; run it via `batch_step4.sh`, or inline with `main.py --step4`.

**MeisterRender** lives in the **git submodule** `components/render/MeisterRender` ([InternRobotics/SIM1MeisterRender](https://github.com/InternRobotics/SIM1MeisterRender), `main`). Use **`git clone --recurse-submodules`** in [Step 2](#step-2--clone-the-repository) so it is checked out automatically.

**Environment:** use the same **`sim1`** env; the render stack is installed by **`setup.sh`** unless you set **`SIM1_SKIP_RENDER=1`** (see comments in `setup.sh`). Package-to-step mapping: [`components/render/README.md`](components/render/README.md).

```bash
conda activate sim1
python components/render/main.py --root_dir ./replay/my_run_0001
bash components/render/batch_step4.sh ./replay/my_run_0001

# Optional: Step 4 inside main after each Step 3
# python components/render/main.py --root_dir ./replay/my_run_0001 --step4
```

### Asset Configuration

Rendering assets (backgrounds, tables, materials) default to **`assets/render/`** under the repo root (from `download_assets.sh`). Override with paths **relative to the repo root**, for example:

```bash
export SIM1_BG_ROOT=./assets/random/bg
export SIM1_TABLE_ROOT=./assets/random/table
export SIM1_MAT_ROOT=./assets/random/mat
```

---

## Data Conversion

After Step 4 rendering, trajectories are stored under **`replay/<session>/out_updated/<record_id>/`** as LMDB + `meta_info.pkl`. To convert them into a **LeRobot v2** dataset for training, use **`components/lmdb2lerobot/`**.

**One-time environment** (separate conda env `lerobot`, Python 3.12 — see full docs for details):

```bash
bash components/lmdb2lerobot/setup_conda_lerobot.sh
conda activate lerobot
```

**Single session → LeRobot dataset:**

```bash
bash components/lmdb2lerobot/run_local.sh \
  --src ./replay/my_session/out_updated \
  --out ./replay/my_session/lerobot_dataset
```

**Batch / multi-GPU** (optional): `components/lmdb2lerobot/run_batch.sh` — see [`components/lmdb2lerobot/README.md`](components/lmdb2lerobot/README.md).

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
│   └── lmdb2lerobot/           # LMDB → LeRobot v2 (+ sim2real); remove_static_frames.py (trim static frames)
│
├── scripts/                    # Post-processing scripts
│   ├── smooth_trajectory_multi_thread.py   # Kalman smooth (used by run_pipeline.sh)
│   ├── filter_joint_unreachable.py         # EE reachability (with --position-randomize)
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

- [x] **Simulation assets** — Robot URDFs, cloth meshes, render assets on Hugging Face ([Sim1 assets](https://huggingface.co/InternRobotics/Sim1_Assets)); see [Download assets](#step-4--download-assets) and `download_assets.sh`.
- [x] **Public datasets** — Pre-generated trajectories / rendered data ([Sim1 dataset](https://huggingface.co/datasets/InternRobotics/Sim1_Dataset)) and related releases.
- [x] **Data generation pipeline** — Generate → smooth → replay → filter with `run_pipeline.sh`; optional `--position-randomize` ([Quick Start — Data Generation](#quick-start--data-generation)).
- [x] **Training utilities** — Policy / trajectory code under `module_train/` (e.g. discriminator, generator).

### Planned

- [ ] **Upgrade to latest Newton** — Bump bundled `newton/` to upstream; adapt API changes in envs/tasks/components.
- [ ] **Integrate libuipc solver** — Optional [libuipc](https://github.com/libuipc/libuipc) cloth/deformable backend for richer contact and friction.

---

## Citation

If you use **Sim1** (code, assets, or datasets) in research, please cite the paper below. **Code:** [github.com/InternRobotics/SIM1](https://github.com/InternRobotics/SIM1). **Project page:** [internrobotics.github.io/sim1.github.io](https://internrobotics.github.io/sim1.github.io/).

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

Unless otherwise noted, all resources and code in this repository are licensed under the **[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)**. **Language data** is licensed under **[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)**. **[Newton](https://github.com/newton-physics/newton)** and other third-party components follow their respective distribution licenses; see e.g. [newton/LICENSE.md](newton/LICENSE.md).
