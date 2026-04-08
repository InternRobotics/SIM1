# SIM1 rendering (USD → Blender → photorealistic data)

Turns replay output (**`npz/` + `usd/`** under a session folder) into blended scenes and, optionally, **path-traced MP4s + LMDB** for training.

**Submodule:** `components/render/MeisterRender` tracks the **`main`** branch of [InternRobotics/SIM1MeisterRender](https://github.com/InternRobotics/SIM1MeisterRender) (see [.gitmodules](../../.gitmodules) at repo root).

---

## Minimal run (Steps 1–3, CPU)

From the **Sim1 repo root** (after [Installation](../../README.md#installation) and `bash download_assets.sh`):

```bash
conda activate sim1
python components/render/main.py --root_dir ./replay/<your_session>
```

Use the folder that contains `npz/` and `usd/` (e.g. `replay/pipeline_output_0001`).

---

## Step 4 (GPU, MeisterRender)

**1. Get MeisterRender** (if `components/render/MeisterRender` is missing):

```bash
cd components/render
git clone https://github.com/InternRobotics/SIM1MeisterRender.git MeisterRender
cd ../..
```

Or from repo root: `git submodule update --init --recursive` (see [.gitmodules](../../.gitmodules)).

**2. Render** (after Steps 1–3 produced `blend_out/` etc.):

```bash
bash components/render/batch_step4.sh ./replay/<your_session> "Fold the shirt"
```

To run Step 4 inline with `main.py`, use `--step4` (see `python components/render/main.py --help`).

---

## Multi-GPU (optional)

```bash
bash components/render/main_parallel.sh ./replay/<your_session> <num_gpus>
# Step 4 in parallel:
bash components/render/batch_step4_parallel.sh ./replay/<your_session> "Fold the shirt" <num_gpus> 0
```

Logs: `/tmp/sim1_worker_*.log`, `/tmp/sim1_step4_worker_*.log`.

---

## Assets

Robot URDF plus scene assets (HDRI, glTF tables/materials) are resolved from **`SIM1_ASSETS_ROOT`** (default `<repo>/assets/`). Step 3 uses **`random/{bg,table,mat}/`** under that root by default — see **`scripts/sim1_asset_paths.py`**. Set **`SIM1_BG_ROOT`**, **`SIM1_TABLE_ROOT`**, or **`SIM1_MAT_ROOT`** only if you need non-default paths.

---

## Pre-render filtering (optional)

Same as the main repo: `scripts/filter_cloth_quality.py`, and with replay randomization also `scripts/filter_joint_unreachable.py`. See [Data Generation](../../README.md#quick-start--data-generation) in the root README.

---

## More options

```bash
python components/render/main.py --help
```

Examples: `--shard_id` / `--num_shards`, `--step3 no_random`, `--language_instruction "..."`.
