# lmdb2lerobot

Convert simulation Step-4 output (**`lmdb/` + `meta_info.pkl`**) into a **LeRobot v2** dataset.

---

## One-shot workflow

`run_local.sh` runs **three** stages by default:

1. **LMDB → LeRobot** (`lmdb2lerobot_arx_sim.py`)
2. **sim2real** parquet post-process (`sim2real.py`) — skip with `--skip-sim2real`
3. **Remove near-static frames** (`remove_static_frames.py`, in-place on the dataset) — skip with `--keep-static-frames`

### 1. Environment (once)

```bash
bash components/lmdb2lerobot/setup_conda_lerobot.sh
conda activate lerobot
```

First run downloads PyTorch and LeRobot; expect on the order of 10–30 minutes.

### 2. Convert one directory

From the **repository root** (edit paths to your session):

```bash
conda activate lerobot
bash components/lmdb2lerobot/run_local.sh \
  --src ./replay/<session>/out_updated \
  --out ./replay/<session>/lerobot_dataset
```

- **`--src`**: full `out_updated` tree, or a single episode folder that contains `lmdb/`.
- **`--out`**: LeRobot dataset root; removed and recreated if it already exists.
- **Static frames**: removed automatically after step 2 (tune with `--static-threshold-ratio`, `--static-workers`; use `--keep-static-frames` to keep all frames).

More flags: `bash components/lmdb2lerobot/run_local.sh --help`

---

## Manual — remove_static_frames only

If you already have a LeRobot dataset and want to trim static frames **without** re-running the full script (or to re-run with different thresholds):

```bash
conda activate lerobot
python components/lmdb2lerobot/remove_static_frames.py /path/to/lerobot_dataset
```

**Back up the dataset first** — this edits parquet, videos, and `meta/` in place.

---

## Optional — batch conversion

```bash
conda activate lerobot
bash components/lmdb2lerobot/run_batch.sh --scan /path/to/parent --out-root /path/to/lerobot_outputs --workers 4
```

Details: `bash components/lmdb2lerobot/run_batch.sh --help`

---

## Input layouts (auto-detected)

```
out_updated/
  000000/lmdb/data.mdb
  000001/lmdb/data.mdb
  ...

# or single-episode folder with lmdb/
# or legacy .../out/<timestamp>/lmdb/data.mdb
```

---

## Output layout (summary)

```
lerobot_dataset/
  data/chunk-000/episode_*.parquet
  videos/chunk-000/images.rgb.{head,hand_left,hand_right}/episode_*.mp4
  meta/info.json, episodes.jsonl, tasks.jsonl
```

`observation.state` and `action` are **14-D**: left arm (×6) + left gripper (×1) + right arm (×6) + right gripper (×1).
