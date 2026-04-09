#!/usr/bin/env python3
"""
Remove near-static frames from a LeRobot v2-style dataset.

Uses consecutive differences in ``observation.state`` (L2 norm). Adjacent pairs whose
change falls below ``threshold_ratio * (max_delta - min_delta)`` drop the *later* frame.
Matching frames are removed from MP4s under ``videos/``; parquet row indices,
timestamps, and ``meta/`` summaries are updated in place.
"""

import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
import cv2
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict


def calculate_state_change(state1, state2):
    """L2 distance between two state vectors."""
    s1 = np.asarray(state1, dtype=float)
    s2 = np.asarray(state2, dtype=float)
    return np.linalg.norm(s2 - s1)


def find_static_frames(state_sequence, threshold_ratio=0.01):
    """
    Return 0-based frame indices to remove (the *later* frame of each static pair).

    ``threshold_ratio`` scales the span of per-step deltas:
    ``threshold = threshold_ratio * (max(delta) - min(delta))``.
    """
    if len(state_sequence) <= 1:
        return set()

    changes = []
    for i in range(len(state_sequence) - 1):
        change = calculate_state_change(state_sequence[i], state_sequence[i + 1])
        changes.append(change)

    if not changes:
        return set()

    changes_array = np.array(changes)
    total_range = changes_array.max() - changes_array.min()

    if total_range < 1e-10:
        return set()

    threshold = total_range * threshold_ratio

    # If delta between frame i and i+1 is below threshold, drop frame i+1 (keep i).
    frames_to_delete = set()
    for i, change in enumerate(changes):
        if change < threshold:
            frames_to_delete.add(i + 1)  # drop the later frame
    
    return frames_to_delete


def remove_frames_from_parquet(parquet_path, frames_to_delete, global_index_start=None):
    """
    Drop rows from a parquet episode file; optionally rewrite global ``index`` column.

    Returns the number of rows removed.
    """
    if not frames_to_delete:
        return 0
    
    table = pq.read_table(parquet_path)
    num_rows = table.num_rows
    
    keep_mask = np.ones(num_rows, dtype=bool)
    for idx in frames_to_delete:
        if 0 <= idx < num_rows:
            keep_mask[idx] = False

    indices = np.where(keep_mask)[0]
    if len(indices) == num_rows:
        return 0

    filtered_table = table.take(pa.array(indices))

    # Renumber ``frame_index`` (per-episode); optionally ``index`` (dataset-global).
    # Recompute ``timestamp`` from inferred FPS so rows stay evenly spaced in time.
    new_num_rows = filtered_table.num_rows
    
    fps = None
    if 'timestamp' in table.column_names and table.num_rows > 1:
        original_timestamps = table['timestamp'].to_pylist()
        if len(original_timestamps) > 1:
            time_diffs = [
                original_timestamps[i + 1] - original_timestamps[i]
                for i in range(len(original_timestamps) - 1)
                if original_timestamps[i + 1] > original_timestamps[i]
            ]
            if time_diffs:
                avg_interval = sum(time_diffs) / len(time_diffs)
                if avg_interval > 0:
                    fps = 1.0 / avg_interval

    if fps is None or fps <= 0:
        fps = 30.0

    arrays = []
    for field in filtered_table.schema:
        col_name = field.name
        if col_name == 'frame_index':
            new_indices = pa.array(range(new_num_rows), type=field.type)
            arrays.append(new_indices)
        elif col_name == 'index' and global_index_start is not None:
            new_indices = pa.array(
                range(global_index_start, global_index_start + new_num_rows), type=field.type
            )
            arrays.append(new_indices)
        elif col_name == 'timestamp':
            new_timestamps = [i / fps for i in range(new_num_rows)]
            new_timestamps_array = pa.array(new_timestamps, type=field.type)
            arrays.append(new_timestamps_array)
        else:
            arrays.append(filtered_table[col_name])

    filtered_table = pa.Table.from_arrays(arrays, schema=filtered_table.schema)

    temp_path = parquet_path.with_suffix('.parquet.tmp')
    try:
        pq.write_table(
            filtered_table,
            temp_path,
            compression='snappy',
            use_dictionary=True,
            write_statistics=True,
            version='2.6',
        )
        test_table = pq.read_table(temp_path)
        if test_table.num_rows != len(indices):
            raise ValueError("Row count mismatch after filtering")
        temp_path.replace(parquet_path)
        return num_rows - len(indices)
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise e


def remove_frames_from_video(video_path, frames_to_delete):
    """Rewrite ``video_path`` MP4 without the given frame indices. Returns success flag."""
    if not frames_to_delete or not video_path.exists():
        return True
    
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        keep_mask = np.ones(total_frames, dtype=bool)
        for idx in frames_to_delete:
            if 0 <= idx < total_frames:
                keep_mask[idx] = False
        
        temp_path = video_path.with_suffix('.mp4.tmp')
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(str(temp_path), fourcc, fps, (width, height))

        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(temp_path), fourcc, fps, (width, height))
        
        if not out.isOpened():
            cap.release()
            return False
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if keep_mask[frame_idx]:
                out.write(frame)
            
            frame_idx += 1
        
        cap.release()
        out.release()
        
        temp_path.replace(video_path)
        return True
        
    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        if 'temp_path' in locals() and temp_path.exists():
            temp_path.unlink()
        return False


def write_table_atomic(parquet_path, table):
    """Write parquet via temp file and atomic replace."""
    temp_path = parquet_path.with_suffix('.parquet.tmp')
    try:
        pq.write_table(
            table,
            temp_path,
            compression='snappy',
            use_dictionary=True,
            write_statistics=True,
            version='2.6',
        )
        test_table = pq.read_table(temp_path)
        if test_table.num_rows != table.num_rows:
            raise ValueError("Row count mismatch after parquet write")
        temp_path.replace(parquet_path)
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise e


def process_chunk_dataset(data_parquet_dir, videos_dir, meta_dir, threshold_ratio=0.01):
    """
    Process chunk-style LeRobot dataset where frames are stored in file-XXX.parquet.
    Returns episode_results list compatible with existing summary code.
    """
    parquet_files = sorted(data_parquet_dir.rglob('*.parquet'))
    if not parquet_files:
        print(f"Warning: no parquet files under {data_parquet_dir}")
        return []

    episode_rows = defaultdict(list)
    for pf in parquet_files:
        table = pq.read_table(pf)
        required = {'episode_index', 'observation.state'}
        if not required.issubset(set(table.column_names)):
            continue

        episode_col = table['episode_index'].to_pylist()
        state_col = table['observation.state'].to_pylist()
        if 'frame_index' in table.column_names:
            frame_col = table['frame_index'].to_pylist()
        else:
            frame_col = list(range(table.num_rows))
        if 'index' in table.column_names:
            index_col = table['index'].to_pylist()
        else:
            index_col = list(range(table.num_rows))

        for row_idx in range(table.num_rows):
            ep = int(episode_col[row_idx])
            episode_rows[ep].append(
                (int(frame_col[row_idx]), int(index_col[row_idx]), pf, row_idx, state_col[row_idx])
            )

    episode_indices = sorted(episode_rows.keys())
    print(f"Found {len(episode_indices)} episodes (chunk layout)")
    if not episode_indices:
        return []

    print(f"Dataset root: {data_parquet_dir.parent}")
    print(f"threshold_ratio: {threshold_ratio}")
    print("-" * 50)

    delete_map = defaultdict(set)  # parquet_path -> set(row_idx)
    episode_results = []
    for ep_idx in episode_indices:
        rows = episode_rows[ep_idx]
        rows.sort(key=lambda x: (x[0], x[1]))
        states = [r[4] for r in rows]
        frames_to_delete = find_static_frames(states, threshold_ratio)

        if not frames_to_delete:
            episode_results.append(
                {
                    'episode_index': ep_idx,
                    'status': 'no_static_frames',
                    'deleted_frames': 0,
                    'original_length': len(states),
                    'new_length': len(states),
                }
            )
            continue

        for seq_pos in frames_to_delete:
            _, _, pf, row_idx, _ = rows[seq_pos]
            delete_map[pf].add(row_idx)

        episode_results.append(
            {
                'episode_index': ep_idx,
                'status': 'success',
                'deleted_frames': len(frames_to_delete),
                'original_length': len(states),
                'new_length': len(states) - len(frames_to_delete),
            }
        )

    # 1) Rewrite data parquet files
    for pf in tqdm(sorted(delete_map.keys()), desc="Rewrite chunk parquet"):
        rows_to_delete = delete_map[pf]
        if not rows_to_delete:
            continue
        table = pq.read_table(pf)
        keep_mask = np.ones(table.num_rows, dtype=bool)
        for idx in rows_to_delete:
            if 0 <= idx < table.num_rows:
                keep_mask[idx] = False
        kept = np.where(keep_mask)[0]
        if len(kept) == table.num_rows:
            continue
        new_table = table.take(pa.array(kept))
        write_table_atomic(pf, new_table)

    # 2) Rewrite corresponding chunk videos (same row indices per chunk file)
    video_keys = ['images.rgb.head', 'images.rgb.hand_left', 'images.rgb.hand_right']
    for pf in tqdm(sorted(delete_map.keys()), desc="Rewrite chunk videos"):
        rows_to_delete = delete_map[pf]
        if not rows_to_delete:
            continue
        rel = pf.relative_to(data_parquet_dir).with_suffix('.mp4')
        for video_key in video_keys:
            video_path = videos_dir / video_key / rel
            if video_path.exists():
                remove_frames_from_video(video_path, rows_to_delete)

    # 3) Rebuild index/frame_index/timestamp across chunk files
    fps = 30.0
    info_file = meta_dir / 'info.json'
    old_total_frames = None
    if info_file.exists():
        try:
            with open(info_file, 'r') as f:
                info = json.load(f)
            fps = float(info.get('fps', fps))
            old_total_frames = info.get('total_frames')
        except Exception:
            pass

    global_index = 0
    ep_counters = defaultdict(int)
    parquet_files = sorted(data_parquet_dir.rglob('*.parquet'))
    for pf in tqdm(parquet_files, desc="Reindex parquet"):
        table = pq.read_table(pf)
        if table.num_rows == 0:
            continue

        ep_col = table['episode_index'].to_pylist() if 'episode_index' in table.column_names else None
        new_frame_index = None
        new_timestamp = None
        if ep_col is not None:
            new_frame_index = []
            for ep in ep_col:
                ep = int(ep)
                fid = ep_counters[ep]
                new_frame_index.append(fid)
                ep_counters[ep] = fid + 1
            new_timestamp = [fid / fps for fid in new_frame_index]

        arrays = []
        for field in table.schema:
            col_name = field.name
            if col_name == 'index':
                arrays.append(pa.array(range(global_index, global_index + table.num_rows), type=field.type))
            elif col_name == 'frame_index' and new_frame_index is not None:
                arrays.append(pa.array(new_frame_index, type=field.type))
            elif col_name == 'timestamp' and new_timestamp is not None:
                arrays.append(pa.array(new_timestamp, type=field.type))
            else:
                arrays.append(table[col_name])
        new_table = pa.Table.from_arrays(arrays, schema=table.schema)
        write_table_atomic(pf, new_table)
        global_index += table.num_rows

    # 4) Update meta/info.json and meta/episodes parquet basic fields
    episode_lengths = dict(ep_counters)
    total_frames = sum(episode_lengths.values())

    if info_file.exists():
        with open(info_file, 'r') as f:
            info = json.load(f)
        info['total_frames'] = total_frames
        info['total_episodes'] = len(episode_lengths)
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=4)

    episodes_parquets = sorted((meta_dir / 'episodes').rglob('*.parquet')) if (meta_dir / 'episodes').exists() else []
    if episodes_parquets:
        episode_order = sorted(episode_lengths.keys())
        start_map = {}
        end_map = {}
        running = 0
        for ep in episode_order:
            start_map[ep] = running
            running += episode_lengths[ep]
            end_map[ep] = running

        for ep_pf in episodes_parquets:
            table = pq.read_table(ep_pf)
            if 'episode_index' not in table.column_names:
                continue
            ep_ids = [int(x) for x in table['episode_index'].to_pylist()]
            old_lengths = table['length'].to_pylist() if 'length' in table.column_names else [None] * len(ep_ids)

            arrays = []
            for field in table.schema:
                name = field.name
                if name == 'length':
                    vals = [int(episode_lengths.get(ep, old_lengths[i] if old_lengths[i] is not None else 0)) for i, ep in enumerate(ep_ids)]
                    arrays.append(pa.array(vals, type=field.type))
                elif name == 'dataset_from_index':
                    vals = [int(start_map.get(ep, 0)) for ep in ep_ids]
                    arrays.append(pa.array(vals, type=field.type))
                elif name == 'dataset_to_index':
                    vals = [int(end_map.get(ep, 0)) for ep in ep_ids]
                    arrays.append(pa.array(vals, type=field.type))
                elif name.endswith('/from_timestamp'):
                    vals = [float(start_map.get(ep, 0) / fps) for ep in ep_ids]
                    arrays.append(pa.array(vals, type=field.type))
                elif name.endswith('/to_timestamp'):
                    vals = [float(end_map.get(ep, 0) / fps) for ep in ep_ids]
                    arrays.append(pa.array(vals, type=field.type))
                elif name.startswith('stats/') and name.endswith('/count'):
                    col = table[name].to_pylist()
                    vals = []
                    for i, ep in enumerate(ep_ids):
                        old_v = col[i]
                        old_len = old_lengths[i]
                        new_len = int(episode_lengths.get(ep, old_len if old_len is not None else 0))
                        if isinstance(old_v, list) and len(old_v) > 0 and old_len is not None and old_v[0] == old_len:
                            old_v = list(old_v)
                            old_v[0] = new_len
                        vals.append(old_v)
                    arrays.append(pa.array(vals, type=field.type))
                else:
                    arrays.append(table[name])
            new_table = pa.Table.from_arrays(arrays, schema=table.schema)
            write_table_atomic(ep_pf, new_table)

    # Optional: keep stats.json total count in sync when possible
    stats_file = meta_dir / 'stats.json'
    if stats_file.exists():
        try:
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            if old_total_frames is not None:
                for _, v in stats.items():
                    if isinstance(v, dict) and 'count' in v:
                        c = v['count']
                        if isinstance(c, list) and len(c) > 0 and c[0] == old_total_frames:
                            c[0] = total_frames
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=4)
        except Exception as e:
            print(f"Warning: failed to update stats.json count fields: {e}")

    return episode_results


def process_episode(data_dir_str, videos_dir_str, episode_index, threshold_ratio=0.01):
    """Process one episode (paths as str for multiprocessing pickle). Returns a status dict."""
    data_dir = Path(data_dir_str)
    videos_dir = Path(videos_dir_str)
    
    parquet_files = list(data_dir.rglob(f'episode_{episode_index:06d}.parquet'))
    if not parquet_files:
        return {'episode_index': episode_index, 'status': 'not_found', 'deleted_frames': 0}
    
    parquet_path = parquet_files[0]
    
    try:
        table = pq.read_table(parquet_path)
        
        if 'observation.state' not in table.column_names:
            return {'episode_index': episode_index, 'status': 'no_state_col', 'deleted_frames': 0}
        
        state_sequence = table['observation.state'].to_pylist()

        frames_to_delete = find_static_frames(state_sequence, threshold_ratio)
        
        if not frames_to_delete:
            return {'episode_index': episode_index, 'status': 'no_static_frames', 'deleted_frames': 0}
        
        deleted_count = remove_frames_from_parquet(
            parquet_path, frames_to_delete, global_index_start=None
        )

        video_keys = ['images.rgb.head', 'images.rgb.hand_left', 'images.rgb.hand_right']
        video_success = True
        ep_name = f'episode_{episode_index:06d}.mp4'
        for video_key in video_keys:
            direct = videos_dir / video_key / ep_name
            if direct.exists():
                video_path = direct
            else:
                found = list(videos_dir.rglob(f'{video_key}/{ep_name}'))
                video_path = found[0] if found else None
            if video_path is not None:
                if not remove_frames_from_video(video_path, frames_to_delete):
                    video_success = False
        
        return {
            'episode_index': episode_index,
            'status': 'success' if video_success else 'video_error',
            'deleted_frames': deleted_count,
            'original_length': len(state_sequence),
            'new_length': len(state_sequence) - deleted_count
        }
        
    except Exception as e:
        return {'episode_index': episode_index, 'status': 'error', 'error': str(e), 'deleted_frames': 0}


def update_meta_files(meta_dir, episode_results):
    """Refresh ``episodes.jsonl``, ``episodes_stats.jsonl`` (if present), and ``info.json``."""
    meta_dir = Path(meta_dir)
    
    results_map = {r['episode_index']: r for r in episode_results if 'new_length' in r}

    episodes_file = meta_dir / 'episodes.jsonl'
    episodes = []
    total_frames = 0
    
    if episodes_file.exists():
        with open(episodes_file, 'r') as f:
            for line in f:
                if line.strip():
                    episodes.append(json.loads(line.strip()))
        
        for ep in episodes:
            ep_idx = ep['episode_index']
            if ep_idx in results_map:
                new_length = results_map[ep_idx]['new_length']
                ep['length'] = new_length
            total_frames += ep['length']
    else:
        for r in episode_results:
            if 'new_length' in r:
                episodes.append({
                    'episode_index': r['episode_index'],
                    'tasks': ['Dex_new'],
                    'length': r['new_length']
                })
                total_frames += r['new_length']
        
        episodes.sort(key=lambda x: x['episode_index'])

    with open(episodes_file, 'w') as f:
        for ep in episodes:
            f.write(json.dumps(ep) + '\n')
    
    stats_file = meta_dir / 'episodes_stats.jsonl'
    if stats_file.exists():
        stats = []
        with open(stats_file, 'r') as f:
            for line in f:
                if line.strip():
                    stats.append(json.loads(line.strip()))
        
        for stat in stats:
            ep_idx = stat.get('episode_index')
            if ep_idx is not None and ep_idx in results_map:
                new_length = results_map[ep_idx]['new_length']
                
                if 'stats' in stat:
                    for stat_key, stat_value in stat['stats'].items():
                        if isinstance(stat_value, dict) and 'count' in stat_value:
                            if isinstance(stat_value['count'], list) and len(stat_value['count']) > 0:
                                stat_value['count'][0] = new_length
                            else:
                                stat_value['count'] = [new_length]

                if 'length' in stat:
                    stat['length'] = new_length
        
        with open(stats_file, 'w') as f:
            for stat in stats:
                f.write(json.dumps(stat) + '\n')
    
    info_file = meta_dir / 'info.json'
    if info_file.exists():
        with open(info_file, 'r') as f:
            info = json.load(f)
        
        info['total_frames'] = total_frames
        info['total_episodes'] = len(episodes)
        
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=4)
    
    return total_frames


def update_global_indices(data_dir_str, episode_results):
    """Reassign contiguous global ``index`` across all episode parquet files."""
    data_dir = Path(data_dir_str)
    
    results_map = {}
    for r in episode_results:
        ep_idx = r['episode_index']
        if 'new_length' in r:
            results_map[ep_idx] = r['new_length']
        elif r['status'] in ['no_static_frames', 'not_found', 'no_state_col']:
            parquet_files = list(data_dir.rglob(f'episode_{ep_idx:06d}.parquet'))
            if parquet_files:
                try:
                    table = pq.read_table(parquet_files[0])
                    results_map[ep_idx] = table.num_rows
                except Exception:
                    pass

    all_parquet_files = sorted(data_dir.rglob('episode_*.parquet'))
    for pf in all_parquet_files:
        name = pf.stem
        if name.startswith('episode_'):
            try:
                ep_idx = int(name.split('_')[1])
                if ep_idx not in results_map:
                    try:
                        table = pq.read_table(pf)
                        results_map[ep_idx] = table.num_rows
                    except Exception:
                        pass
            except Exception:
                pass

    sorted_episodes = sorted(results_map.keys())

    global_index_start = 0
    for ep_idx in sorted_episodes:
        new_length = results_map[ep_idx]
        
        parquet_files = list(data_dir.rglob(f'episode_{ep_idx:06d}.parquet'))
        if not parquet_files:
            global_index_start += new_length
            continue
        
        parquet_path = parquet_files[0]
        
        try:
            table = pq.read_table(parquet_path)

            if table.num_rows != new_length:
                print(
                    f"Warning: episode {ep_idx} row count ({table.num_rows}) "
                    f"does not match expected ({new_length})"
                )

            if 'index' not in table.column_names:
                global_index_start += new_length
                continue

            arrays = []
            for field in table.schema:
                col_name = field.name
                if col_name == 'index':
                    new_indices = pa.array(
                        range(global_index_start, global_index_start + table.num_rows), type=field.type
                    )
                    arrays.append(new_indices)
                else:
                    arrays.append(table[col_name])

            updated_table = pa.Table.from_arrays(arrays, schema=table.schema)

            temp_path = parquet_path.with_suffix('.parquet.tmp')
            try:
                pq.write_table(
                    updated_table,
                    temp_path,
                    compression='snappy',
                    use_dictionary=True,
                    write_statistics=True,
                    version='2.6',
                )
                test_table = pq.read_table(temp_path)
                if test_table.num_rows != table.num_rows:
                    raise ValueError("Row count mismatch after updating index")
                temp_path.replace(parquet_path)
            except Exception as e:
                if temp_path.exists():
                    temp_path.unlink()
                raise e
            
            global_index_start += table.num_rows

        except Exception as e:
            print(f"Warning: failed to update global index for episode {ep_idx}: {e}")
            global_index_start += new_length
            continue


def print_episode_frame_counts(episode_results):
    """
    Print per-episode frame counts before/after static-frame removal.
    """
    valid = [
        r for r in episode_results
        if 'episode_index' in r and 'original_length' in r and 'new_length' in r
    ]
    if not valid:
        print("\nFrame counts: no per-episode length data available.")
        return

    valid = sorted(valid, key=lambda x: x['episode_index'])
    print("\nFrame counts per episode (before -> after, removed):")
    total_before = 0
    total_after = 0
    for r in valid:
        ep = int(r['episode_index'])
        before = int(r['original_length'])
        after = int(r['new_length'])
        removed = int(r.get('deleted_frames', max(0, before - after)))
        total_before += before
        total_after += after
        print(f"  episode_{ep:06d}: {before} -> {after} (removed {removed})")

    print(
        f"Total frames: {total_before} -> {total_after} "
        f"(removed {total_before - total_after})"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Remove near-static frames from a LeRobot dataset root "
            "(expects data/, videos/, meta/ subdirectories)."
        )
    )
    parser.add_argument(
        'data_dir',
        type=str,
        help='LeRobot dataset root directory (parent of data/, videos/, meta/).',
    )
    parser.add_argument(
        '--threshold_ratio',
        type=float,
        default=0.00001,
        help=(
            'Fraction of (max adjacent-state delta - min adjacent-state delta) below which '
            'the later frame is dropped. Default: %(default)s.'
        ),
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of parallel worker processes (default: %(default)s). Use 1 to disable.',
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise ValueError(f"Dataset directory does not exist: {data_dir}")

    data_parquet_dir = data_dir / 'data'
    videos_dir = data_dir / 'videos'
    meta_dir = data_dir / 'meta'

    if not data_parquet_dir.exists():
        raise ValueError(f"Missing data/ under dataset root: {data_parquet_dir}")

    parquet_files = sorted(data_parquet_dir.rglob('*.parquet'))
    if not parquet_files:
        print(f"Warning: no parquet files under {data_parquet_dir}")
        return

    episode_indices = set()
    for pf in parquet_files:
        name = pf.stem
        if name.startswith('episode_'):
            try:
                ep_idx = int(name.split('_')[1])
                episode_indices.add(ep_idx)
            except (ValueError, IndexError):
                continue

    episode_indices = sorted(episode_indices)

    if not episode_indices:
        has_chunk_like_parquet = any(pf.stem.startswith('file-') for pf in parquet_files)
        if has_chunk_like_parquet:
            episode_results = process_chunk_dataset(
                data_parquet_dir=data_parquet_dir,
                videos_dir=videos_dir,
                meta_dir=meta_dir,
                threshold_ratio=args.threshold_ratio,
            )
            total_deleted = sum(r.get('deleted_frames', 0) for r in episode_results)
            success_count = sum(
                1 for r in episode_results if r['status'] in ['success', 'no_static_frames']
            )
            print_episode_frame_counts(episode_results)
            print("\nDone.")
            print(f"Successful episodes: {success_count}/{len(episode_results)}")
            print(f"Total frames removed: {total_deleted}")
            return

    print(f"Found {len(episode_indices)} episodes")
    print(f"Dataset root: {data_dir}")
    print(f"threshold_ratio: {args.threshold_ratio}")
    print(f"workers: {args.workers}")
    print("-" * 50)

    episode_results = []

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    process_episode,
                    str(data_parquet_dir),
                    str(videos_dir),
                    ep_idx,
                    args.threshold_ratio,
                ): ep_idx
                for ep_idx in episode_indices
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="Episodes"):
                result = future.result()
                episode_results.append(result)
                if result['status'] == 'error':
                    print(
                        f"\nError episode {result['episode_index']}: "
                        f"{result.get('error', 'Unknown error')}"
                    )
    else:
        for ep_idx in tqdm(episode_indices, desc="Episodes"):
            result = process_episode(
                str(data_parquet_dir), str(videos_dir), ep_idx, args.threshold_ratio
            )
            episode_results.append(result)
            if result['status'] == 'error':
                print(f"\nError episode {ep_idx}: {result.get('error', 'Unknown error')}")

    total_deleted = sum(r.get('deleted_frames', 0) for r in episode_results)
    success_count = sum(1 for r in episode_results if r['status'] == 'success')

    print_episode_frame_counts(episode_results)
    print("\nDone.")
    print(f"Successful episodes: {success_count}/{len(episode_indices)}")
    print(f"Total frames removed: {total_deleted}")

    print("\nUpdating global index column...")
    update_global_indices(str(data_parquet_dir), episode_results)
    print("Global index update complete.")

    if meta_dir.exists():
        print("\nUpdating meta/")
        total_frames = update_meta_files(meta_dir, episode_results)
        print(f"total_frames in meta: {total_frames}")
    else:
        print("\nWarning: meta/ not found; skipped meta updates")


if __name__ == '__main__':
    main()
