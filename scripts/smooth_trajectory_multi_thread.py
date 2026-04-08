#!/usr/bin/env python3
"""
Trajectory smoothing script (multi-process).
Smooth npz trajectory data and reduce hand jitter noise.
"""

import argparse
import numpy as np
from pathlib import Path
from scipy.signal import savgol_filter
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed


def kalman_filter_1d(observations, process_variance=1e-5, measurement_variance=1e-3):
    n = len(observations)
    if n == 0:
        return observations
    filtered = np.zeros(n)
    filtered[0] = observations[0]
    P = 1.0
    for i in range(1, n):
        predicted = filtered[i-1]
        P_predicted = P + process_variance
        K = P_predicted / (P_predicted + measurement_variance)
        filtered[i] = predicted + K * (observations[i] - predicted)
        P = (1 - K) * P_predicted
    return filtered


def smooth_trajectory(data, method='kalman', window_length=None, polyorder=3, 
                     process_variance=1e-5, measurement_variance=1e-3, axis=0):
    n_frames = data.shape[axis]
    if method == 'kalman':
        if data.ndim == 1:
            return kalman_filter_1d(data, process_variance, measurement_variance)
        elif data.ndim == 2:
            smoothed = np.zeros_like(data)
            for i in range(data.shape[1]):
                smoothed[:, i] = kalman_filter_1d(data[:, i], process_variance, measurement_variance)
            return smoothed
        else:
            smoothed = np.zeros_like(data)
            for i in range(data.shape[1]):
                for j in range(data.shape[2]):
                    smoothed[:, i, j] = kalman_filter_1d(data[:, i, j], process_variance, measurement_variance)
            return smoothed
    elif method == 'savgol':
        if window_length is None:
            window_length = max(5, min(51, (n_frames // 20) * 2 + 1))
        else:
            if window_length % 2 == 0:
                window_length += 1
            window_length = min(window_length, n_frames)
        polyorder = min(polyorder, window_length - 1)
        if window_length < 3:
            return data
        if data.ndim == 1:
            return savgol_filter(data, window_length, polyorder, axis=0)
        elif data.ndim == 2:
            smoothed = np.zeros_like(data)
            for i in range(data.shape[1]):
                smoothed[:, i] = savgol_filter(data[:, i], window_length, polyorder, axis=0)
            return smoothed
        else:
            smoothed = np.zeros_like(data)
            for i in range(data.shape[1]):
                for j in range(data.shape[2]):
                    smoothed[:, i, j] = savgol_filter(data[:, i, j], window_length, polyorder, axis=0)
            return smoothed
    else:
        raise ValueError(f"Unknown method: {method}. Use 'kalman' or 'savgol'")


def process_npz_file(input_path, output_path, method='kalman', window_length=None, 
                     polyorder=3, process_variance=1e-5, measurement_variance=1e-3):
    data = np.load(input_path, allow_pickle=True)
    smoothed_data = {}
    original_data = {}
    for key in data.keys():
        arr = data[key]
        if isinstance(arr, np.ndarray) and arr.ndim >= 1:
            if arr.ndim >= 2 and arr.shape[0] > 1:
                original_data[key] = arr.copy()
                smoothed_arr = smooth_trajectory(
                    arr, method=method, window_length=window_length, polyorder=polyorder,
                    process_variance=process_variance, measurement_variance=measurement_variance, axis=0
                )
                smoothed_data[key] = smoothed_arr
            else:
                original_data[key] = arr.copy()
                smoothed_data[key] = arr.copy()
        else:
            original_data[key] = arr
            smoothed_data[key] = arr
    np.savez_compressed(output_path, **smoothed_data)
    return original_data, smoothed_data


def visualize_trajectory_comparison(original_data, smoothed_data, output_dir, filename):
    vis_dir = Path(output_dir) / 'visualizations'
    vis_dir.mkdir(parents=True, exist_ok=True)
    traj_keys = []
    if "joint_q" in original_data:
        arr = original_data["joint_q"]
        if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[0] > 1:
            traj_keys.append("joint_q")
    if not traj_keys:
        return
    for key in traj_keys:
        original = original_data[key]
        smoothed = smoothed_data[key]
        n_features = original.shape[1]
        n_frames = original.shape[0]
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_features == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_rows > 1 else ([axes] if n_cols == 1 else axes)
        time = np.arange(n_frames)
        for i in range(n_features):
            ax = axes[i]
            ax.plot(time, original[:, i], 'b-', alpha=0.5, linewidth=1, label='Original')
            ax.plot(time, smoothed[:, i], 'r-', linewidth=1.5, label='Smoothed')
            ax.set_xlabel('Frame')
            ax.set_ylabel(f'{key}[{i}]')
            ax.set_title(f'{key} - dim {i}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        for i in range(n_features, len(axes)):
            axes[i].axis('off')
        plt.tight_layout()
        output_path = vis_dir / f'{filename}_{key}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


def worker_process(npz_file, output_dir, args_dict):
    """Worker: process one file (must be top-level for pickle)."""
    output_file = Path(output_dir) / npz_file.name
    try:
        original_data, smoothed_data = process_npz_file(
            npz_file,
            output_file,
            method=args_dict['method'],
            window_length=args_dict['window_length'],
            polyorder=args_dict['polyorder'],
            process_variance=args_dict['process_variance'],
            measurement_variance=args_dict['measurement_variance']
        )
        if args_dict['visualize']:
            filename_stem = npz_file.stem
            visualize_trajectory_comparison(
                original_data,
                smoothed_data,
                output_dir,
                filename_stem
            )
        return f"Success: {npz_file.name}"
    except Exception as e:
        return f"Error: {npz_file.name} - {str(e)}"


def main():
    parser = argparse.ArgumentParser(description='Smooth npz trajectories (multi-process)')
    parser.add_argument('input_dir', type=str, help='Input directory')
    parser.add_argument('output_dir', type=str, help='Output directory')
    parser.add_argument('--window_length', type=int, default=None)
    parser.add_argument('--method', type=str, default='kalman', choices=['kalman', 'savgol'])
    parser.add_argument('--polyorder', type=int, default=3)
    parser.add_argument('--process_variance', type=float, default=1e-5)
    parser.add_argument('--measurement_variance', type=float, default=3e-4)
    parser.add_argument('--pattern', type=str, default='*.npz')
    parser.add_argument('--visualize', action='store_true', help='Generate visualization images')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(input_dir.glob(args.pattern))
    if not npz_files:
        print(f"Warning: no files matching {args.pattern} in {input_dir}")
        return

    print(f"Found {len(npz_files)} npz files, using {args.workers} workers...")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Visualize: {'on' if args.visualize else 'off'}")
    if args.visualize:
        print(f"Images: {output_dir}/visualizations")
    print("-" * 50)

    args_dict = {
        'method': args.method,
        'window_length': args.window_length,
        'polyorder': args.polyorder,
        'process_variance': args.process_variance,
        'measurement_variance': args.measurement_variance,
        'visualize': args.visualize
    }

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(worker_process, f, output_dir, args_dict) for f in npz_files]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            result = future.result()
            if "Error" in result:
                print(f"\n{result}")

    print(f"\nDone. Processed {len(npz_files)} files")
    print(f"Output: {output_dir}")
    if args.visualize:
        print(f"Visualizations: {output_dir / 'visualizations'}")


if __name__ == '__main__':
    main()