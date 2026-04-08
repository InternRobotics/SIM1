import os
import numpy as np
from .utils import read_json_file, save_json_file, makedir
import json
import matplotlib.pyplot as plt

import os
import numpy as np


class Splitter:
    def __init__(self, data_folder, task_split_cfg):
        self.data_folder = os.path.join(data_folder, 'npz')
        self.output_folder = os.path.join(data_folder, 'segments')
        makedir(self.output_folder)
        self.task_split_cfg = read_json_file(task_split_cfg)

    def detect_segments(self, fname, gripper_left, gripper_right):
        N = len(gripper_left)
        tasks = list(self.task_split_cfg.keys())

        start = 0
        taskid = 0
        left_flag = True
        right_flag = True
        segments = {}
        
        for i in range(1, N):
            thresh = self.task_split_cfg[tasks[taskid]]["gripper_thresh"]
            gripper = self.task_split_cfg[tasks[taskid]]["gripper"]
            task_signal = self.task_split_cfg[tasks[taskid]]["task_signal"]

            if gripper == 'left':
                if (gripper_left[i] < thresh) == left_flag:
                    segments.update({
                        task_signal:[start, i-1]
                    })

                    left_flag = not left_flag
                    start = i
                    taskid += 1

            elif gripper == 'right':
                if (gripper_right[i] < thresh) == right_flag:
                    segments.update({
                        task_signal:[start, i-1]
                    })

                    right_flag = not right_flag
                    start = i
                    taskid += 1

            elif gripper == 'both':

                if (gripper_right[i] < thresh) == right_flag and (gripper_left[i] < thresh) == left_flag:
                    segments.update({
                        task_signal:[start, i-1]
                    })

                    left_flag = not left_flag
                    right_flag = not right_flag
                    start = i
                    taskid += 1

            else:
                segments.update({
                        "reset":[start, N]
                    })
                break

        return segments

    def split(self):
        for fname in os.listdir(self.data_folder):
            if not fname.endswith('.npz'):
                continue
            
            path = os.path.join(self.data_folder, fname)
            data = np.load(path) 
            
            arr = data[data.files[1]]
            gripper_left = arr[:, 0]
            gripper_right = arr[:, 1] 

            segments = self.detect_segments(fname, gripper_left, gripper_right) 

            outjson = os.path.join(self.output_folder, fname.replace('.npz','.json'))
            save_json_file(outjson, segments)

class SplitterFine:
    """
    Initial segment split (stable/moving) and merge adjacent moving segments.
    """

    def __init__(self, data_folder, stable_thresh=0.01, min_length=3, visualize=False):
        self.data_folder = os.path.join(data_folder, 'npz')
        self.output_folder = os.path.join(data_folder, "segments_fine")
        makedir(self.output_folder)

        self.stable_thresh = stable_thresh
        self.min_length = min_length
        self.visualize = visualize


    def detect_state_segments(self, gripper_left, gripper_right):
        N = len(gripper_left)
        segments = []

        state = "stable"
        start = 0

        for i in range(1, N):
            dl = abs(gripper_left[i] - gripper_left[i-1])
            dr = abs(gripper_right[i] - gripper_right[i-1])
            curr_state = "stable" if (dl <= self.stable_thresh and dr <= self.stable_thresh) else "moving"

            if curr_state != state:
                if i - start >= self.min_length:
                    segments.append({"start": start, "end": i-1, "state": state})
                start = i
                state = curr_state

        if N - start >= self.min_length:
            segments.append({"start": start, "end": N-1, "state": state})

        merged_segments = []
        for seg in segments:
            if not merged_segments:
                merged_segments.append(seg)
            else:
                last = merged_segments[-1]
                if seg['state'] == last['state']:
                    last['end'] = seg['end']
                else:
                    merged_segments.append(seg)

        return merged_segments

    def postprocess_segments(self, segments, min_stable=15):
        if not segments:
            return segments

        # Phase 1: connect head and tail
        new_segments = [segments[0].copy()]

        for i in range(1, len(segments)):
            prev = new_segments[-1]
            curr = segments[i].copy()

            if curr['start'] > prev['end'] + 1:
                if curr['state'] == "moving":
                    curr['start'] = prev['end'] + 1
                else:
                    prev['end'] = curr['start'] - 1

            new_segments.append(curr)

        # Phase 2: merge adjacent same-state
        final_segments = []
        for seg in new_segments:
            if not final_segments:
                final_segments.append(seg)
            else:
                last = final_segments[-1]
                if seg['state'] == last['state'] and seg['start'] <= last['end'] + 1:
                    last['end'] = seg['end']
                else:
                    final_segments.append(seg)

        # Phase 3: merge short stable segments
        merged_segments = []
        i = 0
        while i < len(final_segments):
            seg = final_segments[i]

            if (seg['state'] == 'stable' and 
                (seg['end'] - seg['start'] + 1) < min_stable and
                i > 0 and i < len(final_segments) - 1 and
                final_segments[i-1]['state'] == 'moving' and
                final_segments[i+1]['state'] == 'moving'):

                # Merge three segments
                merged = {
                    'state': 'moving',
                    'start': final_segments[i-1]['start'],
                    'end':   final_segments[i+1]['end']
                }

                # Replace three with merged
                merged_segments.pop()  # remove previous moving
                merged_segments.append(merged)

                # Skip stable + next moving
                i += 2
            else:
                merged_segments.append(seg)
                i += 1

        return merged_segments

    def visualize_segments(self, gripper_left, gripper_right, segments, save_path):
        plt.figure(figsize=(10,4))
        plt.plot(gripper_left, label="Left Gripper")
        plt.plot(gripper_right, label="Right Gripper")
        colors = {'stable': '#a6cee3', 'moving': '#fb9a99'}

        for seg in segments:
            start, end, state = seg['start'], seg['end'], seg['state']
            plt.axvspan(start, end, color=colors[state], alpha=0.3)

        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

    def split(self):
        for fname in os.listdir(self.data_folder):
            if not fname.endswith(".npz"):
                continue

            path = os.path.join(self.data_folder, fname)
            data = np.load(path)

            try:
                arr = data[data.files[1]]
                gripper_left = arr[:, 0]
                gripper_right = arr[:, 1]
            except Exception as e:
                arr = data[data.files[0]]
                gripper_left = arr[:, 6]
                gripper_right = arr[:, -1]

            # detect
            segments = self.detect_state_segments(gripper_left, gripper_right)
            segments = self.postprocess_segments(segments)
            
            # # ===== Print each segment length =====
            # print(f"[Fine] {fname}: {len(segments)} segments")
            # for idx, seg in enumerate(segments):
            #     seg_length = seg['end'] - seg['start'] + 1
            #     print(f"  Segment {idx}: state={seg['state']}, start={seg['start']}, end={seg['end']}, length={seg_length}")

            # # ===== Save JSON directly (skip len==19 check) =====
            # outjson = os.path.join(self.output_folder, fname.replace(".npz", ".json"))
            # with open(outjson, "w") as f:
            #     json.dump({"segments": segments}, f, indent=2)
            # print(f"[Fine] {fname}: JSON saved to {outjson}")
            
            # if self.visualize:
            #     outpng = os.path.join(self.output_folder, fname.replace(".npz", ".png"))
            #     self.visualize_segments(gripper_left, gripper_right, segments, outpng)
            #     print(f"[Fine] Visualization saved {outpng}")
                
            
            if len(segments) == 19:  # three-fold
            # if len(segments) == 9:  # short pants and mat
                # JSON
                outjson = os.path.join(self.output_folder, fname.replace(".npz", ".json"))
                with open(outjson, "w") as f:
                    json.dump({"segments": segments}, f, indent=2)
                # print(f"[Fine] {fname}: {len(segments)} segments saved {outjson}")
            else:
                print(f"[Fine] {fname}: {len(segments)} skip")
            
            if self.visualize:
                outpng = os.path.join(self.output_folder, fname.replace(".npz", ".png"))
                self.visualize_segments(gripper_left, gripper_right, segments, outpng)
                print(f"[Fine] Visualization saved {outpng}")
            
class SplitterDex:
    """
    Generic fine-grained splitter (Dex/Flexible), no segment count limit.
    Splits gripper signal into stable/moving and saves all valid results.
    Before save: automatically drop last 18 segments.
    """

    def __init__(self, data_folder, stable_thresh=0.01, min_length=3, min_stable=15, visualize=False):
        self.data_folder = os.path.join(data_folder, 'npz')
        self.output_folder = os.path.join(data_folder, "segments_dex")
        makedir(self.output_folder)

        self.stable_thresh = stable_thresh
        self.min_length = min_length
        self.min_stable = min_stable
        self.visualize = visualize
        self.drop_last_n = 18  # drop last 18 segments before save

    def detect_state_segments(self, gripper_left, gripper_right):
        N = len(gripper_left)
        if N == 0:
            return []

        segments = []
        state = "stable"
        start = 0

        for i in range(1, N):
            dl = abs(gripper_left[i] - gripper_left[i-1])
            dr = abs(gripper_right[i] - gripper_right[i-1])
            curr_state = "stable" if (dl <= self.stable_thresh and dr <= self.stable_thresh) else "moving"

            if curr_state != state:
                if i - start >= self.min_length:
                    segments.append({"start": start, "end": i - 1, "state": state})
                start = i
                state = curr_state

        # Append last segment
        if N - start >= self.min_length:
            segments.append({"start": start, "end": N - 1, "state": state})

        # Merge adjacent same-state segments (initial)
        merged = []
        for seg in segments:
            if not merged:
                merged.append(seg)
            else:
                last = merged[-1]
                if seg['state'] == last['state'] and seg['start'] <= last['end'] + 1:
                    last['end'] = seg['end']
                else:
                    merged.append(seg)
        return merged

    def postprocess_segments(self, segments):
        if not segments:
            return segments

        # Phase 1: no gaps between segments
        new_segments = [segments[0].copy()]
        for i in range(1, len(segments)):
            prev = new_segments[-1]
            curr = segments[i].copy()
            if curr['start'] > prev['end'] + 1:
                # Bridge gap
                if curr['state'] == "moving":
                    curr['start'] = prev['end'] + 1
                else:
                    prev['end'] = curr['start'] - 1
            new_segments.append(curr)

        # Phase 2: merge adjacent same-state again
        final_segments = []
        for seg in new_segments:
            if not final_segments:
                final_segments.append(seg)
            else:
                last = final_segments[-1]
                if seg['state'] == last['state'] and seg['start'] <= last['end'] + 1:
                    last['end'] = seg['end']
                else:
                    final_segments.append(seg)

        # Phase 3: merge short stable between two moving
        merged_segments = []
        i = 0
        while i < len(final_segments):
            seg = final_segments[i]
            if (seg['state'] == 'stable' and
                (seg['end'] - seg['start'] + 1) < self.min_stable and
                i > 0 and i < len(final_segments) - 1 and
                final_segments[i - 1]['state'] == 'moving' and
                final_segments[i + 1]['state'] == 'moving'):

                # Merge three segments into one moving
                merged = {
                    'state': 'moving',
                    'start': final_segments[i - 1]['start'],
                    'end': final_segments[i + 1]['end']
                }
                merged_segments.pop()  # remove previous moving
                merged_segments.append(merged)
                i += 2  # skip stable and next moving
            else:
                merged_segments.append(seg)
                i += 1

        return merged_segments

    def visualize_segments(self, gripper_left, gripper_right, segments, save_path):
        plt.figure(figsize=(10, 4))
        plt.plot(gripper_left, label="Left Gripper", alpha=0.8)
        plt.plot(gripper_right, label="Right Gripper", alpha=0.8)
        colors = {'stable': '#a6cee3', 'moving': '#fb9a99'}

        for seg in segments:
            start, end = seg['start'], seg['end']
            plt.axvspan(start, end, color=colors[seg['state']], alpha=0.3, label=seg['state'])

        # Avoid duplicate legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

    def split(self):
        for fname in os.listdir(self.data_folder):
            if not fname.endswith(".npz"):
                continue

            path = os.path.join(self.data_folder, fname)
            data = np.load(path)

            # Auto-adapt data format
            try:
                arr = data[data.files[1]]
                gripper_left = arr[:, 0]
                gripper_right = arr[:, 1]
            except (IndexError, KeyError, ValueError):
                # Fallback: try first array
                arr = data[data.files[0]]
                if arr.shape[1] >= 7:
                    gripper_left = arr[:, 6]
                    gripper_right = arr[:, -1]
                else:
                    print(f"[Dex] Skip {fname}: cannot parse gripper data.")
                    continue

            # Segment
            raw_segments = self.detect_state_segments(gripper_left, gripper_right)
            segments = self.postprocess_segments(raw_segments)

            if not segments:
                print(f"[Dex] Skip {fname}: no valid segments after postprocessing.")
                continue

            # Drop last 18 segments
            if len(segments) > self.drop_last_n:
                segments_to_save = segments[:-self.drop_last_n]
            else:
                segments_to_save = []  # drop all (including exactly 18 or fewer)

            # Save JSON (even if empty, for alignment)
            outjson = os.path.join(self.output_folder, fname.replace(".npz", ".json"))
            with open(outjson, "w") as f:
                json.dump({"segments": segments_to_save}, f, indent=2)

            print(f"[Dex] Original: {len(segments)}, Saved: {len(segments_to_save)} segments for {fname}")

            # Visualize (optional)
            if self.visualize:
                outpng = os.path.join(self.output_folder, fname.replace(".npz", ".png"))
                self.visualize_segments(gripper_left, gripper_right, segments_to_save, outpng)
                print(f"[Dex] Visualization saved to {outpng}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SIM1-DataGen Splitter")
    parser.add_argument("data_folder", type=str, help="Root data folder")
    parser.add_argument("--mode", type=str, default="fine", choices=["fine", "dex", "baseline"])
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--task_split_cfg", type=str, default=None)
    args = parser.parse_args()

    if args.mode == "fine":
        splitter = SplitterFine(args.data_folder, visualize=args.visualize)
    elif args.mode == "dex":
        splitter = SplitterDex(args.data_folder, visualize=args.visualize)
    else:
        if args.task_split_cfg is None:
            print("Error: --task_split_cfg is required for baseline mode")
            sys.exit(1)
        splitter = Splitter(args.data_folder, args.task_split_cfg)
    splitter.split()