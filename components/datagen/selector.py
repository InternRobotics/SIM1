import os
import numpy as np
import random
from .utils import read_json_file, save_json_file, makedir, rand_int_with_seed

class Selector:
    def __init__(self, data_folder,  task_split_cfg, seed=12580):
        self.src_folder = data_folder
        self.input_folder = os.path.join(data_folder, "segments")
        self.output_folder = os.path.join(data_folder, 'temp_trajs')
        makedir(self.output_folder)

        self.task_split_cfg = read_json_file(task_split_cfg)
        self.tasks = list(self.task_split_cfg.keys())

        self.segments = {self.task_split_cfg[task]["task_signal"]: {} for task in self.tasks}

    def gen_segment_data(self, save_file=True):
        for fname in os.listdir(self.input_folder):
            if not fname.endswith('.json'):
                continue
            
            fpath = os.path.join(self.input_folder,  fname)
            data = read_json_file(fpath) 

            for task in self.segments:
                self.segments[task].update(
                    {fname.split('.')[0]: data[task]}
                )

        if save_file:
            outjson = os.path.join(self.src_folder, 'segments.json')
            save_json_file(outjson, self.segments)

    def random_gen_data(self, total_num=1000):
        tasks = list(self.segments.keys())
        for i in range(total_num):
            new_traj_data = {}
            for task in tasks:
                id = rand_int_with_seed(0, len(self.segments[task])-1)
                ids = list(self.segments[task].keys())

                record_id = ids[id]
                new_traj_data.update(
                    {
                        task:{"record_id": record_id,
                              "segment": self.segments[task][record_id]}
                    }
                )

            outjson = os.path.join(self.output_folder, '{:0>6d}.json'.format(i))
            save_json_file(outjson, new_traj_data)

    def select(self, gen_data_num):
        self.gen_segment_data()
        self.random_gen_data(gen_data_num)

class SelectorFine:
    def __init__(self, data_folder, seed=12580):

        self.input_folder = os.path.join(data_folder, "segments_fine")
        self.output_folder = os.path.join(data_folder, 'temp_trajs')
        makedir(self.output_folder)

        random.seed(seed)

        # {subtask_i: {record_id: {"segment":[s,e], "intermediate":int or None}}}
        self.subtask_pool = {}

    # ---------- Split one JSON and return with intermediate ----------
    def split_segments(self, segs):
        """
        Return list of segments, each dict:
            (start, end, intermediate)
        intermediate = end of moving segment
        """
        result = []

        # First segment stands alone, no moving/stable pair
        first = segs[0]
        # result.append((first["start"], first["end"], None))

        # Following segments are moving + stable pairs
        i = 1
        while i < len(segs) - 1:
            curr = segs[i]
            nxt = segs[i + 1]

            if curr["state"] == "moving" and nxt["state"] == "stable":
                s = curr["start"]
                e = nxt["end"]
                intermediate = curr["end"]   # end of moving segment (key point)
                result.append((s, e, intermediate))
                i += 2
            else:
                i += 1

        return result   # total 10 segments

    # ---------- Read one JSON and add to subtask_pool ----------
    def process_one_file(self, json_path):
        data = read_json_file(json_path)
        segs = data["segments"]

        fname = os.path.basename(json_path).replace(".json", "")
        subsegments = self.split_segments(segs)

        for i, (s, e, interm) in enumerate(subsegments):
            key = f"subtask_{i}"

            if key not in self.subtask_pool:
                self.subtask_pool[key] = {}

            self.subtask_pool[key][fname] = {
                "segment": [s, e],
                "intermediate": interm
            }

    # ---------- Build subtask_pool ----------
    def build_pool(self):
        for fname in os.listdir(self.input_folder):
            if fname.endswith(".json"):
                self.process_one_file(os.path.join(self.input_folder, fname))

        print(f"[INFO] processed {len(self.subtask_pool)} subtasks (each with several records)")

    # ---------- Randomly compose one trajectory ----------
    def random_combine_once(self):
        result = {}

        for subtask, candidates in self.subtask_pool.items():

            record_id = random.choice(list(candidates.keys()))
            item = candidates[record_id]

            result[subtask] = {
                "record_id": record_id,
                "segment": item["segment"],
                "intermediate": item["intermediate"]
            }

        return result

    # ---------- Generate multiple trajectories ----------
    def random_generate(self, num=10):
        for i in range(num):
            combined = self.random_combine_once()
            save_path = os.path.join(self.output_folder, f"{i:06d}.json")
            save_json_file(save_path, combined)
            print(f"[OK] saved {save_path}")

    # ---------- Run full pipeline ----------
    def select(self, num=10):
        self.build_pool()
        self.random_generate(num)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SIM1-DataGen Selector")
    parser.add_argument("data_folder", type=str, help="Root data folder")
    parser.add_argument("--task_split_cfg", type=str, default="./components/datagen/configs/lift_cloth_manip.json")
    parser.add_argument("--num", type=int, default=1000, help="Number of trajectories")
    args = parser.parse_args()
    selector = Selector(args.data_folder, args.task_split_cfg)
    selector.gen_segment_data()
    selector.random_gen_data(args.num)
