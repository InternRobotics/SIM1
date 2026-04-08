import argparse
import gc
import logging
import os
import pickle
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import imageio  # imageio[ffmpeg]
import lmdb
import numpy as np
import torch

try:
    from lerobot.common.datasets.compute_stats import (
        auto_downsample_height_width,
        get_feature_stats,
        sample_indices,
    )
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.common.datasets.utils import (
        check_timestamps_sync,
        get_episode_data_index,
        validate_episode_buffer,
    )

    _LEROBOT_LEGACY_COMMON = True
except ModuleNotFoundError:
    from lerobot.datasets.compute_stats import (
        auto_downsample_height_width,
        get_feature_stats,
        sample_indices,
    )
    from lerobot.datasets.feature_utils import validate_episode_buffer
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    _LEROBOT_LEGACY_COMMON = False
    check_timestamps_sync = None  # type: ignore[assignment,misc]
    get_episode_data_index = None  # type: ignore[assignment,misc]


"""
Store camera images and robot states as a combined observation.
Args:
    observation: images (camera), states (robot state)
    actions: joint, gripper, ee_pose
"""
FEATURES = {
    "images.rgb.head": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channel"],
    },
    "images.rgb.hand_left": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channel"],
    },
    "images.rgb.hand_right": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channel"],
    },
    "states.left_joint.position": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["left_joint_0", "left_joint_1", "left_joint_2", "left_joint_3", "left_joint_4", "left_joint_5",],
    },
    "states.left_gripper.position": {
        "dtype": "float32",
        "shape": (1,),
        "names": ["left_gripper_0",],
    },
    "states.right_joint.position": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["right_joint_0", "right_joint_1", "right_joint_2", "right_joint_3", "right_joint_4", "right_joint_5",],
    },
    "states.right_gripper.position": {
        "dtype": "float32",
        "shape": (1,),
        "names": ["right_gripper_0",],
    },
    "actions.left_joint.position": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["left_joint_0", "left_joint_1", "left_joint_2", "left_joint_3", "left_joint_4", "left_joint_5",],
    },
    "actions.left_gripper.position": {
        "dtype": "float32",
        "shape": (1,),
        "names": ["left_gripper_0",],
    },
    "actions.right_joint.position": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["right_joint_0", "right_joint_1", "right_joint_2", "right_joint_3", "right_joint_4", "right_joint_5",],
    },
    "actions.right_gripper.position": {
        "dtype": "float32",
        "shape": (1,),
        "names": ["right_gripper_0", ],
    },
    "master_actions.left_joint.position": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["left_joint_0", "left_joint_1", "left_joint_2", "left_joint_3", "left_joint_4", "left_joint_5",],
    },
    "master_actions.left_gripper.position": {
        "dtype": "float32",
        "shape": (1,),
        "names": ["left_gripper_0",],
    },
    "master_actions.left_gripper.openness": {
        "dtype": "float32",
        "shape": (1,),
        "names": ["left_gripper_0",],
    },
    "master_actions.right_joint.position": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["right_joint_0", "right_joint_1", "right_joint_2", "right_joint_3", "right_joint_4", "right_joint_5",],
    },
    "master_actions.right_gripper.position": {
        "dtype": "float32",
        "shape": (1,),
        "names": ["right_gripper_0", ],
    },
    "master_actions.right_gripper.openness": {
        "dtype": "float32",
        "shape": (1,),
        "names": ["right_gripper_0",],
    },

}

class ARXLift2Dataset(LeRobotDataset):
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        download_videos: bool = True,
        local_files_only: bool = False,
        video_backend: str | None = None,
    ):
        kwargs = dict(
            repo_id=repo_id,
            root=root,
            episodes=episodes,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
            tolerance_s=tolerance_s,
            download_videos=download_videos,
            video_backend=video_backend,
        )
        if _LEROBOT_LEGACY_COMMON:
            kwargs["local_files_only"] = local_files_only
        super().__init__(**kwargs)

    def save_episode(
        self,
        episode_data: dict | None = None,
        videos: dict | None = None,
        parallel_encoding: bool = True,
    ) -> None:
        if not _LEROBOT_LEGACY_COMMON:
            if videos is not None:
                self._save_episode_with_external_videos_modern(episode_data, videos)
                return
            super().save_episode(episode_data, parallel_encoding)
            return

        if episode_data is not None:
            episode_buffer = episode_data
        else:
            episode_buffer = self.episode_buffer

        validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)
        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set(tasks))
        episode_index = episode_buffer["episode_index"]

        episode_buffer["index"] = np.arange(self.meta.total_frames, self.meta.total_frames + episode_length)
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

        for task in episode_tasks:
            task_index = self.meta.get_task_index(task)
            if task_index is None:
                self.meta.add_task(task)

        episode_buffer["task_index"] = np.array([self.meta.get_task_index(task) for task in tasks])
        for key, ft in self.features.items():
            if key in ["index", "episode_index", "task_index"] or ft["dtype"] in ["video"]:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key]).squeeze()
        for key in self.meta.video_keys:
            video_path = self.root / self.meta.get_video_file_path(episode_index, key)
            episode_buffer[key] = str(video_path)  # PosixPath -> str
            video_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(videos[key], video_path)
        ep_stats = compute_episode_stats(episode_buffer, self.features)
        self._save_episode_table(episode_buffer, episode_index)
        self.meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats)
        ep_data_index = get_episode_data_index(self.meta.episodes, [episode_index])
        ep_data_index_np = {k: t.numpy() for k, t in ep_data_index.items()}
        check_timestamps_sync(
            episode_buffer["timestamp"],
            episode_buffer["episode_index"],
            ep_data_index_np,
            self.fps,
            self.tolerance_s,
        )
        if episode_data is None:
            self.episode_buffer = self.create_episode_buffer()

    def _save_episode_with_external_videos_modern(
        self, episode_data: dict | None, videos: dict[str, Path | str]
    ) -> None:
        if episode_data is not None:
            raise NotImplementedError("episode_data=... is not supported for external video import on LeRobot 0.5+.")
        self._require_writer("save_episode")
        episode_buffer = self.writer.episode_buffer
        validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)

        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set(tasks))
        episode_index = episode_buffer["episode_index"]

        episode_buffer["index"] = np.arange(self.meta.total_frames, self.meta.total_frames + episode_length)
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

        self.meta.save_episode_tasks(episode_tasks)
        episode_buffer["task_index"] = np.array([self.meta.get_task_index(task) for task in tasks])

        for key, ft in self.meta.features.items():
            if key in ["index", "episode_index", "task_index"] or ft["dtype"] in ["image", "video"]:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key]).squeeze()

        self.writer._wait_image_writer()

        stats_buffer = {k: v for k, v in episode_buffer.items()}
        for key in self.meta.video_keys:
            stats_buffer[key] = str(Path(videos[key]).resolve())
        ep_stats = compute_episode_stats(stats_buffer, self.features)

        ep_metadata = self.writer._save_episode_data(episode_buffer)

        for key in self.meta.video_keys:
            tmp_root = Path(tempfile.mkdtemp(dir=self.root))
            tmp_mp4 = tmp_root / f"{key.replace('/', '_')}.mp4"
            shutil.copy2(Path(videos[key]).resolve(), tmp_mp4)
            ep_metadata.update(self.writer._save_episode_video(key, episode_index, temp_path=tmp_mp4))

        self.meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats, ep_metadata)
        self.writer.clear_episode_buffer(delete_images=False)

    def add_frame(self, frame: dict) -> None:
        for name in frame:
            if isinstance(frame[name], torch.Tensor):
                frame[name] = frame[name].numpy()
        if not _LEROBOT_LEGACY_COMMON:
            self._require_writer("add_frame")
        if _LEROBOT_LEGACY_COMMON:
            if self.episode_buffer is None:
                self.episode_buffer = self.create_episode_buffer()
            buf = self.episode_buffer
        else:
            buf = self.writer.episode_buffer

        frame_index = buf["size"]
        timestamp = frame.pop("timestamp") if "timestamp" in frame else frame_index / self.fps
        buf["frame_index"].append(frame_index)
        buf["timestamp"].append(timestamp)

        for key in frame:
            if key == "task":
                buf["task"].append(frame["task"])
                continue
            if key not in self.features:
                raise ValueError(
                    f"An element of the frame is not in the features. '{key}' not in '{self.features.keys()}'."
                )
            buf[key].append(frame[key])
        buf["size"] += 1

# def crop_resize_no_padding(image, target_size=(480, 640)):
#     """
#     Crop and scale to target size (no padding)
#     :param image: input image (NumPy array)
#     :param target_size: target size (height, width)
#     :return: processed image
#     """
#     h, w = image.shape[:2]
#     target_h, target_w = target_size
#     target_ratio = target_w / target_h  # Target aspect ratio (e.g. 640/480=1.333)

#     # the original image aspect ratio and cropping direction
#     if w / h > target_ratio:  # Original image is wider → crop width
#         crop_w = int(h * target_ratio)  # Calculate crop width based on target aspect ratio
#         crop_h = h
#         start_x = (w - crop_w) // 2  # Horizontal center starting point
#         start_y = 0
#     else:  # Original image is higher → crop height
#         crop_h = int(w / target_ratio)  # Calculate clipping height according to target aspect ratio
#         crop_w = w
#         start_x = 0
#         start_y = (h - crop_h) // 2  # Vertical center starting point

#     # Perform centered cropping (to prevent out-of-bounds)
#     start_x, start_y = max(0, start_x), max(0, start_y)
#     end_x, end_y = min(w, start_x + crop_w), min(h, start_y + crop_h)
#     cropped = image[start_y:end_y, start_x:end_x]

#     # Resize to target size (bilinear interpolation)
#     resized = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
#     return resized


def load_lmdb_data(episode_path: Path, sava_path: Path, fps_factor: int, target_fps: int) -> Optional[Dict]:
    def load_image(txn, key):
        raw = txn.get(key)
        data = pickle.loads(raw)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        # Convert to RGB if necessary
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = crop_resize_no_padding(image, target_size=(480, 640))
        return image
    try:
        env = lmdb.open(
            str(episode_path / "lmdb"),
            readonly=True,
            lock=False,
            max_readers=128,
            readahead=False
        )
        meta_info = pickle.load(open(episode_path/"meta_info.pkl", "rb"))
        with env.begin(write=False) as txn:
            keys = [k for k, _ in txn.cursor()]
            # import pdb; pdb.set_trace()
            qpos_keys = ['states.left_gripper.position', 'states.left_joint.position', 'states.right_gripper.position', 'states.right_joint.position']
            master_action_keys = ['master_actions.left_gripper.openness', 'master_actions.left_gripper.position', 'master_actions.left_joint.position', 'master_actions.right_gripper.openness', 'master_actions.right_gripper.position', 'master_actions.right_joint.position']
            image_keys = ['images.rgb.head', 'images.rgb.hand_left', 'images.rgb.hand_right']
            total_steps = []
            for image_key in image_keys:
                # keys_image_per_step = sorted([k for k in keys if image_key.encode() in k])
                keys_image_per_step = meta_info['keys'][image_key]
                total_steps.append(len(keys_image_per_step))
            
            state_action_dict = {}
            ### qpos
            for key in qpos_keys:
                state_action_dict[key] = pickle.loads(txn.get(key.encode()))
                state_action_dict[key] = np.stack(state_action_dict[key])
                total_steps.append(len(state_action_dict[key]))
            state_keys = list(state_action_dict.keys())
            ### next qpos as action
            for k in state_keys:
                state_action_dict[k.replace("states", "actions")] = np.concatenate([state_action_dict[k][1:, :], state_action_dict[k][-1, :][None,:]], axis=0)
            ### master action
            for key in master_action_keys:
                state_action_dict[key] = pickle.loads(txn.get(key.encode()))
                if np.isscalar(state_action_dict[key]):
                    state_action_dict[key] = np.array([state_action_dict[key]]).astype("float32")
                state_action_dict[key] = np.stack(state_action_dict[key])
                total_steps.append(len(state_action_dict[key]))
            unique_steps = list(set(total_steps))
            # import pdb; pdb.set_trace()
            print("episode_path:", episode_path)
            print("total_steps: ", total_steps)
            assert len(unique_steps) == 1 and unique_steps[0]>0, f"no data found or qpos / image steps mismatch in {episode_path}"
            assert np.max(np.abs(state_action_dict["states.left_joint.position"])) < 2 * np.pi
            assert np.max(np.abs(state_action_dict["states.right_joint.position"])) < 2 * np.pi
            selected_steps = [step for step in range(unique_steps[0]) if step % fps_factor == 0]
            frames = []
            image_observations = {}
            for image_key in image_keys:
                image_observations[image_key] = []
            start_time = time.time()
            for step_index, step in enumerate(selected_steps):
                step_str = f"{step:04d}"
                data_dict = {}
                for key, value in state_action_dict.items():
                    data_dict[key] = value[step]
                data_dict["task"] = meta_info['language_instruction']
                frames.append(data_dict)
                # import pdb; pdb.set_trace()
                for image_key in image_keys:
                    image_key_step_encode = f"{image_key}/{step_str}".encode()
                    if not image_key_step_encode in keys:
                        raise ValueError(f"Image key {image_key_step_encode} not found in LMDB keys.")
                    image_observations[image_key].append(load_image(txn, image_key_step_encode))
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Loaded image observations from {episode_path}")
        env.close()
        if not frames:
            return None
        os.makedirs(sava_path, exist_ok=True)
        # episode_idx = "0000000"
        os.makedirs(sava_path/episode_path.name, exist_ok=True)
        video_paths = {}
        for image_key in image_keys:
            h_ori, w_ori =  image_observations[image_key][0].shape[:2]
            if w_ori == 1280:
                w_tgt = w_ori//2
                h_tgt = h_ori//2
            else:
                w_tgt = w_ori
                h_tgt = h_ori
            imageio.mimsave(
                sava_path/episode_path.name/f'{image_key.replace(".", "_")}.mp4', 
                image_observations[image_key], 
                fps=target_fps,
                # codec="libsvtav1",
                # codec="libx264",
                # ffmpeg_params=[
                #     "-crf", "28",              # quality (0-63, default 30)
                #     "-preset", "8",            # speed preset (0-13, higher = faster but lower compression)
                #     # "-g", "240",             # keyframe interval (recommended >= 8x fps)
                #     "-pix_fmt", "yuv420p",     # pixel format for broad compatibility
                #     "-movflags", "+faststart",  # move metadata to file start for streaming
                #     # "-threads", "8",         # thread count
                #     "-vf", f"scale={w_tgt}:{h_tgt}",
                #     "-y",                      # overwrite existing output file
                # ]
            )
            video_paths[image_key] = sava_path/episode_path.name/f'{image_key.replace(".", "_")}.mp4'
        # imageio.mimsave(sava_path/episode_path.name/'hand_left.mp4', image_observations["images.rgb.hand_left"], fps=target_fps)
        # imageio.mimsave(sava_path/episode_path.name/'hand_right.mp4', image_observations["images.rgb.hand_right"], fps=target_fps)
        print(f"imageio.mimsave completed for {episode_path}")

        return {
            "frames": frames,
            "videos": video_paths,
        }

    except Exception as e:
        logging.error(f"Failed to load or process LMDB data: {e}")
        return None


def get_all_tasks(src_path: Path, output_path: Path) -> Tuple[Path, Path]:
    output_path.mkdir(exist_ok=True)
    yield (src_path, output_path)


def discover_episode_dirs(src_path: Path) -> list[str]:
    """
    Find episode roots (each must contain lmdb/data.mdb + meta_info.pkl).

    Supported layouts:
    1) Step-4 out_updated: src_path/000000/lmdb, src_path/000001/lmdb, ...
    2) Single episode: src_path itself is the episode root (src_path/lmdb).
    3) Legacy cluster layout: src_path/<run>/out/<timestamp>/lmdb.
    """
    src_path = src_path.resolve()
    if (src_path / "lmdb" / "data.mdb").exists():
        return [src_path.as_posix()]

    legacy: list[str] = []
    for ep_path in sorted(p for p in src_path.iterdir() if p.is_dir()):
        out_dir = ep_path / "out"
        if not out_dir.is_dir():
            continue
        for subdir in sorted(p for p in out_dir.iterdir() if p.is_dir()):
            if (subdir / "lmdb" / "data.mdb").exists():
                legacy.append(subdir.as_posix())
    if legacy:
        return legacy

    flat: list[str] = []
    for d in sorted(p for p in src_path.iterdir() if p.is_dir()):
        if (d / "lmdb" / "data.mdb").exists():
            flat.append(d.as_posix())
    return flat

def compute_episode_stats(episode_data: Dict[str, List[str] | np.ndarray], features: Dict) -> Dict:
    ep_stats = {}
    for key, data in episode_data.items():
        if features[key]["dtype"] == "string":
            continue
        elif features[key]["dtype"] in ["image", "video"]:
            ep_ft_array = sample_images(data)
            axes_to_reduce = (0, 2, 3)  # keep channel dim
            keepdims = True
        else:
            ep_ft_array = data  # data is already a np.ndarray
            axes_to_reduce = 0  # compute stats over the first axis
            keepdims = data.ndim == 1  # keep as np.array

        ep_stats[key] = get_feature_stats(ep_ft_array, axis=axes_to_reduce, keepdims=keepdims)
        if features[key]["dtype"] in ["image", "video"]:
            ep_stats[key] = {
                k: v if k == "count" else np.squeeze(v / 255.0, axis=0) for k, v in ep_stats[key].items()
            }
    return ep_stats


def _load_video_frames_chw(video_path: str) -> np.ndarray:
    """Decode all frames as uint8 array [T, C, H, W] (RGB, channel-first). Uses OpenCV so we do not rely on torchvision.io.VideoReader (often unavailable)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(np.transpose(rgb, (2, 0, 1)))
    finally:
        cap.release()
    if not frames:
        raise ValueError(f"No frames decoded from video: {video_path}")
    return np.stack(frames, axis=0)


def sample_images(input):
    if isinstance(input, (str, Path)):
        video_path = os.fspath(input)
        frames_array = _load_video_frames_chw(video_path)
        sampled_indices = sample_indices(len(frames_array))
        images = None
        for i, idx in enumerate(sampled_indices):
            img = frames_array[idx]
            img = auto_downsample_height_width(img)
            if images is None:
                images = np.empty((len(sampled_indices), *img.shape), dtype=np.uint8)
            images[i] = img
    elif type(input) is np.ndarray:
        frames_array = input[:, None, :, :]  # Shape: [T, C, H, W]
        sampled_indices = sample_indices(len(frames_array))
        images = None
        for i, idx in enumerate(sampled_indices):
            img = frames_array[idx]
            img = auto_downsample_height_width(img)
            if images is None:
                images = np.empty((len(sampled_indices), *img.shape), dtype=np.uint8)
            images[i] = img
    return images


def load_local_dataset(episode_path: str, save_path:str, origin_fps=30, target_fps=30):
    fps_factor = origin_fps // target_fps
    # print(f"fps downsample factor: {fps_factor}")
    # logging.info(f"fps downsample factor: {fps_factor}")
    # for format_str in [f"{episode_id:07d}", f"{episode_id:06d}", str(episode_id)]:
    #     episode_path = Path(src_path) / format_str
    #     save_path = Path(save_path) / format_str
    #     if episode_path.exists():
    #         break
    # else:
    #     logging.warning(f"Episode directory not found for ID {episode_id}")
    #     return None, None
    episode_path = Path(episode_path)
    if not episode_path.exists():
        logging.warning(f"{episode_path} does not exist")
        return None, None
        
    if not (episode_path / "lmdb/data.mdb").exists():
        logging.warning(f"LMDB data not found for episode {episode_path}")
        return None, None
    
    raw_dataset = load_lmdb_data(episode_path, save_path, fps_factor, target_fps)
    if raw_dataset is None:
        return None, None
    frames = raw_dataset["frames"] # states, actions, task
    videos = raw_dataset["videos"] # image paths
    ## check the frames
    for camera_name, video_path in videos.items():
        if not os.path.exists(video_path):
            logging.error(f"Video file {video_path} does not exist.")
            print(f"Camera {camera_name} Video file {video_path} does not exist.")
            return None, None
    return frames, videos


def save_as_lerobot_dataset(task: tuple[Path, Path], repo_id, num_threads, debug, origin_fps=30,  target_fps=30, robot_type="piper", delete_downsampled_videos=True):
    src_path, save_path = task
    print(f"**Processing collected** {src_path}")
    print(f"**saving to** {save_path}")
    if save_path.exists():
        print(f"Output directory {save_path} already exists, removing it.")
        logging.warning(f"Output directory {save_path} already exists, removing it.")
        shutil.rmtree(save_path)
        # print(f"Output directory {save_path} already exists.")
        # return 

    dataset = ARXLift2Dataset.create(
        repo_id=f"{repo_id}",
        root=save_path,
        fps=target_fps,
        robot_type=robot_type,
        features=FEATURES,
    )
    try:
        _run_save_episodes_into_dataset(
            dataset,
            src_path,
            save_path,
            num_threads,
            debug,
            origin_fps,
            target_fps,
            delete_downsampled_videos,
        )
    finally:
        fin = getattr(dataset, "finalize", None)
        if callable(fin):
            fin()


def _run_save_episodes_into_dataset(
    dataset: ARXLift2Dataset,
    src_path: Path,
    save_path: Path,
    num_threads: int,
    debug: bool,
    origin_fps: int,
    target_fps: int,
    delete_downsampled_videos: bool,
) -> None:
    all_episode_paths = discover_episode_dirs(src_path)
    if not all_episode_paths:
        raise FileNotFoundError(
            f"No LMDB episodes found under {src_path}. Expected one of:\n"
            "  - out_updated layout : <src_path>/000000/lmdb/data.mdb, ...\n"
            "  - single episode     : <src_path>/lmdb/data.mdb\n"
            "  - legacy layout      : <src_path>/<run>/out/<id>/lmdb/data.mdb"
        )
    print(f"Found {len(all_episode_paths)} episode(s): {all_episode_paths[:5]}{'...' if len(all_episode_paths) > 5 else ''}")

    if debug:
        for i in range(1):
            frames, videos = load_local_dataset(episode_path=all_episode_paths[i], save_path=save_path, origin_fps=origin_fps, target_fps=target_fps)
            if frames is None or videos is None:
                print(f"Skipping episode {all_episode_paths[i]} due to missing data.")
                continue
            for frame_data in frames:
                dataset.add_frame(frame_data)
            dataset.save_episode(videos=videos)
            if delete_downsampled_videos:
                for _, video_path in videos.items():
                    parent_dir = os.path.dirname(video_path)
                    try:
                        shutil.rmtree(parent_dir)
                        # os.remove(video_path)
                        # print(f"Successfully deleted: {parent_dir}")
                        print(f"Successfully deleted: {video_path}")
                    except Exception as e:
                        pass  # Handle the case where the directory might not exist or is already deleted

    else:
        counter_episodes_uncomplete = 0
        for batch_index in range(len(all_episode_paths) // num_threads + 1):
            batch_episode_paths = all_episode_paths[batch_index * num_threads : (batch_index + 1) * num_threads]
            if len(batch_episode_paths) == 0:
                continue
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                future_to_path = {
                    executor.submit(
                        load_local_dataset,
                        episode_path=episode_path,
                        save_path=save_path,
                        origin_fps=origin_fps,
                        target_fps=target_fps,
                    ): episode_path
                    for episode_path in batch_episode_paths
                }
                results: Dict[str, Any] = {}
                for fut in as_completed(future_to_path):
                    ep = future_to_path[fut]
                    results[ep] = fut.result()
                # LeRobotDataset is not thread-safe; write episodes in stable order.
                for episode_path in batch_episode_paths:
                    frames, videos = results[episode_path]
                    if frames is None or videos is None:
                        counter_episodes_uncomplete += 1
                        print(f"Skipping episode {episode_path} due to missing data.")
                        continue
                    for frame_data in frames:
                        dataset.add_frame(frame_data)
                    dataset.save_episode(videos=videos)
                    gc.collect()
                    print(f"Finished episode {episode_path}")
                    if delete_downsampled_videos:
                        for _, video_path in videos.items():
                            parent_dir = os.path.dirname(video_path)
                            try:
                                shutil.rmtree(parent_dir)
                                print(f"Successfully deleted: {parent_dir}")
                            except Exception:
                                pass
        print("counter_episodes_uncomplete:", counter_episodes_uncomplete)

def main(src_path, save_path, repo_id, num_threads=4, debug=False, origin_fps=30, target_fps=30):
    logging.info("Scanning for episodes ...")
    tasks = get_all_tasks(src_path, save_path)
    if debug:
        task = next(tasks)
        save_as_lerobot_dataset(task, repo_id, num_threads=num_threads, debug=debug, origin_fps=origin_fps, target_fps=target_fps)
    else:
        for task in tasks:
            save_as_lerobot_dataset(task, repo_id, num_threads=num_threads, debug=debug, origin_fps=origin_fps, target_fps=target_fps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert LMDB + meta_info (Step 4 out_updated) to LeRobot v2 dataset."
    )
    parser.add_argument(
        "--src_path",
        type=str,
        required=True,
        help="Parent of episode dirs, e.g. replay/my_run_0001/out_updated (contains 000000/lmdb, ...), "
             "or a single episode directory .../out_updated/000000.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Output LeRobot dataset root (created from scratch; existing directory is removed).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Process only the first discovered episode (smoke test).",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=4,
        help="Parallel LMDB/video decode workers per batch (I/O bound); dataset writes are sequential.",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="arx_sim_local",
        help="LeRobot dataset repo_id string stored in metadata.",
    )
    parser.add_argument(
        "--origin_fps",
        type=int,
        default=60,
        help="Frame rate of the source LMDB / Step-4 render output (default: 60).",
    )
    parser.add_argument(
        "--target_fps",
        type=int,
        default=60,
        help="Frame rate written into the LeRobot dataset; must evenly divide origin_fps (default: 60).",
    )
    args = parser.parse_args()
    assert int(args.origin_fps) % int(args.target_fps) == 0, \
        "origin_fps must be an integer multiple of target_fps"
    start_time = time.time()
    main(
        src_path=Path(args.src_path),
        save_path=Path(args.save_path),
        repo_id=args.repo_id,
        num_threads=args.num_threads,
        debug=args.debug,
        origin_fps=args.origin_fps,
        target_fps=args.target_fps
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total time taken: {elapsed_time:.2f} seconds")

    