import os
import sys

# MeisterRender uses flat imports (engine, api, …); add vendored tree to path.
_MEISTER_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MeisterRender")
if _MEISTER_ROOT not in sys.path:
    sys.path.insert(0, _MEISTER_ROOT)

import numpy as np
import cv2
import lmdb
import pickle
import imageio

from pathlib import Path

from api import RenderEngine
from tqdm import tqdm


def _save_mp4_preview(path: str, frames: list, fps: int = 60) -> None:
    """Write MP4; imageio needs imageio[ffmpeg]. Falls back to OpenCV VideoWriter."""
    if not frames:
        return
    try:
        imageio.mimsave(path, frames, fps=fps)
        return
    except (ValueError, OSError) as e:
        print(f"[step_04] imageio MP4 failed ({e}); trying OpenCV VideoWriter ...")
    h, w = int(frames[0].shape[0]), int(frames[0].shape[1])
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV VideoWriter could not open {path} (install ffmpeg or check codecs)")
    for f in frames:
        bgr = cv2.cvtColor(f.astype(np.uint8), cv2.COLOR_RGB2BGR)
        writer.write(bgr)
    writer.release()


def step_04(language_instruction, usd_name, render_blender_file, root_dir):
    """
    Align with render_acone_updated.py:
    - Output root: <root_dir>/out_updated/<usd_name>/ (not out/ + timestamp)
    - LMDB: json_data (intrinsics), robot2env_pose, camera2env_pose + proprio/action/images
    - NPZ joint_q / openness / base_transform truncated to rendered frame count
    - demo.mp4 at fps=60
    """
    exts_path = os.path.join(root_dir, "camera", f"{usd_name}_exts.npz")
    npz_path = os.path.join(root_dir, "npz", f"{usd_name}.npz")
    save_root_path = os.path.join(root_dir, "out_updated")

    render_engine = RenderEngine("MeshPathtracer")
    # "MeshRasterizer"/"MeshRaytracer"/"MeshPathtracer"

    render_engine.engine_launch()
    render_engine.Load_scene(render_blender_file)

    k_matrix = np.array(
        [
            [433.89, 0.0, 320],
            [0.0, 433.38, 240],
            [0.0, 0.0, 1.0],
        ]
    )
    k_matrix_w = np.array(
        [
            [433.89, 0.0, 320],
            [0.0, 433.38, 240],
            [0.0, 0.0, 1.0],
        ]
    )
    frame_action = {
        "camera_action": {
            "Camera": {"k_matrix": k_matrix},
            "Camera.001": {"k_matrix": k_matrix_w},
            "Camera.002": {"k_matrix": k_matrix_w},
        }
    }
    render_engine.render_frame(frame_action)

    cams = np.load(exts_path)
    primary_exts = cams["primary_exts"]
    left_wrist_exts = cams["left_wrist_exts"]
    right_wrist_exts = cams["right_wrist_exts"]
    n_frames = int(primary_exts.shape[0])
    print(
        f"[step_04] Path tracing {n_frames} frames × 3 cameras (MeshPathtracer); "
        "first frames load textures/HDRIs — Blender logs may appear before tqdm moves."
    )

    primary_images = []
    left_wrist_images = []
    right_wrist_images = []

    for idx in tqdm(range(n_frames), desc="Step4 render (path trace)", unit="frame"):
        frame_action = {
            "camera_action": {
                "Camera": {"camera_poses": [primary_exts[idx]]},
                "Camera.001": {"camera_poses": [left_wrist_exts[idx]]},
                "Camera.002": {"camera_poses": [right_wrist_exts[idx]]},
            }
        }
        render_result = render_engine.render_frame(frame_action)
        primary_images.append(render_result["rgb_image"]["Camera"][0][:, :, :3])
        left_wrist_images.append(render_result["rgb_image"]["Camera.001"][0][:, :, :3])
        right_wrist_images.append(render_result["rgb_image"]["Camera.002"][0][:, :, :3])

    render_engine.engine_finish()

    num_frames = len(primary_images)
    npz_file = np.load(npz_path, allow_pickle=True)
    joint_q = npz_file["joint_q"][:num_frames]
    openness = npz_file["openness"][:num_frames]
    if "base_transform" in npz_file:
        base_transform = np.array(npz_file["base_transform"][:num_frames])
    else:
        base_transform = np.zeros((num_frames, 7), dtype=np.float64)
        print("[step_04] Warning: 'base_transform' missing in npz; using zeros for robot2env_pose offset.")

    x_off, y_off, z_off = (float(base_transform[0][0]), float(base_transform[0][1]), float(base_transform[0][2]))
    T_off = np.array(
        [
            [1.0, 0.0, 0.0, x_off],
            [0.0, 1.0, 0.0, y_off],
            [0.0, 0.0, 1.0, z_off],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    max_size = int(1 * 1024**4)
    save_root_path = Path(save_root_path)
    # Folder name = record id (usd_name), same as render_acone_updated.py
    save_dir = save_root_path / usd_name
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[step_04] Writing LMDB / MP4 under {save_dir.resolve()} ...")

    log_path_lmdb = save_dir / "lmdb"
    meta_info = {
        "keys": {"proprio_data": [], "action_data": []},
        "max_size": max_size,
        "language_instruction": language_instruction,
        "detailed_language_instruction": language_instruction,
    }

    lmdb_env = lmdb.open(str(log_path_lmdb), map_size=max_size)
    txn = lmdb_env.begin(write=True)

    ### intrinsics ###
    intrinsics_params = {
        "hand_left_camera_params": k_matrix_w,
        "hand_right_camera_params": k_matrix_w,
        "head_camera_params": k_matrix,
    }
    txn.put("json_data".encode("utf-8"), pickle.dumps(intrinsics_params))

    ### robot2env_pose ###
    robot2env_pose_tfs = [T_off for _ in range(len(primary_images))]
    txn.put(b"robot2env_pose", pickle.dumps(robot2env_pose_tfs))

    ### camera_extrinsics (per frame: left, right, head) ###
    camera2env_pose_tfs = []
    for frame_idx in range(len(primary_images)):
        primary_ext = primary_exts[frame_idx]
        hand_left_ext = left_wrist_exts[frame_idx]
        hand_right_ext = right_wrist_exts[frame_idx]
        camera2env_pose_tfs.append(hand_left_ext)
        camera2env_pose_tfs.append(hand_right_ext)
        camera2env_pose_tfs.append(primary_ext)
    txn.put(b"camera2env_pose", pickle.dumps(camera2env_pose_tfs))

    ### proprio ###
    left_qpos = [qpos[3:9] for qpos in joint_q]
    txn.put(b"states.left_joint.position", pickle.dumps(left_qpos))
    meta_info["keys"]["proprio_data"].append(b"states.left_joint.position")

    right_qpos = [qpos[11:17] for qpos in joint_q]
    txn.put(b"states.right_joint.position", pickle.dumps(right_qpos))
    meta_info["keys"]["proprio_data"].append(b"states.right_joint.position")

    left_gripper_position = [qpos[9:10] for qpos in joint_q]
    txn.put(b"states.left_gripper.position", pickle.dumps(left_gripper_position))
    meta_info["keys"]["proprio_data"].append(b"states.left_gripper.position")

    right_gripper_position = [qpos[17:18] for qpos in joint_q]
    txn.put(b"states.right_gripper.position", pickle.dumps(right_gripper_position))
    meta_info["keys"]["proprio_data"].append(b"states.right_gripper.position")

    ### action ###
    left_qpos_action = left_qpos[:-1] + [left_qpos[-1]]
    txn.put(b"actions.left_joint.position", pickle.dumps(left_qpos_action))
    txn.put(b"master_actions.left_joint.position", pickle.dumps(left_qpos_action))
    meta_info["keys"]["action_data"].append(b"actions.left_joint.position")
    meta_info["keys"]["action_data"].append(b"master_actions.left_joint.position")

    right_qpos_action = right_qpos[:-1] + [right_qpos[-1]]
    txn.put(b"actions.right_joint.position", pickle.dumps(right_qpos_action))
    txn.put(b"master_actions.right_joint.position", pickle.dumps(right_qpos_action))
    meta_info["keys"]["action_data"].append(b"actions.right_joint.position")
    meta_info["keys"]["action_data"].append(b"master_actions.right_joint.position")

    left_gripper_position_action = left_gripper_position[:-1] + [left_gripper_position[-1]]
    txn.put(b"actions.left_gripper.position", pickle.dumps(left_gripper_position_action))
    txn.put(b"master_actions.left_gripper.position", pickle.dumps(left_gripper_position_action))
    meta_info["keys"]["action_data"].append(b"actions.left_gripper.position")
    meta_info["keys"]["action_data"].append(b"master_actions.left_gripper.position")

    right_gripper_position_action = right_gripper_position[:-1] + [right_gripper_position[-1]]
    txn.put(b"actions.right_gripper.position", pickle.dumps(right_gripper_position_action))
    txn.put(b"master_actions.right_gripper.position", pickle.dumps(right_gripper_position_action))
    meta_info["keys"]["action_data"].append(b"actions.right_gripper.position")
    meta_info["keys"]["action_data"].append(b"master_actions.right_gripper.position")

    left_openness = openness[:, 0].tolist()
    right_openness = openness[:, 1].tolist()

    txn.put(b"actions.left_gripper.openness", pickle.dumps(left_openness))
    txn.put(b"master_actions.left_gripper.openness", pickle.dumps(left_openness))
    meta_info["keys"]["action_data"].append(b"actions.left_gripper.openness")
    meta_info["keys"]["action_data"].append(b"master_actions.left_gripper.openness")

    txn.put(b"actions.right_gripper.openness", pickle.dumps(right_openness))
    txn.put(b"master_actions.right_gripper.openness", pickle.dumps(right_openness))
    meta_info["keys"]["action_data"].append(b"actions.right_gripper.openness")
    meta_info["keys"]["action_data"].append(b"master_actions.right_gripper.openness")

    ### images ###
    head_key = "images.rgb.head"
    hand_left_key = "images.rgb.hand_left"
    hand_right_key = "images.rgb.hand_right"

    root_img_path = save_dir / head_key
    root_img_path.mkdir(parents=True, exist_ok=True)
    meta_info["keys"][head_key] = []
    for i, image in enumerate(tqdm(primary_images, desc=head_key)):
        step_id = str(i).zfill(4)
        txn.put(
            f"{head_key}/{step_id}".encode("utf-8"),
            pickle.dumps(cv2.imencode(".jpg", image.astype(np.uint8))[1]),
        )
        meta_info["keys"][head_key].append(f"{head_key}/{step_id}".encode("utf-8"))
    _save_mp4_preview(os.path.join(root_img_path, "demo.mp4"), primary_images, fps=60)

    root_img_path = save_dir / hand_left_key
    root_img_path.mkdir(parents=True, exist_ok=True)
    meta_info["keys"][hand_left_key] = []
    for i, image in enumerate(tqdm(left_wrist_images, desc=hand_left_key)):
        step_id = str(i).zfill(4)
        txn.put(
            f"{hand_left_key}/{step_id}".encode("utf-8"),
            pickle.dumps(cv2.imencode(".jpg", image.astype(np.uint8))[1]),
        )
        meta_info["keys"][hand_left_key].append(f"{hand_left_key}/{step_id}".encode("utf-8"))
    _save_mp4_preview(os.path.join(root_img_path, "demo.mp4"), left_wrist_images, fps=60)

    root_img_path = save_dir / hand_right_key
    root_img_path.mkdir(parents=True, exist_ok=True)
    meta_info["keys"][hand_right_key] = []
    for i, image in enumerate(tqdm(right_wrist_images, desc=hand_right_key)):
        step_id = str(i).zfill(4)
        txn.put(
            f"{hand_right_key}/{step_id}".encode("utf-8"),
            pickle.dumps(cv2.imencode(".jpg", image.astype(np.uint8))[1]),
        )
        meta_info["keys"][hand_right_key].append(f"{hand_right_key}/{step_id}".encode("utf-8"))
    _save_mp4_preview(os.path.join(root_img_path, "demo.mp4"), right_wrist_images, fps=60)

    meta_info["num_steps"] = len(primary_images)
    txn.commit()
    lmdb_env.close()
    pickle.dump(meta_info, open(os.path.join(save_dir, "meta_info.pkl"), "wb"))
    print(f"[step_04] Saved: {save_dir.resolve()}  (lmdb/, meta_info.pkl, */demo.mp4)")
