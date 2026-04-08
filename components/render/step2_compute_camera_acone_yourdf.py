# Camera extrinsics with optional pitch randomization on the primary camera
import numpy as np
import sys
import os
from tqdm import tqdm
import yourdfpy

_rdir = os.path.dirname(os.path.abspath(__file__))
if _rdir not in sys.path:
    sys.path.insert(0, _rdir)
from asset_paths import get_acone_urdf_path

def get_relative_pose_with_joints(robot, link_a_name, link_b_name, joint_angles=None):
    """
    Relative transform from link_a to link_b at given joint angles.

    Args:
        joint_angles: dict {joint_name: angle_in_radians}
    """
    if joint_angles:
        robot.update_cfg(joint_angles)

    T_a = robot.get_transform(link_a_name)
    T_b = robot.get_transform(link_b_name)
    
    T_a_inv = np.linalg.inv(T_a)
    T_relative = T_a_inv @ T_b
    
    return T_relative

def step_02(root_dir, filename, urdf_path=None):
    if urdf_path is None:
        urdf_path = get_acone_urdf_path()
    raw_dir = f"npz"
    camera = f"camera"
    os.makedirs(os.path.join(root_dir, camera), exist_ok=True)
    joint_seq_path = os.path.join(root_dir, raw_dir, f"{filename}.npz")

    joint_seq = np.load(joint_seq_path)['joint_q']
    M, N = joint_seq.shape
    print(f"Loaded joint sequence: {M} steps, {N} joints")

    base_transform = np.load(joint_seq_path)['base_transform']
    x_off, y_off, z_off = base_transform[0][:3]

    T_off = np.array([
        [1.0, 0.0, 0.0, x_off],
        [0.0, 1.0, 0.0, y_off],
        [0.0, 0.0, 1.0, z_off],
        [0.0, 0.0, 0.0, 1.0]
    ])

    # Base extrinsics for primary camera
    Tcp = np.array([
        [0.00, 0.87, -0.49, 0.16],
        [-1.00, 0.00, 0.00, 0.00],
        [0.00, 0.49, 0.87, 0.25],
        [0.00, 0.00, 0.00, 1.00],
    ])
    Tcl = np.array([
        [0.00, 0.50, -0.86, 0.06],
        [-1.00, 0.00, 0.00, 0.00],
        [-0.00, 0.86, 0.50, 0.073],
        [0.00, 0.00, 0.00, 1.00]
    ])
    Tcr = np.array([
        [0.00, 0.50, -0.86, 0.06],
        [-1.00, 0.00, 0.00, 0.00],
        [-0.00, 0.86, 0.50, 0.073],
        [0.00, 0.00, 0.00, 1.00]
    ])

    # ===== Shared pitch jitter (rotation about camera +X) =====
    # max_pitch_noise_deg = 5.0  # max jitter in degrees
    max_pitch_noise_deg = 0.0  # max jitter in degrees
    max_pitch_noise_rad = np.deg2rad(max_pitch_noise_deg)
    pitch_noise_rad = np.random.uniform(-max_pitch_noise_rad, max_pitch_noise_rad)
    pitch_noise_deg = np.rad2deg(pitch_noise_rad)
    
    print(f"Camera pitch perturbation applied: {pitch_noise_deg:.2f} degrees")
    print(f"  -> Rotation axis: Camera's local X-axis (pitch)")
    print(f"  -> Effect: Changes camera viewing angle up/down without moving position")
    print(f"  -> Left/right wrist cameras: NO perturbation applied")

    # Rotation about camera +X
    c, s = np.cos(pitch_noise_rad), np.sin(pitch_noise_rad)
    R_pitch = np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s,  c]
    ])

    # Apply jitter on the right (in camera frame)
    Tcp_rot = Tcp[:3, :3]
    Tcp_trans = Tcp[:3, 3]
    Tcp_rot_perturbed = Tcp_rot @ R_pitch  # post-multiply = local camera rotation

    # Full 4x4 extrinsics after jitter
    Tcp_perturbed = np.eye(4)
    Tcp_perturbed[:3, :3] = Tcp_rot_perturbed
    Tcp_perturbed[:3, 3] = Tcp_trans

    robot = yourdfpy.URDF.load(urdf_path)
    primary_poses = np.zeros((M, 4, 4))
    left_wrist_poses = np.zeros((M, 4, 4))
    right_wrist_poses = np.zeros((M, 4, 4))
    
    for step in tqdm(range(M), desc="Computing relative poses with pitch perturbation"):
        joint = joint_seq[step]
        target_joint = {
            "joint1": joint[0], "joint2": joint[1], "joint3": joint[2], "joint4": 0.46,
            "left_joint11": joint[3], "left_joint12": joint[4], "left_joint13": joint[5],
            "left_joint14": joint[6], "left_joint15": joint[7], "left_joint16": joint[8],
            "left_joint17": joint[9], "left_joint18": joint[10],
            "right_joint21": joint[11], "right_joint22": joint[12], "right_joint23": joint[13],
            "right_joint24": joint[14], "right_joint25": joint[15], "right_joint26": joint[16],
            "right_joint27": joint[17], "right_joint28": joint[18]
        }

        T_l = get_relative_pose_with_joints(robot, "base_link", "left_link16", target_joint)
        T_r = get_relative_pose_with_joints(robot, "base_link", "right_link26", target_joint)
        T_p = get_relative_pose_with_joints(robot, "base_link", "body4", target_joint)

        # Primary: perturbed extrinsics
        primary_poses[step] = T_off @ T_p @ Tcp_perturbed
        # Wrist cameras: no jitter
        left_wrist_poses[step] = T_off @ T_l @ Tcl
        right_wrist_poses[step] = T_off @ T_r @ Tcr

    np.savez(os.path.join(root_dir, camera, f"{filename}_exts.npz"), 
             primary_exts=primary_poses, 
             left_wrist_exts=left_wrist_poses, 
             right_wrist_exts=right_wrist_poses)
    print(f"Saved camera extrinsics to {os.path.join(root_dir, camera, f'{filename}_exts.npz')}")
    print(f"Primary camera pitch perturbation: {pitch_noise_deg:.2f}° (fixed for entire sequence)")