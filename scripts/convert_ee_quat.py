import numpy as np
import warp as wp
import newton
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from sim1_asset_paths import get_acone_urdf_path

# === Path configuration ===
# These are used only when running this script directly.
# When imported as a module, callers provide paths explicitly.
INPUT_DIR = None
OUTPUT_DIR = None

# === Robot configuration (same as original script) ===
ROBOT_BASE_HEIGHT = 0.17
controllable_joint_names = {
    "joint4",
    "left_joint11", "left_joint12", "left_joint13", "left_joint14", "left_joint15", "left_joint16", 
    "right_joint21", "right_joint22", "right_joint23", "right_joint24", "right_joint25", "right_joint26", 
}
gripper_joint_names = {
    "left_joint17", "left_joint18",
    "right_joint27", "right_joint28",
}
left_ee_body_names = {"left_link16"}
right_ee_body_names = {"right_link26"}

# === Build model and resolve end-effector body indices ===
def build_model_and_get_ee_indices():
    builder = newton.ModelBuilder()
    builder.add_urdf(
        get_acone_urdf_path(),
        floating=False,
        enable_self_collisions=False,
        xform=wp.transform(p=wp.vec3(0.0, 0.0, ROBOT_BASE_HEIGHT))
    )
    lee_index = next(i for i, name in enumerate(builder.body_key) if name in left_ee_body_names)
    ree_index = next(i for i, name in enumerate(builder.body_key) if name in right_ee_body_names)
    model = builder.finalize(requires_grad=False)
    return model, lee_index, ree_index

# === FK + quaternion end-effector pose (instead of Euler angles) ===
def fk_to_quaternion(joint_q, model, lee_index, ree_index):
    state = model.state()
    wp_q = wp.array(joint_q, dtype=float, device=model.device)
    wp_qd = wp.zeros_like(wp_q)
    newton.eval_fk(model, wp_q, wp_qd, state)
    body_q = state.body_q.numpy()

    left_tf = body_q[lee_index]   # [px, py, pz, qw, qx, qy, qz]
    right_tf = body_q[ree_index]

    # Position (3,)
    left_pos = left_tf[:3]
    right_pos = right_tf[:3]

    # Quaternion: convert to [qx, qy, qz, qw] (4,)
    left_quat = np.array([left_tf[4], left_tf[5], left_tf[6], left_tf[3]], dtype=np.float32)
    right_quat = np.array([right_tf[4], right_tf[5], right_tf[6], right_tf[3]], dtype=np.float32)

    # Concatenate to 7D: [x, y, z, qx, qy, qz, qw]
    left_ee = np.concatenate([left_pos, left_quat])   # (7,)
    right_ee = np.concatenate([right_pos, right_quat]) # (7,)

    return left_ee, right_ee

# === Main conversion ===
def convert_ee_quat(input_dir, output_dir):
    model, lee_index, ree_index = build_model_and_get_ee_indices()
    npz_files = [f for f in os.listdir(input_dir) if f.endswith(".npz")]
    os.makedirs(output_dir, exist_ok=True)
    all_vecs = []

    for npz_file in sorted(npz_files):
        input_path = os.path.join(input_dir, npz_file)
        output_path = os.path.join(output_dir, npz_file)

        print(f"Processing {npz_file}...")

        # Load raw trajectory
        data = np.load(input_path)
        joint_q_seq = data["joint_q"]      # (n, dof)
        openness_seq = data["openness"]    # (n, 2) → [left_open, right_open]

        ee_pos_list = []
        for i in range(len(joint_q_seq)):
            joint_q = joint_q_seq[i]
            openness = openness_seq[i]  # [left, right]

            # FK → position + quaternion
            left_ee, right_ee = fk_to_quaternion(joint_q, model, lee_index, ree_index)

            # Append gripper openness (1-D each)
            left_full = np.concatenate([left_ee, [openness[0]]])   # (8,)
            right_full = np.concatenate([right_ee, [openness[1]]]) # (8,)

            frame_ee = np.concatenate([left_full, right_full])    # (16,)
            ee_pos_list.append(frame_ee)

        ee_pos_array = np.array(ee_pos_list, dtype=np.float32)  # (n, 16)
        all_vecs.append(ee_pos_array)

        # Save
        np.savez(output_path, ee_pos=ee_pos_array)
        print(f" Saved: {output_path}")

    cat = np.concatenate(all_vecs, axis=0)  # (N, 16)
    mean = cat.mean(axis=0)
    std = cat.std(axis=0) + 1e-6

    norm_path = os.path.join(output_dir, "norm_stats.npz")
    print("Saving normalization stats to:", norm_path)
    np.savez(norm_path, mean=mean, std=std)

    print(" All done!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert joint trajectories to EE quaternion format")
    parser.add_argument("input_dir", type=str, help="Input directory with .npz files")
    parser.add_argument("output_dir", type=str, help="Output directory for converted .npz files")
    args = parser.parse_args()
    convert_ee_quat(args.input_dir, args.output_dir)