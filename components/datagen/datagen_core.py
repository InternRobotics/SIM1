import os
import sys
import numpy as np
from tqdm import tqdm

# Repo root (for sim1_asset_paths — HF bundle: model/flow_ckpt_three.pth)
_DATAGEN_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_DATAGEN_DIR))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from sim1_asset_paths import get_flow_ckpt_three_path

from components.function import fk, solve_ik
from components.datagen.splitter import  Splitter, SplitterFine
from components.datagen.selector import Selector, SelectorFine
from .utils import makedir, interpolate_position, quaternion_lerp, interpolate_linear, read_json_file
import torch
import warp as wp
import newton.ik as ik

class DataGenerator:

    def __init__(self, env, splitter, selector, data_folder, iterpolation_time=1, fps=60,use_dp=False):
        self.env = env
        self.model = env.model
        self.data_folder = data_folder
        self.input_folder = os.path.join(data_folder, 'temp_trajs')
        self.output_folder = os.path.join(data_folder, 'gen')
        self.iterpolation_time = iterpolation_time
        self.fps = fps
        self.ind = 0
        self.use_dp = use_dp

        self.splitter = splitter
        self.selector = selector

        makedir(self.output_folder)

        use_dp = True
        if use_dp:
            from .traj_df.src.utils import Normalizer14
            from .traj_df.src.simple_diffusion import simpleDiffusion
            from .traj_df.src.models.diffusion import UNet
            from scripts.convert_ee_quat import build_model_and_get_ee_indices, convert_ee_quat
            

            norm_path = os.path.join(data_folder, 'ee_pos')
            if os.path.exists(norm_path):
                print("Loading normalization stats from:", norm_path)
            else:
                print("Calculating normalization stats...")
                input_dir = os.path.join(data_folder, 'npz')
                output_dir = os.path.join(data_folder, 'ee_pos')
                convert_ee_quat(input_dir, output_dir)
            self.normalizer = Normalizer14(norm_path)


            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            DiffusionUNet = UNet().to(device) 
            flow = simpleDiffusion(DiffusionUNet).to(device)

            ckpt_path = get_flow_ckpt_three_path()
            if not os.path.isfile(ckpt_path):
                raise FileNotFoundError(
                    f"Missing diffusion checkpoint: {ckpt_path}\n"
                    "Run `bash download_assets.sh` or set SIM1_ASSETS_ROOT to your Hugging Face bundle."
                )
            flow.load(ckpt_path)
            self.flow = flow
            self.fm_his_len = 1

    def generate_new_traj_w_df(self, traj):
        """
        generate_new_traj (DF version):
        - Supports segment + optional intermediate in traj (split into moving + stable).
        - Uses FM (diffusion) for stable segments via flow.infer_step(batch); fallback to npz ground-truth on failure.
        - Output format unchanged: joint_q / openness / base_transform.
        """

        def feature_to_components(pred_feat, src_joint_template):
            """
            inverse of build_feature_vector:
            - pred_feat: (L,16)
            - returns base (L,7), openness (L,2), joint_q_full (L, robot_joint_q_cnt)
            The produced joint_q fills first 7 dims with predicted; remaining dims copied from src_joint_template.
            Modify to match your robot joint mapping.
            """
            L = pred_feat.shape[0]
            pred_base = pred_feat[:, :7]
            pred_openness = pred_feat[:, 7:9]
            pred_joint_take = pred_feat[:, 9:16]  # 7 dims

            # build full joint vector by copying a template and replacing first take_n dims
            fallback = np.asarray(src_joint_template).reshape(-1)
            full = np.tile(fallback.reshape(1, -1), (L, 1)).astype(np.float32)
            take_n = pred_joint_take.shape[1]
            full[:, :take_n] = pred_joint_take
            return pred_base.astype(np.float32), pred_openness.astype(np.float32), full.astype(np.float32)
        

        normalizer = self.normalizer

        flow = self.flow
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # prepare subsegments: split if intermediate exists
        tasks = list(traj.keys())
        subsegments = []
        
        for t_idx, tk in enumerate(tasks):
            info = traj[tk]
            rec = info['record_id']
            s0 = info['segment'][0]
            s1 = info['segment'][1]
            inter = info.get('intermediate', None)
            if inter is None:
                subsegments.append({'record_id': rec, 'start': s0, 'end': s1, 'type': 'stable', 'orig_task': tk})
            else:
                # split into moving (s0->inter) and stable (inter->s1)
                subsegments.append({'record_id': rec, 'start': s0, 'end': inter, 'type': 'moving', 'orig_task': tk})
                subsegments.append({'record_id': rec, 'start': inter, 'end': s1, 'type': 'stable', 'orig_task': tk})

        # init output sequences
        self.new_joint_q_seq = np.empty((0, self.env.robot_joint_q_cnt), dtype=np.float32)
        self.new_ee_pos_seq = np.empty((0, 16), dtype=np.float32)
        segment_labels = []  # 'moving' | 'stable' | 'phase1' for visualization

        # npz cache
        npz_cache = {}
        def load_npz(record_id):
            if record_id in npz_cache:
                return npz_cache[record_id]
            path = os.path.join(self.data_folder, "ee_pos", f"{record_id}.npz")
            d = np.load(path)
            npz_cache[record_id] = d
            return d

        i = 0
        # parameters for FM
        fm_his_len = getattr(self, 'fm_his_len', 8)

        while i < len(subsegments):
            seg = subsegments[i]
            rec = seg['record_id']
            start = seg['start']
            end = seg['end']
            typ = seg['type']

            data = load_npz(rec)
            ee_pos = data[data.files[0]]
            joint_seq = ee_pos

            # if moving -> append raw frames (consistent with original)
            if typ == 'moving':
                seg_start = start
                seg_end = end
                src_joint_slice = joint_seq[seg_start:seg_end+1, :]
                self.new_ee_pos_seq = np.vstack((self.new_ee_pos_seq, src_joint_slice))
                segment_labels.extend(['moving'] * len(src_joint_slice))
                i += 1
                # if last moving segment needs to append next segment's tail we handle at natural loop end similar to original
                if i == len(subsegments):
                    # nothing extra
                    pass
                continue

            # typ == 'stable' : decide whether to use FM
            use_fm = False
            prev_is_moving = (i - 1 >= 0 and subsegments[i-1]['type'] == 'moving')
            next_is_moving = (i + 1 < len(subsegments) and subsegments[i+1]['type'] == 'moving')
            if i == 0 or (prev_is_moving and next_is_moving):
                use_fm = True

            if not use_fm or flow is None:
                # fallback: append ground-truth frames from npz
                # NOTE: original code used interpolation between segments; here when using intermediate we treat stable as own frames
                seg_start = start
                seg_end = end
                if seg_end > seg_start:
                    src_joint_slice = joint_seq[seg_start:seg_end, :]  # stable: frames start..end-1 (same convention as before)
                    self.new_ee_pos_seq = np.vstack((self.new_ee_pos_seq, src_joint_slice))
                    segment_labels.extend(['stable'] * len(src_joint_slice))
                i += 1
                continue

            # use FM to synthesize stable segment
            L = max(0, end - start + 1)  # number of frames to generate
            if L <= 0:
                i += 1
                continue
            
            # build src (stable first frame), tgt (next moving start first frame if exists), traj (ground truth padded), mask, lengths
            src_feat = data['ee_pos'][start]
            if (i+1 < len(subsegments)) and subsegments[i+1]['type'] == 'moving':
                next_data = load_npz(subsegments[i+1]['record_id'])
                tgt_feat = next_data['ee_pos'][subsegments[i+1]['start']]
            else:
                tgt_feat = data['ee_pos'][end if end < len(joint_seq) else len(joint_seq)-1]

            # traj (ground truth features for stable) padded to fm_max_m
            feats = []
            for idx in range(start, end + 1):
                feats.append(data['ee_pos'][idx])
            if len(feats) == 0:
                # nothing to generate, skip
                i += 1
                continue
            feats = np.stack(feats, axis=0)  # (L,16)
            max_m = feats.shape[0]
            traj_padded = np.zeros((max_m, 16), dtype=np.float32)
            traj_padded[:feats.shape[0], :] = feats

            mask = np.zeros((max_m,), dtype=np.float32)
            mask[:feats.shape[0]] = 1.0
            lengths = np.array([feats.shape[0]], dtype=np.int32)

            # history: walk backwards collecting fm_his_len frames (from current record then previous subsegments)
            his_data = self.new_ee_pos_seq
            seq_len = len(his_data)

            if his_data is None or len(his_data) == 0:
                history_list = [src_feat.copy() for _ in range(fm_his_len)]
            elif seq_len >= fm_his_len:
                history_list = his_data[-fm_his_len:]
            else:
                last_elem = his_data[-1]
                pad_len = fm_his_len - seq_len
                history_list = list(his_data) + [last_elem.copy()] * pad_len
            history_np = np.stack(history_list[-fm_his_len:], axis=0)  # (his_len,16)

            batch = {
                'history': np.expand_dims(history_np.astype(np.float32), 0),  # (1, his_len, 16)
                'src': np.expand_dims(src_feat.astype(np.float32), 0),        # (1,16)
                'tgt': np.expand_dims(tgt_feat.astype(np.float32), 0),        # (1,16)
                'traj': np.expand_dims(traj_padded.astype(np.float32), 0),    # (1,max_m,16)
                'mask': np.expand_dims(mask.astype(np.float32), 0),          # (1,max_m)
                'lengths': np.expand_dims(lengths, 0),                       # (1,)
                'record_ids': [rec]
            }

            # normalize if possible
            batch['history'] = normalizer.normalize(batch['history'])
            batch['src'] = normalizer.normalize(batch['src'])
            batch['tgt'] = normalizer.normalize(batch['tgt'])
            batch['traj'] = normalizer.normalize(batch['traj'])

            # move tensors to device if flow expects torch tensors
            # we call flow.infer_step(batch) per your provided API: x_pred, mask = flow.infer_step(batch)
            # try to convert numpy arrays to torch tensors if flow expects torch
            batch_torch = {}
            for k, v in batch.items():
                if isinstance(v, np.ndarray):
                    batch_torch[k] = torch.from_numpy(v).to(device)
                else:
                    batch_torch[k] = v  # e.g., record_ids list

            xpred, pred_mask = flow.infer_step(batch_torch)
            
            # xpred -> np (expected shape (1, max_m, 16) or (max_m,16))
            if isinstance(xpred, torch.Tensor):
                xpred_np = xpred.detach().cpu().numpy()
            else:
                xpred_np = np.asarray(xpred)
            if xpred_np.ndim == 3:
                pred_seq = xpred_np[0]
            elif xpred_np.ndim == 2:
                pred_seq = xpred_np
            else:
                # unexpected shape: fallback
                raise RuntimeError(f"FM returned unexpected shape {xpred_np.shape}")
            # visualize_ee_pos(pred_seq, mask)
            

            L_valid = feats.shape[0]
            pred_valid = pred_seq[:L_valid-1, :]  # (L,16)
            pred_valid = normalizer.denormalize(pred_valid)
            self.new_ee_pos_seq = np.vstack((self.new_ee_pos_seq, pred_valid))
            segment_labels.extend(['stable'] * len(pred_valid))
            i += 1

        # visualize_ee_pos(self.new_ee_pos_seq)
        # input()
        # Smooth EE trajectory: remove jumps + phase1 interpolation
        result = self.smooth_and_interpolate_ee16(
            self.new_ee_pos_seq,
            transition_duration_sec=0.5,
            segment_labels=segment_labels
        )
        self.new_ee_pos_seq, segment_labels = result
        # Visualize moving/stable/phase1 segments and save
        save_path = os.path.join(self.output_folder, f"ee_pos_segments_{self.ind:06d}.png")
        # visualize_segments_ee_pos(self.new_ee_pos_seq, segment_labels, save_path=save_path)

        self.ee16_to_joint()

        start_record_id = subsegments[0]['record_id']
        start_record_path = os.path.join(self.data_folder, "npz", f"{start_record_id}.npz")
        start_record_data = np.load(start_record_path)['base_transform'][0]

        target_len = len(self.new_ee_pos_seq)
        start_record_data = start_record_data[None, ...]   # (1, d)
        self.new_base_transform_seq = np.repeat(start_record_data, target_len, axis=0)

        # Save data (calls save_data)
        self.save_data()
        
    def generate_new_traj(self, traj):
        self.new_joint_q_seq = np.empty((0, self.env.robot_joint_q_cnt), dtype=np.float32)
        self.new_openness_seq = np.empty((0, 2), dtype=np.float32)
        self.new_base_transform_seq = np.empty((0, 7), dtype=np.float32)

        tasks = list(traj.keys())
        for i in range(len(tasks)-1):
            
            # Compute interpolation segment
            start_record_id = traj[tasks[i]]['record_id']
            end_record_id = traj[tasks[i+1]]['record_id']
            start = traj[tasks[i]]['segment'][1]
            end = traj[tasks[i+1]]['segment'][0]

            # Load start segment data
            start_record_path = os.path.join(self.data_folder, "npz", f"{start_record_id}.npz")
            start_record_data = np.load(start_record_path) 
            start_joint_q_seq = start_record_data[start_record_data.files[0]]
            start_openness_seq = start_record_data[start_record_data.files[1]]
            start_joint_q = start_joint_q_seq[start]
            start_openness = start_openness_seq[start]
            
            # Load end segment data
            end_record_path = os.path.join(self.data_folder, "npz", f"{end_record_id}.npz")
            end_record_data = np.load(end_record_path) 
            end_joint_q_seq = end_record_data[end_record_data.files[0]]
            end_openness_seq = end_record_data[end_record_data.files[1]]
            end_joint_q = end_joint_q_seq[end]
            end_openness = end_openness_seq[end]
            # assert 0 <= start < len(start_openness_seq)
            # assert 0 <= end < len(end_openness_seq)

            start_left_tf, start_right_tf = fk(start_joint_q, self.model, self.env.lee_index, self.env.ree_index)
            end_left_tf, end_right_tf = fk(end_joint_q, self.model, self.env.lee_index, self.env.ree_index)
            # Interpolate
            left_pos = interpolate_position(start_left_tf[:3], end_left_tf[:3], duration=self.iterpolation_time, fps=self.fps)
            right_pos = interpolate_position(start_right_tf[:3], end_right_tf[:3], duration=self.iterpolation_time, fps=self.fps)
            left_quat = quaternion_lerp(start_left_tf[3:], end_left_tf[3:], duration=self.iterpolation_time, fps=self.fps)
            right_quat = quaternion_lerp(start_right_tf[3:], end_right_tf[3:], duration=self.iterpolation_time, fps=self.fps)
            left_openness = interpolate_linear(start_openness[0], end_openness[0], duration=self.iterpolation_time, fps=self.fps)
            right_openness = interpolate_linear(start_openness[1], end_openness[1], duration=self.iterpolation_time, fps=self.fps)

            inter_left_tfs = np.concatenate((left_pos[1:-1, :], left_quat[1:-1, :]), axis=1)
            inter_right_tfs = np.concatenate((right_pos[1:-1, :], right_quat[1:-1, :]), axis=1)
            inter_openness = np.column_stack((left_openness[1:-1], right_openness[1:-1]))
            
            inter_joint_q_seq = np.empty((0, len(start_joint_q)), dtype=np.float32)  

            base_transform = start_record_data[start_record_data.files[2]][0] 
            inter_base_transform_seq = np.empty((0, 7), dtype=np.float32)
                
            for left_tf, right_tf, openness in zip(inter_left_tfs, inter_right_tfs, inter_openness):
                new_joint_q = solve_ik(
                    left_tf, right_tf, openness,
                    self.model, self.env.lee_index, self.env.ree_index,
                    self.env.controllable_joint_indices, 
                    self.env.left_gripper_joint_indices, 
                    self.env.right_gripper_joint_indices
                )
  
                inter_joint_q_seq = np.vstack((inter_joint_q_seq, new_joint_q))
                inter_base_transform_seq = np.vstack((inter_base_transform_seq, base_transform))
            
            start = traj[tasks[i]]['segment'][0]
            end = traj[tasks[i]]['segment'][1]
            src_joint_q_seq = start_joint_q_seq[start:end+1, :]
            src_openness_seq = start_openness_seq[start:end+1, :]
            src_base_transform_seq = start_record_data[start_record_data.files[2]][start:end+1, :]
            
            self.new_joint_q_seq = np.vstack((self.new_joint_q_seq, src_joint_q_seq))
            self.new_joint_q_seq = np.vstack((self.new_joint_q_seq, inter_joint_q_seq))
            self.new_openness_seq = np.vstack((self.new_openness_seq, src_openness_seq))
            self.new_openness_seq = np.vstack((self.new_openness_seq, inter_openness))
            self.new_base_transform_seq = np.vstack((self.new_base_transform_seq, src_base_transform_seq))
            self.new_base_transform_seq = np.vstack((self.new_base_transform_seq, inter_base_transform_seq))
            
            if i == len(tasks)-2:
                
                start = traj[tasks[i+1]]['segment'][0]
                end = traj[tasks[i+1]]['segment'][1]
                src_joint_q_seq = end_joint_q_seq[start:end+1, :]
                src_openness_seq = end_openness_seq[start:end+1, :]
                src_base_transform_seq = end_record_data[end_record_data.files[2]][start:end+1, :]
                self.new_joint_q_seq = np.vstack((self.new_joint_q_seq, src_joint_q_seq))
                self.new_openness_seq = np.vstack((self.new_openness_seq, src_openness_seq))
                self.new_base_transform_seq = np.vstack((self.new_base_transform_seq, src_base_transform_seq))
        
        self.save_data()
        
    def ee16_to_joint(self):
        """
        Convert sequence of 16-D end-effector vectors -> joint angle sequence using IK.

        Input per frame (16 dims):
        [ lx, ly, lz,
            lqx, lqy, lqz, lqw,  # left quat order produced by your fk_to_quaternion
            l_open,
            rx, ry, rz,
            rqx, rqy, rqz, rqw,  # right quat
            r_open ]

        Output:
        joint_q_seq: (N, model.joint_coord_count) numpy array (float32)
        """
        model = self.model
        device = model.device
        ee16 = self.new_ee_pos_seq

        # Ensure ee16 is numpy array
        ee16 = np.asarray(ee16, dtype=np.float32)
        # Handle various shapes to avoid len() on scalar TypeError
        if ee16.ndim == 0:
            # No valid trajectory, return empty
            return np.zeros((0, model.joint_coord_count), dtype=np.float32)
        if ee16.ndim == 1:
            ee16 = ee16[np.newaxis, :]

        N = ee16.shape[0]
        if N == 0:
            return np.zeros((0, model.joint_coord_count), dtype=np.float32)

        # indices from env
        lee_index = self.env.lee_index
        ree_index = self.env.ree_index
        controllable_indices = self.env.controllable_joint_indices
        left_gripper_indices = self.env.left_gripper_joint_indices
        right_gripper_indices = self.env.right_gripper_joint_indices

        # gripper limits (same as your code)
        GRIPPER_LIMIT_LOWER = 0.001
        GRIPPER_LIMIT_UPPER = 0.044

        joint_q_seq = []

        for i in range(N):
            vec = ee16[i]  # length 16

            # parse left and right
            # left: pos(3) + quat(qx,qy,qz,qw) + open
            left_pos = vec[0:3]
            left_qx, left_qy, left_qz, left_qw = vec[3:7]
            left_open = vec[7]

            # right: starts at 8
            right_pos = vec[8:11]
            right_qx, right_qy, right_qz, right_qw = vec[11:15]
            right_open = vec[15]

            # convert to wp.vec4 order (qw, qx, qy, qz) as required in your IK
            target_lee_pos = wp.array([wp.vec3(left_pos[0], left_pos[1], left_pos[2])],
                                    dtype=wp.vec3, device=device)
            target_lee_rot = wp.array([wp.vec4(left_qw, left_qx, left_qy, left_qz)],
                                    dtype=wp.vec4, device=device)

            target_ree_pos = wp.array([wp.vec3(right_pos[0], right_pos[1], right_pos[2])],
                                    dtype=wp.vec3, device=device)
            target_ree_rot = wp.array([wp.vec4(right_qw, right_qx, right_qy, right_qz)],
                                    dtype=wp.vec4, device=device)

            # --- IK objectives (same as your code) ---
            total_residuals = 2 * 6 + model.joint_coord_count
            l_pos_obj = ik.IKPositionObjective(
                link_index=lee_index, link_offset=wp.vec3(0.0),
                target_positions=target_lee_pos,
                n_problems=1, total_residuals=total_residuals, residual_offset=0,
            )
            l_rot_obj = ik.IKRotationObjective(
                link_index=lee_index, link_offset_rotation=wp.quat_identity(),
                target_rotations=target_lee_rot,
                n_problems=1, total_residuals=total_residuals, residual_offset=3,
            )
            r_pos_obj = ik.IKPositionObjective(
                link_index=ree_index, link_offset=wp.vec3(0.0),
                target_positions=target_ree_pos,
                n_problems=1, total_residuals=total_residuals, residual_offset=6,
            )
            r_rot_obj = ik.IKRotationObjective(
                link_index=ree_index, link_offset_rotation=wp.quat_identity(),
                target_rotations=target_ree_rot,
                n_problems=1, total_residuals=total_residuals, residual_offset=9,
            )
            obj_joint_limits = ik.IKJointLimitObjective(
                joint_limit_lower=model.joint_limit_lower,
                joint_limit_upper=model.joint_limit_upper,
                n_problems=1, total_residuals=total_residuals,
                residual_offset=12, weight=10.0,
            )

            # --- Solve IK ---
            ik_joint_q = wp.zeros((1, model.joint_coord_count), dtype=float, device=device)
            solver = ik.IKSolver(
                model=model,
                joint_q=ik_joint_q,
                objectives=[l_pos_obj, l_rot_obj, r_pos_obj, r_rot_obj, obj_joint_limits],
                lambda_initial=0.1,
                jacobian_mode=ik.IKJacobianMode.MIXED,
            )
            solver.solve(iterations=24)

            # --- Attach gripper openness ---
            result = ik_joint_q.numpy()[0].copy()

            # map openness scalar [0..1] (or if already physical pos) to gripper servo pos
            # However earlier you used recorder opennes -> left_pos = lower + open*(upper-lower)
            left_pos_val = GRIPPER_LIMIT_LOWER + float(left_open) * (GRIPPER_LIMIT_UPPER - GRIPPER_LIMIT_LOWER)
            right_pos_val = GRIPPER_LIMIT_LOWER + float(right_open) * (GRIPPER_LIMIT_UPPER - GRIPPER_LIMIT_LOWER)

            result[left_gripper_indices] = left_pos_val
            result[right_gripper_indices] = right_pos_val

            joint_q_seq.append(result)
        self.new_joint_q_seq = np.array(joint_q_seq, dtype=np.float32)
        self.new_openness_seq = np.array(ee16[:, [7,-1]], dtype=np.float32)
        
        # return np.array(joint_q_seq, dtype=np.float32)

    # def save_data(self):
    #     file_name = os.path.join(self.output_folder, '{:0>6d}.npz'.format(self.ind))

    #     np.savez(
    #             file_name,
    #             joint_q=self.new_joint_q_seq,
    #             openness=self.new_openness_seq,
    #             base_transform=self.new_base_transform_seq
    #         )
    
    def save_data(self):
        file_name = os.path.join(self.output_folder, '{:0>6d}.npz'.format(self.ind))
        
        # 1.5x frame-rate interpolation (slow down arm)
        original_length = len(self.new_joint_q_seq)
        if original_length > 0:
            target_length = int(np.ceil(original_length * 1.5))

            def interpolate_2d_sequence(seq, target_len):
                if seq.shape[0] <= 1 or target_len <= seq.shape[0]:
                    return seq
                old_x = np.arange(seq.shape[0])
                new_x = np.linspace(0, seq.shape[0] - 1, target_len)
                new_seq = np.zeros((target_len, seq.shape[1]), dtype=np.float32)
                for i in range(seq.shape[1]):
                    new_seq[:, i] = np.interp(new_x, old_x, seq[:, i])
                return new_seq
            
            # Interpolate the three key sequences
            self.new_joint_q_seq = interpolate_2d_sequence(self.new_joint_q_seq, target_length)
            self.new_openness_seq = interpolate_2d_sequence(self.new_openness_seq, target_length)
            self.new_base_transform_seq = interpolate_2d_sequence(self.new_base_transform_seq, target_length)
            
            print(f"[Speed Adjustment] Trajectory slowed down: {original_length} frames → {target_length} frames (1.5x duration)")
        # ==============================================
        
        # Save interpolated data (structure unchanged)
        np.savez(
            file_name,
            joint_q=self.new_joint_q_seq,
            openness=self.new_openness_seq,
            base_transform=self.new_base_transform_seq
        )
    
    def _estimate_frames_per_meter_from_ee_pos(self, vel_threshold=1e-4):
        """From reference ee_pos, estimate frames-per-meter for dynamic interpolation."""
        if hasattr(self, "_frames_per_meter_cache") and self._frames_per_meter_cache is not None:
            return self._frames_per_meter_cache

        data_folder = getattr(self, "data_folder", None)
        if data_folder is None:
            # Fallback when reference data unavailable: ~30 frames per meter
            self._frames_per_meter_cache = 30.0
            return self._frames_per_meter_cache

        ee_dir = os.path.join(data_folder, "ee_pos")
        if not os.path.isdir(ee_dir):
            # Reference dir not found, use default
            self._frames_per_meter_cache = 30.0
            return self._frames_per_meter_cache

        total_dist = 0.0
        total_frames = 0

        for fname in os.listdir(ee_dir):
            if not fname.endswith(".npz"):
                continue
            fpath = os.path.join(ee_dir, fname)
            try:
                d = np.load(fpath)
            except Exception:
                continue

            # Support different key names
            if "ee_pos" in d.files:
                ee = d["ee_pos"]
            else:
                ee = d[d.files[0]]

            # Only accept 2D (T, 16) or (T, D)
            if ee.ndim != 2 or ee.shape[0] < 2 or ee.shape[1] < 11:
                continue

            left_pos_ref = ee[:, :3]
            right_pos_ref = ee[:, 8:11]
            vel_left_ref = np.linalg.norm(np.diff(left_pos_ref, axis=0), axis=1)
            vel_right_ref = np.linalg.norm(np.diff(right_pos_ref, axis=0), axis=1)
            max_vel_ref = np.maximum(vel_left_ref, vel_right_ref)

            # Only count frames with real motion (filter near-static noise)
            moving_mask = max_vel_ref > vel_threshold
            if not np.any(moving_mask):
                continue

            total_dist += float(max_vel_ref[moving_mask].sum())
            total_frames += int(moving_mask.sum())

        # Use default ratio if no valid stats
        if total_dist <= 0.0 or total_frames <= 0:
            self._frames_per_meter_cache = 30.0
        else:
            self._frames_per_meter_cache = total_frames / total_dist

        return self._frames_per_meter_cache

    def smooth_and_interpolate_ee16(self, ee16_seq, transition_duration_sec=1.0, segment_labels=None):
        """
        For 16D EE sequence: insert phase1 linear interpolation only at stable-moving boundaries.
        If segment_labels given, return (smoothed, new_labels) for visualization.
        """
        if len(ee16_seq) == 0:
            return (ee16_seq, segment_labels) if segment_labels is not None else ee16_seq

        fps = self.fps
        # Time-based minimum 0; interpolation frames driven by jump distance
        base_trans_frames = 0

        # Estimate frames per meter from reference ee_pos
        frames_per_meter = self._estimate_frames_per_meter_from_ee_pos()


        if len(ee16_seq) < 2:
            return (ee16_seq, segment_labels) if segment_labels is not None else ee16_seq

        # Interpolate only at stable-moving boundaries (segment type change)
        if segment_labels is not None and len(segment_labels) == len(ee16_seq):
            segment_labels_arr = np.array(segment_labels)
            boundary = segment_labels_arr[:-1] != segment_labels_arr[1:]
            jump_indices = np.where(boundary)[0]
        else:
            jump_indices = np.array([], dtype=np.int64)

        # If no jump detected, return original (with first-order smoothing)
        if len(jump_indices) == 0:
            smoothed = self._apply_first_order_smoothing(ee16_seq)
            return (smoothed, segment_labels) if segment_labels is not None else smoothed

        # Insert transition: two-phase interpolation at each jump
        new_seq = []
        new_labels = [] if segment_labels is not None else None
        last_end = 0
        
        for idx in jump_indices:
            # Append all frames before jump
            chunk = ee16_seq[last_end:idx+1]
            new_seq.append(chunk)
            if new_labels is not None:
                new_labels.extend(segment_labels[last_end:idx+1])
            
            # Start and end frames of jump
            start_frame = ee16_seq[idx].copy()
            end_frame = ee16_seq[idx+1].copy()

            # Spatial distance of jump (both EEs)
            start_pos = np.concatenate([start_frame[:3], start_frame[8:11]])
            end_pos = np.concatenate([end_frame[:3], end_frame[8:11]])
            jump_dist = float(np.linalg.norm(end_pos - start_pos))

            # Interpolation frames from jump distance (larger distance -> more frames)
            dynamic_trans_frames = int(max(base_trans_frames, jump_dist * frames_per_meter))
            if dynamic_trans_frames < 2:
                dynamic_trans_frames = 2

            # Phase1: position interpolation, gripper at start; last frame = end position for phase2
            ts_phase1 = np.linspace(0, 1, dynamic_trans_frames + 2)[1:]
            phase1_frames = []
            for t in ts_phase1:
                interp_frame = start_frame.copy()
                for dim in range(len(start_frame)):
                    if dim != 7 and dim != 15:
                        interp_frame[dim] = (1 - t) * start_frame[dim] + t * end_frame[dim]
                phase1_frames.append(interp_frame)
            if phase1_frames:
                # First frame = start_frame (connects to previous chunk)
                for dim in range(len(start_frame)):
                    if dim != 7 and dim != 15:
                        phase1_frames[0][dim] = start_frame[dim]
                # Last frame = end_frame (connects to phase2)
                for dim in range(len(start_frame)):
                    if dim != 7 and dim != 15:
                        phase1_frames[-1][dim] = end_frame[dim]

            if phase1_frames:
                new_seq.append(np.stack(phase1_frames, axis=0))
                if new_labels is not None:
                    new_labels.extend(['phase1'] * len(phase1_frames))

            # Phase2: gripper interpolation, position at end_frame; last frame = end_frame
            ts_phase2 = np.linspace(0, 1, dynamic_trans_frames + 2)[1:]
            phase2_frames = []
            for t in ts_phase2:
                interp_frame = end_frame.copy()
                interp_frame[7] = (1 - t) * start_frame[7] + t * end_frame[7]
                interp_frame[15] = (1 - t) * start_frame[15] + t * end_frame[15]
                phase2_frames.append(interp_frame)
            if phase2_frames:
                # First frame = end position + start gripper (connects to phase1)
                phase2_frames[0][7] = start_frame[7]
                phase2_frames[0][15] = start_frame[15]
                # Last frame = end_frame (connects to next segment)
                phase2_frames[-1] = end_frame.copy()

            if phase2_frames:
                new_seq.append(np.stack(phase2_frames, axis=0))
                if new_labels is not None:
                    new_labels.extend(['phase2'] * len(phase2_frames))

            last_end = idx + 1
        
        # Append final segment
        new_seq.append(ee16_seq[last_end:])
        if new_labels is not None:
            new_labels.extend(segment_labels[last_end:])
        smoothed = np.concatenate(new_seq, axis=0)
        # First-order smoothing on result (position dims; dims 7/15 unchanged)
        smoothed = self._apply_first_order_smoothing(smoothed)
        return (smoothed, new_labels) if new_labels is not None else smoothed

    # def _apply_first_order_smoothing(self, seq, window_length=21, polyorder=2):
    #     """Savitzky-Golay for smooth velocity (first-order continuity).
    #     """
    #     from scipy.signal import savgol_filter
    #     if len(seq) < window_length or window_length <= polyorder:
    #         return seq
    #     try:
    #         if window_length % 2 == 0:
    #             window_length -= 1
    #         smoothed = savgol_filter(seq, window_length=window_length, polyorder=polyorder, axis=0, mode='interp')
    #         return smoothed.astype(np.float32)
    #     except Exception as e:
    #         print(f"[Warning] Smoothing failed: {e}. Returning original.")
    #         return seq
    
    # First-order derivative smoothing for jump removal
    def _apply_first_order_smoothing(self, seq, window_length=21, polyorder=2):
        """
        Savitzky-Golay for smooth velocity. Dims 7 and 15 (gripper) unchanged; others smoothed.
        """
        from scipy.signal import savgol_filter
        if len(seq) < window_length or window_length <= polyorder:
            return seq
        try:
            if window_length % 2 == 0:
                window_length -= 1

            smoothed = seq.copy()
            position_dims = [i for i in range(seq.shape[1]) if i != 7 and i != 15]

            if position_dims:
                smoothed[:, position_dims] = savgol_filter(
                    seq[:, position_dims], 
                    window_length=window_length, 
                    polyorder=polyorder, 
                    axis=0, 
                    mode='interp'
                )

            # Dims 7 and 15 unchanged (from copy)
            return smoothed.astype(np.float32)
        except Exception as e:
            print(f"[Warning] Smoothing failed: {e}. Returning original.")
            return seq
        
    def data_gen(self, gen_data_number=10000):

        self.splitter.split()
        
        self.selector.select(gen_data_number)

        for fname in tqdm(os.listdir(self.input_folder),
                                     desc="Data Generation in Progress",
                                     ncols=100):
            if not fname.endswith('.json'):
                continue
            
            fpath = os.path.join(self.input_folder, fname)
            traj = read_json_file(fpath) 
            if self.use_dp:
                self.generate_new_traj_w_df(traj)
            else:
                self.generate_new_traj(traj)
            self.ind += 1

def visualize_segments_ee_pos(arr, segment_labels, save_path="ee_pos_segments.png"):
    """
    Visualize ee_pos by segment (moving/stable/phase1) with different colors; save figure.

    Parameters:
        arr (np.ndarray): shape (n, 16)
        segment_labels (list): length n, each 'moving' | 'stable' | 'phase1'
        save_path (str): output path
    """
    import matplotlib.pyplot as plt
    assert arr.ndim == 2 and arr.shape[1] == 16
    assert len(arr) == len(segment_labels)

    colors = {'moving': '#1f77b4', 'stable': '#2ca02c', 'phase1': '#ff7f0e', 'phase2': '#d62728'}

    def get_runs(labels):
        runs = []
        start = 0
        for i in range(1, len(labels) + 1):
            if i == len(labels) or labels[i] != labels[start]:
                runs.append((start, i, labels[start]))
                start = i
        return runs

    runs = get_runs(segment_labels)
    plt.figure(figsize=(16, 9))

    for dim in range(6, 16):
        for s, e, label in runs:
            x = np.arange(s, e)
            if len(x) == 0:
                continue
            c = colors.get(label, '#888888')
            plt.plot(x, arr[s:e, dim], color=c)

    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=colors['moving'], lw=2, label='moving'),
                      Line2D([0], [0], color=colors['stable'], lw=2, label='stable'),
                      Line2D([0], [0], color=colors['phase1'], lw=2, label='phase1'),
                      Line2D([0], [0], color=colors['phase2'], lw=2, label='phase2')]
    plt.legend(handles=legend_elements)
    plt.title("EE Pos Segments (moving / stable / phase1 / phase2)")
    plt.xlabel("timestep")
    plt.ylabel("value")
    plt.grid(True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Saved] Segment visualization saved to {save_path}")


def visualize_ee_pos(arr, mask=None, save_path="new_ee_pos_seq_masked.png"):
    """
    Visualize n x 16 array; plot only mask==1 points and lines.
    
    Parameters:
        arr (np.ndarray): shape (n, 16)
        mask (np.ndarray): shape (n,), only elements == 1 will be plotted
        save_path (str): path to save output figure
    """
    import matplotlib.pyplot as plt
    # Check input
    assert isinstance(arr, np.ndarray), "arr must be numpy array"
    assert arr.ndim == 2 and arr.shape[1] == 16, "arr shape must be (N, 16)"

    if mask is not None:
        # Select mask==1 data
        idx = mask == 1
        arr_masked = arr[idx]
        x_masked = np.arange(len(arr))[idx]
    else:
        arr_masked = arr
        x_masked = np.arange(len(arr))

    plt.figure(figsize=(16, 9))

    # Plot
    for i in range(16):
        if i <6:
            continue
        # Plot mask==1 points and lines
        plt.plot(x_masked, arr_masked[:, i], label=f"dim {i}")
        plt.scatter(x_masked, arr_masked[:, i], s=10)

    plt.title("new_ee_pos_seq Visualization (Masked)")
    plt.xlabel("timestep")
    plt.ylabel("value")
    plt.grid(True)
    plt.legend(ncol=4)


    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Saved] Visualization saved to {save_path}")