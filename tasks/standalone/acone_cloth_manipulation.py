import time
import warp as wp
import newton
import numpy as np
import newton.ik as ik

import os
from enum import IntEnum

from components.function.warp_kernel import update_control_kernel
from envs.lift2_short_shirt import left_ee_body_names, left_gripper_joint_names, right_ee_body_names, right_gripper_joint_names


class AconeClothManip:
    def __init__(
        self,
        viewer,
        env,
        recorder,
        fps=60,
        sim_substeps=10,
        mode="teleopration",
        save_video=True,
        save_usd=True,
        folder_name="default",
        input_npz_path=None,   # optional input NPZ path
        replay_session_dir=None,  # if set: save under <replay_session_dir>/{npz,usd,video}/ (replay batch layout)
    ):
        self.name = "acone_cloth_manip"
        self.input_npz_base = os.path.splitext(os.path.basename(input_npz_path))[0] if input_npz_path else "default"
        self.fps = fps
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_frame = 0
        self.sim_substeps = sim_substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self._substep_index = 0
        self.last_step_time = None
        self.fps_window_size = 30

        self.mode = mode
        self.is_finished = False
        self.save_video = save_video
        self.queue_executing = False
        self.trajectory_queue_size = 60
        self.save_usd = save_usd
        self.folder_name = folder_name
        self.replay_session_dir = os.path.abspath(replay_session_dir) if replay_session_dir else None

        self.viewer = viewer
        self.env = env
        self.recorder = recorder

        self.model = env.model
        self.ik_graph = None
        self.physics_graph = None
        
        
        self._setup_contact_filtering()
        self._init_gpu_buffers()
        self._setup_viewer()
        self._setup_viewer_usd()

        self.state = self.model.state()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        
        self.control = self.model.control()

        self._init_end_effector_control()
        self._init_trajectory_queue()

        self._ik_pos_buffer = np.zeros((1, 3), dtype=np.float32)
        self._ik_rot_buffer = np.zeros((1, 4), dtype=np.float32)

        self._setup_ik_objectives()

        # Variables the IK solver will update
        self.ik_joint_q = wp.array(self.model.joint_q, shape=(1, self.model.joint_coord_count))
        self.ik_joint_qd = wp.array(self.model.joint_qd, shape=(self.model.joint_dof_count))
        self.ik_iters = 24

        self._init_solvers()
        self.capture()
        
        # For detecting particle motion
        self.last_particle_pos = None  # shape: (N, 3)
        self.last_left_open = 1.0
        self.last_right_open = 1.0
        print(f" input_npz_path = {input_npz_path}")
        print(f" self.input_npz_base = {self.input_npz_base}")

        # Save original home (must be before augmentation)
        home_index = 1
        original_home = recorder.joint_q_seq[home_index].copy()
        original_openness = recorder.openness_seq[home_index].copy()

        # Apply rigid transform with grasp-point compensation
        self.recorder = self.random_init_augmentation(recorder)
        
        # Save original base_transform if present
        self.original_base_transform = None
        if input_npz_path and os.path.exists(input_npz_path):
            with np.load(input_npz_path) as orig_data:
                if 'base_transform' in orig_data:
                    self.original_base_transform = orig_data['base_transform'].copy()
                    print(f"[INFO] Loaded base_transform from {input_npz_path}")
                else:
                    print(f"[WARNING] 'base_transform' not found in {input_npz_path}")

        # Get augmented home (same index)
        augmented_home = self.recorder.joint_q_seq[home_index].copy()
        augmented_openness = self.recorder.openness_seq[home_index].copy()

        # Interpolation duration (seconds)
        interp_duration = 1.5
        interp_frames = int(interp_duration * self.fps)
        interp_joint_seq = []
        interp_openness_seq = []
        for i in range(interp_frames):
            alpha = float(i) / max(1, interp_frames - 1)
            q = (1 - alpha) * original_home + alpha * augmented_home
            o = (1 - alpha) * original_openness + alpha * augmented_openness
            interp_joint_seq.append(q)
            interp_openness_seq.append(o)

        interp_joint_seq = np.array(interp_joint_seq, dtype=np.float32)
        interp_openness_seq = np.array(interp_openness_seq, dtype=np.float32)

        # Key: concatenate only augmented trajectory from home_index (not from 0)
        if self.recorder.joint_q_seq.shape[0] > home_index:
            remaining_joint = self.recorder.joint_q_seq[home_index:]
            remaining_openness = self.recorder.openness_seq[home_index:]
        else:
            # If trajectory has fewer frames than home_index, use interpolation only
            remaining_joint = np.empty((0, original_home.shape[0]), dtype=np.float32)
            remaining_openness = np.empty((0, 2), dtype=np.float32)

        # Concatenate
        self.recorder.joint_q_seq = np.concatenate([interp_joint_seq, remaining_joint], axis=0)
        self.recorder.openness_seq = np.concatenate([interp_openness_seq, remaining_openness], axis=0)

        print(f"[INFO] Interpolated {interp_frames} frames ({interp_duration}s) from original[1] -> augmented[1], then run augmented[1:]")
                 
     
    # Original: translation and rotation 
    def random_init_augmentation(self, recorder):
        
        pos_rand = self.env.randomizer.rand_pos         # (3,)
        rot_rand = self.env.randomizer.rand_rot         # (3,) Euler [rx, ry, rz]

        # # ---- Convert Euler angle to quaternion ----
        # rand_quat = R.from_euler('xyz', rot_rand).as_quat()  # (x, y, z, w)
        # # Convert to (w, x, y, z) for your quat_mul
        rot_rand = np.array([rot_rand[3], rot_rand[0], rot_rand[1], rot_rand[2]], dtype=np.float32)

        def quat_mul(q1, q2):
            w1, x1, y1, z1 = q1
            w2, x2, y2, z2 = q2
            return np.array([
                w1*w2 - x1*x2 - y1*y2 - z1*z2,
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2
            ], dtype=np.float32)
        raw_ee_pos = self.joint_2_ee_pos(recorder)  # [lx,ly,lz,qw,qx,qy,qz, rx,ry,rz,qw,qx,qy,qz]
        raw = raw_ee_pos
        N = raw.shape[0]
        transformed = raw.copy()

        for i in range(N):

            # ===== Left EE =====
            l_pos = raw[i, 0:3]
            l_quat = raw[i, 3:7]   # (w, x, y, z)
            l_pos_new = l_pos + pos_rand 
            l_quat_new = quat_mul(rot_rand, l_quat)

            # ===== Right EE =====
            r_pos = raw[i, 7:10]
            r_quat = raw[i, 10:14]  # (w, x, y, z)
            r_pos_new = r_pos + pos_rand
            r_quat_new = quat_mul(rot_rand, r_quat)

            # ---- write back ----
            transformed[i, 0:3]   = l_pos_new
            transformed[i, 3:7]   = l_quat_new
            transformed[i, 7:10]  = r_pos_new
            transformed[i, 10:14] = r_quat_new

        # Convert back to joint angles
        recorder.joint_q_seq = self.ee_pos_2_joint(transformed, recorder=recorder)
        return recorder

            
    def joint_2_ee_pos(self, recorder):
        """FK: extract left/right end-effector 7D poses (quaternion) from full joint_q_seq."""
        joint_q_seq = recorder.joint_q_seq  # (N, D)
        model = self.model
        device = model.device
        state = model.state()
        joint_qd = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=device)

        # Get indices from env (consistent with Example)
        lee_index = self.env.lee_index
        ree_index = self.env.ree_index

        ee_pos_list = []
        for i, joint_q in enumerate(joint_q_seq):
            q_wp = wp.array(joint_q, dtype=wp.float32, device=device)
            newton.eval_fk(model, q_wp, joint_qd, state)
            body_q = state.body_q.numpy()
            left_tf = body_q[lee_index]   # [x, y, z, qw, qx, qy, qz]
            right_tf = body_q[ree_index]
            ee_pos_list.append(np.concatenate([left_tf, right_tf]))
        return np.array(ee_pos_list, dtype=np.float32)  # (N, 14)


    def ee_pos_2_joint(self, ee_pos, recorder=None):
        """IK: rebuild full joint_q_seq from 14D end-effector poses + openness."""
        model = self.model
        device = model.device

        # Get from env (consistent with Example)
        lee_index = self.env.lee_index
        ree_index = self.env.ree_index
        controllable_indices = self.env.controllable_joint_indices
        left_gripper_indices = self.env.left_gripper_joint_indices
        right_gripper_indices = self.env.right_gripper_joint_indices

        # Gripper limits (consistent with Example)
        GRIPPER_LIMIT_LOWER = 0.001
        GRIPPER_LIMIT_UPPER = 0.044

        # Get openness from recorder
        if recorder is not None:
            openness_seq = recorder.openness_seq  # (N, 2)
        else:
            openness_seq = np.ones((len(ee_pos), 2), dtype=np.float32)

        joint_q_seq = []

        for i in range(len(ee_pos)):
            left_tf = ee_pos[i, :7]    # [x, y, z, qw, qx, qy, qz]
            right_tf = ee_pos[i, 7:14]
            openness = openness_seq[i] if i < len(openness_seq) else [1.0, 1.0]

            # Build IK targets (vec3/vec4)
            target_lee_pos = wp.array([wp.vec3(left_tf[0], left_tf[1], left_tf[2])], dtype=wp.vec3, device=device)
            target_lee_rot = wp.array([wp.vec4(left_tf[3], left_tf[4], left_tf[5], left_tf[6])], dtype=wp.vec4, device=device)
            target_ree_pos = wp.array([wp.vec3(right_tf[0], right_tf[1], right_tf[2])], dtype=wp.vec3, device=device)
            target_ree_rot = wp.array([wp.vec4(right_tf[3], right_tf[4], right_tf[5], right_tf[6])], dtype=wp.vec4, device=device)

            # Set IK objectives
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

            # Solve IK
            ik_joint_q = wp.zeros((1, model.joint_coord_count), dtype=float, device=device)
            solver = ik.IKSolver(
                model=model,
                joint_q=ik_joint_q,
                objectives=[l_pos_obj, l_rot_obj, r_pos_obj, r_rot_obj, obj_joint_limits],
                lambda_initial=0.1,
                jacobian_mode=ik.IKJacobianMode.MIXED,
            )
            solver.solve(iterations=24)

            # Add gripper
            result = ik_joint_q.numpy()[0].copy()
            left_open, right_open = openness
            left_pos = GRIPPER_LIMIT_LOWER + left_open * (GRIPPER_LIMIT_UPPER - GRIPPER_LIMIT_LOWER)
            right_pos = GRIPPER_LIMIT_LOWER + right_open * (GRIPPER_LIMIT_UPPER - GRIPPER_LIMIT_LOWER)
            result[left_gripper_indices] = left_pos
            result[right_gripper_indices] = right_pos

            joint_q_seq.append(result)

            # if i % 50 == 0:
            #     print(f"IK: {i}/{len(ee_pos)}")

        return np.array(joint_q_seq, dtype=np.float32)
        
    def apply_rand_transform(self, poses, rand_pos, rand_rot):
        """
        Apply global rigid transform to 14D EE poses (7D + 7D: pos + quat).
        
        Args:
            poses: np.ndarray, shape (N, 14)
                [lx,ly,lz,qw,qx,qy,qz,  rx,ry,rz,qw,qx,qy,qz]
            rand_pos: array-like, shape (3,)
            rand_rot: array-like, shape (4,) in [w,x,y,z] (scalar-first quat)
        
        Returns:
            transformed_poses: np.ndarray, shape (N, 14)
        """
        import numpy as np

        def quat_mul(q1, q2):
            w1, x1, y1, z1 = q1
            w2, x2, y2, z2 = q2
            return np.array([
                w1*w2 - x1*x2 - y1*y2 - z1*z2,
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2
            ])

        def quat_rot_vec(q, v):
            w, x, y, z = q
            vx, vy, vz = v
            return np.array([
                2*(x*(x*vx + y*vy + z*vz) + w*(w*vx + y*vz - z*vy)),
                2*(y*(x*vx + y*vy + z*vz) + w*(w*vy - x*vz + z*vx)),
                2*(z*(x*vx + y*vy + z*vz) + w*(w*vz + x*vy - y*vx))
            ])

        poses = np.asarray(poses, dtype=np.float32)
        assert poses.shape[1] == 14, f"Expected (N,14), got {poses.shape}"
        rand_pos = np.asarray(rand_pos, dtype=np.float32)
        rand_rot = np.asarray(rand_rot, dtype=np.float32)

        N = poses.shape[0]
        out = np.empty_like(poses)

        for i in range(N):
            # Left EE
            l_pos = poses[i, 0:3]
            l_quat = poses[i, 3:7]
            l_pos_new = quat_rot_vec(rand_rot, l_pos) + rand_pos
            l_quat_new = quat_mul(rand_rot, l_quat)

            # Right EE
            r_pos = poses[i, 7:10]
            r_quat = poses[i, 10:14]
            r_pos_new = quat_rot_vec(rand_rot, r_pos) + rand_pos
            r_quat_new = quat_mul(rand_rot, r_quat)

            out[i, 0:3] = l_pos_new
            out[i, 3:7] = l_quat_new
            out[i, 7:10] = r_pos_new
            out[i, 10:14] = r_quat_new

        return out

    def _setup_contact_filtering(self):
        """Prepare data for dynamic contact filtering.

        Collect shape IDs for gripper and end-effector (fl_link6, fr_link6).
        Contact filtering is applied at runtime based on gripper state.
        """
        joint_child_np = self.model.joint_child.numpy()
        COLLIDE_PARTICLES = 1 << 2  # ShapeFlags.COLLIDE_PARTICLES
        
        def _collect_shape_ids_from_joints(joint_name_set):
            bodies = [int(joint_child_np[i]) for i, key in enumerate(self.model.joint_key) if key in joint_name_set]
            shape_ids = set()
            for body_id in bodies:
                if body_id in self.model.body_shapes:
                    shape_ids.update(int(shape_id) for shape_id in self.model.body_shapes[body_id])
            return shape_ids
        
        def _collect_shape_ids_from_bodies(body_name_set):
            bodies = [i for i, key in enumerate(self.model.body_key) if key in body_name_set]
            shape_ids = set()
            for body_id in bodies:
                if body_id in self.model.body_shapes:
                    shape_ids.update(int(shape_id) for shape_id in self.model.body_shapes[body_id])
            return shape_ids
        
        # Collect left/right gripper shapes (including end-effector)
        left_gripper_shapes = _collect_shape_ids_from_joints(left_gripper_joint_names)
        left_ee_shapes = _collect_shape_ids_from_bodies(left_ee_body_names)
        left_shapes_combined = left_gripper_shapes | left_ee_shapes
        
        right_gripper_shapes = _collect_shape_ids_from_joints(right_gripper_joint_names)
        right_ee_shapes = _collect_shape_ids_from_bodies(right_ee_body_names)
        right_shapes_combined = right_gripper_shapes | right_ee_shapes

        # Convert to numpy arrays
        if left_shapes_combined:
            self.left_gripper_shape_ids = np.array(sorted(left_shapes_combined), dtype=np.int32)
        else:
            self.left_gripper_shape_ids = np.zeros((0,), dtype=np.int32)
            
        if right_shapes_combined:
            self.right_gripper_shape_ids = np.array(sorted(right_shapes_combined), dtype=np.int32)
        else:
            self.right_gripper_shape_ids = np.zeros((0,), dtype=np.int32)
        
        self.COLLIDE_PARTICLES = COLLIDE_PARTICLES
        
        print(f"\n=== Contact filtering configured ===")
        print(f"Left gripper shapes (incl. fl_link6): {len(self.left_gripper_shape_ids)}")
        print(f"Right gripper shapes (incl. fr_link6): {len(self.right_gripper_shape_ids)}")
        print(f"Dynamic contact filtering: disable gripper/EE collision during opening")

    def _init_gpu_buffers(self):
        """Initialize GPU buffers for control and IK."""
        self.left_gripper_joint_indices_wp = wp.array(
            self.env.left_gripper_joint_indices, dtype=int, device=self.model.device
        )
        self.right_gripper_joint_indices_wp = wp.array(
            self.env.right_gripper_joint_indices, dtype=int, device=self.model.device
        )
        self.controllable_joint_indices_wp = wp.array(
            self.env.controllable_joint_indices, dtype=int, device=self.model.device
        )
        self.q0_frame_wp = wp.zeros(self.model.joint_coord_count, dtype=float, device=self.model.device)
        self.q_target_frame_wp = wp.zeros(self.model.joint_coord_count, dtype=float, device=self.model.device)
        self.gripper_params_wp = wp.zeros(4, dtype=float, device=self.model.device)

        self.left_gripper_shape_ids_wp = wp.array(
            self.left_gripper_shape_ids, dtype=int, device=self.model.device
        )
        self.right_gripper_shape_ids_wp = wp.array(
            self.right_gripper_shape_ids, dtype=int, device=self.model.device
        )

    def _setup_viewer(self):
        self.viewer.set_model(self.model)
        self.viewer.vsync = True
        if isinstance(self.viewer, newton.viewer.ViewerGL):
            camera_pos = type(self.viewer.camera.pos)(0.32, -0.01, 1.15)
            self.viewer.camera.pos = camera_pos
            self.viewer.camera.pitch = -60
            self.viewer.camera.fov = 100
            self.viewer.camera.yaw = 0
        if self.mode == "replay":
            self.viewer.show_ui = False
            
         
    def _setup_viewer_usd(self):
        if self.save_usd:
            # Save USD locally under project replay directory
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            replay_root = os.path.join(project_root, "replay")

            if self.replay_session_dir:
                usd_dir = os.path.join(self.replay_session_dir, "usd")
            elif self.folder_name:
                usd_dir = os.path.join(replay_root, f"usd_{self.folder_name}")
            else:
                usd_dir = os.path.join(replay_root, "SIM1-Sim", "position1", "usd")

            os.makedirs(usd_dir, exist_ok=True)
            self.usd_output_path = os.path.join(usd_dir, f"{self.input_npz_base}.usd")
            self.viewer_usd = newton.viewer.ViewerUSD(output_path=self.usd_output_path)
            self.viewer_usd.set_model(self.model)
            print(f"USD output will be saved in: {self.usd_output_path}")
        
    def _init_end_effector_control(self):
        self.open_left_gripper = 1.0
        self.open_right_gripper = 1.0
        self.left_gripper_state = 1.0
        self.right_gripper_state = 1.0

        body_q_np = self.state.body_q.numpy()
        initial_lee_tf = wp.transform(*body_q_np[self.env.lee_index])
        initial_ree_tf = wp.transform(*body_q_np[self.env.ree_index])
        
        self.clamp_stats = {
            'pos_clamp_count': 0,
            'rot_clamp_count': 0,
            'total_frames': 0,
            'max_pos_distance': 0.0,
            'max_rot_angle': 0.0
        }
        
        self.gizmo_lee_tf = wp.transform(
            wp.transform_get_translation(initial_lee_tf),
            wp.transform_get_rotation(initial_lee_tf)
        )
        self.gizmo_ree_tf = wp.transform(
            wp.transform_get_translation(initial_ree_tf),
            wp.transform_get_rotation(initial_ree_tf)
        )
        self.prev_gizmo_lee_tf = wp.transform(
            wp.transform_get_translation(initial_lee_tf),
            wp.transform_get_rotation(initial_lee_tf)
        )
        self.prev_gizmo_ree_tf = wp.transform(
            wp.transform_get_translation(initial_ree_tf),
            wp.transform_get_rotation(initial_ree_tf)
        )
        self.lee_tf = wp.transform(
            wp.transform_get_translation(initial_lee_tf),
            wp.transform_get_rotation(initial_lee_tf)
        )
        self.ree_tf = wp.transform(
            wp.transform_get_translation(initial_ree_tf),
            wp.transform_get_rotation(initial_ree_tf)
        )

    def _init_trajectory_queue(self):
        self.lee_tf_queue = [wp.transform() for _ in range(self.trajectory_queue_size)]
        self.ree_tf_queue = [wp.transform() for _ in range(self.trajectory_queue_size)]
        self.queue_index = 0
        
        self.cached_lee_target_tf = self.lee_tf
        self.cached_ree_target_tf = self.ree_tf
        
        self.trajectory_alphas = np.linspace(
            1.0 / self.trajectory_queue_size,
            1.0,
            self.trajectory_queue_size,
            dtype=np.float32
        )

    def _setup_ik_objectives(self):
        total_residuals = 2 * 6 + self.model.joint_coord_count
        
        def _q2v4(q):
            return wp.vec4(q[0], q[1], q[2], q[3])

        self.l_pos_obj = ik.IKPositionObjective(
            link_index=self.env.lee_index,
            link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=wp.array([wp.transform_get_translation(self.lee_tf)], dtype=wp.vec3),
            n_problems=1,
            total_residuals=total_residuals,
            residual_offset=0,
        )

        self.l_rot_obj = ik.IKRotationObjective(
            link_index=self.env.lee_index,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([_q2v4(wp.transform_get_rotation(self.lee_tf))], dtype=wp.vec4),
            n_problems=1,
            total_residuals=total_residuals,
            residual_offset=3,
        )

        self.r_pos_obj = ik.IKPositionObjective(
            link_index=self.env.ree_index,
            link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=wp.array([wp.transform_get_translation(self.ree_tf)], dtype=wp.vec3),
            n_problems=1,
            total_residuals=total_residuals,
            residual_offset=6,
        )

        self.r_rot_obj = ik.IKRotationObjective(
            link_index=self.env.ree_index,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([_q2v4(wp.transform_get_rotation(self.ree_tf))], dtype=wp.vec4),
            n_problems=1,
            total_residuals=total_residuals,
            residual_offset=9,
        )

        self.obj_joint_limits = ik.IKJointLimitObjective(
            joint_limit_lower=self.model.joint_limit_lower,
            joint_limit_upper=self.model.joint_limit_upper,
            n_problems=1,
            total_residuals=total_residuals,
            residual_offset=12,
            weight=10.0,
        )

    def _init_solvers(self):
        self.solver = ik.IKSolver(
            model=self.model,
            joint_q=self.ik_joint_q,
            objectives=[
                self.l_pos_obj,
                self.l_rot_obj,
                self.r_pos_obj,
                self.r_rot_obj,
                self.obj_joint_limits
            ],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianMode.MIXED,
        )

        self.use_mujoco_cpu = False
        self.rigid_solver = newton.solvers.SolverMuJoCo(
            self.model,
            njmax=150000,
            solver='newton',
            cone="elliptic",
            use_mujoco_cpu=self.use_mujoco_cpu,
            use_mujoco_contacts=False,
            contact_stiffness_time_const=self.sim_dt
        )

        self.model.edge_rest_angle.zero_()
        self.cloth_solver = newton.solvers.SolverVBD(
            self.model,
            iterations=16,
            self_contact_radius=self.env.self_contact_radius,
            self_contact_margin=self.env.self_contact_margin,
            handle_self_contact=True,
            vertex_collision_buffer_pre_alloc=32,
            edge_collision_buffer_pre_alloc=64,
            integrate_with_external_rigid_solver=True,
            collision_detection_interval=-1,
        )

    def _has_gizmo_moved(self, threshold_pos=0.001, threshold_rot=0.1):
        lee_pos_curr = wp.transform_get_translation(self.gizmo_lee_tf)
        lee_pos_prev = wp.transform_get_translation(self.prev_gizmo_lee_tf)
        lee_pos_diff = wp.length(lee_pos_curr - lee_pos_prev)
        
        lee_rot_curr = wp.transform_get_rotation(self.gizmo_lee_tf)
        lee_rot_prev = wp.transform_get_rotation(self.prev_gizmo_lee_tf)
        lee_rot_dot = abs(wp.dot(lee_rot_curr, lee_rot_prev))
        lee_rot_diff = 2.0 * wp.acos(wp.clamp(lee_rot_dot, -1.0, 1.0))
        
        ree_pos_curr = wp.transform_get_translation(self.gizmo_ree_tf)
        ree_pos_prev = wp.transform_get_translation(self.prev_gizmo_ree_tf)
        ree_pos_diff = wp.length(ree_pos_curr - ree_pos_prev)
        
        ree_rot_curr = wp.transform_get_rotation(self.gizmo_ree_tf)
        ree_rot_prev = wp.transform_get_rotation(self.prev_gizmo_ree_tf)
        ree_rot_dot = abs(wp.dot(ree_rot_curr, ree_rot_prev))
        ree_rot_diff = 2.0 * wp.acos(wp.clamp(ree_rot_dot, -1.0, 1.0))
        
        return (lee_pos_diff > threshold_pos or lee_rot_diff > threshold_rot or
                ree_pos_diff > threshold_pos or ree_rot_diff > threshold_rot)

    def capture(self):
        self.ik_graph = None
        if wp.get_device().is_cuda and not self.use_mujoco_cpu:
            with wp.ScopedCapture() as capture:
                self.ik_simulate()
            self.ik_graph = capture.graph

        self.physics_graph = None
        if wp.get_device().is_cuda and not self.use_mujoco_cpu:
            with wp.ScopedCapture() as capture:
                self.physics_simulate()
            self.physics_graph = capture.graph

    def _update_gripper_collision_filtering_cpu(self):
        left_opening = float(self.left_gripper_state + 0.005) < float(self.open_left_gripper)
        right_opening = float(self.right_gripper_state + 0.005) < float(self.open_right_gripper)
        
        shape_flags_np = self.model.shape_flags.numpy()
        
        for shape_id in self.left_gripper_shape_ids:
            if left_opening or self.viewer.is_key_down("1"):
                shape_flags_np[shape_id] = shape_flags_np[shape_id] & ~self.COLLIDE_PARTICLES
            else:
                shape_flags_np[shape_id] = shape_flags_np[shape_id] | self.COLLIDE_PARTICLES
        
        for shape_id in self.right_gripper_shape_ids:
            if right_opening or self.viewer.is_key_down("2"):
                shape_flags_np[shape_id] = shape_flags_np[shape_id] & ~self.COLLIDE_PARTICLES
            else:
                shape_flags_np[shape_id] = shape_flags_np[shape_id] | self.COLLIDE_PARTICLES

        # Camera switching (keep as-is)
        if right_opening or self.viewer.is_key_down("3"):
            if isinstance(self.viewer, newton.viewer.ViewerGL):
                self.viewer.camera.pos = type(self.viewer.camera.pos)(0.8, -0.01, 1.8)
                self.viewer.camera.pitch = -89.9
                self.viewer.camera.fov = 60
                self.viewer.camera.yaw = 0
        if right_opening or self.viewer.is_key_down("4"):
            if isinstance(self.viewer, newton.viewer.ViewerGL):
                self.viewer.camera.pos = type(self.viewer.camera.pos)(0.70, 1.31, 0.86)
                self.viewer.camera.pitch = -1.8
                self.viewer.camera.fov = 60
                self.viewer.camera.yaw = 268.7
        if right_opening or self.viewer.is_key_down("5"):
            if isinstance(self.viewer, newton.viewer.ViewerGL):
                self.viewer.camera.pos = type(self.viewer.camera.pos)(0.32, -0.01, 1.15)
                self.viewer.camera.pitch = -60
                self.viewer.camera.fov = 100
                self.viewer.camera.yaw = 0

        self.model.shape_flags.assign(shape_flags_np)

    def _push_targets_from_gizmos_queue_no_clamp(self):
        if hasattr(self.viewer, "is_key_down"):
            if self.viewer.is_key_down("1"):
                self.open_left_gripper -= 0.05
            else:
                self.open_left_gripper += 0.05

            if self.viewer.is_key_down("2"):
                self.open_right_gripper -= 0.05
            else:
                self.open_right_gripper += 0.05
                
            self.open_left_gripper = np.clip(self.open_left_gripper, 0.0, 1.0)
            self.open_right_gripper = np.clip(self.open_right_gripper, 0.0, 1.0)
        
        gizmo_moved = self._has_gizmo_moved(threshold_pos=0.001)

        if gizmo_moved:
            target_lee_tf = self.gizmo_lee_tf
            target_ree_tf = self.gizmo_ree_tf
            current_lee_tf = self.cached_lee_target_tf
            current_ree_tf = self.cached_ree_target_tf
            
            lee_pos_start = wp.transform_get_translation(current_lee_tf)
            lee_pos_end = wp.transform_get_translation(target_lee_tf)
            lee_rot_start = wp.transform_get_rotation(current_lee_tf)
            lee_rot_end = wp.transform_get_rotation(target_lee_tf)
            
            ree_pos_start = wp.transform_get_translation(current_ree_tf)
            ree_pos_end = wp.transform_get_translation(target_ree_tf)
            ree_rot_start = wp.transform_get_rotation(current_ree_tf)
            ree_rot_end = wp.transform_get_rotation(target_ree_tf)
            
            for i, alpha in enumerate(self.trajectory_alphas):
                alpha_f = float(alpha)
                lee_pos_interp = lee_pos_start + alpha_f * (lee_pos_end - lee_pos_start)
                lee_rot_interp = wp.quat_slerp(lee_rot_start, lee_rot_end, alpha_f)
                self.lee_tf_queue[i] = wp.transform(lee_pos_interp, lee_rot_interp)
                
                ree_pos_interp = ree_pos_start + alpha_f * (ree_pos_end - ree_pos_start)
                ree_rot_interp = wp.quat_slerp(ree_rot_start, ree_rot_end, alpha_f)
                self.ree_tf_queue[i] = wp.transform(ree_pos_interp, ree_rot_interp)
            
            self.queue_executing = True
            self.queue_index = 0
        
        self.prev_gizmo_lee_tf = wp.transform(
            wp.transform_get_translation(self.gizmo_lee_tf),
            wp.transform_get_rotation(self.gizmo_lee_tf)
        )
        self.prev_gizmo_ree_tf = wp.transform(
            wp.transform_get_translation(self.gizmo_ree_tf),
            wp.transform_get_rotation(self.gizmo_ree_tf)
        )
        
        if self.queue_executing and self.queue_index < len(self.lee_tf_queue):
            target_lee_tf = self.lee_tf_queue[self.queue_index]
            target_ree_tf = self.ree_tf_queue[self.queue_index]
            
            lee_pos = wp.transform_get_translation(target_lee_tf)
            lee_rot = wp.transform_get_rotation(target_lee_tf)
            ree_pos = wp.transform_get_translation(target_ree_tf)
            ree_rot = wp.transform_get_rotation(target_ree_tf)
            
            self._ik_pos_buffer[0] = [lee_pos[0], lee_pos[1], lee_pos[2]]
            self.l_pos_obj.target_positions.assign(self._ik_pos_buffer)
            self._ik_rot_buffer[0] = [lee_rot[0], lee_rot[1], lee_rot[2], lee_rot[3]]
            self.l_rot_obj.target_rotations.assign(self._ik_rot_buffer)
            
            self._ik_pos_buffer[0] = [ree_pos[0], ree_pos[1], ree_pos[2]]
            self.r_pos_obj.target_positions.assign(self._ik_pos_buffer)
            self._ik_rot_buffer[0] = [ree_rot[0], ree_rot[1], ree_rot[2], ree_rot[3]]
            self.r_rot_obj.target_rotations.assign(self._ik_rot_buffer)
            
            self.cached_lee_target_tf = target_lee_tf
            self.cached_ree_target_tf = target_ree_tf
            
            self.queue_index += 1
            if self.queue_index >= len(self.lee_tf_queue):
                self.queue_executing = False
                self.queue_index = 0

    def ik_simulate(self):
        self.solver.solve(iterations=self.ik_iters)

    def physics_simulate(self):
        for s in range(self.sim_substeps):
            self._substep_index = s
            current_q_flat = self.state_0.joint_q.reshape((self.model.joint_coord_count,))
            left_count = int(self.left_gripper_joint_indices_wp.shape[0])
            right_count = int(self.right_gripper_joint_indices_wp.shape[0])
            controllable_count = int(self.controllable_joint_indices_wp.shape[0])
            dim = max(max(left_count, right_count), controllable_count)
            wp.launch(
                kernel=update_control_kernel,
                dim=dim,
                inputs=[
                    self.model.joint_limit_lower,
                    self.model.joint_limit_upper,
                    self.left_gripper_joint_indices_wp,
                    self.right_gripper_joint_indices_wp,
                    self.controllable_joint_indices_wp,
                    self.q0_frame_wp,
                    self.q_target_frame_wp,
                    current_q_flat,
                    self.control.joint_target,
                    self.ik_joint_qd,
                    self.state_0.joint_qd,
                    left_count,
                    right_count,
                    controllable_count,
                    self.sim_substeps,
                    int(self._substep_index),
                    self.sim_dt,
                    self.gripper_params_wp,
                    int(1),
                ],
                device=self.model.device,
            )

            self.contacts = self.model.collide(self.state_0)
            self.state_0.clear_forces()
            self.state_1.clear_forces()

            particle_count = self.model.particle_count
            self.model.particle_count = 0
            self.rigid_solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0.particle_f.zero_()
            self.model.particle_count = particle_count

            self.contacts = self.model.collide(self.state_0, soft_contact_margin=self.env.cloth_body_contact_margin)
            self.cloth_solver.step(self.state_0, self.state_1, None, self.contacts, self.sim_dt)
            (self.state_0, self.state_1) = (self.state_1, self.state_0)

    def stop_teleopration_device(self):
        if hasattr(self, 'right_quest_controller'):
            self.right_quest_controller.stop()
        if hasattr(self, 'left_quest_controller'):
            self.left_quest_controller.stop()
    
    def step(self):
        if self.sim_time == 0.0:
            newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)  # state_0 particle_f, particle_q
            self.left_gripper_state = self.open_left_gripper
            self.right_gripper_state = self.open_right_gripper

        self._push_targets_from_gizmos_queue_no_clamp()

        if self.ik_graph:
            wp.capture_launch(self.ik_graph)
        else:
            self.ik_simulate()

        ik_joint_q = self.ik_joint_q.flatten()
        ik_joint_q_np = ik_joint_q.numpy()
        if self.mode == "replay":
            if self.sim_frame < self.recorder.joint_q_seq.shape[0]:
                cur_joint_q = self.recorder.joint_q_seq[self.sim_frame]
                cur_gripper_q = self.recorder.openness_seq[self.sim_frame]
            else:
                cur_joint_q = self.recorder.joint_q_seq[-1]
                cur_gripper_q = self.recorder.openness_seq[-1]
                self.is_finished = True
            

            # new acone 
            ik_joint_q_np[self.env.controllable_joint_indices] = cur_joint_q[(list(range(3,9)) + list(range(11,17)))]  
            self.open_left_gripper = cur_gripper_q[0]
            self.open_right_gripper = cur_gripper_q[1]
            ik_joint_q.assign(ik_joint_q_np)
            print(self.open_left_gripper, self.open_right_gripper)
        
        state_q_flat = self.state_0.joint_q.reshape((self.model.joint_coord_count,))
        wp.copy(self.q0_frame_wp, state_q_flat)
        wp.copy(self.q_target_frame_wp, ik_joint_q)

        gripper_params_host = np.array([
            float(self.left_gripper_state),
            float(self.open_left_gripper),
            float(self.right_gripper_state),
            float(self.open_right_gripper),
        ], dtype=np.float32)
        self.gripper_params_wp.assign(gripper_params_host)
        
        self._update_gripper_collision_filtering_cpu()
        self.cloth_solver.rebuild_bvh(self.state_0)

        if self.physics_graph:
            wp.capture_launch(self.physics_graph)
        else:
            self.physics_simulate()
            
        self.sim_time += self.frame_dt
        self.sim_frame += 1
        
        current_time = time.time()
        if self.last_step_time is not None:
            step_duration = current_time - self.last_step_time
            current_fps = 1.0 / step_duration if step_duration > 0 else 0

        self.last_step_time = current_time

        if self.mode == 'teleopration':
            self.recorder.record_data(self.state_0, self.open_left_gripper, self.open_right_gripper)
        
        self.left_gripper_state = self.open_left_gripper
        self.right_gripper_state = self.open_right_gripper
        
    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_gizmo("left_target_tcp", self.gizmo_lee_tf)
        self.viewer.log_gizmo("right_target_tcp", self.gizmo_ree_tf)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

        if self.save_video:
            self.recorder.record_frame(self.viewer)

        if self.save_usd:
            print("save usd frame:", self.sim_frame)
            self.viewer_usd.begin_frame(self.sim_time)
            self.viewer_usd.log_state(self.state_0)
            self.viewer_usd.end_frame()
            # Removed self.viewer_usd.close() here
    
        

    def finished(self):

        # Save replay data to project replay directory
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        REPLAY_ROOT = os.path.join(project_root, "replay")
        # 1. Save raw data (teleoperation mode)
        if self.mode == "teleopration":
            self.recorder.save_data()
        
        # 2. Save video
        if self.save_video:
            if self.replay_session_dir:
                video_dir = os.path.join(self.replay_session_dir, "video")
            elif self.folder_name:
                video_dir = os.path.join(REPLAY_ROOT, f"video_{self.folder_name}")
            else:
                video_dir = os.path.join(REPLAY_ROOT, "SIM1-Sim", "video")
            os.makedirs(video_dir, exist_ok=True)
            video_path = os.path.join(video_dir, f"{self.input_npz_base}.mp4")
            self.recorder.save_video(video_path)
            print(f"[INFO] Video saved to: {video_path}")
        
        # 3. Save augmented NPZ (replay mode)
        if self.mode == "replay" and self.save_usd:
            if self.replay_session_dir:
                npz_dir = os.path.join(self.replay_session_dir, "npz")
            elif self.folder_name:
                npz_dir = os.path.join(REPLAY_ROOT, f"npz_{self.folder_name}")
            else:
                npz_dir = os.path.join(REPLAY_ROOT, "SIM1-Sim", "npz")
            os.makedirs(npz_dir, exist_ok=True)
            aug_npz_path = os.path.join(npz_dir, f"{self.input_npz_base}.npz")
            
            # Key names without _seq suffix
            save_dict = {
                'joint_q': self.recorder.joint_q_seq,
                'openness': self.recorder.openness_seq,
            }
            if self.original_base_transform is not None:
                save_dict['base_transform'] = self.original_base_transform
            
            np.savez_compressed(aug_npz_path, **save_dict)
            print(f"[INFO] Augmented NPZ (with base_transform) saved to: {aug_npz_path}")
        
        # 4. Close USD recorder
        if hasattr(self, 'viewer_usd') and self.viewer_usd is not None:
            try:
                self.viewer_usd.close()
                print(f"[INFO] USD recording finalized: {self.usd_output_path}")
            except Exception as e:
                print(f"[ERROR] Failed to close USD viewer: {e}")
                
