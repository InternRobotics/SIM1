import time

import warp as wp
import newton
import numpy as np
import newton.ik as ik

from components.function.warp_kernel import update_control_kernel
from envs.lift2_short_shirt import left_ee_body_names, left_gripper_joint_names, right_ee_body_names, right_gripper_joint_names

class Lift2ClothManip:
    # Base step; Shift doubles speed
    # Normal ~0.36 m/frame, with Shift ~0.72 m/frame
    KEYBOARD_MOVE_STEP = 0.01
    KEYBOARD_ROT_STEP = 0.05

    def __init__(self, viewer, env, recorder, fps=60, sim_substeps=24, mode="teleopration", save_image=True):
        self.name = "lift_cloth_manip"
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
        self.save_image = save_image
        self.queue_executing = False
        self.trajectory_queue_size = 60

        self.viewer = viewer
        self.env = env
        self.recorder = recorder

        self.model = env.model
        self.ik_graph = None
        self.physics_graph = None

        self._setup_contact_filtering()
        self._init_gpu_buffers()
        self._setup_viewer()

        self.state = self.model.state()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.control = self.model.control()

        # Min EE height: safe lower bound from table height + thickness to avoid penetration
        self.min_ee_z = None
        try:
            table_z = float(getattr(self.env, "TABLE_Z_BASE", 0.0))
            table_hz = float(getattr(self.env, "TABLE_SIZE", (0.0, 0.0, 0.0))[2])
            # Table top + 1.5cm margin
            self.min_ee_z = table_z + table_hz + 0.015
        except Exception:
            self.min_ee_z = None

        # Wrist joint indices in joint_q (indices 3 and 9 in controllable_joint_indices)
        self.left_wrist_q_index = None
        self.right_wrist_q_index = None
        # Joints 3 and 9 in controllable_joint_indices
        if len(self.env.controllable_joint_indices) > 3:
            self.left_wrist_q_index = int(self.env.controllable_joint_indices[3])
        if len(self.env.controllable_joint_indices) > 9:
            self.right_wrist_q_index = int(self.env.controllable_joint_indices[9])

        # Store initial joint and cloth state for R-key reset
        self._initial_joint_q = wp.array(self.state_0.joint_q, dtype=float, device=self.model.device)
        self._initial_joint_qd = wp.array(self.state_0.joint_qd, dtype=float, device=self.model.device)
        if self.model.particle_count > 0:
            self._initial_particle_q = wp.array(self.state_0.particle_q, dtype=float, device=self.model.device)
            self._initial_particle_qd = wp.array(self.state_0.particle_qd, dtype=float, device=self.model.device)
        else:
            self._initial_particle_q = None
            self._initial_particle_qd = None

        self._init_end_effector_control()

        self._left_gripper_closed = False
        self._right_gripper_closed = False
        self._prev_x_down = False
        self._prev_m_down = False
        self._prev_r_down = False

        self._init_trajectory_queue()

        self._ik_pos_buffer = np.zeros((1, 3), dtype=np.float32)
        self._ik_rot_buffer = np.zeros((1, 4), dtype=np.float32)

        self._setup_ik_objectives()

        # Variables the IK solver will update
        self.ik_joint_q = wp.array(self.model.joint_q, shape=(1, self.model.joint_coord_count))
        self.ik_joint_qd = wp.array(self.model.joint_qd, shape=(self.model.joint_dof_count))
        self.ik_iters = 24

        self._init_solvers()
        
        # Capture CUDA graphs for better performance
        self.capture()

    def _setup_contact_filtering(self):
        """Prepare data for dynamic contact filtering.
        Collect shape IDs for grippers and end-effectors (fl_link6, fr_link6).
        Collision control is applied at runtime based on gripper state.
        """
        joint_child_np = self.model.joint_child.numpy()
        COLLIDE_PARTICLES = 1 << 2  # ShapeFlags.COLLIDE_PARTICLES
        
        def _collect_shape_ids_from_joints(joint_name_set):
            """Collect shape IDs for the given joint name set."""
            bodies = [int(joint_child_np[i]) for i, key in enumerate(self.model.joint_key) if key in joint_name_set]
            shape_ids = set()
            for body_id in bodies:
                if body_id in self.model.body_shapes:
                    shape_ids.update(int(shape_id) for shape_id in self.model.body_shapes[body_id])
            return shape_ids
        
        def _collect_shape_ids_from_bodies(body_name_set):
            """Collect shape IDs for the given body name set."""
            bodies = [i for i, key in enumerate(self.model.body_key) if key in body_name_set]
            shape_ids = set()
            for body_id in bodies:
                if body_id in self.model.body_shapes:
                    shape_ids.update(int(shape_id) for shape_id in self.model.body_shapes[body_id])
            return shape_ids
        
        # Collect left/right gripper shapes (including end-effectors)
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
        print(f"Left gripper shapes (incl. fl_link6): {len(self.left_gripper_shape_ids)}, IDs: {self.left_gripper_shape_ids.tolist()}")
        print(f"  - from gripper joints: {sorted(left_gripper_shapes)}")
        print(f"  - from fl_link6 body: {sorted(left_ee_shapes)}")
        print(f"Right gripper shapes (incl. fr_link6): {len(self.right_gripper_shape_ids)}, IDs: {self.right_gripper_shape_ids.tolist()}")
        print(f"  - from gripper joints: {sorted(right_gripper_shapes)}")
        print(f"  - from fr_link6 body: {sorted(right_ee_shapes)}")
        print(f"Dynamic contact filtering: disable gripper/EE collision while opening/releasing")
    

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

        # GPU buffers for collision filtering
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
            camera_pos = type(self.viewer.camera.pos)(0.01, -0.01, 1.60)
            self.viewer.camera.pos = camera_pos
            self.viewer.camera.pitch = -48
            self.viewer.camera.fov = 60
            self.viewer.camera.yaw = 0
        if self.mode == "replay":
            self.viewer.show_ui = False
        if self.mode == "teleopration":
            self.viewer.show_ui = False
            self._override_viewer_keys()

    def _override_viewer_keys(self):
        """Override Newton viewer keybindings: free WASD/QE for robot control, remove Q=quit."""
        import types

        viewer = self.viewer
        original_on_key_press = viewer.on_key_press

        def new_on_key_press(symbol, modifiers):
            if viewer.ui and viewer.ui.is_capturing():
                return
            try:
                import pyglet
            except Exception:
                return
            # Use G for GUI toggle to avoid conflict with right gripper H key
            if symbol == pyglet.window.key.G:
                viewer.show_ui = not viewer.show_ui
            elif symbol == pyglet.window.key.SPACE:
                viewer._paused = not viewer._paused
            elif symbol == pyglet.window.key.F:
                viewer._frame_camera_on_model()
            elif symbol == pyglet.window.key.ESCAPE:
                viewer.renderer.close()

        for i, cb in enumerate(viewer.renderer._key_callbacks):
            if cb is original_on_key_press or (hasattr(cb, '__self__') and cb.__self__ is viewer):
                viewer.renderer._key_callbacks[i] = new_on_key_press
                break

        def _update_camera_arrows_only(self_viewer, dt):
            """Camera movement with arrow keys only (WASD freed for robot control)."""
            if self_viewer.ui and self_viewer.ui.is_capturing():
                return
            import pyglet

            forward = np.array(self_viewer.camera.get_front(), dtype=np.float32)
            right = np.array(self_viewer.camera.get_right(), dtype=np.float32)
            up = np.array(self_viewer.camera.get_up(), dtype=np.float32)
            forward -= up * float(np.dot(forward, up))
            right -= up * float(np.dot(right, up))
            fn = float(np.linalg.norm(forward))
            ln = float(np.linalg.norm(right))
            if fn > 1.0e-6:
                forward /= fn
            if ln > 1.0e-6:
                right /= ln

            desired = np.zeros(3, dtype=np.float32)
            if self_viewer.renderer.is_key_down(pyglet.window.key.UP):
                desired += forward
            if self_viewer.renderer.is_key_down(pyglet.window.key.DOWN):
                desired -= forward
            if self_viewer.renderer.is_key_down(pyglet.window.key.LEFT):
                desired -= right
            if self_viewer.renderer.is_key_down(pyglet.window.key.RIGHT):
                desired += right

            dn = float(np.linalg.norm(desired))
            if dn > 1.0e-6:
                desired = desired / dn * self_viewer._cam_speed
            else:
                desired[:] = 0.0

            tau = max(1.0e-4, float(self_viewer._cam_damp_tau))
            self_viewer._cam_vel += (desired - self_viewer._cam_vel) * (dt / tau)
            dv = type(self_viewer.camera.pos)(*self_viewer._cam_vel)
            self_viewer.camera.pos += dv * dt

        viewer._update_camera = types.MethodType(_update_camera_arrows_only, viewer)

    def _init_end_effector_control(self):
        """Initialize end effector transforms and gripper states."""
        # Initialize gripper openness
        self.open_left_gripper = 1.0
        self.open_right_gripper = 1.0
        self.left_gripper_state = 1.0
        self.right_gripper_state = 1.0

        # Get initial end effector transforms
        body_q_np = self.state.body_q.numpy()
        initial_lee_tf = wp.transform(*body_q_np[self.env.lee_index])
        initial_ree_tf = wp.transform(*body_q_np[self.env.ree_index])
        
        # Statistics for debugging displacement limiting
        self.clamp_stats = {
            'pos_clamp_count': 0,
            'rot_clamp_count': 0,
            'total_frames': 0,
            'max_pos_distance': 0.0,
            'max_rot_angle': 0.0
        }
        
        # Create independent transform objects for gizmo control
        self.gizmo_lee_tf = wp.transform(
            wp.transform_get_translation(initial_lee_tf),
            wp.transform_get_rotation(initial_lee_tf)
        )
        self.gizmo_ree_tf = wp.transform(
            wp.transform_get_translation(initial_ree_tf),
            wp.transform_get_rotation(initial_ree_tf)
        )
        
        # Previous frame transforms for displacement limiting
        self.prev_gizmo_lee_tf = wp.transform(
            wp.transform_get_translation(initial_lee_tf),
            wp.transform_get_rotation(initial_lee_tf)
        )
        self.prev_gizmo_ree_tf = wp.transform(
            wp.transform_get_translation(initial_ree_tf),
            wp.transform_get_rotation(initial_ree_tf)
        )
        
        # Initial end effector transforms for IK objectives
        self.lee_tf = wp.transform(
            wp.transform_get_translation(initial_lee_tf),
            wp.transform_get_rotation(initial_lee_tf)
        )
        self.ree_tf = wp.transform(
            wp.transform_get_translation(initial_ree_tf),
            wp.transform_get_rotation(initial_ree_tf)
        )

    def _init_trajectory_queue(self):
        """Initialize trajectory queue for INTERACTIVE_QUEUE mode."""
        self.lee_tf_queue = [wp.transform() for _ in range(self.trajectory_queue_size)]
        self.ree_tf_queue = [wp.transform() for _ in range(self.trajectory_queue_size)]
        self.queue_index = 0
        
        # Cache targets to avoid GPU->CPU sync
        self.cached_lee_target_tf = self.lee_tf
        self.cached_ree_target_tf = self.ree_tf
        
        # Pre-compute interpolation alphas
        self.trajectory_alphas = np.linspace(
            1.0 / self.trajectory_queue_size,
            1.0,
            self.trajectory_queue_size,
            dtype=np.float32
        )

    def _setup_ik_objectives(self):
        """Setup IK objectives for dual-arm control."""
        total_residuals = 2 * 6 + self.model.joint_coord_count
        
        def _q2v4(q):
            return wp.vec4(q[0], q[1], q[2], q[3])

        # Left end effector position objective
        self.l_pos_obj = ik.IKPositionObjective(
            link_index=self.env.lee_index,
            link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=wp.array([wp.transform_get_translation(self.lee_tf)], dtype=wp.vec3),
            n_problems=1,
            total_residuals=total_residuals,
            residual_offset=0,
        )

        # Left end effector rotation objective
        self.l_rot_obj = ik.IKRotationObjective(
            link_index=self.env.lee_index,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([_q2v4(wp.transform_get_rotation(self.lee_tf))], dtype=wp.vec4),
            n_problems=1,
            total_residuals=total_residuals,
            residual_offset=3,
        )

        # Right end effector position objective
        self.r_pos_obj = ik.IKPositionObjective(
            link_index=self.env.ree_index,
            link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=wp.array([wp.transform_get_translation(self.ree_tf)], dtype=wp.vec3),
            n_problems=1,
            total_residuals=total_residuals,
            residual_offset=6,
        )

        # Right end effector rotation objective
        self.r_rot_obj = ik.IKRotationObjective(
            link_index=self.env.ree_index,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([_q2v4(wp.transform_get_rotation(self.ree_tf))], dtype=wp.vec4),
            n_problems=1,
            total_residuals=total_residuals,
            residual_offset=9,
        )

        # Joint limit objective
        self.obj_joint_limits = ik.IKJointLimitObjective(
            joint_limit_lower=self.model.joint_limit_lower,
            joint_limit_upper=self.model.joint_limit_upper,
            n_problems=1,
            total_residuals=total_residuals,
            residual_offset=12,
            weight=10.0,
        )

    def _init_solvers(self):
        """Initialize IK, rigid body, and cloth solvers."""
        # IK solver
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

        # Rigid body solver
        self.use_mujoco_cpu = False
        self.rigid_solver = newton.solvers.SolverMuJoCo(
            self.model,
            njmax=150000,
            # ncon_per_world=150000,
            solver='newton',
            cone="elliptic",
            use_mujoco_cpu=self.use_mujoco_cpu,
            use_mujoco_contacts=False,
            contact_stiffness_time_const=self.sim_dt
        )

        # Cloth solver
        self.model.edge_rest_angle.zero_()
        # VBD iterations (aligned with cloth solver): 12
        self.cloth_solver = newton.solvers.SolverVBD(
            self.model,
            iterations=12,
            self_contact_radius=self.env.self_contact_radius,
            self_contact_margin=self.env.self_contact_margin,
            handle_self_contact=True,
            vertex_collision_buffer_pre_alloc=32,
            edge_collision_buffer_pre_alloc=64,
            integrate_with_external_rigid_solver=True,
            collision_detection_interval=-1,
        )

    def _has_gizmo_moved(self, threshold_pos=0.001, threshold_rot=0.1):
        """Check if gizmo has moved significantly since last recorded position."""
        # Check left end effector
        lee_pos_curr = wp.transform_get_translation(self.gizmo_lee_tf)
        lee_pos_prev = wp.transform_get_translation(self.prev_gizmo_lee_tf)
        lee_pos_diff = wp.length(lee_pos_curr - lee_pos_prev)
        
        lee_rot_curr = wp.transform_get_rotation(self.gizmo_lee_tf)
        lee_rot_prev = wp.transform_get_rotation(self.prev_gizmo_lee_tf)
        lee_rot_dot = abs(wp.dot(lee_rot_curr, lee_rot_prev))
        lee_rot_diff = 2.0 * wp.acos(wp.clamp(lee_rot_dot, -1.0, 1.0))
        
        # Check right end effector
        ree_pos_curr = wp.transform_get_translation(self.gizmo_ree_tf)
        ree_pos_prev = wp.transform_get_translation(self.prev_gizmo_ree_tf)
        ree_pos_diff = wp.length(ree_pos_curr - ree_pos_prev)
        
        ree_rot_curr = wp.transform_get_rotation(self.gizmo_ree_tf)
        ree_rot_prev = wp.transform_get_rotation(self.prev_gizmo_ree_tf)
        ree_rot_dot = abs(wp.dot(ree_rot_curr, ree_rot_prev))
        ree_rot_diff = 2.0 * wp.acos(wp.clamp(ree_rot_dot, -1.0, 1.0))
        
        # Return True if any significant movement detected (only end effector, not gripper)
        return (lee_pos_diff > threshold_pos or lee_rot_diff > threshold_rot or
                ree_pos_diff > threshold_pos or ree_rot_diff > threshold_rot)


    def capture(self):
        """Capture IK and physics simulation into CUDA graphs for better performance."""
        self.ik_graph = None
        if wp.get_device().is_cuda and not self.use_mujoco_cpu:
            # Capture IK simulation
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
        
        # Get numpy view of shape_flags
        shape_flags_np = self.model.shape_flags.numpy()

        # Update left gripper collision: disable only while opening/releasing; enable when closed for grasp
        for shape_id in self.left_gripper_shape_ids:
            if left_opening:
                # Opening/releasing: disable collision
                shape_flags_np[shape_id] = shape_flags_np[shape_id] & ~self.COLLIDE_PARTICLES
            else:
                # Closed/maintaining: enable collision
                shape_flags_np[shape_id] = shape_flags_np[shape_id] | self.COLLIDE_PARTICLES
        
        # Right gripper same logic
        for shape_id in self.right_gripper_shape_ids:
            if right_opening:
                shape_flags_np[shape_id] = shape_flags_np[shape_id] & ~self.COLLIDE_PARTICLES
            else:
                shape_flags_np[shape_id] = shape_flags_np[shape_id] | self.COLLIDE_PARTICLES
        
        # Camera presets only on keys 3/4/5 (no auto-zoom on gripper toggle)
        if self.viewer.is_key_down("3"):
            # self.viewer.set_model(self.model)
            self.viewer.vsync = True
            if isinstance(self.viewer, newton.viewer.ViewerGL):
                camera_pos = type(self.viewer.camera.pos)(0.8, -0.01, 1.8)
                self.viewer.camera.pos = camera_pos
                self.viewer.camera.pitch = -89.9
                self.viewer.camera.fov = 60
                self.viewer.camera.yaw = 0
        if self.viewer.is_key_down("4"):
            self.viewer.vsync = True
            if isinstance(self.viewer, newton.viewer.ViewerGL):

                camera_pos = type(self.viewer.camera.pos)(0.70, 1.31, 0.86)
                self.viewer.camera.pos = camera_pos
                self.viewer.camera.pitch = -1.8
                self.viewer.camera.fov = 60
                self.viewer.camera.yaw = 268.7
        if self.viewer.is_key_down("5"):
            self.viewer.vsync = True
            if isinstance(self.viewer, newton.viewer.ViewerGL):
                camera_pos = type(self.viewer.camera.pos)(0.01, -0.01, 1.60)
                self.viewer.camera.pos = camera_pos
                # self.viewer.camera.pitch = -80
                self.viewer.camera.pitch = -48
                self.viewer.camera.fov = 60
                self.viewer.camera.yaw = 0

        # Sync flags back to GPU
        self.model.shape_flags.assign(shape_flags_np)

    def _update_gizmos_from_keyboard(self):
        """Update gizmo transforms and gripper states from keyboard input."""
        if not hasattr(self.viewer, "is_key_down"):
            return

        # Shift doubles speed
        speed_scale = 2.0 if self.viewer.is_key_down("shift") else 1.0
        step = self.KEYBOARD_MOVE_STEP * speed_scale
        rot_step = self.KEYBOARD_ROT_STEP * speed_scale

        # Left EE: WASD=XY, QE=Z
        lx = ly = lz = 0.0
        if self.viewer.is_key_down("w"):
            lx += step
        if self.viewer.is_key_down("s"):
            lx -= step
        if self.viewer.is_key_down("a"):
            ly += step
        if self.viewer.is_key_down("d"):
            ly -= step
        if self.viewer.is_key_down("e"):
            lz += step
        if self.viewer.is_key_down("q"):
            lz -= step

        if lx != 0 or ly != 0 or lz != 0:
            pos = wp.transform_get_translation(self.gizmo_lee_tf)
            new_pos = wp.vec3(pos[0] + lx, pos[1] + ly, pos[2] + lz)
            # Left arm: min height and no negative X/Y to avoid table penetration / out-of-bounds
            if self.min_ee_z is not None and new_pos[2] < self.min_ee_z:
                new_pos = wp.vec3(new_pos[0], new_pos[1], self.min_ee_z)
            # X lower bound: not below arm base (usually 0)
            base_x = 0.0
            if new_pos[0] < base_x:
                new_pos = wp.vec3(base_x, new_pos[1], new_pos[2])
            # Y lower bound: not below table center (usually 0)
            table_y = getattr(self.env, "TABLE_XY", (0.0, 0.0))[1]
            if new_pos[1] < table_y:
                new_pos = wp.vec3(new_pos[0], table_y, new_pos[2])
            self.gizmo_lee_tf = wp.transform(
                new_pos,
                wp.transform_get_rotation(self.gizmo_lee_tf),
            )

        # Left EE pitch: Z/C (pitch; IK follows)
        left_rot_delta = 0.0
        # Z = up, C = down
        if self.viewer.is_key_down("z"):
            left_rot_delta += rot_step
        if self.viewer.is_key_down("c"):
            left_rot_delta -= rot_step
        if left_rot_delta != 0.0:
            rot = wp.transform_get_rotation(self.gizmo_lee_tf)
            # Pitch around local Y axis
            local_y = wp.vec3(0.0, 1.0, 0.0)
            axis_world = wp.quat_rotate(rot, local_y)
            delta_q = wp.quat_from_axis_angle(axis_world, left_rot_delta)
            new_rot = wp.mul(delta_q, rot)
            pos = wp.transform_get_translation(self.gizmo_lee_tf)
            self.gizmo_lee_tf = wp.transform(pos, new_rot)

        # Right EE: U/J/H/K = X/Y, Y/I = Z (key layout shifted left)
        rx = ry = rz = 0.0
        if self.viewer.is_key_down("u"):
            rx += step
        if self.viewer.is_key_down("j"):
            rx -= step
        if self.viewer.is_key_down("h"):
            ry += step
        if self.viewer.is_key_down("k"):
            ry -= step
        if self.viewer.is_key_down("i"):
            rz += step
        if self.viewer.is_key_down("y"):
            rz -= step

        if rx != 0 or ry != 0 or rz != 0:
            pos = wp.transform_get_translation(self.gizmo_ree_tf)
            new_pos = wp.vec3(pos[0] + rx, pos[1] + ry, pos[2] + rz)
            # Right arm: same X lower bound (arm base)
            base_x = 0.0
            if new_pos[0] < base_x:
                new_pos = wp.vec3(base_x, new_pos[1], new_pos[2])
            if self.min_ee_z is not None and new_pos[2] < self.min_ee_z:
                new_pos = wp.vec3(new_pos[0], new_pos[1], self.min_ee_z)
            self.gizmo_ree_tf = wp.transform(
                new_pos,
                wp.transform_get_rotation(self.gizmo_ree_tf),
            )

        # Right EE pitch: B/M (B=up, M=down)
        right_rot_delta = 0.0
        if self.viewer.is_key_down("b"):
            right_rot_delta += rot_step
        if self.viewer.is_key_down("m"):
            right_rot_delta -= rot_step
        if right_rot_delta != 0.0:
            rot = wp.transform_get_rotation(self.gizmo_ree_tf)
            # Pitch around local Y axis
            local_y = wp.vec3(0.0, 1.0, 0.0)
            axis_world = wp.quat_rotate(rot, local_y)
            delta_q = wp.quat_from_axis_angle(axis_world, right_rot_delta)
            new_rot = wp.mul(delta_q, rot)
            pos = wp.transform_get_translation(self.gizmo_ree_tf)
            self.gizmo_ree_tf = wp.transform(pos, new_rot)

        # Gripper toggle: X=left, N=right (edge-triggered)
        x_down = self.viewer.is_key_down("x")
        if x_down and not self._prev_x_down:
            self._left_gripper_closed = not self._left_gripper_closed
        self._prev_x_down = x_down

        m_down = self.viewer.is_key_down("n")
        if m_down and not self._prev_m_down:
            self._right_gripper_closed = not self._right_gripper_closed
        self._prev_m_down = m_down

        if self._left_gripper_closed:
            self.open_left_gripper -= 0.2
        else:
            self.open_left_gripper += 0.2
        self.open_left_gripper = np.clip(self.open_left_gripper, 0.0, 1.0)

        if self._right_gripper_closed:
            self.open_right_gripper -= 0.2
        else:
            self.open_right_gripper += 0.2
        self.open_right_gripper = np.clip(self.open_right_gripper, 0.0, 1.0)

    def _push_targets_from_gizmos_queue_no_clamp(self):
        """Read gizmo-updated transform and push IK targets (no interpolation, direct follow)."""
        self._update_gizmos_from_keyboard()

        # Use gizmo directly as IK target (no interpolation queue) for direct follow
        target_lee_tf = self.gizmo_lee_tf
        target_ree_tf = self.gizmo_ree_tf

        # Update IK target (left)
        lee_pos = wp.transform_get_translation(target_lee_tf)
        lee_rot = wp.transform_get_rotation(target_lee_tf)
        self._ik_pos_buffer[0, 0] = lee_pos[0]
        self._ik_pos_buffer[0, 1] = lee_pos[1]
        self._ik_pos_buffer[0, 2] = lee_pos[2]
        self.l_pos_obj.target_positions.assign(self._ik_pos_buffer)

        self._ik_rot_buffer[0, 0] = lee_rot[0]
        self._ik_rot_buffer[0, 1] = lee_rot[1]
        self._ik_rot_buffer[0, 2] = lee_rot[2]
        self._ik_rot_buffer[0, 3] = lee_rot[3]
        self.l_rot_obj.target_rotations.assign(self._ik_rot_buffer)

        # Update IK target (right)
        ree_pos = wp.transform_get_translation(target_ree_tf)
        ree_rot = wp.transform_get_rotation(target_ree_tf)
        self._ik_pos_buffer[0, 0] = ree_pos[0]
        self._ik_pos_buffer[0, 1] = ree_pos[1]
        self._ik_pos_buffer[0, 2] = ree_pos[2]
        self.r_pos_obj.target_positions.assign(self._ik_pos_buffer)

        self._ik_rot_buffer[0, 0] = ree_rot[0]
        self._ik_rot_buffer[0, 1] = ree_rot[1]
        self._ik_rot_buffer[0, 2] = ree_rot[2]
        self._ik_rot_buffer[0, 3] = ree_rot[3]
        self.r_rot_obj.target_rotations.assign(self._ik_rot_buffer)

        # Cache current targets for stats or other logic
        self.cached_lee_target_tf = target_lee_tf
        self.cached_ree_target_tf = target_ree_tf

        # No queue used
        self.queue_executing = False
        self.queue_index = 0

        # Store previous frame gizmo for _has_gizmo_moved etc.
        self.prev_gizmo_lee_tf = wp.transform(
            wp.transform_get_translation(self.gizmo_lee_tf),
            wp.transform_get_rotation(self.gizmo_lee_tf)
        )
        self.prev_gizmo_ree_tf = wp.transform(
            wp.transform_get_translation(self.gizmo_ree_tf),
            wp.transform_get_rotation(self.gizmo_ree_tf)
        )

    def ik_simulate(self):
        self.solver.solve(iterations=self.ik_iters)

    def physics_simulate(self):
        """Run physics simulation for all substeps."""
        for s in range(self.sim_substeps):
            self._substep_index = s

            # Update controls for this substep on device
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

            # Collide and clear forces; filter gripper contacts if opened
            self.contacts = self.model.collide(self.state_0)
            self.state_0.clear_forces()
            self.state_1.clear_forces()

            # Temporarily clear particle info for rigid solver
            particle_count = self.model.particle_count
            self.model.particle_count = 0

            # Rigid body step
            self.rigid_solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # Recover particle info
            self.state_0.particle_f.zero_()
            self.model.particle_count = particle_count

            # Cloth step
            self.contacts = self.model.collide(self.state_0, soft_contact_margin=self.env.cloth_body_contact_margin)
            self.cloth_solver.step(self.state_0, self.state_1, None, self.contacts, self.sim_dt)

            # Swap state
            (self.state_0, self.state_1) = (self.state_1, self.state_0)

    def reset(self):
        """Restore robot joints and cloth to initial state at script start."""
        wp.synchronize()

        # Restore joints
        wp.copy(self.state_0.joint_q, self._initial_joint_q)
        wp.copy(self.state_0.joint_qd, self._initial_joint_qd)
        wp.copy(self.state_1.joint_q, self._initial_joint_q)
        wp.copy(self.state_1.joint_qd, self._initial_joint_qd)

        # Restore cloth particles (position + zero velocity)
        if self._initial_particle_q is not None:
            wp.copy(self.state_0.particle_q, self._initial_particle_q)
            wp.copy(self.state_1.particle_q, self._initial_particle_q)
            self.state_0.particle_qd.zero_()
            self.state_1.particle_qd.zero_()

        # Clear residual forces to avoid post-reset jitter
        self.state_0.clear_forces()
        self.state_1.clear_forces()
        if hasattr(self.state_0, 'particle_f'):
            self.state_0.particle_f.zero_()
        if hasattr(self.state_1, 'particle_f'):
            self.state_1.particle_f.zero_()

        # Recompute FK to refresh derived state
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.state_1.joint_q, self.state_1.joint_qd, self.state_1)

        # Reset time, queue, and end-effector control state
        self.sim_time = 0.0
        self.sim_frame = 0
        self.queue_executing = False
        self.queue_index = 0

        self.open_left_gripper = 1.0
        self.open_right_gripper = 1.0
        self.left_gripper_state = 1.0
        self.right_gripper_state = 1.0
        self._left_gripper_closed = False
        self._right_gripper_closed = False

        # Re-init gizmo / end-effector pose
        self._init_end_effector_control()

        # Rebuild cloth BVH for collision with new particle positions
        self.cloth_solver.rebuild_bvh(self.state_0)

        # Re-capture CUDA graph (state changed, old graph may be stale)
        wp.synchronize()
        self.capture()

    def stop_teleopration_device(self):
        self.right_quest_controller.stop()
        self.left_quest_controller.stop()

    def step(self):
        """Execute one simulation step."""
        if self.sim_time == 0.0:
            newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
            self.left_gripper_state = self.open_left_gripper
            self.right_gripper_state = self.open_right_gripper

        self._push_targets_from_gizmos_queue_no_clamp()

        # IK step, update self.ik_joint_q as the target pose
        if self.ik_graph:
            wp.capture_launch(self.ik_graph)
        else:
            self.ik_simulate()

        ik_joint_q = self.ik_joint_q.flatten()
        # reply mode: direct assign recorder data
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
            ik_joint_q_np[self.env.controllable_joint_indices] = cur_joint_q[(list(range(3,9))+list(range(11,17)))]  
            # old lift2
            # ik_joint_q_np[self.env.controllable_joint_indices] = cur_joint_q[(list(range(0,6))+list(range(8,14)))]  
            
            self.open_left_gripper = cur_gripper_q[0]
            self.open_right_gripper = cur_gripper_q[1]

        # Write modified ik_joint_q_np back to Warp array (teleop or replay)
        ik_joint_q.assign(ik_joint_q_np)

        # Copy data directly on GPU to avoid CPU-GPU transfer
        # Use warp.copy for efficient GPU-to-GPU copy
        state_q_flat = self.state_0.joint_q.reshape((self.model.joint_coord_count,))
        wp.copy(self.q0_frame_wp, state_q_flat)
        wp.copy(self.q_target_frame_wp, ik_joint_q)

        # Update gripper params on host then transfer once (small data, acceptable overhead)
        # [left_prev, left_target, right_prev, right_target]
        # Note: This is a small 4-element array, the overhead is minimal
        gripper_params_host = np.array([
            float(self.left_gripper_state),
            float(self.open_left_gripper),
            float(self.right_gripper_state),
            float(self.open_right_gripper),
        ], dtype=np.float32)
        self.gripper_params_wp.assign(gripper_params_host)
        
        self._update_gripper_collision_filtering_cpu()
        self.cloth_solver.rebuild_bvh(self.state_0)

        # Physics step for all substeps (loop is inside physics_simulate for CUDA graph)
        if self.physics_graph:
            # print("Launching physics simulation from CUDA graph...")
            wp.capture_launch(self.physics_graph)
        else:
            self.physics_simulate()
            
        self.sim_time += self.frame_dt
        self.sim_frame += 1
        
        # Calculate and print FPS
        current_time = time.time()
        if self.last_step_time is not None:
            step_duration = current_time - self.last_step_time
            current_fps = 1.0 / step_duration if step_duration > 0 else 0            
            # Print FPS info (skip frequent printing in GPU-optimized interactive modes for better performance)

        self.last_step_time = current_time

        if self.mode == 'teleopration':
            self.recorder.record_data(self.state_0, self.open_left_gripper, self.open_right_gripper)
        
        self.left_gripper_state = self.open_left_gripper
        self.right_gripper_state = self.open_right_gripper


    def render(self):
        self.viewer.begin_frame(self.sim_time)

        self.viewer.log_gizmo("left_target_tcp", self.gizmo_lee_tf)
        self.viewer.log_gizmo("right_target_tcp", self.gizmo_ree_tf)

        # newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.viewer.log_state(self.state_0)

        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

        if self.save_image:
            self.recorder.save_image(self.viewer, self.sim_frame)

        wp.synchronize()

    def finished(self):
        self.recorder.save_data()
    