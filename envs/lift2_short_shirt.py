import os
import sys

# Repo root must be on path for sim1_asset_paths (Hugging Face asset root: SIM1_ASSETS_ROOT).
_ENV_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_ENV_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from sim1_asset_paths import get_acone_urdf_path, get_cloth_usdc_path

import newton
import warp as wp
import json
import numpy as np
from pxr import Usd, UsdGeom
from components.randomization.utils import get_random_value, set_body_transform

# Joint name sets
fixed_joint_names = {
    # "base_link", "body4", etc. — currently empty
}

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
left_gripper_joint_names = {"left_joint17", "left_joint18"}
right_ee_body_names = {"right_link26"}
right_gripper_joint_names = {"right_joint27", "right_joint28"}

_ACONE_URDF_PATH = get_acone_urdf_path()
_CLOTH_USDC_PATH = get_cloth_usdc_path()


def load_json(config_file):
    """Load physics parameters from a JSON configuration file."""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        print(f"✓ Successfully loaded configuration from: {config_file}")
        return config.get('physics_parameters', {})
    except FileNotFoundError:
        print(f"⚠ Warning: Configuration file not found: {config_file}")
        print(f"  Using default parameters instead.")
        return {}
    except json.JSONDecodeError as e:
        print(f"⚠ Warning: Failed to parse JSON in {config_file}: {e}")
        print(f"  Using default parameters instead.")
        return {}
    except Exception as e:
        print(f"⚠ Warning: Error loading configuration file: {e}")
        print(f"  Using default parameters instead.")
        return {}


class LiftShortShirt:
    IK_ITERATIONS = 24

    # Joint control gains
    ARM_JOINT_KE = 3000.0
    ARM_JOINT_KD = 10.0
    GRIPPER_JOINT_KE = 3000.0
    GRIPPER_JOINT_KD = 10.0

    # Gripper limits
    GRIPPER_LIMIT_LOWER = 0.001
    GRIPPER_LIMIT_UPPER = 0.044

    # Robot pose
    ROBOT_BASE_HEIGHT = 0.17  # meters

    # Table configuration
    TABLE_XY = (0.65, 0.0)
    TABLE_Z_BASE = 0.4
    TABLE_Z_RANDOM_RANGE = (0.00, 0.00)  # Currently fixed; change to e.g. (-0.07, 0.03) for real randomness for table
    TABLE_SIZE = (0.5, 1.0, 0.4)  # half extents (hx, hy, hz)

    # Cloth properties (world-space base; must stay in sync with TABLE_* above)
    # CLOTH_POS = (0.645, -0.32, 0.82)  # world coordinates base (will be adjusted)
    # CLOTH_POS = (0.645, -0.32, 1.2)  # alternate height
    CLOTH_POS = (0.660, 0.0, 0.78)  # alternate layout
    CLOTH_ROTATION_ANGLE = np.pi * 0.5
    CLOTH_DENSITY = 2
    CLOTH_SCALE = 0.001

    def __init__(self, randomizer, physics_json=None,cloth_pos_rot=None):
        self._env = newton.ModelBuilder()
        self.randomizer = randomizer
        self.cloth_pos_rot = cloth_pos_rot  # shape (6,) or None

        # Sample table height offset BEFORE creating environment
        self.table_z_offset = self._sample_table_height_offset()

        config_params = None
        if physics_json:
            config_params = load_json(physics_json)
        self._set_physical_parameters(config_params)
        self._create_env()
        self._configure_joint_controllers()

    def _sample_table_height_offset(self):
        """Sample random offset for table height within allowed range."""
        return get_random_value(self.TABLE_Z_RANDOM_RANGE)

    @property
    def model(self):
        model = self._env.finalize(requires_grad=False)
        model.soft_contact_ke = self.soft_contact_ke
        model.soft_contact_kd = self.soft_contact_kd
        model.soft_contact_mu = self.self_contact_friction
        return model

    def _set_physical_parameters(self, physics_params=None):
        defaults = {
            'cloth_particle_radius': 0.008,
            'cloth_body_contact_margin': 0.01,
            'self_contact_radius': 0.002,
            'self_contact_margin': 0.003,
            'soft_contact_ke': 500.0,
            'soft_contact_kd': 5e-3,
            'robot_friction': 1.5,
            'table_friction': 0.0,
            'self_contact_friction': 0.25,
            'tri_ke': 1e2,
            'tri_ka': 1e2,
            'tri_kd': 1.5e-6,
            'bending_ke': 1e-4,
            'bending_kd': 1e-3,
        }
        if physics_params:
            print(f"Loading physics parameters from config file:")
            for key, value in physics_params.items():
                if key in defaults:
                    defaults[key] = value
                    print(f"  {key}: {value}")
        for key, value in defaults.items():
            setattr(self, key, value)

    def _create_env(self):
        # Add robot
        self._env.add_urdf(
            _ACONE_URDF_PATH,
            floating=False,
            enable_self_collisions=False,
            xform=wp.transform(p=wp.vec3(0.0, 0.0, self.ROBOT_BASE_HEIGHT))
        )

        self._identify_robot_joint_indices()

        # Compute table center Z and adjust half-height to avoid ground penetration
        table_z = self.TABLE_Z_BASE + self.table_z_offset / 2.0
        original_hz = self.TABLE_SIZE[2]
        # Ensure table bottom (table_z - hz) >= 0 → hz <= table_z
        effective_hz = min(original_hz, table_z) if table_z > 0 else 1e-4

        table_pos = wp.vec3(self.TABLE_XY[0], self.TABLE_XY[1], table_z)

        body_table = self._env.add_body()
        self._env.add_joint_fixed(-1, body_table)
        table_cfg = newton.ModelBuilder.ShapeConfig(
            mu=self.table_friction,
            has_particle_collision=True,
            has_shape_collision=False,
        )
        self._env.add_shape_box(
            body_table,
            xform=wp.transform(p=table_pos, q=wp.quat_identity()),
            hx=self.TABLE_SIZE[0],
            hy=self.TABLE_SIZE[1],
            hz=effective_hz,  # ← Dynamically adjusted to prevent ground penetration
            cfg=table_cfg
        )

        # Set material properties for all shapes
        for i in range(len(self._env.shape_material_mu)):
            self._env.shape_material_mu[i] = 1.0
            self._env.shape_material_ka[i] = 0.002
            self._env.shape_is_solid[i] = True

        
        usd_stage = Usd.Stage.Open(_CLOTH_USDC_PATH)
        usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/root/Mesh"))

        
        mesh_points = np.array(usd_geom.GetPointsAttr().Get())
        mesh_indices = np.array(usd_geom.GetFaceVertexIndicesAttr().Get())
        print("=== Cloth Information ===")
        print(f"vertices={mesh_points.shape}, faces={mesh_indices.shape}")
        
        # Cache cloth triangles for downstream (e.g. recording / analysis)
        self.cloth_triangles = mesh_indices.reshape(-1, 3)  # shape: (F, 3)
        
        
        pos_rand, rot_rand = self.randomizer.rand_pos, self.randomizer.rand_rot
        self.pos_rand = pos_rand
        self.rot_rand = rot_rand
        print(f"[ENV] Applying cloth randomization: pos={self.randomizer.rand_pos}, rot={self.randomizer.rand_rot}")
        # Default: CLOTH_POS + fixed fold rotation (see class constants above).
        # Optional cloth_pos_rot (6,) = [x, y, z, euler_x, euler_y, euler_z] in radians replaces
        # position and base rotation (e.g. replay from NPZ `pos_rot` recorded at data collection).
        rot_base = wp.quat_from_axis_angle(wp.vec3(-1.0, 0.0, 0.0), self.CLOTH_ROTATION_ANGLE)
        z_table_sync = self.table_z_offset / 2.0
        if self.cloth_pos_rot is not None:
            pr = np.asarray(self.cloth_pos_rot, dtype=np.float64).reshape(-1)
            if pr.size != 6:
                raise ValueError(f"cloth_pos_rot must have shape (6,), got {pr.shape}")
            cloth_pos_base = wp.vec3(float(pr[0]), float(pr[1]), float(pr[2]))
            rot_base = newton.utils.quat_from_euler(
                wp.vec3(float(pr[3]), float(pr[4]), float(pr[5])), 0, 1, 2
            )
            z_table_sync = 0.0
        else:
            cloth_pos_base = wp.vec3(*self.CLOTH_POS)

        # Synchronize cloth Z with table: table center uses TABLE_Z_BASE + table_z_offset/2
        cloth_base = wp.vec3(
            cloth_pos_base.x,
            cloth_pos_base.y,
            cloth_pos_base.z + z_table_sync,
        )
        pos_final = cloth_base + self.pos_rand
        rot_final = wp.mul(self.rot_rand, rot_base)
        
        
        # Log final cloth pose (world position + quaternion xyzw)
        pos_final_np = np.array([pos_final[0], pos_final[1], pos_final[2]], dtype=np.float32)
        rot_final_np = np.array([
            float(rot_final[0]), float(rot_final[1]),
            float(rot_final[2]), float(rot_final[3])
        ], dtype=np.float32)

        print(f"[INFO] Cloth final position (world): [{pos_final_np[0]:.4f}, {pos_final_np[1]:.4f}, {pos_final_np[2]:.4f}]")
        print(f"[INFO] Cloth final rotation (xyzw): [{rot_final_np[0]:.4f}, {rot_final_np[1]:.4f}, {rot_final_np[2]:.4f}, {rot_final_np[3]:.4f}]")
        print(f"[INFO] Rotation norm: {np.linalg.norm(rot_final_np):.6f}")

        vertices = [wp.vec3(v) for v in mesh_points]
        self._env.add_cloth_mesh(
            vertices=vertices,
            indices=mesh_indices,
            rot=rot_final,
            pos=pos_final,
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=self.CLOTH_DENSITY,
            scale=self.CLOTH_SCALE,
            tri_ke=self.tri_ke,
            tri_ka=self.tri_ka,
            tri_kd=self.tri_kd,
            edge_ke=self.bending_ke,
            edge_kd=self.bending_kd,
            particle_radius=self.cloth_particle_radius,
        )

        self._env.color()
        self._env.add_ground_plane()

    def _identify_robot_joint_indices(self):
        for i in range(self._env.body_count):
            if self._env.body_key[i] in left_ee_body_names:
                self.lee_index = i
            if self._env.body_key[i] in right_ee_body_names:
                self.ree_index = i

        self.fixed_joint_indices = np.array([], dtype=int)
        self.controllable_joint_indices = np.array([], dtype=int)
        self.left_gripper_joint_indices = np.array([], dtype=int)
        self.right_gripper_joint_indices = np.array([], dtype=int)

        cnt = 0
        for i in range(self._env.joint_count):
            dof_dim = self._env.joint_dof_dim[i]
            dof_start = cnt
            dof_end = cnt + dof_dim[0] + dof_dim[1]

            for j in range(dof_start, dof_end):
                joint_key = self._env.joint_key[i]
                if joint_key in fixed_joint_names:
                    self.fixed_joint_indices = np.append(self.fixed_joint_indices, j)
                elif joint_key in controllable_joint_names:
                    self.controllable_joint_indices = np.append(self.controllable_joint_indices, j)
                elif joint_key in left_gripper_joint_names:
                    self.left_gripper_joint_indices = np.append(self.left_gripper_joint_indices, j)
                elif joint_key in right_gripper_joint_names:
                    self.right_gripper_joint_indices = np.append(self.right_gripper_joint_indices, j)

            cnt += dof_dim[0] + dof_dim[1]

        print(f"joint dq cnt check: {cnt} == {self._env.joint_dof_count}")
        print("fixed joint", self.fixed_joint_indices)
        print("controllable joint", self.controllable_joint_indices)
        print("left gripper joint", self.left_gripper_joint_indices)
        print("right gripper joint", self.right_gripper_joint_indices)

        self.robot_joint_q_cnt = len(self._env.joint_q)

    def _configure_joint_controllers(self):
        # PD gains + per-step control.joint_target_pos (no JointMode in current Newton)
        for i in self.controllable_joint_indices:
            self._env.joint_target_ke[i] = self.ARM_JOINT_KE
            self._env.joint_target_kd[i] = self.ARM_JOINT_KD

        for i in self.fixed_joint_indices:
            self._env.joint_target_ke[i] = 0.0
            self._env.joint_target_kd[i] = 0.0
            self._env.joint_limit_lower[i] = 0
            self._env.joint_limit_upper[i] = 0

        gripper_indices = np.concatenate((
            self.left_gripper_joint_indices,
            self.right_gripper_joint_indices
        ))
        for i in gripper_indices:
            self._env.joint_limit_lower[i] = self.GRIPPER_LIMIT_LOWER
            self._env.joint_limit_upper[i] = self.GRIPPER_LIMIT_UPPER
            self._env.joint_target_ke[i] = self.GRIPPER_JOINT_KE
            self._env.joint_target_kd[i] = self.GRIPPER_JOINT_KD