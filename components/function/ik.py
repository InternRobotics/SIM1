import numpy as np
import warp as wp
import newton.ik as ik_module

ROBOT_BASE_HEIGHT = 0.17
GRIPPER_LIMIT_LOWER = 0.001
GRIPPER_LIMIT_UPPER = 0.044
IK_ITERATIONS = 24

def solve_ik(left_tf, right_tf, openness, model, lee_index, ree_index,
             controllable_indices, left_gripper_indices, right_gripper_indices):
    # Build position/rotation targets (vec3/vec4)
    target_lee_pos = wp.array([wp.vec3(left_tf[0], left_tf[1], left_tf[2])], dtype=wp.vec3, device=model.device)
    target_lee_rot = wp.array([wp.vec4(left_tf[3], left_tf[4], left_tf[5], left_tf[6])], dtype=wp.vec4, device=model.device)
    target_ree_pos = wp.array([wp.vec3(right_tf[0], right_tf[1], right_tf[2])], dtype=wp.vec3, device=model.device)
    target_ree_rot = wp.array([wp.vec4(right_tf[3], right_tf[4], right_tf[5], right_tf[6])], dtype=wp.vec4, device=model.device)

    # Configure IK objectives (reuse Example._setup_ik_objectives)
    total_residuals = 2 * 6 + model.joint_coord_count
    l_pos_obj = ik_module.IKPositionObjective(
        link_index=lee_index,
        link_offset=wp.vec3(0.0),
        target_positions=target_lee_pos,
        n_problems=1,
        total_residuals=total_residuals,
        residual_offset=0,
    )
    l_rot_obj = ik_module.IKRotationObjective(
        link_index=lee_index,
        link_offset_rotation=wp.quat_identity(),
        target_rotations=target_lee_rot,
        n_problems=1,
        total_residuals=total_residuals,
        residual_offset=3,
    )
    r_pos_obj = ik_module.IKPositionObjective(
        link_index=ree_index,
        link_offset=wp.vec3(0.0),
        target_positions=target_ree_pos,
        n_problems=1,
        total_residuals=total_residuals,
        residual_offset=6,
    )
    r_rot_obj = ik_module.IKRotationObjective(
        link_index=ree_index,
        link_offset_rotation=wp.quat_identity(),
        target_rotations=target_ree_rot,
        n_problems=1,
        total_residuals=total_residuals,
        residual_offset=9,
    )
    obj_joint_limits = ik_module.IKJointLimitObjective(
        joint_limit_lower=model.joint_limit_lower,
        joint_limit_upper=model.joint_limit_upper,
        n_problems=1,
        total_residuals=total_residuals,
        residual_offset=12,
        weight=10.0,
    )

    # Solve IK
    ik_joint_q = wp.zeros((1, model.joint_coord_count), dtype=float, device=model.device)
    solver = ik_module.IKSolver(
        model=model,
        joint_q=ik_joint_q,
        objectives=[l_pos_obj, l_rot_obj, r_pos_obj, r_rot_obj, obj_joint_limits],
        lambda_initial=0.1,
        jacobian_mode=ik_module.IKJacobianMode.MIXED,
    )
    solver.solve(iterations=IK_ITERATIONS)

    # Add gripper joints
    result = ik_joint_q.numpy()[0].copy()
    left_open, right_open = openness
    left_pos = GRIPPER_LIMIT_LOWER + left_open * (GRIPPER_LIMIT_UPPER - GRIPPER_LIMIT_LOWER)
    right_pos = GRIPPER_LIMIT_LOWER + right_open * (GRIPPER_LIMIT_UPPER - GRIPPER_LIMIT_LOWER)
    result[left_gripper_indices] = left_pos
    result[right_gripper_indices] = right_pos

    return result