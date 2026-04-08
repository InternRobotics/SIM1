import numpy as np
import warp as wp
import newton

def fk(joint_q, model, lee_index, ree_index):
    state = model.state()
    wp_q = wp.array(joint_q, dtype=float, device=model.device)
    wp_qd = wp.zeros_like(wp_q)
    newton.eval_fk(model, wp_q, wp_qd, state)
    body_q = state.body_q.numpy()
    left_tf = body_q[lee_index].copy()  # [x, y, z, qw, qx, qy, qz]
    right_tf = body_q[ree_index].copy()
    return left_tf, right_tf