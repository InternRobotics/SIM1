import warp as wp
import numpy as np
import random
import math

def get_random_value(range):
    min_value, max_value = range
    value = random.uniform(min_value, max_value)
    return value

def get_wp_transform(tf):
    pos = wp.vec3(tf[0], tf[1], tf[2])
    quat = wp.quat(tf[3], tf[4], tf[5], tf[6])
    return wp.transform(pos, quat)

def get_body_transform(state, body_id, use_numpy=True):
    """
    state: newton state
    id: index of body in model
    transfrom: 7 Dof, [translate, orientation]->[x, y, z, i, j, k, w]
    """
    if use_numpy:
        return state.body_q.numpy()[body_id]
    else:
        tf = state.body_q.numpy()[body_id]
        pos = wp.vec3(tf[0], tf[1], tf[2])
        quat = wp.quat(tf[3], tf[4], tf[5], tf[6])
    return wp.transform(pos, quat)

# def set_body_transform(state, body_id, transform):
#     """
#     state: newton state
#     id: index of body in model
#     transfrom: 7 Dof, [translate, orientation]->[x, y, z, i, j, k, w]
#     """
#     return state.body_q[wp.array((body_id,), dtype=wp.int32)].assign(transform)

def set_body_transform(state, body_id, transform):
    """
    state: newton state
    id: index of body in model
    transfrom: 7 Dof, [translate, orientation]->[x, y, z, i, j, k, w]
    """
    return state.body_q[wp.array((body_id,), dtype=wp.int32)].assign(transform)
