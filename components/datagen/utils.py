import os
import json
import random
import numpy as np

def read_json_file(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    return data

def save_json_file(save_file_path: str, data):
    with open(save_file_path, 'w') as f:
        json.dump(data, f, indent=2)

    return data

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def rand_int_with_seed(min_val, max_val, seed=None):
    if seed is not None:
        random.seed(seed)
    return random.randint(min_val, max_val)


def interpolate_position(start, end, duration, fps=60):
    start = np.array(start)
    end = np.array(end)
    
    num_points = int(duration * fps)
    t = np.linspace(0, 1, num_points)
    
    result = start + np.outer(t, (end - start))
    
    return result

def quaternion_lerp(start, end, duration, fps=60):
    start = np.array(start)
    end = np.array(end)
    
    if np.dot(start, end) < 0:
        end = -end
    
    num_points = int(duration * fps)
    t = np.linspace(0, 1, num_points)

    result = start + np.outer(t, (end - start))
    
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    result = result / norms
    
    return result

def interpolate_linear(start, end, duration, fps=60):
    num_points = int(duration * fps)
    result = np.linspace(start, end, num_points)
    return result
