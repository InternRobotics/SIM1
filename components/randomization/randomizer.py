import warp as wp
import numpy as np
import newton
from .utils import get_random_value, set_body_transform


BASE_POS_RANGE = [[0, 0], [0, 0], [0, 0]]
BASE_ROT_RANGE = [[0, 0], [0, 0], [0, 0]]

# Simple randomization module: randomize position and rotation of target objects in env
class Randomizer:

    def __init__(self, pos_rot=None, pos_range=BASE_POS_RANGE, rot_range=BASE_ROT_RANGE, axis=[1, 1, 1]):
        """
        env: 
        pos_range:[[x_min, x_max],[y_min, y_max],[z_min, z_max]]
        rot_range:[[x_min, x_max],[y_min, y_max],[z_min, z_max]]
        """
        self.pos_range = pos_range
        self.rot_range = rot_range
        self.axis = axis

        if pos_rot is None:
            self.rand_pos, self.rand_rot = self.randomize()
        else:
            self.rand_pos, self.rand_rot = self.get_pos_rot(pos_rot)
            
    def get_pos_rot(self, pos_rot):
        pos = pos_rot[:3]
        rot = pos_rot[3:]

        pos_new = wp.vec3(pos[0], pos[1], pos[2])
        rot = wp.vec3(rot[0], rot[1], rot[2])
        # rot_new = eular2quat(rot)
        rot_new = newton.utils.quat_from_euler(rot, 0, 1, 2)
        return pos_new, rot_new
    
    def _postition_randomize(self):
        x_new = get_random_value(self.pos_range[0])
        y_new = get_random_value(self.pos_range[1])
        z_new = get_random_value(self.pos_range[2])
        return wp.vec3(x_new, y_new, z_new)

    def _orientation_randomize(self):
        x_new = np.deg2rad(get_random_value(self.rot_range[0]))
        y_new = np.deg2rad(get_random_value(self.rot_range[1]))
        z_new = np.deg2rad(get_random_value(self.rot_range[2]))
        
        q_x = wp.quat_from_axis_angle(wp.vec3(1.0*self.axis[0], 0.0, 0.0), float(x_new))
        q_y = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0*self.axis[1], 0.0), float(y_new))
        q_z = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0*self.axis[2]), float(z_new))
        q_f = wp.quat_identity()
        for q in [q_x, q_y, q_z]:
            q_f = wp.mul(q, q_f)
        return q_f
    
    def randomize(self):
        pos_new = self._postition_randomize()
        rot_new = self._orientation_randomize()
        return pos_new, rot_new
        
        


