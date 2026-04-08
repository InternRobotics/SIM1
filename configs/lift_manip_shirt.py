import newton
from tasks import Lift2ClothManip
from components.recorder import DualArmRecorder
from components.randomization import Randomizer
from envs.lift2_short_shirt import LiftShortShirt


class LiftManipShortShirt:
    def __init__(self):
        self.task_name = "lift_manip_shirt"

        # Interactive teleoperation with OpenGL Viewer GUI
        self.viewer = newton.viewer.ViewerGL()

        # Randomizer aligned with replay.py
        # No rigid transform: cloth position/pose fixed at calibration
        self.randomizer = Randomizer(
            pos_range=[[-0.0, 0.0], [-0.0, 0.0], [0.0, 0.0]],
            rot_range=[[0, 0], [0, 0], [0, 0]],
            axis=[-1, 1, 1],
        )

        # Short-shirt environment
        self.env = LiftShortShirt(self.randomizer)

        # Dual-arm recorder for teleoperation data
        self.recorder = DualArmRecorder(self.task_name, self.env.robot_joint_q_cnt)

    def finalize(self):
        task = Lift2ClothManip(self.viewer, self.env, self.recorder)
        return task