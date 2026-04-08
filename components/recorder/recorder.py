import os
import time
import numpy as np
from newton.utils import RecorderModelAndState
from components.function import dump_gl_frame_image
import glob
import cv2

def get_timestamp(mode='timestamp'):
    current_time = time.asctime(time.localtime(time.time()))
    tmp = current_time.split(" ")
    if "" in tmp:
        tmp.remove("")
    week, month, day, ctime, year = tmp
    
    if mode == 'timestamp':
        return ctime.replace(":","-")
    else:
        return f"{year}-{month}-{day}" 

class BaseRecorder:

    def __init__(self, task_name, data_root):
        self.newton_recorder = RecorderModelAndState()
        self.data_root = data_root
        self.save_root = os.path.join(data_root, task_name)

    def record_state(self, state):
        self.init_state = state
        self.newton_recorder.record(state)

    def record_model(self, model):
        self.newton_recorder.record_model(model)

    def playback_to_init(self, state):
        self.newton_recorder.history = []
        self.newton_recorder.record(self.init_state)
        self.newton_recorder.playback(state, 0)
        # self.newton_recorder.playback_model(model)
        # self.newton_recorder.deserialized_model = None

    



class SingleArmRecorder(BaseRecorder):

    def __init__(self, task_name, joint_dof_count, data_root="./dataset", mode="teleoperation"):
        super().__init__(task_name, data_root)
        self.joint_dof_count = joint_dof_count
        self.batch = get_timestamp('batch')
        self.joint_q_seq = np.empty((0, self.joint_dof_count), dtype=np.float32)
        
        if not os.path.exists(os.path.join(self.save_root, self.batch)) and mode == 'teleoperation':
            os.makedirs(os.path.join(self.save_root, self.batch))

    def record_data(self, state):
        joint_q_np = state.joint_q.numpy()
        self.joint_q_seq = np.vstack((self.joint_q_seq, joint_q_np[0:self.joint_dof_count]))
    
    def save_data(self):
        file_name = os.path.join(self.save_root, self.batch, f"{get_timestamp()}.npz")
        
        np.savez(file_name, joint_q=self.joint_q_seq)
        print(f"Saved data at {file_name}")

    def load_from_file(self, filename):
        with np.load(filename) as data:
            self.joint_q_seq  = data['joint_q']


class DualArmRecorder(BaseRecorder):
    def __init__(self, task_name, joint_dof_count, data_root="./dataset", mode="teleoperation", fps=30):
        super().__init__(task_name, data_root)
        self.joint_dof_count = joint_dof_count
        self.batch = get_timestamp('batch')
        self.fps = fps  # FPS of recorded video

        self.joint_q_seq = np.empty((0, self.joint_dof_count), dtype=np.float32)
        self.openness_seq = np.empty((0, 2), dtype=np.float32)
        self.base_transform_seq = np.empty((0, 7), dtype=np.float32)
        self.image_frames = []  # Buffer for video frames

        if not os.path.exists(os.path.join(self.save_root, self.batch)) and mode == 'teleoperation':
            os.makedirs(os.path.join(self.save_root, self.batch))

    def record_data(self, state, open_left_gripper, open_right_gripper):
        joint_q_np = state.joint_q.numpy()
        body_q_np = state.body_q.numpy()
        base_transform = body_q_np[0]

        self.joint_q_seq = np.vstack((self.joint_q_seq, joint_q_np[0:self.joint_dof_count]))
        self.openness_seq = np.vstack((self.openness_seq, np.array([open_left_gripper, open_right_gripper])))
        self.base_transform_seq = np.vstack((self.base_transform_seq, base_transform))
    def record_frame(self, viewer):
        width = viewer.renderer._screen_width
        height = viewer.renderer._screen_height
        #rgba_image = dump_gl_frame_image(width, height)
        #viewer.get_frame()
        rgba_image = viewer.get_frame()  # Use viewer API to grab RGBA frame
        rgba_image = rgba_image.to("cpu").numpy()
        if rgba_image is None or rgba_image.size == 0:
            print("[ERROR] Captured frame is empty!")
            return
            
        bgr_image = cv2.cvtColor(rgba_image, cv2.COLOR_RGBA2BGR)
        self.image_frames.append(bgr_image)
        print(f"[DEBUG] Recorded frame {len(self.image_frames)}: {bgr_image.shape}")

    def save_data(self):
        file_name = os.path.join(self.save_root, self.batch, f"{get_timestamp()}.npz")
        np.savez(
            file_name,
            joint_q=self.joint_q_seq,
            openness=self.openness_seq,
            base_transform=self.base_transform_seq
        )
        print(f"Saved data at {file_name}")

    def save_video(self, output_path=None):
        if not self.image_frames:  # Use image_frames buffer
            print("[WARN] No video frames to save.")
            return

        if output_path is None:
            video_dir = os.path.join(self.save_root, self.batch)
            os.makedirs(video_dir, exist_ok=True)
            output_path = os.path.join(video_dir, "output.mp4")
        else:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Use OpenCV to write video (we already have BGR frames)
        height, width, _ = self.image_frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))
        for frame in self.image_frames:  # 
            video_writer.write(frame)
        video_writer.release()
        print(f"[INFO] Video saved to: {output_path}")
    def load_from_file(self, filename):
        self.cur_data_file = filename.split("/")[-1].split(".")[0]
        with np.load(filename) as data:
            self.joint_q_seq = data['joint_q']
            self.openness_seq = data['openness']
            self.base_transform_seq = data['base_transform']

