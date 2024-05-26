import threading
import time

import numpy as np
from oculus_reader.reader import OculusReader
from scipy.spatial.transform import Rotation as R


def rmat_to_quat(rot_mat, degrees=False):
    quat = R.from_matrix(rot_mat).as_quat()
    return quat


def quat_diff(target, source):
    result = R.from_quat(target) * R.from_quat(source).inv()
    return result.as_quat()


class VRController(object):
    def __init__(
        self,
        right_controller: bool = True,
        max_lin_vel: float = 1,
        max_rot_vel: float = 1,
        max_gripper_vel: float = 1,
        spatial_coeff: float = 1,
        pos_action_gain: float = 0.25,
        rot_action_gain: float = 0.5,
        gripper_action_gain: float = 1,
        min_magnitude: float = 0,
        control_hz=10,
        robot_orientation="gripper_in_front",
    ):
        self.oculus_reader = OculusReader()
        self.vr_to_global_mat = np.eye(4)
        self.max_lin_vel = max_lin_vel
        self.max_rot_vel = max_rot_vel
        self.max_gripper_vel = max_gripper_vel
        self.spatial_coeff = spatial_coeff
        self.pos_action_gain = pos_action_gain
        self.rot_action_gain = rot_action_gain
        self.gripper_action_gain = gripper_action_gain
        self.min_magnitude = min_magnitude

        # robot_orientation is defined by where the gripper
        # is with respect to the teleoperator when facing forward
        if robot_orientation == "left":
            self.global_to_env_mat = np.asarray(
                [[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float32
            )
        elif robot_orientation == "right":
            self.global_to_env_mat = np.asarray(
                [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float32
            )
        elif robot_orientation == "front":
            self.global_to_env_mat = np.asarray(
                [[0, -1, 0, 0], [-1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float32
            )

        self.control_hz = control_hz

        self.controller_id = "r" if right_controller else "l"
        self.reset_orientation = True
        self.reset_state()

        self._running = True
        self.thread = threading.Thread(target=self._update_internal_state, args=(), daemon=True)
        self.thread.start()

    def reset_state(self):
        self._state = {
            "poses": {},
            "buttons": {"A": False, "B": False},
            "movement_enabled": False,
            "controller_on": True,
        }
        self.update_sensor = True
        self.reset_origin = True
        self.robot_origin = None
        self.vr_origin = None
        self.vr_state = None

    def _update_internal_state(self, num_wait_sec=5):
        last_read_time = time.time()
        while self._running:
            # Regulate Read Frequency #
            time.sleep(1 / self.control_hz)

            # Read Controller
            time_since_read = time.time() - last_read_time
            poses, buttons = self.oculus_reader.get_transformations_and_buttons()
            self._state["controller_on"] = time_since_read < num_wait_sec
            if poses == {}:
                continue

            # Determine Control Pipeline #
            toggled = self._state["movement_enabled"] != buttons["RG"]
            self.update_sensor = self.update_sensor or buttons["RG"]
            self.reset_orientation = self.reset_orientation or buttons["RJ"]
            self.reset_origin = self.reset_origin or toggled

            # Save Info #
            self._state["poses"] = poses
            self._state["buttons"] = buttons
            self._state["movement_enabled"] = buttons["RG"]
            self._state["controller_on"] = True
            last_read_time = time.time()

            # Update Definition Of "Forward" #
            stop_updating = self._state["buttons"]["RJ"] or self._state["movement_enabled"]
            if self.reset_orientation:
                # print("[research] Resetting joystick orientation")
                rot_mat = np.asarray(self._state["poses"][self.controller_id])
                if stop_updating:
                    self.reset_orientation = False
                try:
                    self.vr_to_global_mat = np.linalg.inv(rot_mat)
                except np.linalg.LinAlgError:
                    pass

    def _process_reading(self):
        rot_mat = np.asarray(self._state["poses"][self.controller_id])
        rot_mat = self.global_to_env_mat @ self.vr_to_global_mat @ rot_mat
        vr_pos = self.spatial_coeff * rot_mat[:3, 3]
        vr_quat = rmat_to_quat(rot_mat[:3, :3])
        vr_gripper = self._state["buttons"]["rightTrig"][0]

        self.vr_state = {"pos": vr_pos, "quat": vr_quat, "gripper": vr_gripper}

    def _calculate_action(self, ee_pos, ee_quat):
        if self.update_sensor:
            self._process_reading()
            self.update_sensor = False

        if self.reset_origin:
            self.robot_origin = {"pos": ee_pos, "quat": ee_quat}
            self.vr_origin = {"pos": self.vr_state["pos"], "quat": self.vr_state["quat"]}
            self.reset_origin = False

        # Position computations
        robot_pos_offset = ee_pos - self.robot_origin["pos"]
        target_pos_offset = self.vr_state["pos"] - self.vr_origin["pos"]
        pos_action = target_pos_offset - robot_pos_offset
        scale_pos_action = pos_action * self.pos_action_gain

        # Rotation computations
        robot_quat_offset = quat_diff(ee_quat, self.robot_origin["quat"])
        target_quat_offset = quat_diff(self.vr_state["quat"], self.vr_origin["quat"])
        quat_action = quat_diff(target_quat_offset, robot_quat_offset)
        euler_action = R.from_quat(quat_action).as_euler("xyz")
        scale_euler_action = euler_action * self.rot_action_gain

        # Gripper action
        gripper_action = np.array([(self.vr_state["gripper"]) * self.gripper_action_gain])

        return np.concatenate((scale_pos_action, scale_euler_action, gripper_action))

    def get_info(self):
        return {
            "user_set_success": self._state["buttons"]["A"],
            "user_set_failure": self._state["buttons"]["B"],
            "movement_enabled": self._state["movement_enabled"],
            "controller_on": self._state["controller_on"],
        }

    def predict(self, obs, wait_for_movement_enabled=True):
        if wait_for_movement_enabled:
            while not self._state["movement_enabled"]:
                time.sleep(1 / self.control_hz)
        elif not self._state["movement_enabled"]:
            return None

        if self._state["poses"] == {}:
            action = np.zeros(7)
            return action
        else:
            action = self._calculate_action(obs["state"]["ee_pos"], obs["state"]["ee_quat"])
            if np.linalg.norm(action[:-1]) < self.min_magnitude:
                action = None
            return action

    def __del__(self):
        self._running = False
        self.thread.join()
