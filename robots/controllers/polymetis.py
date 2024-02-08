import time
from copy import copy
from functools import cached_property
from typing import List, Union

import gym
import numpy as np
from scipy.spatial.transform import Rotation as ScipyRotation

from .controller import Controller

try:
    import polymetis
    import torch

    POLYMETIS_IMPORTED = True
except ImportError:
    print("[research] Skipping polymetis and torch. One of the packages was not found")
    POLYMETIS_IMPORTED = False


class Rotation(ScipyRotation):
    """Extend scipy's Rotation class to support rot6d."""

    @classmethod
    def from_rot6d(cls, rot6d: Union[np.ndarray, List]):
        if isinstance(rot6d, list):
            rot6d = np.array(rot6d)
        a1, a2 = rot6d[..., :3], rot6d[..., 3:]
        # Gram-Schmidt
        b1 = a1 / np.linalg.norm(a1, axis=-1)
        b2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
        b2 = b2 / np.linalg.norm(b2, axis=-1)
        b3 = np.cross(b1, b2, axis=-1)
        return super().from_matrix(np.stack((b1, b2, b3), axis=-2))

    def as_rot6d(self):
        matrix = self.as_matrix()
        batch_dim = matrix.shape[:-2]
        return matrix[..., :2, :].reshape((*batch_dim, 6))


class PolyMetisController(Controller):
    # Define the joint bounds for the Franka Robot
    JOINT_LOW = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159], dtype=np.float32)
    JOINT_HIGH = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159], dtype=np.float32)

    JOINT_DELTA_LOW = 1 / 4.1 * np.array([-2, -1, -1.5, -1.25, -3, -1.5, -3])
    JOINT_DELTA_HIGH = 1 / 4.1 * np.array([2, 1, 1.5, 1.25, 3, 1.5, 3])

    HOME = np.array([0.0, -np.pi / 4.0, 0.0, -3.0 * np.pi / 4.0, 0.0, np.pi / 2.0, 0], dtype=np.float32)

    def __init__(
        self,
        ip_address: str = "localhost",
        max_cartesian_delta: float = 0.05,
        max_orientation_delta: float = 0.2,
        workspace: List[List[float]] = [[0.1, -0.4, -0.05], [1.0, 0.4, 1.0]],
    ):
        self.ip_address = ip_address
        self.workspace = workspace
        # These are coarse coordinate-wise bounds on xyz delta and
        # orientation delta and are not recommended to be changed.
        self.max_cartesian_delta = max_cartesian_delta
        self.max_orientation_delta = max_orientation_delta

        self._robot = None
        self._gripper = None

        self._last_joint_pos_desired = None
        self._last_gripper_pos_desired = None
        self._last_state = None
        self.state = None

    @cached_property
    def action_spaces(self):
        def _make_action_space_with_gripper(low, high):
            low, high = np.concatenate((low, [0])), np.concatenate((high, [1]))
            return gym.spaces.Box(low=low, high=high, dtype=np.float32)

        action_spaces = dict()
        action_spaces["JOINT_IMPEDANCE"] = _make_action_space_with_gripper(self.JOINT_LOW, self.JOINT_HIGH)
        action_spaces["JOINT_DELTA"] = _make_action_space_with_gripper(self.JOINT_DELTA_LOW, self.JOINT_DELTA_HIGH)

        # Cartesian Euler Impedance
        low = np.concatenate((np.array(self.workspace[0]), np.array([-np.pi, -np.pi, -np.pi])), dtype=np.float32)
        high = np.concatenate((np.array(self.workspace[1]), np.array([np.pi, np.pi, np.pi])), dtype=np.float32)
        action_spaces["CARTESIAN_EULER_IMPEDANCE"] = _make_action_space_with_gripper(low, high)

        # Cartesian Rot6D Impedance
        low = np.concatenate((np.array(self.workspace[0]), np.array([-1, -1, -1, -1, -1, -1])), dtype=np.float32)
        high = np.concatenate((np.array(self.workspace[1]), np.array([1, 1, 1, 1, 1, 1])), dtype=np.float32)
        action_spaces["CARTESIAN_ROT6D_IMPEDANCE"] = _make_action_space_with_gripper(low, high)

        # Cartesian Euler Delta
        low = np.array(3 * [-self.max_cartesian_delta] + 3 * [-self.max_orientation_delta], dtype=np.float32)
        high = np.array(3 * [self.max_cartesian_delta] + 3 * [self.max_orientation_delta], dtype=np.float32)
        action_spaces["CARTESIAN_EULER_DELTA"] = _make_action_space_with_gripper(low, high)

        return action_spaces

    @cached_property
    def observation_space(self):
        return gym.spaces.Dict(
            {
                "joint_pos": gym.spaces.Box(low=self.JOINT_LOW, high=self.JOINT_HIGH, dtype=np.float32),
                "joint_vel": gym.spaces.Box(
                    low=-np.inf * self.JOINT_LOW, high=np.inf * self.JOINT_HIGH, dtype=np.float32
                ),
                "ee_pos": gym.spaces.Box(
                    low=self.CARTESIAN_EULER_LOW[:3], high=self.CARTESIAN_EULER_HIGH[:3], dtype=np.float32
                ),
                "ee_quat": gym.spaces.Box(low=-np.ones(4), high=np.ones(4), dtype=np.float32),
                "gripper_pos": gym.spaces.Box(low=np.array([0.0]), high=np.array([1.0]), dtype=np.float32),
            }
        )

    @property
    def robot(self):
        """
        Lazy load of polymetis interface so that we can initialize the environment without
        actually having a polymetis installation or a polymetis server running
        """
        if self._robot is None:
            assert POLYMETIS_IMPORTED, "Attempted to load robot without polymetis package."
            self._robot = polymetis.RobotInterface(ip_address=self.ip_address, enforce_version=False)
            self._robot.set_home_pose(torch.Tensor(self.HOME))
        return self._robot

    @property
    def gripper(self):
        if self._gripper is None:
            assert POLYMETIS_IMPORTED, "Attempted to load gripper without polymetis package."
            self._gripper = polymetis.GripperInterface(ip_address=self.ip_address)
            if hasattr(self.gripper, "metadata") and hasattr(self.gripper.metadata, "max_width"):
                # Should grab this from robotiq2f
                self._max_gripper_width = self._gripper.metadata.max_width
            else:
                self._max_gripper_width = 0.08  # FrankaHand Value

        return self._gripper

    def update_gripper(self, gripper_action, blocking=False):
        # We always run the gripper in absolute position
        self.gripper.goto(
            width=self._max_gripper_width * (1 - gripper_action), speed=0.1, force=0.01, blocking=blocking
        )
        return gripper_action

    def update(self, action: np.ndarray, controller_type: str):
        """
        Updates the robot controller with the action
        """

        # Load values that will be used globally
        messages = []
        euler_delta_action_space = self.action_spaces["CARTESIAN_EULER_DELTA"]

        # Special case: If we are in Cartesian-Euler-Delta mode,
        # clip the action coordinate-wise
        if controller_type == "CARTESIAN_EULER_DELTA":
            clipped_action = np.clip(action, euler_delta_action_space.low, euler_delta_action_space.high)
            if not np.allclose(action, clipped_action):
                messages.append("cartesian_euler_delta_clipped")
            action = clipped_action

        robot_action, gripper_action = action[:-1], action[-1]

        # Regardless of the action space, we will call `update_desired_joint_positions`
        # in Polymetis. So we need to compute the desired joint position in all cases.
        # Along the way, we will compute equivalent actions to produce the same motion
        # for the other controller types.

        if controller_type in ("JOINT_IMPEDANCE", "JOINT_DELTA"):
            # Compute joint_pos_desired and joint_delta_desired
            if controller_type == "JOINT_IMPEDANCE":
                joint_pos_desired = robot_action
                joint_delta_desired = joint_pos_desired - self.state["joint_pos"]
            elif controller_type == "JOINT_DELTA":
                joint_delta_desired = robot_action
                joint_pos_desired = self.state["joint_pos"] + joint_delta_desired

            joint_pos_desired = torch.from_numpy(joint_pos_desired)

            # Compute ee_pos_desired, ee_rot_desired, ee_pos_delta_desired, and ee_rot_delta_desired
            ee_pos_desired, ee_quat_desired = self.robot.robot_model.forward_kinematics(joint_pos_desired)
            ee_pos_desired, ee_quat_desired = ee_pos_desired.numpy(), ee_quat_desired.numpy()
            ee_pos_delta_desired = ee_pos_desired - self.state["ee_pos"]
            ee_rot_desired = Rotation.from_quat(ee_quat_desired)
            ee_rot_delta_desired = Rotation.from_quat(self.state["ee_quat"]).inv() * ee_rot_desired

            ik_success = True

        elif controller_type in ("CARTESIAN_EULER_IMPEDANCE", "CARTESIAN_ROT6D_IMPEDANCE", "CARTESIAN_EULER_DELTA"):
            # Compute ee_pos_desired, ee_rot_desired, ee_pos_delta_desired, and ee_rot_delta_desired
            if controller_type == "CARTESIAN_EULER_IMPEDANCE":
                ee_pos_desired, ee_rot_desired = robot_action[:3], Rotation.from_euler("xyz", robot_action[3:])
                ee_pos_delta_desired = ee_pos_desired - self.state["ee_pos"]
                ee_rot_delta_desired = Rotation.from_quat(self.state["ee_quat"]).inv() * ee_rot_desired
            elif controller_type == "CARTESIAN_ROT6D_IMPEDANCE":
                ee_pos_desired, ee_rot_desired = robot_action[:3], Rotation.from_rot6d(robot_action[3:])
                ee_pos_delta_desired = ee_pos_desired - self.state["ee_pos"]
                ee_rot_delta_desired = Rotation.from_quat(self.state["ee_quat"]).inv() * ee_rot_desired
            elif controller_type == "CARTESIAN_EULER_DELTA":
                ee_pos_delta_desired, ee_rot_delta_desired = robot_action[:3], Rotation.from_euler(
                    "xyz", robot_action[3:]
                )
                ee_pos_desired = self.state["ee_pos"] + ee_pos_delta_desired
                ee_rot_desired = Rotation.from_quat(self.state["ee_quat"]) * ee_rot_delta_desired

            # Compute joint_pos_desired and joint_delta_desired
            joint_pos_desired, ik_success = self.robot.solve_inverse_kinematics(
                torch.from_numpy(ee_pos_desired).double(),
                torch.from_numpy(ee_rot_desired.as_quat()).double(),
                self.robot.get_joint_positions(),
            )
            joint_delta_desired = joint_pos_desired.numpy() - self.state["joint_pos"]

            ik_success = ik_success.item()

        else:
            raise NotImplementedError("A controller type has been used that is not implemented.")

        # Gripper action clipping
        gripper_pos_desired = np.clip(gripper_action, 0, 1).reshape(
            [
                1,
            ]
        )
        if not gripper_pos_desired.item() == gripper_action:
            messages.append("gripper_action_clipped")

        # Handle various possible failures of the polymetis controller:

        # Workspace clipping
        ee_pos_euler_desired = np.concatenate([ee_pos_desired, ee_rot_desired.as_euler("xyz")])
        clipped_ee_pos_euler_desired = np.clip(
            ee_pos_euler_desired, euler_delta_action_space.low, euler_delta_action_space.high
        )
        if not np.allclose(ee_pos_euler_desired, clipped_ee_pos_euler_desired):
            messages.append("workspace_constraints_violated")
            desired_actions, new_message = self.update(
                np.concatenate([clipped_ee_pos_euler_desired, gripper_pos_desired]), "CARTESIAN_EULER_IMPEDANCE"
            )
            if new_message is not None:
                messages.append(new_message)
            return desired_actions, " ".join(messages) if len(messages) > 0 else None

        # IK failure
        if not ik_success:
            messages.append("ik_failed")
            desired_actions, new_message = self.update(self._last_joint_pos_desired, controller_type="JOINT_IMPEDANCE")
            if new_message is not None:
                messages.append(new_message)
            return desired_actions, " ".join(messages) if len(messages) > 0 else None

        # Joints clipping
        clipped_joint_pos_desired = np.clip(joint_pos_desired, self.JOINT_LOW, self.JOINT_HIGH)
        if not np.allclose(joint_pos_desired, clipped_joint_pos_desired):
            messages.append("joint_limits_violated")
            desired_actions, new_message = self.update(
                np.concatenate([clipped_joint_pos_desired, gripper_pos_desired]), controller_type="JOINT_IMPEDANCE"
            )
            if new_message is not None:
                messages.append(new_message)
            return desired_actions, " ".join(messages) if len(messages) > 0 else None

        # Joint delta clipping
        clipped_joint_delta_desired = np.clip(joint_delta_desired, self.JOINT_DELTA_LOW, self.JOINT_DELTA_HIGH)
        if not np.allclose(joint_delta_desired, clipped_joint_delta_desired):
            messages.append("joint_delta_limits_violated")
            desired_actions, new_message = self.update(
                np.concatenate([clipped_joint_delta_desired, gripper_pos_desired]), controller_type="JOINT_DELTA"
            )
            if new_message is not None:
                messages.append(new_message)
            return desired_actions, " ".join(messages) if len(messages) > 0 else None

        # We return the equivalent actions that could have been taken with different
        # controller_types.
        desired_actions = {
            "JOINT_IMPEDANCE": np.concatenate([joint_pos_desired, gripper_pos_desired]),
            "JOINT_DELTA": np.concatenate([joint_delta_desired, gripper_pos_desired]),
            "CARTESIAN_EULER_IMPEDANCE": np.concatenate(
                [ee_pos_desired, ee_rot_desired.as_euler("xyz"), gripper_pos_desired]
            ),
            "CARTESIAN_ROT6D_IMPEDANCE": np.concatenate(
                [ee_pos_desired, ee_rot_desired.as_rot6d(), gripper_pos_desired]
            ),
            "CARTESIAN_EULER_DELTA": np.concatenate(
                [ee_pos_delta_desired, ee_rot_delta_desired.as_euler("xyz"), gripper_pos_desired]
            ),
        }

        # Update the gripper. Note that the gripper action is always represented as an absolute desired position.
        self._last_gripper_pos_desired = gripper_pos_desired
        self.update_gripper(gripper_pos_desired.item(), blocking=False)

        # Update the robot
        self._last_joint_pos_desired = joint_pos_desired
        self.robot.update_desired_joint_positions(joint_pos_desired)

        return desired_actions, " ".join(messages) if len(messages) > 0 else None

    def get_action(self):
        """
        Returns a dictionary of *achieved* actions for each controller type
        between the current state and the last state.
        Note: We always use *desired* absolute position for the gripper only.
        """
        gripper_pos_desired = self._last_gripper_pos_desired
        joint_pos_achieved = self.state["joint_pos"]
        joint_delta_achieved = joint_pos_achieved - self._last_state["joint_pos"]
        ee_pos_achieved = self.state["ee_pos"]
        ee_pos_delta_achieved = ee_pos_achieved - self._last_state["ee_pos"]
        ee_rot_achieved = Rotation.from_quat(self.state["ee_quat"])
        ee_rot_delta_achieved = Rotation.from_quat(self._last_state["ee_quat"]).inv() * ee_rot_achieved
        ee_euler_achieved = ee_rot_achieved.as_euler("xyz")
        ee_euler_delta_achieved = ee_rot_delta_achieved.as_euler("xyz")
        ee_rot6d_achieved = ee_rot_achieved.as_rot6d()

        return {
            "JOINT_IMPEDANCE": np.concatenate([joint_pos_achieved, gripper_pos_desired]),
            "JOINT_DELTA": np.concatenate([joint_delta_achieved, gripper_pos_desired]),
            "CARTESIAN_EULER_IMPEDANCE": np.concatenate([ee_pos_achieved, ee_euler_achieved, gripper_pos_desired]),
            "CARTESIAN_ROT6D_IMPEDANCE": np.concatenate([ee_pos_achieved, ee_rot6d_achieved, gripper_pos_desired]),
            "CARTESIAN_EULER_DELTA": np.concatenate(
                [ee_pos_delta_achieved, ee_euler_delta_achieved, gripper_pos_desired]
            ),
        }

    def get_state(self):
        """
        Returns the robot state dictionary.
        For VR support MUST include [ee_pos, ee_quat]
        """
        robot_state = self.robot.get_robot_state()
        ee_pos, ee_quat = self.robot.robot_model.forward_kinematics(torch.Tensor(robot_state.joint_positions))
        gripper_state = self.gripper.get_state()
        gripper_pos = 1 - (gripper_state.width / self._max_gripper_width)
        self._updated = False
        self._last_state = copy(self.state)
        self.state = dict(
            joint_pos=np.array(robot_state.joint_positions, dtype=np.float32),
            joint_vel=np.array(robot_state.joint_velocities, dtype=np.float32),
            ee_pos=ee_pos.numpy(),
            ee_quat=ee_quat.numpy(),
            gripper_pos=np.asarray([gripper_pos], dtype=np.float32),
        )
        return self.state

    def calculate_fingertip_pos(self, ee_pos, ee_quat):
        home_fingertip_offset = np.array([0, 0, -0.17])
        ee_euler = Rotation.from_quat(ee_quat).as_euler("xyz") - np.array([-np.pi, 0, 0])
        fingertip_offset = Rotation.from_euler("xyz", ee_euler).as_matrix() @ home_fingertip_offset
        fingertip_pos = ee_pos + fingertip_offset
        return fingertip_pos

    @property
    def fingertip_pos(self):
        return self.calculate_fingertip_pos(self.state["ee_pos"], self.state["ee_quat"])

    def reset(self, randomize: bool = True):
        if self.robot.is_running_policy():
            self.robot.terminate_current_policy()
        self.update_gripper(0, blocking=True)  # Close the gripper
        print("resetting")
        self.robot.go_home()
        time.sleep(1.0)

        if randomize:
            # Get the current position and then add some noise to it
            joint_positions = self.robot.get_joint_positions()
            # Update the desired joint positions
            high = 0.15 * np.ones(self.JOINT_LOW.shape[0], dtype=np.float32)
            noise = np.random.uniform(low=-high, high=high)
            randomized_joint_positions = np.array(joint_positions, dtype=np.float32) + noise
            self.robot.move_to_joint_positions(torch.from_numpy(randomized_joint_positions))

        self.robot.start_joint_impedance()

    def eval(self, fn_name, *args, **kwargs):
        if hasattr(self, fn_name):
            fn = getattr(self, fn_name)
            if callable(fn):
                return fn(*args, **kwargs)
            else:
                return fn
        elif fn_name.startswith("robot."):
            return getattr(self.robot, fn_name.replace("robot.", ""))(*args, **kwargs)
