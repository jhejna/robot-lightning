from functools import cached_property
from typing import Optional

import gym
import time
import numpy as np
from typing import List, Union

from .controller import Controller
from scipy.spatial.transform import Rotation as ScipyRotation

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
        return matrix[..., :2, :].reshape(batch_dim + (6,))


class PolyMetisController(Controller):
    # Define the bounds for the Franka Robot

    CARTESIAN_EULER_LOW = np.array([0.1, -0.4, -0.05, -np.pi, -np.pi, -np.pi], dtype=np.float32)
    CARTESIAN_EULER_HIGH = np.array([1.0, 0.4, 1.0, np.pi, np.pi, np.pi], dtype=np.float32)

    CARTESIAN_ROT6D_LOW = np.array([0.1, -0.4, -0.05, -1, -1, -1, -1, -1, -1], dtype=np.float32)
    CARTESIAN_ROT6D_HIGH = np.array([1.0, 0.4, 1.0, 1, 1, 1, 1, 1, 1], dtype=np.float32)

    JOINT_LOW = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159], dtype=np.float32)
    JOINT_HIGH = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159], dtype=np.float32)

    HOME = np.array([0.0, -np.pi / 4.0, 0.0, -3.0 * np.pi / 4.0, 0.0, np.pi / 2.0, 0], dtype=np.float32)

    def __init__(
        self, ip_address: str = "localhost", controller_type: str = "CARTESIAN_EULER_DELTA", max_delta: float = 0.05
    ):
        self.ip_address = ip_address
        self.controller_type = controller_type
        assert controller_type in {
            "JOINT_IMPEDANCE",
            "JOINT_DELTA",
            "CARTESIAN_EULER_IMPEDANCE",
            "CARTESIAN_EULER_DELTA",
            "CARTESIAN_ROT6D_IMPEDANCE"
        }
        self.max_delta = max_delta

        self._robot = None
        self._gripper = None

    @cached_property
    def observation_space(self):
        return gym.spaces.Dict(
            {
                "joint_pos": gym.spaces.Box(low=self.JOINT_LOW, high=self.JOINT_HIGH, dtype=np.float32),
                "joint_vel": gym.spaces.Box(
                    low=-np.inf * self.JOINT_LOW, high=np.inf * self.JOINT_HIGH, dtype=np.float32
                ),
                "ee_pos": gym.spaces.Box(low=self.CARTESIAN_EULER_LOW[:3], high=self.CARTESIAN_EULER_HIGH[:3], dtype=np.float32),
                "ee_quat": gym.spaces.Box(low=np.zeros(4), high=np.ones(4), dtype=np.float32),
                "gripper_pos": gym.spaces.Box(low=np.array([0.0]), high=np.array([1.0]), dtype=np.float32),
            }
        )

    def get_action_bounds(self, controller_type):
        if controller_type == "JOINT_IMPEDANCE":
            low, high = self.JOINT_LOW, self.JOINT_HIGH
        elif controller_type == "JOINT_DELTA":
            high = self.max_delta * np.ones(self.JOINT_LOW.shape[0], dtype=np.float32)
            low = -high
        elif controller_type == "CARTESIAN_EULER_IMPEDANCE":
            low, high = self.CARTESIAN_EULER_LOW, self.CARTESIAN_EULER_HIGH
        elif controller_type == "CARTESIAN_ROT6D_IMPEDANCE":
            low, high = self.CARTESIAN_ROT6D_LOW, self.CARTESIAN_ROT6D_HIGH
        elif controller_type == "CARTESIAN_EULER_DELTA":
            high = self.max_delta * np.ones(6, dtype=np.float32)
            high[3:] = 4 * self.max_delta
            low = -high
        else:
            raise ValueError("Invalid Controller type provided")
        # Add the gripper action space
        low = np.concatenate((low, [0]))
        high = np.concatenate((high, [1]))

        return low, high

    @cached_property
    def action_space(self):
        low, high = self.get_action_bounds(self.controller_type)
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

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
        gripper_action = max(min(gripper_action, 1), 0)
        self.gripper.goto(
            width=self._max_gripper_width * (1 - gripper_action), speed=0.1, force=0.01, blocking=blocking
        )
        return gripper_action

    def update(self, action: np.ndarray, controller_type: Optional[str] = None):
        """
        Updates the robot controller with the action
        """
        
        if controller_type == None:
            controller_type = self.controller_type
        
        # Clip based on the controller_type specified in the call,
        # otherwise use the default controller_type
        low, high = self.get_action_bounds(controller_type)
        action = np.clip(action, low, high)

        robot_action, gripper_action = action[:-1], action[-1]

        # Regardless of the action space, we will call `update_desired_joint_positions`
        # in Polymetis. So we need to compute the desired joint position in all cases.
        # Along the way, we will compute equivalent actions to produce the same motion
        # for the other controller types.

        if controller_type in ["JOINT_IMPEDANCE", "JOINT_DELTA"]:

            # Compute joint_pos_desired and joint_delta_desired
            if controller_type == "JOINT_IMPEDANCE":
                joint_pos_desired = robot_action
                joint_delta_desired = joint_pos_desired - self.state['joint_pos']
            elif controller_type == "JOINT_DELTA":
                joint_delta_desired = robot_action
                joint_pos_desired = self.state["joint_pos"] + joint_delta_desired
                joint_pos_desired = np.clip(joint_pos_desired, self.JOINT_LOW, self.JOINT_HIGH)
            
            # Compute ee_pos_desired, ee_rot_desired, ee_pos_delta_desired, and ee_rot_delta_desired
            ee_pos_desired, ee_quat_desired = self.robot.robot_model.forward_kinematics(torch.from_numpy(joint_pos_desired)).numpy()
            ee_pos_delta_desired = ee_pos_desired - self.state['ee_pos']
            ee_rot_desired = Rotation.from_quat(ee_quat_desired)
            ee_rot_delta_desired = ee_rot_desired / Rotation.from_quat(self.state['ee_quat']) # TODO: Check I am dividing in the right order.

            success = True

        elif controller_type in ["CARTESIAN_EULER_IMPEDANCE", "CARTESIAN_ROT6D_IMPEDANCE", "CARTESIAN_EULER_DELTA"]:

            # Compute ee_pos_desired, ee_rot_desired, ee_pos_delta_desired, and ee_rot_delta_desired
            if controller_type == "CARTESIAN_EULER_IMPEDANCE":
                ee_pos_desired, ee_rot_desired = robot_action[:3], Rotation.from_euler('xyz', robot_action[3:])
                ee_pos_delta_desired = ee_pos_desired - self.state['ee_pos']
                ee_rot_delta_desired = ee_rot_desired / Rotation.from_quat(self.state['ee_quat'])
            elif controller_type == "CARTESIAN_ROT6D_IMPEDANCE":
                ee_pos_desired, ee_rot_desired = robot_action[:3], Rotation.from_rot6d(robot_action[3:])
                ee_pos_delta_desired = ee_pos_desired - self.state['ee_pos']
                ee_rot_delta_desired = ee_rot_desired / Rotation.from_quat(self.state['ee_quat'])
            elif controller_type == "CARTESIAN_EULER_DELTA":
                ee_pos_delta_desired, ee_rot_delta_desired = robot_action[:3], Rotation.from_euler('xyz', robot_action[3:])
                ee_pos_desired = self.state["ee_pos"] + ee_pos_delta_desired
                ee_pos_desired = np.clip(ee_pos_desired, self.CARTESIAN_EULER_LOW[:3], self.CARTESIAN_EULER_HIGH[:3])
                ee_rot_desired = Rotation.from_quat(self.state["ee_quat"]) * ee_rot_delta_desired
            
            # Compute joint_pos_desired and joint_delta_desired
            joint_pos_desired, success = self.robot.solve_inverse_kinematics(
                torch.from_numpy(ee_pos_desired), torch.from_numpy(ee_rot_desired.as_quat()), self.robot.get_joint_positions()
            )
            joint_delta_desired = joint_pos_desired - self.state['joint_pos']
            
            success = success.item()

        # Update the gripper. Note that the gripper action is always an absolute desired position.
        gripper_pos_desired = self.update_gripper(gripper_action, blocking=False).reshape([1])

        # We return the equivalent actions that could have been taken with different
        # controller_types.
        equivalent_actions = {
            "JOINT_IMPEDANCE": np.concatenate([joint_pos_desired, gripper_pos_desired]),
            "JOINT_DELTA": np.concatenate([joint_delta_desired, gripper_pos_desired]),
            "CARTESIAN_EULER_IMPEDANCE": np.concatenate([ee_pos_desired, ee_rot_desired.as_euler('xyz'), gripper_pos_desired]),
            "CARTESIAN_ROT6D_IMPEDANCE": np.concatenate([ee_pos_desired, ee_rot_desired.as_rot6d(), gripper_pos_desired]),
            "CARTESIAN_EULER_DELTA": np.concatenate([ee_pos_delta_desired, ee_rot_delta_desired.as_euler('xyz'), gripper_pos_desired]),
            "success": success,
        }

        # Update the robot
        if success:
            self.robot.update_desired_joint_positions(joint_pos_desired)

        return equivalent_actions

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
        self.state = dict(
            joint_pos=np.array(robot_state.joint_positions, dtype=np.float32),
            joint_vel=np.array(robot_state.joint_velocities, dtype=np.float32),
            ee_pos=ee_pos.numpy(),
            ee_quat=ee_quat.numpy(),
            gripper_pos=np.asarray([gripper_pos], dtype=np.float32),
        )
        return self.state

    def reset(self, randomize: bool = True):
        if self.robot.is_running_policy():
            self.robot.terminate_current_policy()
        self.update_gripper(0, blocking=True)  # Close the gripper
        print("resetting")
        self.robot.go_home()
        time.sleep(3.0)

        if randomize:
            # Get the current position and then add some noise to it
            joint_positions = self.robot.get_joint_positions()
            # Update the desired joint positions
            high = 0.15 * np.ones(self.JOINT_LOW.shape[0], dtype=np.float32)
            noise = np.random.uniform(low=-high, high=high)
            randomized_joint_positions = np.array(joint_positions, dtype=np.float32) + noise
            self.robot.move_to_joint_positions(torch.from_numpy(randomized_joint_positions))

        self.robot.start_joint_impedance()

    def evaluate_command(self, fn_name, *args, **kwargs):
        return getattr(self.robot, fn_name)(*args, **kwargs)
