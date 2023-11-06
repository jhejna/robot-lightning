from functools import cached_property

import gym
import numpy as np
import torch
import zerorpc
from scipy.spatial.transform import Rotation

try:
    from polymetis import GripperInterface, RobotInterface

    POLYMETIS_IMPORTED = True
except ImportError:
    print("[research] Skipping polymetis, package not found")
    POLYMETIS_IMPORTED = False

def parse_to_lists(item):
    if isinstance(item, (dict, gym.spaces.Dict)):
        return {k: parse_to_lists(v) for k, v in item.items()}
    elif isinstance(item, np.ndarray):
        return item.tolist()
    elif isinstance(item, gym.spaces.Box):
        return dict(low=item.low.tolist(), high=item.high.tolist())
    else:
        raise ValueError("Invalid item passed to parse_to_list")
    
class PolyMetisController(object):
    # Define the bounds for the Franka Robot
    EE_LOW = np.array([0.1, -0.4, -0.05, -np.pi, -np.pi, -np.pi], dtype=np.float32)
    EE_HIGH = np.array([1.0, 0.4, 1.0, np.pi, np.pi, np.pi], dtype=np.float32)  # np.pi at end
    JOINT_LOW = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159], dtype=np.float32)
    JOINT_HIGH = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159], dtype=np.float32)
    HOME = np.array([0.0, -np.pi / 4.0, 0.0, -3.0 * np.pi / 4.0, 0.0, np.pi / 2.0, np.pi / 4.0], dtype=np.float32)

    def __init__(
        self, ip_address: str = "localhost", controller_type: str = "CARTESIAN_IMPEDANCE", max_delta: float = 0.05
    ):
        self.ip_address = ip_address
        self.controller_type = controller_type
        assert controller_type in {"CARTESIAN_IMPEDANCE", "JOINT_IMPEDANCE", "CARTESIAN_DELTA", "JOINT_DELTA"}
        self.max_delta = max_delta

        self._robot = None
        self._gripper = None

    @cached_property
    def observation_space(self):
        return gym.spaces.Dict(
            {
                "joint_positions": gym.spaces.Box(low=self.JOINT_LOW, high=self.JOINT_HIGH, dtype=np.float32),
                "joint_velocities": gym.spaces.Box(
                    low=-np.inf * self.JOINT_LOW, high=np.inf * self.JOINT_HIGH, dtype=np.float32
                ),
                "ee_pos": gym.spaces.Box(low=self.EE_LOW[:3], high=self.EE_HIGH[:3], dtype=np.float32),
                "ee_quat": gym.spaces.Box(low=np.zeros(4), high=np.ones(4), dtype=np.float32),
                "gripper_pos": gym.spaces.Box(low=np.array([0.0]), high=np.array([1.0]), dtype=np.float32),
            }
        )

    @cached_property
    def action_space(self):
        if self.controller_type == "JOINT_IMPEDANCE":
            low, high = self.JOINT_LOW, self.JOINT_HIGH
        elif self.controller_type == "CARTESIAN_IMPEDANCE":
            low, high = self.EE_LOW, self.EE_HIGH
        elif self.controller_type == "JOINT_DELTA":
            high = self.max_delta * np.ones(self.JOINT_LOW.shape[0], dtype=np.float32)
            low = -high
        elif self.controller_type == "CARTESIAN_DELTA":
            high = self.max_delta * np.ones(6, dtype=np.float32)
            high[3:] = 4 * self.max_delta
            low = -high
        else:
            raise ValueError("Invalid Controller type provided")
        # Add the gripper action space
        low = np.concatenate((low, [0]))
        high = np.concatenate((high, [1]))
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def get_observation_space(self):
        return parse_to_lists(self.observation_space)

    def get_action_space(self):
        return parse_to_lists(self.action_space)

    @property
    def robot(self):
        """
        Lazy load of polymetis interface so that we can initialize the environment without
        actually having a polymetis installation or a polymetis server running
        """
        if self._robot is None:
            assert POLYMETIS_IMPORTED, "Attempted to load robot without polymetis package."
            self._robot = RobotInterface(ip_address=self.ip_address, enforce_version=False)
            self._robot.set_home_pose(torch.Tensor(self.HOME))
        return self._robot

    @property
    def gripper(self):
        if self._gripper is None:
            assert POLYMETIS_IMPORTED, "Attempted to load gripper without polymetis package."
            self._gripper = GripperInterface(ip_address=self.ip_address)
            if hasattr(self.gripper, "metadata") and hasattr(self.gripper.metadata, "max_width"):
                # Should grab this from robotiq2f
                self._max_gripper_width = self._gripper.metadata.max_width
            else:
                self._max_gripper_width = 0.08  # FrankaHand Value

        return self._gripper

    def update_gripper(self, gripper_action, blocking=False):
        # We always run the gripper in absolute position
        # self.gripper.get_state()
        gripper_action = max(min(gripper_action, 1), 0)
        # print(self.gripper.get_state(), gripper_action)
        self.gripper.goto(
            width=self._max_gripper_width * (1 - gripper_action), speed=0.1, force=0.01, blocking=blocking
        )

    def update(self, action):
        """
        Updates the robot controller with the action
        """
        action = np.array(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        robot_action, gripper_action = action[:-1], action[-1]

        if self.controller_type == "JOINT_IMPEDANCE":
            self.robot.update_desired_joint_positions(torch.from_numpy(robot_action))
        elif self.controller_type == "CARTESIAN_IMPEDANCE":
            pos, ori = robot_action[:3], Rotation.from_euler("xyz", robot_action[3:]).as_quat()
            self.robot.update_desired_ee_pose(torch.from_numpy(pos), torch.from_numpy(ori))
        elif self.controller_type == "JOINT_DELTA":
            new_joint_positions = self.state["joint_positions"] + robot_action
            new_joint_positions = np.clip(new_joint_positions, self.JOINT_LOW, self.JOINT_HIGH)
            self.robot.update_desired_joint_positions(torch.from_numpy(new_joint_positions))
        elif self.controller_type == "CARTESIAN_DELTA":
            delta_pos, delta_ori = robot_action[:3], robot_action[3:]
            new_pos = self.state["ee_pos"] + delta_pos
            new_pos = np.clip(new_pos, self.EE_LOW[:3], self.EE_HIGH[:3])
            new_pos = torch.from_numpy(new_pos).float()
            # Compute the new quat
            # TODO: this can be made much faster using purpose build methods instead of scipy.
            old_rot = Rotation.from_quat(self.state["ee_quat"])
            delta_rot = Rotation.from_euler("xyz", delta_ori)
            new_rot = delta_rot * old_rot
            new_quat = torch.from_numpy(new_rot.as_quat()).float()
            self.robot.update_desired_ee_pose(new_pos, new_quat)
        else:
            raise ValueError("Invalid Controller type provided")
        # Update the gripper
        self.update_gripper(gripper_action, blocking=False)

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
            joint_positions=np.array(robot_state.joint_positions, dtype=np.float32),
            joint_velocities=np.array(robot_state.joint_velocities, dtype=np.float32),
            ee_pos=ee_pos.numpy(),
            ee_quat=ee_quat.numpy(),
            gripper_pos=np.asarray([gripper_pos], dtype=np.float32),
        )
        return parse_to_lists(self.state)

    def reset(self, randomize: bool = True):
        if self.robot.is_running_policy():
            self.robot.terminate_current_policy()
        self.update_gripper(0, blocking=False)  # Close the gripper
        self.robot.go_home()
        if randomize:
            # Get the current position and then add some noise to it
            joint_positions = self.robot.get_joint_positions()
            # Update the desired joint positions
            high = 0.15 * np.ones(self.JOINT_LOW.shape[0], dtype=np.float32)
            noise = np.random.uniform(low=-high, high=high)
            randomized_joint_positions = np.array(joint_positions, dtype=np.float32) + noise
            self.robot.move_to_joint_positions(torch.from_numpy(randomized_joint_positions))

        if self.controller_type == "JOINT_IMPEDANCE":
            self.robot.start_joint_impedance()
        elif self.controller_type == "CARTESIAN_IMPEDANCE":
            self.robot.start_cartesian_impedance()
        elif self.controller_type == "JOINT_DELTA":
            self.robot.start_joint_impedance()
        elif self.controller_type == "CARTESIAN_DELTA":
            self.robot.start_cartesian_impedance()
        else:
            raise ValueError("Invalid Controller type provided")


if __name__ == "__main__":
    import argparse
    import subprocess

    parser = argparse.ArgumentParser()
    parser.add_argument("--gripper", type=str, help="Gripper type")
    args = parser.parse_args()

    if False:
        # Start the gripper and controller processes
        sudo_pw = input("What is the sudo password?")

        # Kill any existing servers

        # process = subprocess.Popen(
        #     "echo " + sudo_pw + " | sudo -S " + "bash /launch_robot.sh",
        #     stdout=subprocess.PIPE,
        #     stdin=subprocess.PIPE,
        #     shell=True,
        #     executable="/bin/bash",
        #     encoding="utf8",
        # )

    # Then launch the controller.
    client = PolyMetisController()
    s = zerorpc.Server(client)
    s.bind("tcp://0.0.0.0:4242")
    s.run()
