from typing import Dict, List, Union

import gym
import numpy as np
import zerorpc

from .robot import Controller


def parse_from_lists(item: Union[Dict, List]):
    if isinstance(item, list):
        return np.array(item, dtype=np.float32)
    elif isinstance(item, dict) and "low" in item and "high" in item:
        return gym.spaces.Box(low=parse_from_lists(item["low"]), high=parse_from_lists(item["high"]), dtype=np.float32)
    elif isinstance(item, dict):
        return {k: parse_from_lists(v) for k, v in item.items()}
    else:
        raise ValueError("Invalid item passed to parse_from_lists")


class ZeroRPCController(Controller):
    """
    A simple ZeroRPC Controller that communicates with a controller that runs natively on the NUC.
    This perhaps isn't best practice, but many controllers require it:
    1. PolyMetis (since its python verisons are deprecated)
    2. R2D2 (it natively does something like this to get over the woes of polymetis.)
    """

    def __init__(self, ip_address: str = "127.16.0.1", port: int = 4242):
        self.ip_address = ip_address
        self.server = zerorpc.Client(heartbeat=20)
        self.server.connect("tcp://" + self.ip_address + ":" + str(port))
        self._observation_space = gym.spaces.Dict(parse_from_lists(self.server.get_observation_space()))
        self._action_space = parse_from_lists(self.server.get_action_space())

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def update(self, action):
        """
        Updates the robot controller with the action
        """
        self.server.update(action.tolist())

    def get_state(self):
        """
        Returns the robot state dictionary.
        For VR support MUST include [ee_pos, ee_quat]
        Also updates the controllers internal state for delta actions.
        """
        return parse_from_lists(self.server.get_state())

    def reset(self, randomize=False):
        """
        Reset the robot to HOME, randomize if asked for.
        """
        self.server.reset(randomize=randomize)
