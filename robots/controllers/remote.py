from typing import Dict, List, Optional, Union

import gym
import numpy as np
import zerorpc

import robots

from .controller import Controller, DummyController


def parse_from_lists(item: Union[Dict, List]):
    if item is None:
        return None
    elif isinstance(item, list) and len(item) > 0 and isinstance(item[0], float):
        return np.array(item, dtype=np.float32)
    elif isinstance(item, list):
        return [parse_from_lists(v) for v in item]
    elif isinstance(item, tuple):
        return tuple(parse_from_lists(v) for v in item)
    elif isinstance(item, dict) and "low" in item and "high" in item:
        return gym.spaces.Box(low=parse_from_lists(item["low"]), high=parse_from_lists(item["high"]), dtype=np.float32)
    elif isinstance(item, dict):
        return {k: parse_from_lists(v) for k, v in item.items()}
    elif isinstance(item, (bool, str, int, float)):
        return item
    else:
        print(f"Invalid item of type {type(item)} passed to parse_from_lists")
        return None


def parse_to_lists(item):
    if item is None:
        return None
    elif isinstance(item, list):
        return [parse_to_lists(v) for v in item]
    elif isinstance(item, tuple):
        return tuple(parse_to_lists(v) for v in item)
    elif isinstance(item, np.ndarray):
        return item.tolist()
    elif isinstance(item, (dict, gym.spaces.Dict)):
        return {k: parse_to_lists(v) for k, v in item.items()}
    elif isinstance(item, gym.spaces.Box):
        return dict(low=item.low.tolist(), high=item.high.tolist())
    elif isinstance(item, (bool, str, int, float)):
        return item
    else:
        print(f"Invalid item of type {type(item)} passed to parse_to_lists")
        return None


class ZeroRPCClient(Controller):
    """
    A simple ZeroRPC Controller that communicates with a controller that runs natively on the NUC.
    This perhaps isn't best practice, but many controllers require it:
    1. PolyMetis (since its python verisons are deprecated)
    2. R2D2 (it natively does something like this to get over the woes of polymetis.)
    """

    def __init__(
        self,
        controller_class: DummyController,
        controller_kwargs: Optional[Dict] = None,
        ip_address: str = "127.16.0.1",
        port: int = 4242,
    ):
        self.ip_address = ip_address
        self.port = port
        self._client = None
        controller_class = vars(robots)[controller_class] if isinstance(controller_class, str) else controller_class
        controller_kwargs = dict() if controller_kwargs is None else controller_kwargs
        self.controller = controller_class(**controller_kwargs)

    @property
    def client(self):
        if self._client is None:
            self._client = zerorpc.Client(heartbeat=20)
            self._client.connect("tcp://" + self.ip_address + ":" + str(self.port))
        return self._client

    @property
    def observation_space(self):
        return self.controller.observation_space

    @property
    def action_spaces(self):
        """
        Returns the gym action space corresponding to controller_type
        """
        return self.controller.action_spaces

    def update(self, action: np.ndarray, controller_type: Optional[str] = None, ignore_workspace_clipping: bool = False) -> Dict:
        """
        Updates the robot controller with the action
        """
        return parse_from_lists(self.client.update(parse_to_lists(action), controller_type, ignore_workspace_clipping))

    def get_state(self):
        """
        Returns the robot state dictionary.
        For VR support MUST include [ee_pos, ee_quat]
        Also updates the controllers internal state for delta actions.
        """
        return parse_from_lists(self.client.get_state())

    def get_action(self):
        """
        Returns the action actually executed by the robot since the last call to update,
        represented as a dictionary where the keys are all supported controller types.
        """
        return parse_from_lists(self.client.get_action())

    def reset(self, randomize=False):
        """
        Reset the robot to HOME, randomize if asked for.
        """
        self.client.reset(randomize=randomize)

    def eval(self, fn_name, *args, **kwargs):
        args = tuple(parse_to_lists(arg) for arg in args)
        kwargs = {k: parse_to_lists(v) for k, v in kwargs.items()}
        return parse_from_lists(self.client.eval(fn_name, kwargs, *args))


class ZeroRPCServer(Controller):
    def __init__(self, controller_class, controller_kwargs: Optional[Dict] = None, randomize: bool = False):
        controller_class = vars(robots)[controller_class] if isinstance(controller_class, str) else controller_class
        controller_kwargs = dict() if controller_kwargs is None else controller_kwargs
        self.controller = controller_class(**controller_kwargs)
        self.randomize = randomize

    @property
    def observation_space(self):
        return self.controller.observation_space

    @property
    def action_spaces(self):
        """
        Returns the gym action space corresponding to controller_type
        """
        return self.controller.action_spaces

    def update(self, action: List[float], controller_type: Optional[str] = None, ignore_workspace_clipping: bool = False) -> Dict:
        """
        Updates the robot controller with the action
        """
        action = np.array(action, dtype=np.float32)
        return parse_to_lists(self.controller.update(action, controller_type=controller_type, ignore_workspace_clipping=ignore_workspace_clipping))

    def get_state(self):
        """
        Returns the robot state dictionary.
        For VR support MUST include [ee_pos, ee_quat]
        Also updates the controllers internal state for delta actions.
        """
        return parse_to_lists(self.controller.get_state())

    def get_action(self):
        """
        Returns the action actually executed by the robot since the last call to update,
        represented as a dictionary where the keys are all supported controller types.
        """
        return parse_to_lists(self.controller.get_action())

    def reset(self, randomize: Optional[bool] = None):
        """
        Reset the robot to HOME, randomize if asked for.
        """
        if randomize is None:
            randomize = self.randomize
        self.controller.reset(randomize=randomize)

    def eval(self, fn_name, kwargs, *args):
        args = parse_from_lists(args)
        kwargs = parse_from_lists(kwargs)
        return parse_to_lists(self.controller.eval(fn_name, *args, **kwargs))
