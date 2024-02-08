from abc import abstractmethod, abstractproperty
from typing import Dict, Optional

import gym
import numpy as np


class Controller(object):
    """
    Note: controllers should all follow a LazyInit standard.
    """

    @abstractproperty
    def observation_space(self) -> gym.Space:
        raise NotImplementedError

    @abstractproperty
    def action_spaces(self) -> Dict[str, gym.Space]:
        """
        Returns a dictionary of action spaces for different controller types
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, action: np.ndarray, controller_type: Optional[str] = None) -> None:
        """
        Updates the robot controller with the action

        returns: (Dict[str, np.ndarray], Optional[str])
        Where the dict has all the equivalent actions in controller_types and
        the str is a message reported by the controller
        """
        raise NotImplementedError

    @abstractmethod
    def get_state(self):
        """
        Returns the robot state dictionary.
        For VR support MUST include [ee_pos, ee_quat]
        Also updates the controllers internal state for delta actions.
        """
        raise NotImplementedError

    @abstractmethod
    def get_action(self):
        """
        Returns the action actually executed by the robot since the last call to update,
        represented as a dictionary where the keys are all supported controller types.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self, randomize=False) -> None:
        """
        Reset the robot to HOME, randomize if asked for.
        """
        raise NotImplementedError


class DummyController(Controller):
    """
    A dummy controller class used for testing code.
    """

    @property
    def observation_space(self):
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(14,))  # Ex: 7DoF pos and vel

    @property
    def action_spaces(self, controller_type=None):
        return dict(default=gym.spaces.Box(low=-1, high=1, shape=(7,)))

    def update(self, action: np.ndarray, controller_type: Optional[str] = None) -> None:
        return dict(default=action), None

    def get_state(self):
        return self.observation_space.sample()

    def get_action(self):
        return dict(default=self.action_spaces["default"].sample())

    def reset(self, randomize=False):
        pass
