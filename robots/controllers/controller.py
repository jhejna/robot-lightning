from abc import abstractmethod, abstractproperty
import numpy as np
import gym

class Controller(object):
    """
    Note: controllers should all follow a LazyInit standard.
    """
    @abstractproperty
    def observation_space(self) -> gym.Space:
        raise NotImplementedError

    @abstractproperty
    def action_space(self) -> gym.Space:
        raise NotImplementedError

    @abstractmethod
    def update(self, action: np.ndarray) -> None:
        """
        Updates the robot controller with the action
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
    def reset(self, randomize=False) -> None:
        """
        Reset the robot to HOME, randomize if asked for.
        """
        raise NotImplementedError


class DummyController(Controller):
    """
    A dummy controller class used for testing code.
    """

    @abstractproperty
    def observation_space(self):
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(14,))  # Ex: 7DoF pos and vel

    @property
    def action_space(self):
        return gym.spaces.Box(low=-1, high=1, shape=(7,))

    def update(self, action: np.ndarray):
        pass

    def get_state(self):
        return self.observation_space.sample()

    def reset(self, randomize=False):
        pass