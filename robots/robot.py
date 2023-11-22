import time
from typing import Dict, Optional, Type, Union

import gym
import numpy as np

import robots
from robots.controllers.controller import Controller, DummyController

NEW_GYM_API = False if gym.__version__ < "0.26.1" else True


def precise_wait(t_end: float, slack_time: float = 0.001):
    t_start = time.time()
    t_wait = t_end - t_start
    if t_wait > 0:
        t_sleep = t_wait - slack_time
        if t_sleep > 0:
            time.sleep(t_sleep)
        while time.time() < t_end:
            pass


class RobotEnv(gym.Env):
    """
    A simple Gym Environment for controlling robots.
    """

    def __init__(
        self,
        controller_class: Union[str, Type[Controller]] = DummyController,
        controller_kwargs: Optional[Dict] = None,
        random_init: bool = True,
        control_hz: float = 10.0,
        img_width: int = 128,
        img_height: int = 128,
        depth: bool = False,
        cameras: Optional[Dict] = None,
        channels_first: bool = True,
        horizon: Optional[int] = 500,
        normalize_actions: bool = True,
    ):
        self.random_init = random_init
        controller_class = vars(robots)[controller_class] if isinstance(controller_class, str) else controller_class
        self.controller = controller_class(**({} if controller_kwargs is None else controller_kwargs))
        # Add the action space limits.
        self.normalize_actions = normalize_actions
        if self.normalize_actions:
            self.action_space = gym.spaces.Box(
                low=-1, high=1, shape=self.controller.action_space.shape, dtype=np.float32
            )
        else:
            self.action_space = self.controller.action_space

        spaces = dict(state=self.controller.observation_space)

        self.cameras = dict()
        if cameras is not None:
            self.cameras = dict()
            for name, camera_params in cameras.items():
                camera_class, camera_kwargs = camera_params["camera_class"], camera_params["camera_kwargs"]
                self.cameras[name] = vars(robots.cameras)[camera_class](
                    width=img_width, height=img_height, depth=depth, **camera_kwargs
                )
                spaces[name + "_image"] = gym.spaces.Box(
                    low=0, high=255, shape=(img_height, img_width, 3), dtype=np.uint8
                )
                if depth and self.cameras[name].has_depth:
                    spaces[name + "_depth"] = gym.spaces.Box(
                        low=0, high=255, shape=(img_height, img_width, 1), dtype=np.uint8
                    )

        self.observation_space = gym.spaces.Dict(spaces)
        self.horizon = horizon
        self._max_episode_steps = horizon
        self.control_hz = float(control_hz)
        self.channels_first = channels_first
        self._steps = 0

    def _get_obs(self):
        obs = dict(state=self.controller.get_state())
        for name, camera in self.cameras.items():
            frames = camera.get_frames()
            if self.channels_first:
                frames = {k: v.transpose(2, 0, 1) for k, v in frames.items()}
            obs.update({name + "_" + k: v for k, v in frames.items()})
        return obs

    def step(self, action):
        # Immediately update with the action. Note that we scale everything to be between -1 and 1.
        start_time = time.time()
        if self.normalize_actions:
            low, high = self.controller.action_space.low, self.controller.action_space.high
            unscaled_action = low + (0.5 * (action + 1.0) * (high - low))
        else:
            unscaled_action = action

        self.controller.update(unscaled_action)

        end_time = start_time + (1 / self.control_hz)
        precise_wait(end_time)

        self._steps += 1

        terminated = self.horizon is not None and self._steps == self.horizon
        info = dict(discount=1 - float(terminated))

        if NEW_GYM_API:
            # Note that this is following the Gym 0.26 API for termination.
            return self._get_obs(), 0, False, terminated, info
        else:
            return self._get_obs(), 0, terminated, info

    def reset(self):
        self.controller.reset(randomize=self.random_init)
        time.sleep(1.0)
        self._steps = 0
        if NEW_GYM_API:
            return self._get_obs(), dict()
        else:
            return self._get_obs()
