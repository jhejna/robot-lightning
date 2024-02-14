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
        controller_type: str = "CARTESIAN_EULER_DELTA",
        random_init: bool = True,
        control_hz: float = 10.0,
        img_width: int = 128,
        img_height: int = 128,
        depth: bool = False,
        cameras: Optional[Dict] = None,
        channels_first: bool = True,
        horizon: Optional[int] = 500,
    ):
        controller_class = vars(robots)[controller_class] if isinstance(controller_class, str) else controller_class
        self.controller = controller_class(**({} if controller_kwargs is None else controller_kwargs))
        assert controller_type in self.controller.action_spaces, "controller_type not supported by the controller."
        self.default_controller_type = controller_type

        # Add the action space limits.
        self.action_space = self.controller.action_spaces[self.default_controller_type]
        # Construct the observation space
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
                        low=0, high=2**16 - 1, shape=(img_height, img_width, 1), dtype=np.uint16
                    )

        self.observation_space = gym.spaces.Dict(spaces)
        self.random_init = random_init
        self.horizon = horizon
        self._max_episode_steps = horizon  # Added so it looks like we have a gym time limit wrapper.
        self.control_hz = float(control_hz)
        self.channels_first = channels_first
        self._steps = 0

    def _get_frames(self):
        obs_frames = dict()
        for name, camera in self.cameras.items():
            frames = camera.get_frames()
            if self.channels_first:
                frames = {k: v.transpose(2, 0, 1) for k, v in frames.items()}
            obs_frames.update({name + "_" + k: v for k, v in frames.items()})
        return obs_frames

    def _get_obs(self):
        obs = dict(state=self.controller.get_state())
        obs.update(self._get_frames())
        return obs

    def step(self, action, controller_type: Optional[str] = None):
        if self._time is None:
            self._time = time.time()

        controller_type = self.default_controller_type if controller_type is None else controller_type
        desired_action, action_message = self.controller.update(action, controller_type)

        # Make sure get_obs gets called at control_hz
        # (This is true assuming get_obs() takes a constant amount of time)
        end_time = self._time + (1 / self.control_hz)
        precise_wait(end_time)
        self._time = time.time()
        obs = self._get_obs()
        achieved_action = self.controller.get_action()

        self._steps += 1
        terminated = self.horizon is not None and self._steps == self.horizon
        info = dict(discount=1 - float(terminated), desired_action=desired_action, achieved_action=achieved_action, action_message=action_message)

        if info["action_message"] != "":
            print(f"[robots] {info['action_message']}")

        if NEW_GYM_API:
            # Note that this is following the Gym 0.26 API for termination.
            return obs, 0, False, terminated, info
        else:
            return obs, 0, terminated, info

    def reset(self, reset_controller=True):
        if reset_controller:
            self.controller.reset(randomize=self.random_init)
            time.sleep(1.0)

        self._steps = 0
        self._time = None
        obs = self._get_obs()
        if NEW_GYM_API:
            return obs, dict()
        else:
            return obs
