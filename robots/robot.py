import time
from abc import abstractmethod, abstractproperty
from typing import Dict, List, Optional, Type, Union

import cv2
import gym
import numpy as np
import robots

NEW_GYM_API = False if gym.__version__ < "0.26.1" else True


class Controller(object):
    @abstractproperty
    def observation_space(self):
        raise NotImplementedError

    @abstractproperty
    def action_space(self):
        raise NotImplementedError

    @abstractmethod
    def update(self, action):
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
    def reset(self, randomize=False):
        """
        Reset the robot to HOME, randomize if asked for.
        """
        raise NotImplementedError


def precise_wait(t_end: float, slack_time: float = 0.001):
    t_start = time.time()
    t_wait = t_end - t_start
    if t_wait > 0:
        t_sleep = t_wait - slack_time
        if t_sleep > 0:
            time.sleep(t_sleep)
        while time.time() < t_end:
            pass


class Camera(object):
    @abstractmethod
    def get_frame(self) -> np.ndarray:
        pass

    @abstractmethod
    def close(self):
        pass


class OpenCVCamera(Camera):
    def __init__(self, cap, width: int = 64, height: int = 64):
        self._width = width
        self._height = height
        self._cap = cap

    def get_frame(self) -> np.ndarray:
        retval, img = self._cap.read()
        assert retval
        img = cv2.resize(img, (self._width, self._height), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img


class RealSenseCamera(Camera):
    def __init__(self, device, width: int = 64, height: int = 64):
        import pyrealsense2 as rs

        self._pipeline = rs.pipeline()
        self._serial_number = str(device.get_info(rs.camera_info.serial_number))
        self._width, self._height = width, height
        config = rs.config()
        config.enable_device(self._serial_number)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        self._pipeline.start(config)

    def get_frame(self) -> np.ndarray:
        frames = self._pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        assert color_frame
        img = np.asanyarray(color_frame.get_data())
        img = cv2.resize(img, (self._width, self._height), interpolation=cv2.INTER_AREA)
        return img


class RobotEnv(gym.Env):
    """
    A simple Gym Environment for controlling robots.
    """

    def __init__(
        self,
        controller_class: Union[str, Type[Controller]],
        controller_kwargs: Optional[Dict] = None,
        random_init: bool = True,
        control_hz: float = 10.0,
        img_width: int = 224,
        img_height: int = 224,
        cameras: List[str] = ("agent", "wrist"),
        channels_last: bool = True,
        horizon: int = 500,
    ):
        self.random_init = random_init
        controller_class = vars(robots)[controller_class] if isinstance(controller_class, str) else controller_class # TODO figure out how to import elegantly.
        self.controller = controller_class(**({} if controller_kwargs is None else controller_kwargs))
        # Add the action space limits.
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=self.controller.action_space.shape, dtype=np.float32)

        # Get the cameras
        camera_objects = []

        caps = [cv2.VideoCapture(i) for i in range(3)]
        camera_objects.extend([OpenCVCamera(cap, width=img_width, height=img_height) for cap in caps if cap.read()[0]])

        try:
            import pyrealsense2 as rs

            context = rs.context()
            camera_objects.extend([RealSenseCamera(device) for device in list(context.devices)])
        except ImportError:
            print("Warning: pyrealsense2 package not found")

        assert len(camera_objects) == len(cameras), "Found a different number of connected cameras than in `cameras`."

        self.cameras = {k: v for k, v in zip(cameras, camera_objects)}

        spaces = dict(state=self.controller.observation_space)
        for camera in self.cameras.keys():
            spaces[camera] = gym.spaces.Box(low=0, high=255, shape=(img_height, img_width, 3), dtype=np.uint8)
        self.observation_space = gym.spaces.Dict(spaces)

        self.horizon = horizon
        self._max_episode_steps = horizon
        self.control_hz = float(control_hz)
        self.channels_last = channels_last
        self._steps = 0

    def _get_obs(self):
        obs = dict(state=self.controller.get_state())
        for camera_name, camera_object in self.cameras.items():
            frame = camera_object.get_frame()
            if self.channels_last:
                frame = frame.transpose(2, 0, 1)
            obs[camera_name + "_image"] = frame

        return obs

    def step(self, action):
        # Immediately update with the action. Note that we scale everything to be between -1 and 1.
        start_time = time.time()
        low, high = self.controller.action_space.low, self.controller.action_space.high
        unscaled_action = low + (0.5 * (action + 1.0) * (high - low))
        self.controller.update(unscaled_action)

        comp_time = time.time() - start_time
        sleep_left = max(0, (1 / self.control_hz) - comp_time)
        time.sleep(sleep_left)

        self._steps += 1

        terminated = self._steps == self.horizon
        info = dict(discount=1 - float(terminated))

        if NEW_GYM_API:
            # Note that this is following the Gym 0.26 API for termination.
            return self._get_obs(), 0, False, self._steps == self.horizon, info
        else:
            return self._get_obs, 0, terminated, info

    def reset(self):
        self.controller.reset(randomize=self.random_init)
        time.sleep(3.0)
        self._steps = 0
        if NEW_GYM_API:
            return self._get_obs(), dict()
        else:
            return self._get_obs()
