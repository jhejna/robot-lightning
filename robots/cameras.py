import numpy as np
from typing import Dict, Union
import cv2
from abc import abstractmethod, abstractproperty
import threading

try:
    import pyrealsense2 as rs
    IMPORTED_PYREALSENSE = True
except ImportError:
    IMPORTED_PYREALSENSE = False

class Camera(object):

    def __init__(self, width: int, height: int, depth: bool = False):
        self.width = width
        self.height = height
        self.depth = depth

    @abstractproperty
    def has_depth(self):
        raise NotImplementedError

    @abstractmethod
    def get_frames(self) -> Dict[str, np.ndarray]:
        pass

    @abstractmethod
    def close(self):
        pass


class OpenCVCamera(Camera):
    def __init__(self, id: Union[int, str] = None, **kwargs):
        super().__init__(**kwargs)
        self.id = id
        self._running = False

    @property
    def has_depth(self):
        return False

    def _init(self):
        self._running = True
        self._image = None
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()
        
    def _reader(self):
        # We need to parse the CV Camera ID
        self._cap = cv2.VideoCapture(self.id)
        # Values other than default 640x480 have not been tested yet
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.height)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.width)

        while self._running:

            retval, img = self._cap.read()
            if retval:
                
                self.lock.acquire()
                self._image = img
                self.lock.release()

        self._cap.release()

    def get_frames(self) -> np.ndarray:
        if not self._running:
            self._init()
        image = None
        while image is None:
            self.lock.acquire()
            if self._image is not None:
                image = self._image
                self._image = None # set back to None.
            self.lock.release()
        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return dict(image=image)

    def close(self):
        if self._running:
            self._running = False
            self.thread.join()


class RealSenseCamera(Camera):
    def __init__(self, serial_number, **kwargs):
        super().__init__(**kwargs)
        self.serial_number = serial_number

    @property
    def has_depth(self):
        return self.depth

    def _init(self):
        self._running = True
        self._image = None
        self._depth = None
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        pipeline = rs.pipeline()
        config = rs.config()

        config.enable_device(self.serial_number)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        if self.depth:
            config.enable_stream(rs.stream.color, 640, 480, rs.format.z16, 30)
            depth_filters = [rs.spatial_filter(), rs.temporal_filter()]

        while self._running:
            frames = pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            if self.depth:
                depth = aligned_frames.get_depth_frame()
                for rs_filter in depth_filters:
                    depth = rs_filter(depth)
                depth = np.asanyarray(depth.get_data())
            else:
                self._depth = None
            image = aligned_frames.get_color_frame()

            self.lock.acquire()
            if self.depth:
                self._depth = depth
            self._image = image
            self.lock.release()

        # Close the thread
        pipeline.stop()

    def get_frames(self) -> np.ndarray:
        if not self._running:
            self._init()
        image = None
        while image is None:
            self.lock.acquire()
            if self._image is not None:
                image, depth = self._image, self._depth
                self._image, self._depth = None, None # set back to None.
            self.lock.release()

        # Process the image
        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frames = dict(image=image)
        if self.depth:
            depth = cv2.resize(depth, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
            if len(depth.shape) == 2:
                depth = np.expand_dims(depth, axis=-1)
            frames["depth"] = depth
        return frames

    def close(self):
        if self._running:
            self._running = False
            self.thread.join()
