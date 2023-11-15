import argparse

import cv2
from matplotlib import pyplot as plt

from robots.cameras import OpenCVCamera, RealSenseCamera

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=128, help="Image width")
    parser.add_argument("--height", type=int, default=128, help="Image height")

    args = parser.parse_args()

    camera_objects = []

    caps = [cv2.VideoCapture(i) for i in range(3)]
    camera_objects.extend(
        [
            OpenCVCamera(
                cap,
                width=args.width,
                height=args.height,
            )
            for cap in caps
            if cap.read()[0]
        ]
    )

    try:
        import pyrealsense2 as rs

        context = rs.context()
        camera_objects.extend([RealSenseCamera(device, args.width, args.height) for device in list(context.devices)])
    except ImportError:
        print("Warning: pyrealsense2 package not found")

    # Print frames from all of the camera objects.
    fig, axes = plt.subplots(1, len(camera_objects))
    for i, camera in enumerate(camera_objects):
        ax = axes.flat[i]
        ax.imshow(camera.get_frame())
        ax.set_title(str(i))

    plt.tight_layout()
    plt.show()
