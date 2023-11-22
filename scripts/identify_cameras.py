import argparse
import subprocess

from matplotlib import pyplot as plt

from robots.cameras import OpenCVCamera, RealSenseCamera

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=256, help="Image width")
    parser.add_argument("--height", type=int, default=256, help="Image height")

    args = parser.parse_args()

    camera_objects = []

    try:
        import pyrealsense2 as rs

        context = rs.context()
        serial_numbers = [str(device.get_info(rs.camera_info.serial_number)) for device in context.devices]
        camera_objects.extend(
            [
                (serial_number, RealSenseCamera(serial_number, width=args.width, height=args.height, depth=False))
                for serial_number in serial_numbers
            ]
        )
    except ImportError:
        print("Warning: pyrealsense2 package not found")

    devices = subprocess.run(["v4l2-ctl", "--list-devices"], check=False, stdout=subprocess.PIPE).stdout.decode("utf-8")
    lines = devices.split("\n")
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if not line == ""]

    cam_ids = []
    for i in range(len(lines)):
        if lines[i].startswith("/dev") or "RealSense" in lines[i]:
            continue
        cam_ids.append(lines[i + 1])

    camera_objects.extend(
        [
            (
                cam_id,
                OpenCVCamera(
                    cam_id,
                    width=args.width,
                    height=args.height,
                ),
            )
            for cam_id in cam_ids
        ]
    )

    # Print frames from all of the camera objects.
    fig, axes = plt.subplots(1, len(camera_objects))
    for i, (cam_id, camera) in enumerate(camera_objects):
        ax = axes.flat[i]
        ax.imshow(camera.get_frames()["image"])
        ax.set_title(str(cam_id))
        print(i, cam_id)
    plt.tight_layout()
    plt.show()
