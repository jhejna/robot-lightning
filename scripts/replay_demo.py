"""
A simple script for replaying a demonstration
"""

import argparse
import os

import numpy as np
import yaml
from matplotlib import pyplot as plt

import robots

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to demonstration.")
    parser.add_argument("--camera-name", type=str, default="agent", help="Path to demonstration.")
    parser.add_argument("--use-robot", type=int, default=1, help="Set to zero to disable the robot.")
    args = parser.parse_args()

    assert os.path.exists(args.path), "Demo did not exist."
    demo_dir = os.path.dirname(args.path)
    config_path = os.path.join(demo_dir, "config.yaml")

    with open(args.path, "rb") as f:
        data = np.load(f)
        actions = data["action"]
        images = data["obs." + args.camera_name + "_image"]
        is_channels_first = images.shape[-1] != 3

    if args.use_robot:
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.Loader)
        env = robots.RobotEnv(**config)
        env.reset()
        print("[robots] Finished reset.")

    image = images[0]
    if is_channels_first:
        image = image.transpose(1, 2, 0)
    display = plt.imshow(image)
    plt.ion()
    plt.show()
    for i in range(min(actions.shape[0], images.shape[0] - 1)):
        if args.use_robot:
            env.step(actions[i])
        image = images[i + 1]
        if is_channels_first:
            image = image.transpose(1, 2, 0)
        display.set_data(image)
        plt.pause(0.001)

    print("[robots] Done replaying.")
