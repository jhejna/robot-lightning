"""
A simple script for replaying a demonstration
"""
import argparse
import os

import numpy as np
import yaml

import robots

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to demonstration.")
    args = parser.parse_args()

    assert os.path.exists(args.path), "Demo did not exist."
    demo_dir = os.path.dirname(args.path)
    config_path = os.path.join(demo_dir, "config.yaml")

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    env = robots.RobotEnv(**config)

    with open(args.path, "rb") as f:
        data = np.load(f)
        actions = data["action"]

    env.reset()
    for i in range(actions.shape[0]):
        env.step(actions[i])

    print("Done replaying.")
