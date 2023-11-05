import argparse
import datetime
import io
import os
from typing import Dict, Iterable, Tuple

import gym
import numpy as np

import robots

NEW_GYM_API = False if gym.__version__ < "0.26.1" else True


"""
Script for collecting demos.
To be run on the workstation.
"""


def parse_var(s: str) -> Tuple[str]:
    """
    Parse a key, value pair, separated by '='
    """
    items = s.split("=")
    key = items[0].strip()  # we remove blanks around keys, as is logical
    if len(items) > 1:
        # rejoin the rest:
        value = "=".join(items[1:])
    return (key, value)


def parse_vars(items: Iterable) -> Dict:
    """
    Parse a series of key-value pairs and return a dictionary
    """
    d = {}

    if items:
        for item in items:
            key, value = parse_var(item)
            d[key] = value
    return d


def append(lst, item):
    # This takes in a nested list structure and appends everything from item to the nested list structure.
    # It will require lst to have the complete set of keys -- if keys are in item but not in lst,
    # they will not be appended.
    if isinstance(lst, dict):
        assert isinstance(item, dict)
        for k in lst.keys():
            append(lst[k], item[k])
    else:
        lst.append(item)


def save_episode(episode: Dict, path: str, enforce_length: bool = True) -> None:
    # Flatten the dict for saving as a numpy array.
    data = dict()
    for k in episode.keys():
        if k == "obs":
            for obs_key in episode[k].keys():
                data[k + "." + obs_key] = episode[k][obs_key]
        else:
            data[k] = episode[k]

    # Format everything into numpy in case it was saved as a list
    for k in data.keys():
        if isinstance(data[k], np.ndarray) and not data[k].dtype == np.float64:  # Allow float64 carve out.
            continue
        elif isinstance(data[k], list):
            first_value = data[k][0]
            if isinstance(first_value, (np.float64, float)):
                dtype = np.float32  # Detect and convert out float64
            elif isinstance(first_value, (np.ndarray, np.generic)):
                dtype = first_value.dtype
            elif isinstance(first_value, bool):
                dtype = np.bool_
            elif isinstance(first_value, int):
                dtype = np.int64
            data[k] = np.array(data[k], dtype=dtype)
        else:
            raise ValueError("Unsupported type passed to `save_data`.")

    if enforce_length:
        assert len(set(map(len, data.values()))) == 1, "All data keys must be the same length."

    with io.BytesIO() as bs:
        np.savez_compressed(bs, **data)
        bs.seek(0)
        with open(path, "wb") as f:
            f.write(bs.read())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to save demonstrations.")
    parser.add_argument("--horizon", type=int, default=200, help="episode horizon")
    parser.add_argument("--width", type=int, default=128, help="Image width")
    parser.add_argument("--height", type=int, default=128, help="Image height")
    parser.add_argument("--cameras", nargs="+", default=[], help="Cameras connected to the workstation.")
    parser.add_argument("--control-hz", type=float, default=10.0, help="Control Hz")
    parser.add_argument("--controller", type=str, default="ZeroRPCController", help="Controller Class to use")
    parser.add_argument(
        "--lightning-format",
        type=int,
        default=1,
        help="Whether or not to save demos compatible with research-lightning",
    )
    parser.add_argument(
        "--controller-kwargs",
        metavar="KEY=VALUE",
        nargs="+",
        action="append",
        help="Set kv pairs used as args for the controller class.",
    )

    parser.add_argument(
        "--vr-kwargs",
        metavar="KEY=VALUE",
        nargs="+",
        action="append",
        help="Set kv pairs used as args for the controller class.",
    )

    args = parser.parse_args()

    controller_kwargs = dict(controller_type="CARTESIAN_DELTA")
    controller_kwargs.update(parse_vars(args.controller_kwargs))

    vr_kwargs = dict(
        pos_action_gain=3.0,
        rot_action_gain=1.0,
        gripper_action_gain=1.0,
        min_magnitude=0.15,
        robot_orientation="gripper_on_left",
    )
    vr_kwargs.update(parse_vars(args.vr_kwargs))

    controller = vars(robots)[args.controller](parse_vars(args.controller_kwargs))
    env = robots.RobotEnv(
        random_init=True,
        img_width=args.width,
        img_height=args.height,
        control_hz=args.control_hz,
        cameras=args.cameras,
        horizon=args.horizon,
    )
    vr = robots.VRController(
        pos_action_gain=3.0,
        rot_action_gain=1.0,
        gripper_action_gain=1.0,
        min_magnitude=0.15,
        robot_orientation="gripper_on_left",
    )

    num_episodes = 0
    while True:
        done = False

        if args.lightning_format:
            episode = dict(
                obs={k: [] for k, v in env.observation_space.keys()},
                action=[env.action_space.sample()],
                reward=[0.0],
                done=[False],
                discount=[1.0],
            )
        else:
            episode = dict(
                obs={k: [] for k, v in env.observation_space.keys()},
                action=[],
                reward=[],
                done=[],
                discount=[],
            )

        if NEW_GYM_API:
            obs, info = env.reset()
        else:
            obs = env.reset()

        append(episode, dict(obs=obs))

        while not done:
            action = vr.predict(obs)

            if action is not None:
                if NEW_GYM_API:
                    obs, reward, done, terminated, info = env.step(action)
                else:
                    obs, reward, done, info = env.step(action)
                    terminated = False

            controller_info = vr.get_info()
            done = done or controller_info["user_set_success"] or controller_info["user_set_failure"]
            discount = 1.0 - float(terminated)

            append(episode, dict(obs=obs, action=action, reward=reward, done=done, discount=discount))

        ep_len = len(episode["done"])
        num_episodes += 1
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        ep_filename = f"{ts}_{num_episodes}_{ep_len}.npz"
        save_episode(episode, os.path.join(args.path, ep_filename), enforce_length=args.lightning_format)

        to_break = input("Quit (q)?")
        if to_break == "q":
            break

    env.close()
    del env
    del vr
    exit()
