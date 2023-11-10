import argparse
import datetime
import io
import os
from typing import Any, Dict, Iterable, Tuple

import gym
import numpy as np
import yaml

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
        for k in item.keys():
            append(lst[k], item[k])
    else:
        lst.append(item)


def _flatten_dict_helper(flat_dict: Dict, value: Any, prefix: str, separator: str = ".") -> None:
    if isinstance(value, (dict, gym.spaces.Dict)):
        for k in value.keys():
            assert isinstance(k, str), "Can only flatten dicts with str keys"
            _flatten_dict_helper(flat_dict, value[k], prefix + separator + k, separator=separator)
    else:
        flat_dict[prefix[1:]] = value


def flatten_dict(d: Dict, separator: str = ".") -> Dict:
    flat_dict = dict()
    _flatten_dict_helper(flat_dict, d, "", separator=separator)
    return flat_dict


def nest_dict(d: Dict, separator: str = ".") -> Dict:
    nested_d = dict()
    for key in d.keys():
        key_parts = key.split(separator)
        current_d = nested_d
        while len(key_parts) > 1:
            if key_parts[0] not in current_d:
                current_d[key_parts[0]] = dict()
            current_d = current_d[key_parts[0]]
            key_parts.pop(0)
        current_d[key_parts[0]] = d[key]  # Set the value
    return nested_d


def save_episode(data: Dict, path: str, enforce_length: bool = True) -> None:
    # Flatten the dict for saving as a numpy array.
    data = flatten_dict(data)

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
            else:
                dtype = None
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
    parser.add_argument("--config", type=str, required=True, help="Path to config file.")
    parser.add_argument("--instr", type=str, default=None, help="Language instruction.")
    parser.add_argument(
        "--lightning-format",
        type=int,
        default=1,
        help="Whether or not to save demos compatible with research-lightning",
    )
    parser.add_argument(
        "--vr-kwargs",
        metavar="KEY=VALUE",
        action="append",
        help="Set kv pairs used as args for the controller class.",
    )

    args = parser.parse_args()

    vr_kwargs = dict(
        pos_action_gain=3.0,
        rot_action_gain=1.0,
        gripper_action_gain=1.0,
        min_magnitude=0.15,
        robot_orientation="gripper_on_left",
    )
    vr_kwargs.update(parse_vars(args.vr_kwargs))

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    env = robots.RobotEnv(**config)
    vr = robots.VRController(**vr_kwargs)

    os.makedirs(args.path, exist_ok=True)

    num_episodes = 0
    while True:
        done = False

        if args.lightning_format:
            episode = dict(
                action=[env.action_space.sample()],
                reward=[0.0],
                done=[False],
                discount=[1.0],
            )
        else:
            episode = dict(
                action=[],
                reward=[],
                done=[],
                discount=[],
            )
        episode["obs"] = nest_dict({k: [] for k in flatten_dict(env.observation_space).keys()})

        # Reset the environment
        if NEW_GYM_API:
            obs, info = env.reset()
        else:
            obs = env.reset()
        append(episode, dict(obs=obs))

        # See if we want to use language.
        lang = args.instr if args.instr is not None else input("[robots] Language instruction?")
        lang = None if lang == "" else lang
        if lang is not None:
            episode["language_instruction"] = [lang]

        print("[robots] Start episode.")
        while not done:
            action = vr.predict(obs)

            if action is not None:
                # If we have an action from VR, step the environment
                if NEW_GYM_API:
                    obs, reward, done, terminated, info = env.step(action)
                else:
                    obs, reward, done, info = env.step(action)
                    terminated = False

                discount = 1.0 - float(terminated)
                step = dict(obs=obs, action=action, reward=reward, done=done, discount=discount)
                if lang is not None:
                    step["language_instruction"] = lang
                append(episode, step)

            controller_info = vr.get_info()
            done = done or controller_info["user_set_success"] or controller_info["user_set_failure"]

        print("[robots] Finished episode.")
        # Store done and reward at the final timestep
        episode["done"][-1] = True
        episode["reward"][-1] = 1.0

        ep_len = len(episode["done"])
        num_episodes += 1
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        ep_filename = f"{ts}_{num_episodes}_{ep_len}.npz"
        save_episode(episode, os.path.join(args.path, ep_filename), enforce_length=args.lightning_format)

        to_break = input("[robots] Quit (q)?")
        if to_break == "q":
            break

    env.close()
    del env
    del vr
    exit()
