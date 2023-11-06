from typing import Iterator, Tuple, Any, Union, Optional, Dict

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

"""
Borrowing Utils from research lightning
https://github.com/jhejna/research-lightning/blob/main/research/utils/utils.py
"""

def get_from_batch(batch: Any, start: Union[int, np.ndarray], end: Optional[int] = None) -> Any:
    if isinstance(batch, dict):
        return {k: get_from_batch(v, start, end=end) for k, v in batch.items()}
    elif isinstance(batch, (list, tuple)):
        return [get_from_batch(v, start, end=end) for v in batch]
    elif isinstance(batch, np.ndarray):
        if end is None:
            return batch[start]
        else:
            return batch[start:end]
    else:
        raise ValueError("Unsupported type passed to `get_from_batch`")

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


class RTXFrankaDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'agent_image': tfds.features.Image(
                            shape=(128, 128, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                        'wrist_image': tfds.features.Image(
                            shape=(128, 128, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Wrist camera RGB observation.',
                        ),
                        'state': tfds.features.FeatureDict({
                            'joint_positions': tfds.features.Tensor(
                                shape=(7,),
                                dtype=np.float32,
                                doc='Robot joint positions'
                            ),
                            'joint_velocities': tfds.features.Tensor(
                                shape=(7,),
                                dtype=np.float32,
                                doc='Robot joint velocities'
                            ),
                            'ee_pos': tfds.features.Tensor(
                                shape=(3,),
                                dtype=np.float32,
                                doc='Robot end effector position'
                            ),
                            'ee_quat': tfds.features.Tensor(
                                shape=(4,),
                                dtype=np.float32,
                                doc='Robot end effector quaternion'
                            ),
                            'gripper_pos': tfds.features.Tensor(
                                shape=(1,),
                                dtype=np.float32,
                                doc='Robot gripper position'
                            ),
                        })
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Robot action, consists of [3x cartesian delta, '
                            '3x rotation delta, 1x gripper position].',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path='path/to/train/demos/*.npz'),
            'val': self._generate_examples(path='path/to/val/demos/*.npz'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path):
            # load raw data --> this should change for your dataset
            with open(path, 'rb') as f:
                data = np.load(f)
                data = {k: data[k] for k in data.keys()}

            use_lightning_format = len(set(map(len, data.values()))) == 1
            data = nest_dict(data)
            
            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []

            # Language instruction doesn't change, so compute it once.
            language_instruction = data['language_instruction'][0]
            language_embedding = self._embed([language_instruction])[0].numpy()
            ep_len = len(data["done"])
            for i in (range(1, ep_len) if use_lightning_format else range(ep_len)):
                # compute Kona language embedding
                obs_idx = i - 1 if use_lightning_format else i
                step = {
                    "observation": get_from_batch(data["obs"], obs_idx),
                    "action": get_from_batch(data["action"], i),
                    "reward": get_from_batch(data["reward"], i),
                    "discount": get_from_batch(data["discount"], i),
                    "is_first": obs_idx == 0,
                    "is_last": i == ep_len,
                    "language_instruction": language_instruction,
                    "language_embedding": language_embedding
                }
                episode.append(step)

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path
                }
            }

            # if you want to skip an example for whatever reason, simply return None
            return episode_path, sample

        # create list of all examples
        episode_paths = glob.glob(path)

        # for smallish datasets, use single-thread parsing
        for sample in episode_paths:
            yield _parse_example(sample)
