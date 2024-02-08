import argparse
import os
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, type=str, help="Path to dataset collected with `collect_demos.py`")
    parser.add_argument("--percent", default=0.1, type=float, help="Percent to give to validation.")
    args = parser.parse_args()

    assert os.path.exists(args.path), "Dataset path did not exist."

    os.makedirs(os.path.join(args.path, "train"), exist_ok=True)
    os.makedirs(os.path.join(args.path, "val"), exist_ok=True)

    demos = [f for f in os.listdir(args.path) if f.endswith(".npz")]
    random.shuffle(demos)  # shuffle the data
    split = int(args.percent * len(demos))
    train, val = demos[split:], demos[:split]

    # Move all of the files to the correct location
    for demo in train:
        old_path = os.path.join(args.path, demo)
        new_path = os.path.join(args.path, "train", demo)
        os.rename(old_path, new_path)

    for demo in val:
        old_path = os.path.join(args.path, demo)
        new_path = os.path.join(args.path, "val", demo)
        os.rename(old_path, new_path)
