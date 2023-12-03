import os
import argparse
import re


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="rtx_franka", help="Path to the dataset")
    parser.add_argument("--name", type=str, default="rtx_franka", help="Name of the dataset")

    args = parser.parse_args()

    old_folder_path = args.path.rstrip(os.path.sep)
    directory, old_under_name = os.path.split(old_folder_path)

    old_camel_name = "".join([s.capitalize() for s in old_under_name.split("_")])
    new_under_name = args.name
    new_camel_name = "".join([s.capitalize() for s in new_under_name.split("_")])

    # Get all the files we need to rename or move
    old_builder_file = os.path.join(args.path, old_under_name + "_dataset_builder.py")
    assert os.path.exists(old_builder_file), "Invalid path name, could not find " + old_builder_file

    with open(old_builder_file, 'r') as f:
        content = f.read()

    new_content = content.replace(old_camel_name + "Dataset", new_camel_name + "Dataset")

    new_builder_file = os.path.join(args.path, new_under_name + "_dataset_builder.py")
    with open(new_builder_file, 'w') as f:
        f.write(new_content)

    # Unlink the old file
    os.remove(old_builder_file)
    
    # Now rename the folder
    new_folder_path = os.path.join(directory, new_under_name)
    os.rename(old_folder_path, new_folder_path)

    print("Done!")

    