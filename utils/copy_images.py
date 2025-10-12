import os
import shutil
import argparse

def copy_images(source_dir, dest_dir):
    """
    Copies all images from the source directory to the destination directory.

    Args:
        source_dir (str): The path to the source directory.
        dest_dir (str): The path to the destination directory.
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for filename in os.listdir(source_dir):
        source_path = os.path.join(source_dir, filename)
        if os.path.isfile(source_path):
            shutil.copy(source_path, dest_dir)
            print(f"Copied {filename} to {dest_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy images from a source to a destination directory.")
    parser.add_argument("source", help="Source directory for images.")
    parser.add_argument("destination", help="Destination directory for images.")
    args = parser.parse_args()

    copy_images(args.source, args.destination)
