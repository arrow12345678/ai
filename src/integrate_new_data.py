#!/usr/bin/env python3
"""
Integrates newly downloaded and processed images and their YOLO annotations
into the main raw data directory structure.

This script moves files from the staging area (data/newly_downloaded/)
to the project's raw data directories (data/raw/).
"""

import os
import shutil
import glob
import argparse

# --- Configuration Constants ---
DEFAULT_SOURCE_IMAGES_DIR = "data/newly_downloaded/train/"
DEFAULT_SOURCE_LABELS_DIR = "data/newly_downloaded/yolo_labels/"
DEFAULT_DEST_RAW_IMAGES_DIR = "data/raw/images/"
DEFAULT_DEST_RAW_LABELS_DIR = "data/raw/labels/"

# Common image file extensions
IMAGE_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png"]

# --- Helper Functions ---

def ensure_dir(directory_path: str):
    """Creates a directory if it doesn't exist."""
    os.makedirs(directory_path, exist_ok=True)
    print(f"Ensured directory exists: {directory_path}")

def move_files_from_source(source_dir: str, dest_dir: str, file_patterns: list[str], file_type_name: str) -> int:
    """
    Moves files matching specified patterns from a source directory to a destination directory.

    Args:
        source_dir (str): The directory to move files from.
        dest_dir (str): The directory to move files to.
        file_patterns (list[str]): A list of glob patterns (e.g., ["*.jpg", "*.png"]) to match files.
        file_type_name (str): A descriptive name for the type of files being moved (e.g., "image", "label").

    Returns:
        int: The number of files successfully moved.
    """
    if not os.path.isdir(source_dir):
        print(f"Source directory for {file_type_name} files not found: {source_dir}. Skipping.")
        return 0

    moved_count = 0
    print(f"\n--- Moving {file_type_name} files ---")
    print(f"Source: {source_dir}")
    print(f"Destination: {dest_dir}")

    all_files_to_move = []
    for pattern in file_patterns:
        all_files_to_move.extend(glob.glob(os.path.join(source_dir, pattern)))

    # Remove duplicates if patterns overlap (e.g. *.jpeg and *.jpg if symlinked)
    all_files_to_move = sorted(list(set(all_files_to_move)))

    if not all_files_to_move:
        print(f"No {file_type_name} files matching patterns {file_patterns} found in {source_dir}.")
        return 0

    for source_file_path in all_files_to_move:
        file_name = os.path.basename(source_file_path)
        dest_file_path = os.path.join(dest_dir, file_name)

        if os.path.exists(dest_file_path):
            print(f"Warning: {file_type_name} file '{file_name}' already exists in {dest_dir}. It will be overwritten.")

        try:
            shutil.move(source_file_path, dest_file_path)
            # print(f"Moved {file_type_name} file: {file_name}") # Can be too verbose
            moved_count += 1
        except Exception as e:
            print(f"Error moving {file_type_name} file {file_name}: {e}")
            print(f"  Source: {source_file_path}")
            print(f"  Destination: {dest_file_path}")

    print(f"Successfully moved {moved_count} {file_type_name} file(s).")
    return moved_count

# --- Main Execution Block ---

def main(source_images: str, source_labels: str, dest_images: str, dest_labels: str):
    """Main function to orchestrate the integration of new data."""
    print("=" * 50)
    print("Starting Data Integration Process")
    print("=" * 50)

    # 1. Ensure Destination Directories Exist
    print("\n--- Ensuring destination directories exist ---")
    ensure_dir(dest_images)
    ensure_dir(dest_labels)

    # 2. Move Image Files
    num_images_moved = move_files_from_source(
        source_dir=source_images,
        dest_dir=dest_images,
        file_patterns=IMAGE_EXTENSIONS,
        file_type_name="image"
    )

    # 3. Move Label Files
    num_labels_moved = move_files_from_source(
        source_dir=source_labels,
        dest_dir=dest_labels,
        file_patterns=["*.txt"],
        file_type_name="label"
    )

    print("\n--- Integration Summary ---")
    print(f"Total image files moved: {num_images_moved}")
    print(f"Total label files moved: {num_labels_moved}")

    # Optional: Check if the source directories are now empty and suggest cleanup
    if os.path.isdir(source_images) and not os.listdir(source_images):
        print(f"Source image directory is now empty: {source_images}")
    if os.path.isdir(source_labels) and not os.listdir(source_labels):
        print(f"Source label directory is now empty: {source_labels}")

    print("=" * 50)
    print("Data Integration Process Finished")
    print("=" * 50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Integrate new data into the main raw data directories.")
    parser.add_argument(
        "--source_images_dir", type=str, default=DEFAULT_SOURCE_IMAGES_DIR,
        help=f"Source directory for image files. Default: {DEFAULT_SOURCE_IMAGES_DIR}"
    )
    parser.add_argument(
        "--source_labels_dir", type=str, default=DEFAULT_SOURCE_LABELS_DIR,
        help=f"Source directory for YOLO label files. Default: {DEFAULT_SOURCE_LABELS_DIR}"
    )
    parser.add_argument(
        "--dest_images_dir", type=str, default=DEFAULT_DEST_RAW_IMAGES_DIR,
        help=f"Destination directory for image files. Default: {DEFAULT_DEST_RAW_IMAGES_DIR}"
    )
    parser.add_argument(
        "--dest_labels_dir", type=str, default=DEFAULT_DEST_RAW_LABELS_DIR,
        help=f"Destination directory for YOLO label files. Default: {DEFAULT_DEST_RAW_LABELS_DIR}"
    )
    args = parser.parse_args()

    main(
        source_images=args.source_images_dir,
        source_labels=args.source_labels_dir,
        dest_images=args.dest_images_dir,
        dest_labels=args.dest_labels_dir
    )
