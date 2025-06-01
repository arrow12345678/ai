#!/usr/bin/env python3
"""
Downloads new images of vehicle registration plates from the OpenImages dataset.

This script performs the following steps:
1.  Configures paths, URLs, and download parameters.
2.  Ensures necessary directories for metadata and downloads exist.
3.  Downloads OpenImages metadata CSVs (bounding boxes, image URLs) and the
    official OpenImages downloader script if they are not already present.
4.  Processes the bounding box metadata to find images containing the target
    class (vehicle registration plates).
5.  Prepares a list of these image IDs for the downloader script.
6.  Uses the OpenImages downloader.py script to download the actual image files.
7.  Filters the initial bounding box data to include only annotations for
    images that were successfully downloaded.
8.  Saves these filtered annotations to a CSV file in the download directory.
"""

import os
import pandas as pd
import requests
import subprocess
import argparse

# --- Configuration Constants ---
TARGET_CLASS_MID = "/m/01jfm_"  # Vehicle registration plate
METADATA_DIR = "data/openimages_metadata/"
STAGING_DIR = "data/newly_downloaded/"
IMAGES_SUBDIR = os.path.join(STAGING_DIR, "images")
ANNOTATIONS_FILE = os.path.join(STAGING_DIR, "annotations.csv") # Changed to save directly in STAGING_DIR

# URLs for OpenImages data
URL_OI_BOUNDING_BOXES = "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv"
URL_OI_IMAGE_LIST_TRAIN = "https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv" # Not directly used for filtering, but good to have
URL_OI_DOWNLOADER_SCRIPT = "https://raw.githubusercontent.com/openimages/dataset/master/downloader.py"

# Local paths for downloaded metadata and script
BBOX_CSV_PATH = os.path.join(METADATA_DIR, "oidv6-train-annotations-bbox.csv")
# IMAGE_LIST_CSV_PATH = os.path.join(METADATA_DIR, "train-images-boxable-with-rotation.csv") # Not strictly needed for current logic
DOWNLOADER_SCRIPT_PATH = "downloader.py" # Assuming it's downloaded to the current directory or a specified tools dir

# --- Helper Functions ---

def ensure_dir(directory_path: str):
    """Creates a directory if it doesn't exist."""
    os.makedirs(directory_path, exist_ok=True)
    print(f"Ensured directory exists: {directory_path}")

def download_file(url: str, destination_path: str, description: str = "file"):
    """Downloads a file from a URL if it doesn't already exist at the destination."""
    if os.path.exists(destination_path):
        print(f"{description.capitalize()} already exists: {destination_path}")
        return True

    print(f"Downloading {description} from {url} to {destination_path}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors
        with open(destination_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Successfully downloaded {description}.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {description}: {e}")
        if os.path.exists(destination_path): # Clean up partial download
            os.remove(destination_path)
        return False

# --- Main Script Logic ---

def main(max_images: int):
    """Main function to orchestrate the download and processing of new images."""
    print("=" * 50)
    print("Starting New Image Download Process")
    print("=" * 50)

    # 1. Setup Directories
    print("\n--- Setting up directories ---")
    ensure_dir(METADATA_DIR)
    ensure_dir(STAGING_DIR)
    ensure_dir(IMAGES_SUBDIR)
    # No separate annotations subdir, annotations.csv goes into STAGING_DIR

    # 2. Download Metadata & Downloader Script
    print("\n--- Downloading essential files (if needed) ---")
    bbox_download_success = download_file(URL_OI_BOUNDING_BOXES, BBOX_CSV_PATH, "bounding box annotations CSV")
    # download_file(URL_OI_IMAGE_LIST_TRAIN, IMAGE_LIST_CSV_PATH, "image list CSV") # Optional, not used in current logic
    downloader_script_success = download_file(URL_OI_DOWNLOADER_SCRIPT, DOWNLOADER_SCRIPT_PATH, "OpenImages downloader script")

    if not bbox_download_success:
        print("Critical error: Could not download bounding box annotations. Exiting.")
        return
    if not downloader_script_success:
        print("Critical error: Could not download OpenImages downloader script. Exiting.")
        return

    # Make downloader script executable
    if os.path.exists(DOWNLOADER_SCRIPT_PATH):
        try:
            os.chmod(DOWNLOADER_SCRIPT_PATH, 0o755) # rwxr-xr-x
            print(f"Made {DOWNLOADER_SCRIPT_PATH} executable.")
        except OSError as e:
            print(f"Warning: Could not make {DOWNLOADER_SCRIPT_PATH} executable: {e}")


    # 3. Process Metadata to Find Target Images
    print("\n--- Processing metadata to find target images ---")
    try:
        print(f"Loading bounding box data from: {BBOX_CSV_PATH}")
        df_bbox = pd.read_csv(BBOX_CSV_PATH)

        print(f"Filtering for target class MID: {TARGET_CLASS_MID}")
        df_filtered_class = df_bbox[df_bbox['LabelName'] == TARGET_CLASS_MID]

        if df_filtered_class.empty:
            print(f"No images found for class MID {TARGET_CLASS_MID}. Exiting.")
            return

        unique_image_ids = df_filtered_class['ImageID'].unique()
        print(f"Found {len(unique_image_ids)} unique images with target class.")

        if len(unique_image_ids) > max_images:
            image_ids_to_download_list = unique_image_ids[:max_images]
            print(f"Limiting to {max_images} images for download.")
        else:
            image_ids_to_download_list = unique_image_ids
            print(f"Attempting to download all {len(image_ids_to_download_list)} found images.")

        # Store relevant columns for these image IDs
        # We'll filter this again later for successfully downloaded images
        all_target_annotations = df_filtered_class[df_filtered_class['ImageID'].isin(image_ids_to_download_list)][
            ['ImageID', 'XMin', 'XMax', 'YMin', 'YMax', 'IsOccluded', 'IsTruncated', 'IsGroupOf', 'IsDepiction', 'IsInside']
        ]
        print(f"Selected {len(all_target_annotations)} annotations for {len(image_ids_to_download_list)} potential images.")

    except FileNotFoundError:
        print(f"Error: Bounding box CSV not found at {BBOX_CSV_PATH}. Please ensure it's downloaded. Exiting.")
        return
    except Exception as e:
        print(f"An error occurred during metadata processing: {e}")
        return

    # 4. Prepare Image List for OpenImages Downloader
    print("\n--- Preparing image list for downloader ---")
    temp_image_list_file = "image_ids_to_download.txt"
    with open(temp_image_list_file, 'w') as f:
        for img_id in image_ids_to_download_list:
            f.write(f"train/{img_id}\n") # Assuming 'train' split as per CSV
    print(f"Image ID list saved to: {temp_image_list_file}")

    # 5. Download Images
    print("\n--- Downloading images using OpenImages downloader.py ---")
    # Command: python downloader.py image_ids_to_download.txt --download_folder=data/newly_downloaded/images/ --num_processes=5
    # The downloader script automatically creates subfolders like "train_abc123xyz"
    # We want images directly in IMAGES_SUBDIR

    # The downloader.py script from OpenImages creates subdirectories within the download_folder
    # based on the prefix in the image list (e.g., 'train/').
    # So, if IMAGES_SUBDIR is 'data/newly_downloaded/images', it will create 'data/newly_downloaded/images/train/'.
    # This is acceptable.

    download_command = [
        "python", DOWNLOADER_SCRIPT_PATH,
        temp_image_list_file,
        "--download_folder", STAGING_DIR, # downloader.py will create a 'train' subdir within this
        "--num_processes", "5" # Using 5 parallel processes for download
    ]
    # The downloader script expects the images to be listed as "PREFIX/IMAGE_ID"
    # and it will download them to "download_folder/PREFIX/IMAGE_ID.jpg"
    # Our IMAGES_SUBDIR is 'data/newly_downloaded/images', but the downloader creates 'train' inside the specified folder.
    # So if we pass STAGING_DIR ('data/newly_downloaded/'), it will create 'data/newly_downloaded/train/'.
    # We need to adjust the expected path or the argument.
    # Let's have the downloader download into STAGING_DIR, and it will create a 'train' subdirectory there.
    # Then we can move/process images from 'data/newly_downloaded/train/'.

    # The downloader.py script will place images in STAGING_DIR/train/
    # We defined IMAGES_SUBDIR as STAGING_DIR/images. This needs alignment.
    # Let's make the downloader output to `IMAGES_SUBDIR` directly.
    # However, the downloader script might inherently create a "train" subfolder based on the input file format "train/ImageID".
    # Let's test this behavior. For now, assume it creates STAGING_DIR/train/ImageID.jpg

    print(f"Executing command: {' '.join(download_command)}")
    try:
        process = subprocess.run(download_command, check=True, capture_output=True, text=True)
        print("Image download process completed.")
        print("Downloader Output:\n", process.stdout)
        if process.stderr:
            print("Downloader Errors:\n", process.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error during image download subprocess: {e}")
        print("Downloader Output (stdout):\n", e.stdout)
        print("Downloader Output (stderr):\n", e.stderr)
        # Continue to try and process any images that might have been downloaded
    except FileNotFoundError:
        print(f"Error: {DOWNLOADER_SCRIPT_PATH} not found. Ensure it's in the current directory and executable.")
        return

    # Clean up the temporary image ID list file
    try:
        os.remove(temp_image_list_file)
        print(f"Removed temporary image list file: {temp_image_list_file}")
    except OSError as e:
        print(f"Warning: Could not remove temporary image list file {temp_image_list_file}: {e}")

    # 6. Save Filtered Annotations
    print("\n--- Filtering and saving annotations for downloaded images ---")

    # The downloader script places images in STAGING_DIR/train/ for "train/ImageID" entries
    actual_downloaded_images_dir = os.path.join(STAGING_DIR, "train") # Adjust based on downloader.py behavior

    if not os.path.isdir(actual_downloaded_images_dir):
        print(f"Error: Expected download directory {actual_downloaded_images_dir} not found. No images were likely downloaded or path is incorrect.")
        # Try to check IMAGES_SUBDIR if that was the intent for direct downloads
        if os.path.isdir(IMAGES_SUBDIR):
             print(f"Checking alternative configured IMAGES_SUBDIR: {IMAGES_SUBDIR}")
             actual_downloaded_images_dir = IMAGES_SUBDIR # Fallback, though downloader usually creates 'train'
        else:
             print("No valid image directory found. Exiting annotation processing.")
             return

    downloaded_image_files = [f for f in os.listdir(actual_downloaded_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    downloaded_image_ids = [os.path.splitext(f)[0] for f in downloaded_image_files]
    print(f"Found {len(downloaded_image_ids)} images in {actual_downloaded_images_dir}.")

    if not downloaded_image_ids:
        print("No images were successfully downloaded. No annotations to save.")
    else:
        final_annotations = all_target_annotations[all_target_annotations['ImageID'].isin(downloaded_image_ids)]

        # Add file paths to annotations
        # final_annotations['filepath'] = final_annotations['ImageID'].apply(
        #     lambda img_id: os.path.join(actual_downloaded_images_dir, f"{img_id}.jpg") # Assuming jpg
        # )
        # For simplicity, sticking to original request of ImageID, XMin, XMax, YMin, YMax

        final_annotations_subset = final_annotations[['ImageID', 'XMin', 'XMax', 'YMin', 'YMax']]

        try:
            final_annotations_subset.to_csv(ANNOTATIONS_FILE, index=False)
            print(f"Successfully saved {len(final_annotations_subset)} annotations for downloaded images to: {ANNOTATIONS_FILE}")
        except Exception as e:
            print(f"Error saving final annotations: {e}")

    print("=" * 50)
    print("New Image Download Process Finished")
    print("=" * 50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download new images for ALPR training.")
    parser.add_argument(
        "--max_images",
        type=int,
        default=50,
        help="Maximum number of images to download."
    )
    args = parser.parse_args()

    main(max_images=args.max_images)
