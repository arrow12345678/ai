#!/usr/bin/env python3
"""
Converts bounding box annotations from CSV format (produced by download_new_images.py)
to YOLO format text files.

Each image's annotations will be saved in a separate .txt file named after the ImageID.
The YOLO format line is: class_id x_center y_center width height
Coordinates are normalized (0.0 to 1.0).
"""

import os
import pandas as pd
import argparse

# --- Configuration Constants ---
DEFAULT_INPUT_CSV_PATH = "data/newly_downloaded/annotations.csv"
DEFAULT_OUTPUT_YOLO_DIR = "data/newly_downloaded/yolo_labels/"
DEFAULT_CLASS_ID = 0  # Class ID for vehicle registration plate

# --- Helper Functions ---

def ensure_dir(directory_path: str):
    """Creates a directory if it doesn't exist."""
    os.makedirs(directory_path, exist_ok=True)
    print(f"Ensured directory exists: {directory_path}")

# --- Core Processing Function ---

def convert_annotations_to_yolo(input_csv: str, output_dir: str, class_id: int):
    """
    Reads annotations from a CSV file and converts them to YOLO format.

    Args:
        input_csv (str): Path to the input CSV file.
                         Expected columns: ImageID, XMin, XMax, YMin, YMax.
        output_dir (str): Directory where YOLO .txt files will be saved.
        class_id (int): The class ID to use for all annotations.
    """
    print(f"\n--- Starting YOLO Annotation Conversion ---")
    print(f"Input CSV: {input_csv}")
    print(f"Output Directory: {output_dir}")
    print(f"Class ID: {class_id}")

    try:
        df_annotations = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: Input CSV file not found at {input_csv}. Exiting.")
        return
    except Exception as e:
        print(f"Error reading CSV file {input_csv}: {e}. Exiting.")
        return

    if df_annotations.empty:
        print("Input CSV is empty. No annotations to convert.")
        return

    required_columns = ['ImageID', 'XMin', 'XMax', 'YMin', 'YMax']
    if not all(col in df_annotations.columns for col in required_columns):
        print(f"Error: Input CSV must contain the columns: {', '.join(required_columns)}. Found: {', '.join(df_annotations.columns)}. Exiting.")
        return

    # Group annotations by ImageID
    grouped_annotations = df_annotations.groupby('ImageID')

    files_created_count = 0
    annotations_processed_count = 0

    for image_id, group in grouped_annotations:
        yolo_file_path = os.path.join(output_dir, f"{image_id}.txt")
        yolo_lines = []

        for _, row in group.iterrows():
            xmin = row['XMin']
            xmax = row['XMax']
            ymin = row['YMin']
            ymax = row['YMax']

            # Calculate YOLO coordinates (already normalized as per OpenImages standard)
            box_width = xmax - xmin
            box_height = ymax - ymin
            x_center = xmin + (box_width / 2)
            y_center = ymin + (box_height / 2)

            # Format: class_id x_center y_center width height
            yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
            yolo_lines.append(yolo_line)
            annotations_processed_count += 1

        try:
            with open(yolo_file_path, 'w') as f:
                for line in yolo_lines:
                    f.write(line + "\n")
            files_created_count += 1
        except IOError as e:
            print(f"Error writing YOLO file for ImageID {image_id} at {yolo_file_path}: {e}")

    print(f"\n--- Conversion Summary ---")
    print(f"Total annotations processed: {annotations_processed_count}")
    print(f"Number of YOLO annotation files created: {files_created_count} in {output_dir}")
    if files_created_count != len(grouped_annotations):
        print(f"Warning: Expected to create {len(grouped_annotations)} files, but created {files_created_count}.")


# --- Main Execution Block ---

def main():
    """Main function to parse arguments and orchestrate the conversion."""
    parser = argparse.ArgumentParser(description="Convert OpenImages annotations from CSV to YOLO format.")
    parser.add_argument(
        "--input_csv",
        type=str,
        default=DEFAULT_INPUT_CSV_PATH,
        help=f"Path to the input CSV annotations file. Default: {DEFAULT_INPUT_CSV_PATH}"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_YOLO_DIR,
        help=f"Directory to save output YOLO .txt files. Default: {DEFAULT_OUTPUT_YOLO_DIR}"
    )
    parser.add_argument(
        "--class_id",
        type=int,
        default=DEFAULT_CLASS_ID,
        help=f"Class ID for the annotations (e.g., 0 for license plate). Default: {DEFAULT_CLASS_ID}"
    )
    args = parser.parse_args()

    # 1. Setup Output Directory
    ensure_dir(args.output_dir)

    # 2. Process Annotations
    convert_annotations_to_yolo(args.input_csv, args.output_dir, args.class_id)

    print("=" * 50)
    print("YOLO Annotation Processing Finished")
    print("=" * 50)

if __name__ == '__main__':
    main()
