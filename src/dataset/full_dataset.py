import os
import shutil

import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../../'))
import config.config


def merge_datasets(source_dir1, source_dir2, merged_dir):
    # Function to copy contents of one directory into the merged directory
    def copy_contents(src_dir, dest_dir):
        for root, dirs, files in os.walk(src_dir):
            # Calculate the relative path from src_dir
            relative_path = os.path.relpath(root, src_dir)
            # Define the corresponding destination path
            dest_path = os.path.join(dest_dir, relative_path)

            # Create directories in the destination path if they don't exist
            if not os.path.exists(dest_path):
                os.makedirs(dest_path)

            # Copy all files from the current source subfolder to the destination subfolder
            for file_name in files:
                src_file = os.path.join(root, file_name)
                dest_file = os.path.join(dest_path, file_name)

                # If the file already exists in the destination, choose to overwrite or skip
                if os.path.exists(dest_file):
                    print(f"Overwriting {dest_file}")
                shutil.copy2(src_file, dest_file)  # copy2 to preserve metadata

    # Copy contents of both source directories into the merged directory
    copy_contents(source_dir1, merged_dir)
    copy_contents(source_dir2, merged_dir)


if __name__ == "__main__":
    # Ensure the merged directory exists
    if not os.path.exists(config.config.DATASET_DIR):
        os.makedirs(config.config.DATASET_DIR)

    merge_datasets(config.config.MET_DATA_DIR, config.config.VA_DATA_DIR, config.config.DATASET_DIR)