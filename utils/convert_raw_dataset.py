# convert the dataset in data/elearn_raw/train to remove the outer folder "phase", extract the images with their direct parent folders and save all of them in a new folder data/elearn/train
import os
import shutil

# Define paths
raw_data_path = "/home/zichang/repo/PyCIL/data/elearn_raw/train"
new_data_path = "/home/zichang/repo/PyCIL/data/elearn/train"

# Loop through all subdirectories in raw_data_path
for root, dirs, files in os.walk(raw_data_path):
    for file in files:
        # Check if file is an image
        if file.endswith(".jpg") or file.endswith(".png"):
            # Get the parent directory name
            parent_dir = os.path.basename(root)
            # Get the phase directory name
            phase_dir = os.path.basename(os.path.dirname(root))
            # Create new directory path
            new_dir_path = os.path.join(new_data_path, parent_dir)
            # Create new directory if it doesn't exist
            if not os.path.exists(new_dir_path):
                os.makedirs(new_dir_path)
            # Copy the image to the new directory
            old_file_path = os.path.join(root, file)
            new_file_path = os.path.join(new_dir_path, phase_dir+file)
            shutil.copy(old_file_path, new_file_path)
