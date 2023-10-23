import os
import cv2
import numpy as np

# Define the path to the dataset directory
dataset_dir = 'data/elearn_flatten/train'

# Initialize empty lists to store the mean and std of each channel
channel1_mean = []
channel2_mean = []
channel3_mean = []
channel1_std = []
channel2_std = []
channel3_std = []

import glob 
import shutil
    
destination_path = "data/elearn_flatten/train"
pattern = "data/elearn/train/*/*"  
for img in glob.glob(pattern):
    shutil.copy(img, destination_path)
# Loop through each image in the dataset directory
for filename in os.listdir(dataset_dir):
    # Read the image using cv2.imread()
    img = cv2.imread(os.path.join(dataset_dir, filename))
    # Convert the image to float32 using numpy.float32()
    img = np.float32(img)
    # Calculate the mean and std of each channel using numpy.mean() and numpy.std()
    ch1_mean = np.mean(img[:, :, 0])
    ch2_mean = np.mean(img[:, :, 1])
    ch3_mean = np.mean(img[:, :, 2])
    ch1_std = np.std(img[:, :, 0])
    ch2_std = np.std(img[:, :, 1])
    ch3_std = np.std(img[:, :, 2])
    # ch1_mean, ch2_mean, ch3_mean = np.mean(img)[:3]
    # claculate the std of each channel using numpy.std()

    # ch1_std, ch2_std, ch3_std = np.std(img)[:3]
    # Append the mean and std of each channel to their respective lists
    channel1_mean.append(ch1_mean)
    channel2_mean.append(ch2_mean)
    channel3_mean.append(ch3_mean)
    channel1_std.append(ch1_std)
    channel2_std.append(ch2_std)
    channel3_std.append(ch3_std)

# Calculate the overall mean and std of each channel using numpy.mean() and numpy.std() on the lists
mean = [np.mean(channel1_mean), np.mean(channel2_mean), np.mean(channel3_mean)]
std = [np.mean(channel1_std), np.mean(channel2_std), np.mean(channel3_std)]

# Print out the results in the required format
print(f"mean:{mean}, std:{std}")
