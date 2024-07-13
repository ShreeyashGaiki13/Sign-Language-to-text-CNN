import os
import shutil
import splitfolders

# Define the source and destination directories
source_dir = 'D:\\Mini Project\\collected_data'
destination_dir = 'D:\\Mini Project\\modelData'

# Create the destination directory if it doesn't exist
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# Split the dataset into train and test directories directly
splitfolders.ratio(source_dir, output=destination_dir, ratio=(0.8, 0.2), seed=42)

# Define the labels for the symbols
labels = [str(i) for i in range(10)] + ['blank']

# Iterate over the labels and move the images to the appropriate train and test directories
for label in labels:
    for dataset_type in ['train', 'test']:
        source_label_dir = os.path.join(destination_dir, dataset_type, label)
        destination_label_dir = os.path.join(destination_dir, dataset_type, label)
        if not os.path.exists(destination_label_dir):
            os.makedirs(destination_label_dir)
        for filename in os.listdir(source_label_dir):
            shutil.move(os.path.join(source_label_dir, filename), destination_label_dir)
