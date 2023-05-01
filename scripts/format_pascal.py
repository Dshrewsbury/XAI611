import os
import numpy as np
import argparse

NUM_CATEGORIES = 20
CATEGORY_NAMES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
default_load_path = os.path.join(parent_directory, 'data')
parser = argparse.ArgumentParser(description='Format PASCAL 2012 metadata.')
parser.add_argument('--load-path', type=str, default=default_load_path,
                    help='Path to a directory containing a copy of the PASCAL dataset.')
parser.add_argument('--save-path', type=str, default=default_load_path, help='Path to output directory.')
args = parser.parse_args()

# Define dictionaries to map category names to category IDs and vice versa
name_to_id = {name: idx for idx, name in enumerate(CATEGORY_NAMES)}
id_to_name = {idx: name for idx, name in enumerate(CATEGORY_NAMES)}


def process_phase(phase, load_path, save_path):
    # Initialize an empty dictionary for storing annotations and an image list
    annotations = {}
    image_filenames = []

    # Iterate through categories and read annotation files
    for category in name_to_id:
        with open(os.path.join(load_path, 'VOCdevkit', 'VOC2012', 'ImageSets', 'Main', category + '_' + phase + '.txt'),
                  'r') as file:
            for line in file:
                # Extract image ID and label from the current line in the annotation file
                current_line = line.rstrip().split(' ')
                image_id = current_line[0]
                label = current_line[-1]
                image_filename = image_id + '.jpg'

                # If the label indicates the presence of the category in the image, update annotations and image_filenames accordingly
                if int(label) == 1:
                    if image_filename not in annotations:
                        annotations[image_filename] = []
                        image_filenames.append(image_filename)
                    annotations[image_filename].append(name_to_id[category])

    # Sort the image list
    image_filenames.sort()

    # Initialize the label matrix with zeros, with dimensions: number of images x number of categories
    num_images = len(image_filenames)
    label_matrix = np.zeros((num_images, NUM_CATEGORIES))

    # Iterate over the image list to populate the label matrix with 1.0 at the label indices
    for i in range(num_images):
        current_image = image_filenames[i]
        label_indices = np.array(annotations[current_image])
        label_matrix[i, label_indices] = 1.0

    # Save the label matrix and image list for each phase as .npy files in the specified output directory
    np.save(os.path.join(save_path, 'processed_' + phase + '_labels.npy'), label_matrix)
    np.save(os.path.join(save_path, 'processed_' + phase + '_images.npy'), np.array(image_filenames))


# Process both train and validation phases
for phase in ['train', 'val']:
    process_phase(phase, args.load_path, args.save_path)
