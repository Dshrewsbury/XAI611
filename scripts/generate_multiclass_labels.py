import numpy as np
import os


def convert_to_multiclass(label_matrix, rng):
    '''
    Convert the multi-label label matrix with 1s and 0s to a multi-class label matrix with only a single
    positive 1 per instance and the rest equal to 0. The single positive label is chosen at random.

    label_matrix: binary (0/1) label matrix with shape num_items x num_classes
    rng: random number generator to use
    '''
    num_items, num_classes = np.shape(label_matrix)
    multiclass_matrix = np.zeros_like(label_matrix)

    for i in range(num_items):
        positive_indices = np.nonzero(label_matrix[i, :] == 1)[0]
        chosen_positive_idx = rng.choice(positive_indices)
        multiclass_matrix[i, chosen_positive_idx] = 1

    return multiclass_matrix


def load_label_matrix(phase, load_path):
    file_path = os.path.join(load_path, f'processed_{phase}_labels.npy')
    label_matrix = np.load(file_path)
    return label_matrix

def save_label_matrix(observed_label_matrix, phase, save_path):
    file_path = os.path.join(save_path, f'partial_{phase}_labels.npy')
    np.save(file_path, observed_label_matrix)


if __name__ == "__main__":
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_load_path = os.path.join(parent_dir, 'data')

    for phase in ['train', 'val']:
        # Load ground truth binary label matrix
        label_matrix = load_label_matrix(phase, default_load_path)
        assert np.max(label_matrix) == 1
        assert np.min(label_matrix) == 0

        rng = np.random.RandomState(1200)
        multiclass_matrix = convert_to_multiclass(label_matrix, rng)

        # Save observed labels
        save_label_matrix(multiclass_matrix, phase, default_load_path)
