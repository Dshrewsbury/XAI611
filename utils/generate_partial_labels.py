import numpy as np


def get_random_label_indices(labels, target_value, num_samples, rng):
    idx = np.where(labels == target_value)[0]
    return rng.choice(idx, num_samples, replace=False)


def generate_partial_labels(label_matrix, num_pos, num_neg, rng):
    '''
    label_matrix: binary (-1/+1) label matrix with shape num_items x num_classes
    num_pos: number of positive labels to observe for each item
    num_neg: number of negative labes to observe for each item
    rng: random number generator to use
    '''

    # check the observation parameters:
    assert (num_pos == -1) or (num_pos >= 0)
    assert (num_neg == -1) or (num_neg >= 0)

    # check that label_matrix is a binary numpy array:
    assert type(label_matrix) is np.ndarray
    label_values = np.unique(label_matrix)
    assert len(label_values) == 2
    assert 0 in label_values
    assert 1 in label_values
    assert len(np.unique(label_matrix)) == 2

    # apply uniform observation process:
    num_items, num_classes = np.shape(label_matrix)
    label_matrix_obs = np.zeros_like(label_matrix)
    for i in range(num_items):
        idx_pos = get_random_label_indices(label_matrix[i, :], 1.0, num_pos, rng)
        label_matrix_obs[i, idx_pos] = 1.0
        idx_neg = get_random_label_indices(label_matrix[i, :], 0.0, num_neg, rng)
        label_matrix_obs[i, idx_neg] = 0.0

    return label_matrix_obs