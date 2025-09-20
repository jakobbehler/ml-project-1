import os
import json
import numpy as np


# -------------------------------------------------------------------------------------
#  Functions for operations like filtering, normalization and stuff
# -------------------------------------------------------------------------------------

def standardize_selected_columns(x, mask=None):
    """
    
    Standardize the input data feature-wise. A mask can be defined so only the selected features are standardized. 

    Args:
        x: numpy array of shape (num_samples, num_features)
        mask: optional list/array of column indices to normalize.
              If None, all columns are normalized.

    Returns:
        standardized data, shape (num_samples, num_features)

    Example:
    >>> standardize(np.array([[1, 2], [3, 4], [5, 6]]))
    array([[-1.22474487, -1.22474487],
           [ 0.        ,  0.        ],
           [ 1.22474487,  1.22474487]])

    >>> standardize(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), mask=[0, 2])
    array([[-1.22474487,  5.        , -1.22474487],
           [ 0.        ,  8.        ,  0.        ],
           [ 1.22474487, 11.        ,  1.22474487]])
    """
    Z = x.copy().astype(float)

    # If no mask given, normalize all columns
    if mask is None:
        mask = range(Z.shape[1])

    for j in mask:
        col = Z[:, j]
        mean = np.mean(col)
        std = np.std(col)
        if std != 0:
            Z[:, j] = (col - mean) / std

    return Z

# -------------------------------------------------------------------------------------
#  Functions for telling us what feature is categorical, continuous andd so on. 
# -------------------------------------------------------------------------------------

def load_feature_classes(path):
    with open(path, 'r') as f:
        return json.load(f)

def load_feature_names(path):
    with open(path, 'r') as f:
        # Read all lines (column-wise CSV)
        return [line.strip().strip('"') for line in f if line.strip()]

def build_feature_dictionary(classes_path, feature_names_path):
    feature_classes = load_feature_classes(classes_path)
    feature_names = load_feature_names(feature_names_path)

    # Initialize empty index lists for each category
    index_dict = {k: [] for k in feature_classes}

    # Convert each group to a set for faster lookup
    class_sets = {k: set(v) for k, v in feature_classes.items()}

    # Iterate once through feature_names
    for idx, name in enumerate(feature_names):
        for category, feature_set in class_sets.items():
            if name in feature_set:
                index_dict[category].append(idx)

    # Combine names + indices into the final structure
    result = {
        category: {
            "names": [feature_names[i] for i in indices],
            "indices": indices
        }
        for category, indices in index_dict.items()
    }

    return result
