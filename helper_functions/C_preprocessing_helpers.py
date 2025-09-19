import os
import json
import numpy as np

# -----------------
#  Functions for telling us what feature is categorical, continuous andd so on. 
# -----------------



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
