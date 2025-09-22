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

### ORDINAL FEATURES

def clean_ordinal_feature(col):
    """Takes a column of ordinal data and replace values after a gap with NaN. 
        Example: a feature has a scale 1,2,3,4,7 where 7 is "no answer" then 7 is replaced with Null"""

    unique_vals = np.unique(col)
    diffs = np.diff(unique_vals)
    
    # Find first "gap" larger than 1
    gap_idx = np.where(diffs > 1)[0]
    if len(gap_idx) == 0:
        return col  # no gap, return as-is
    
    cutoff = unique_vals[gap_idx[0] + 1]  # first value after the gap
    cleaned = col.astype(float).copy()
    cleaned[col >= cutoff] = np.nan
    return cleaned


### CONTINUOUS FEATURES

def clean_continuous_feature(col, rules):
    """
    Clean a single feature column according to rules.
    Rules can contain:
      - 'replace': dict of value -> new_value
      - 'remove': list of values to set as empty string
      - 'ranges': dict like {"101-199": "lambda v: (v - 100) * 30"}
    """
    
    col = col.astype(object).copy()

    # Apply replacements
    if "replace" in rules:
        for old, new in rules["replace"].items():
            col[col == old] = new

    # Apply removals
    if "remove" in rules:
        for bad in rules["remove"]:
            col[col == bad] = ""

    # Apply range transformations
    if "ranges" in rules:
        for range_key, func_str in rules["ranges"].items():
            start, end = map(int, range_key.split("-"))
            func = eval(func_str)

            # mask: not empty, not nan, within range
            safe_mask = []
            for v in col:
                if v == "" or v is None:
                    safe_mask.append(False)
                else:
                    try:
                        iv = int(v)
                        safe_mask.append(start <= iv <= end)
                    except (ValueError, TypeError):
                        safe_mask.append(False)
            safe_mask = np.array(safe_mask, dtype=bool)

            col[safe_mask] = [func(int(v)) for v in col[safe_mask]]

    return col

def apply_cleaning_continuous_features(X, feature_classes_dictionary, cleaning_rules):
    """
    Apply cleaning rules to dataset X.
    - X: numpy array (N, d)
    - feature_classes_dictionary: dict from ph.build_feature_dictionary
    - cleaning_rules: dict with cleaning instructions
    Returns: cleaned dataset (dtype=object so we can store empty cells)
    """
    X_cleaned = X.astype(object).copy()

    for feat, rules in cleaning_rules.items():
        # Find feature in dictionary
        for group in feature_classes_dictionary.values():
            if feat in group["names"]:
                idx = group["indices"][group["names"].index(feat)]
                col = X_cleaned[:, idx]
                X_cleaned[:, idx] = clean_continuous_feature(col, rules)
                break  # stop searching once found

    return X_cleaned


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

