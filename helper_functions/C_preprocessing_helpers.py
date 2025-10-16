import os
import json
import numpy as np
import csv

# ----------------------------------------------------------------------------------------
# Full Pipeline Functions
# ---------------------------------------------------------------------------------------

def preprocessing_pipeline(
    X,
    feature_names_path,
    classes_path,
    cleaning_rules,
    drop_first=True,
    drop_correlated_mode = 'nothing'
):
    """
    Full preprocessing pipeline:
    - Load metadata (like feature categories)
    - Clean ordinal features
    - Clean continuous features
    - (todo) impute missing values
    - (todo) normalize continuous features
    - Drop unimportant / empty columns
    -Drop highly correlated columns from X based on one of four modes: 'nothing', 'first', 'missing', or 'var'.
    - One-hot encode categorical columns
    
    Returns
    -------
    X_out : np.ndarray
        Transformed feature matrix
    feature_names_out : list[str]
        Updated feature names list in the correct order / indices corresponding to the columns in X_out
    """
    # 1. Load metadata----------------------------------------------------------------------------------------
    
    feature_names = csv_to_list_1D(feature_names_path)
    feature_dict = build_feature_dictionary(classes_path, feature_names)
    correlation_data_path = 'data/correlation_data'

    # 2. Clean ordinal ----------------------------------------------------------------------------------------
    
    print("Cleaning Ordinal Features")
    for j in feature_dict['ordinal']['indices']:
        X[:, j] = clean_ordinal_feature(X[:, j])
    print("Done.")
    print("")

    # 3. Clean continuous----------------------------------------------------------------------------------------
   
    print("Cleaning Continuous Features")
    X = apply_cleaning_continuous_features(X, feature_dict, cleaning_rules)
    print("Done.")
    print("")


    # 4. impute missing values----------------------------------------------------------------------------------------
    # TODO: implement imputation
    X, cols_to_drop = impute_missing_values(X, feature_dict)
    X, feature_names = drop_columns_by_index(X, feature_names, cols_to_drop)
    feature_dict = build_feature_dictionary(classes_path, feature_names)

    
    # 5. normalize continuous----------------------------------------------------------------------------------------
   
    # can only be done if imputation is done

    cont_indices = (
        feature_dict['continuous']['indices'] +
        feature_dict['continuous_but_null_also_a_number']['indices']
    )
    X = standardize_selected_columns(X, mask=cont_indices)


    # 6. Drop unimportant columns----------------------------------------------------------------------------------------

    print("Dropping Invalid Features")
    X, feature_names, feature_dict = remove_unimportant_features(
        X, feature_names, classes_path
    )
    print("Done.")
    print("")

     # 7. Drop correlated columns----------------------------------------------------------------------------------------
    
    X, feature_names, feature_dict = drop_correlated_features(
        X, feature_names, feature_dict, classes_path, correlation_data_path, mode=drop_correlated_mode
    )

    # 8. Drop invalid columns----------------------------------------------------------------------------------------
    
    print("Dropping Invalid Features")
    X, feature_names, feature_dict = remove_unimportant_features(
        X, feature_names, classes_path
    )
    print("Done.")
    print("")

    # 9 . One-hot encode categoricals ---------------------------------------------------------------------------------------
    print("One Hot Encoding Categorical Features")
    print()
    print("This might take a while...")
    print()


    X, feature_names = one_hot_all(X, feature_names)
    feature_dict = build_feature_dictionary(classes_path, feature_names)
    print("OMG finally. One hot encoding is done.")
    print()


    print("All preprocessing is done.")
    print("--------------------------------")
    return X, feature_names, feature_dict


# -------------------------------------------------------------------------------------
#  Functions for operations like filtering, normalization and stuff
# -------------------------------------------------------------------------------------

def standardize_selected_columns(x, mask=None):
    """
    
    Standardize the input data feature-wise. A mask can be defined so only the selected features are standardized, like the continuous features. 

    Args:
        x: numpy array of shape (num_samples, num_features)
        mask: optional list/array of column indices to normalize.
              If None, all columns are normalized.

    Returns:
        standardized data, shape (num_samples, num_features)

    Example:
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

## Dropping Columns

def drop_columns(X, feature_names, cols_to_drop):
    '''
    X: raw numpy array training data
    feature_names: old list of feature names before removal
    cols_to_drop: liust of strings of features to drop
    '''
    # keep only those that still exist

    valid_cols_to_drop = [c for c in cols_to_drop if c in feature_names]

    # find indices of columns to drop
    drop_idx = [feature_names.index(c) for c in cols_to_drop]

    # drop from X
    X_new = np.delete(X, drop_idx, axis=1)

    # update feature names
    feature_names_new = [c for c in feature_names if c not in cols_to_drop]

    return X_new, feature_names_new

## Pipeline for dropping columns from class 'not_displayed_or_unrelated' and keeping dictionary of feature indices updated

def remove_unimportant_features(X, feature_names, classes_path):
    """
    Drop 'not_displayed_or_unrelated' features from X and update metadata.
    """
    # Build dictionary from current state
    feature_dictionary = build_feature_dictionary(classes_path, feature_names)

    # Find what to drop
    drop_set_unrelated = feature_dictionary["not_displayed_or_unrelated"]["names"]

    # Drop from BOTH X and feature_names
    X_new, feature_names_new = drop_columns(X, feature_names, drop_set_unrelated)

    # Rebuild dictionary from updated names
    feature_dictionary_new = build_feature_dictionary(classes_path, feature_names_new)

    return X_new, feature_names_new, feature_dictionary_new


## CATEGORICAL

def one_hot_column(X, feature_names, colname, drop_first=False):
    """
    One-hot encode a single column by name.
    """
    # 1. Find index
    idx = feature_names.index(colname)
    col = X[:, idx].astype(float)

    # 2. Unique categories (ignore NaN for now) 
    categories = np.unique(col[~np.isnan(col)])
    val_to_idx = {val: i for i, val in enumerate(categories)}
    mapped = np.array([val_to_idx[v] if not np.isnan(v) else -1 for v in col])

    # 3. One-hot encode
    one_hot = np.eye(len(categories))[mapped.clip(min=0)]
    if drop_first:
        one_hot = one_hot[:, 1:]
        categories = categories[1:]

    # 4. Update X
    X_left  = X[:, :idx]
    X_right = X[:, idx+1:]
    X_new = np.hstack([X_left, one_hot, X_right])

    # 5. Update feature_names
    new_names = [f"{colname}_{int(c)}" for c in categories]
    feature_names_new = feature_names[:idx] + new_names + feature_names[idx+1:]

    return X_new, feature_names_new

# runs for 4 minutes on my machine because I write bad algorithms
def one_hot_all(x, feature_names):

    classes_path = "data/feature_properties/feature_classes.json"
    feature_dictionary = build_feature_dictionary(classes_path, feature_names)

    categorical_colnames = feature_dictionary['categorical']['names']

    for colname in categorical_colnames:
        x_new, feature_names_new = one_hot_column(x, feature_names, colname, drop_first=True)
        feature_names = feature_names_new
        x = x_new
    
    return x, feature_names


# remove correlated

def drop_correlated_features(X, feature_names, feature_dict, classes_path, correlation_data_path, mode='nothing'):
    """
    Drop features that are too correlated based on precomputed selection files.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    feature_names : list[str]
        Current feature names (aligned with X)
    feature_dict : dict
        Dictionary of feature categories and indices
    classes_path : str
        Path to feature_classes.json
    correlation_data_path : str
        Directory containing correlation-based feature drop lists
    mode : str
        One of ['nothing', 'first', 'missing', 'var']

    Returns
    -------
    X_new : np.ndarray
    feature_names_new : list[str]
    feature_dict_new : dict
    """

    if mode == 'nothing':
        print("Skipping correlated feature dropping (mode='nothing').\n")
        return X, feature_names, feature_dict

    # Choose file based on mode
    file_map = {
        'first': 'features_to_delete_first.csv',
        'missing': 'features_to_delete_missing.csv',
        'var': 'features_to_delete_var.csv'
    }

    filename = file_map.get(mode)
    path = os.path.join(correlation_data_path, filename)

    # Load features to drop
    if not os.path.exists(path):
        print(f"Warning: Correlation file not found at {path}. No features dropped.\n")
        return X, feature_names, feature_dict

   
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        features_to_drop = [row['feature_name'].strip() for row in reader if row.get('feature_name')]

    if not features_to_drop:
        print("No correlated features to drop.\n")
        return X, feature_names, feature_dict

    # Filter out features that no longer exist
    features_to_drop_filtered = [f for f in features_to_drop if f in feature_names]

    if not features_to_drop_filtered:
        print("All correlated features already removed earlier — nothing to drop.\n")
        return X, feature_names, feature_dict
    
    print(f"Dropping {len(features_to_drop)} highly correlated features ({mode} mode)...")

    X_new, feature_names_new = drop_columns(X, feature_names, features_to_drop_filtered)

    feature_dict_new = build_feature_dictionary(classes_path, feature_names_new)

    print("Done.\n")
    return X_new, feature_names_new, feature_dict_new



# -------------------------------------------------------------------------------------
#  Functions for telling us what feature is categorical, continuous andd so on. 
# -------------------------------------------------------------------------------------

import csv

def csv_to_list_1D(path):
    """
    extracts a column CSV from a path and turns it into a python list
    """
    with open(path, newline="") as f:
        reader = csv.reader(f)
        list = [row[0] for row in reader]  # list of strings
    return list

def list_1D_to_csv(items, path):
    """
    Saves a 1D Python list as a one-column CSV file.
    """
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        for item in items:
            writer.writerow([item])



def build_feature_dictionary(classes_path: str, feature_names: list[str]):
    """
    Build dictionary of feature classes and indices to keep track of 
    what feature is where in the numpy array X, which does not keep track of labels

    Args:
        classes_path: path to feature_classes.json (static)
        feature_names: current list of feature names (aligned with X)

    Returns:
        dict: { category: {"names": [...], "indices": [...] } }
    """
    # Load static schema
    with open(classes_path, "r") as f:
        feature_classes = json.load(f)

    # Prepare index lists
    index_dict = {k: [] for k in feature_classes}
    class_sets = {k: set(v) for k, v in feature_classes.items()}

    # Match names to categories
    for idx, name in enumerate(feature_names):
        for category, feature_set in class_sets.items():
            if name in feature_set:
                index_dict[category].append(idx)

    # Build result
    return {
        category: {
            "names": [feature_names[i] for i in indices],
            "indices": indices
        }
        for category, indices in index_dict.items()
    }


def drop_feature_names(feature_names: list[str], drop_list: list[str]) -> list[str]:
    """
    Return a new feature_names list with all names in drop_list removed.

    Args:
        feature_names: list of feature name strings (aligned with X columns)
        drop_list: list of feature names to remove

    Returns:
        new list of feature names
    """
    drop_set = set(drop_list)  # faster lookup
    return [name for name in feature_names if name not in drop_set]

def impute_missing_values(X, feature_dict, threshold=0.45):
    """
    Impute missing values in X according to feature types.
    Assumes missing values are represented as np.nan only.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Input dataset (can be dtype=object).
    feature_dict : dict
        Dictionary from build_feature_dictionary() with keys like 'continuous', 'categorical', etc.
    threshold : float, optional
        Proportion threshold to determine "too many" missing values per column.

    Returns
    -------
    X_imputed : np.ndarray
        Dataset with missing values imputed for columns under threshold.
    cols_too_many_nans : list[int]
        List of column indices with missing proportion >= threshold.
    """
    X_imputed = X.astype(object).copy()
    n_samples = X.shape[0]
    cols_too_many_nans = []

    for category in ['continuous', 'categorical', 'ordinal', 'continuous_but_null_also_a_number']:
        for idx in feature_dict[category]['indices']:
            col = X_imputed[:, idx]

            # Identify np.nan values
            nan_mask = np.isnan(col.astype(float))
            nan_ratio = np.sum(nan_mask, axis=0) / n_samples
            n_missing = nan_mask.sum()

            if nan_ratio >= threshold:
                cols_too_many_nans.append(idx)
                continue  # Skip imputation

            # Quantile-based imputation for continuous data
            if category == 'continuous' or category=='continuous_but_null_also_a_number':
                valid_vals = col[~nan_mask]
                sorted_vals = np.sort(valid_vals)

                if valid_vals.size > 0:
                    # Sample uniform quantiles
                    quantiles = np.random.rand(n_missing)
                    # samples values from the empirical distribution of the non-missing values in that column.
                    sampled_vals = np.quantile(sorted_vals, quantiles)
                    X_imputed[nan_mask, idx] = sampled_vals


            elif category == 'categorical' or category=='ordinal':
                # Build empirical distribution from non-nan values
                valid_vals = col[~nan_mask]
                if valid_vals.size > 0:
                    unique_vals, counts = np.unique(valid_vals, return_counts=True)
                    probs = counts / counts.sum()
                    sampled_vals = np.random.choice(unique_vals, size=nan_mask.sum(), p=probs)
                    X_imputed[nan_mask, idx] = sampled_vals

    return X_imputed, set(cols_too_many_nans)

def drop_columns_by_index(X, feature_names, indices_to_drop):
    """
    Remove columns from X and feature_names by index.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Input feature matrix.
    feature_names : list of str
        List of feature names aligned with X columns.
    indices_to_drop : list of int
        Column indices to drop.

    Returns
    -------
    X_new : np.ndarray
        New array with selected columns removed.
    feature_names_new : list of str
        Feature names with corresponding entries removed.
    """
    # Sort and deduplicate indices (important for np.delete)
    indices_to_drop = sorted(set(indices_to_drop))

    # Drop columns from array
    X_new = np.delete(X, indices_to_drop, axis=1)

    # Drop corresponding feature names
    feature_names = [
        name for i, name in enumerate(feature_names) if i not in indices_to_drop
    ]

    return X_new, feature_names

