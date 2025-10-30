from data.feature_properties.cleaning_rules_continuous import cleaning_rules
import helper_functions.C_preprocessing_helpers as ph
import helper_functions.B_regression_helpers as regressionHelpers
import implementations as regression
import helper_functions.E_scoring_functions as scoring
import numpy as np
from helper_functions.A_loading_helpers import load_csv_data, load_numpy, create_csv_submission
import os

# CONFIGS
np.random.seed(42)
procesed_data_path = 'data/processed/'
x_train_path = procesed_data_path + 'x_train_processed_corrfilter_var.npy'
x_test_path = procesed_data_path +'x_test_processed_corrfilter_var.npy'
y_train_path = procesed_data_path + 'y_train.npy'
ids_path = 'data/test_ids.npy'
output_folder = 'data/test_results/'

model = 'regularized-logistic'
data_correlation_filter = 'none'

# HYPERPARAMS

gamma = 0.4
max_iters = 100
decision_boundary = 0.15


def main():
    
    # load procesed data
    x_training_processed, y_training, x_test, y_training, D, ids = load_data()

    # train model
    if model == 'regularized-logistic':
        w = run_logistic_regression(x_training_processed, y_training, D)

    # apply to test data & convert from 0, 1 to -1, 1
    y_pred = regressionHelpers.sigmoid(x_test @ w)
    y_pred = np.where(y_pred > decision_boundary, 1, -1)

    # format to submission format csv
    filename = output_folder + 'y_submission_test'
    ids = np.arange(0, len(y_pred))
    create_csv_submission(ids, y_pred, filename)


def load_data():
    # load procesed data
    x_training_processed = load_numpy(x_train_path)
    y_training = load_numpy(y_train_path)
    x_test = load_numpy(x_test_path)
    y_training = (y_training + 1) / 2

    ids = load_numpy(ids_path)
    N, D = x_training_processed.shape
    return x_training_processed, y_training, x_test, y_training, D, ids
    
def run_logistic_regression(x_training_processed, y_training, D):
    initial_w = np.random.randn(D)
    w, loss = regression.logistic_regression(y_training, x_training_processed, initial_w, max_iters, gamma)
    return w


if __name__ == "__main__":
    main()
