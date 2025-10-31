from data.feature_properties.cleaning_rules_continuous import cleaning_rules
import helper_functions.C_preprocessing_helpers as ph
import helper_functions.B_regression_helpers as regressionHelpers
import implementations as regression
import helper_functions.E_scoring_functions as scoring
import numpy as np
from helper_functions.A_loading_helpers import (
    load_csv_data,
    load_numpy,
    create_csv_submission,
)
import os

# CONFIGS
np.random.seed(42)
procesed_data_path = "data/processed/"
x_train_path = procesed_data_path + "x_train_processed_corrfilter_var.npy"
x_test_path = procesed_data_path + "x_test_processed_corrfilter_var.npy"
y_train_path = procesed_data_path + "y_train.npy"
ids_path = "data/test_ids.npy"
output_folder = "data/test_results/"

model = "ridge-regression"
data_correlation_filter = "none"

# HYPERPARAMS

downsampling_ratio = 2.5
lambda_ = 1e-4


def main():

    # load procesed data
    x_training_processed, y_training, x_test, D, ids = load_data()

    # downsample negative class
    x_training_processed_down, y_training_down = ph.downsample(
        X_train=x_training_processed, y_train=y_training, num=downsampling_ratio
    )

    # train model
    if model == "ridge-regression":
        w = run_ridge_regression(x_training_processed_down, y_training_down, lambda_)

    # apply to test data
    y_pred = np.sign(x_test @ w)

    # format to submission format csv
    filename = (
        output_folder + "y_submission_ridge_regression_down2_5_lamda1e-4_corrfilterVar"
    )
    create_csv_submission(ids, y_pred, filename)


def load_data():

    # processed data was processed using the preprocesing pipeline function 
    # in C_preprocessing_helpers.py

    # load procesed data
    x_training_processed = load_numpy(x_train_path)

    # y_training = load_numpy(y_train_path)[:, 1]
    y_training = load_numpy(y_train_path)
    x_test = load_numpy(x_test_path)
    # y_training = (y_training + 1) / 2

    ids = load_numpy(ids_path)
    N, D = x_training_processed.shape
    return x_training_processed, y_training, x_test, D, ids


def run_ridge_regression(x_training_processed, y_training, lambda_):
    w, loss = regression.ridge_regression(y_training, x_training_processed, lambda_)
    return w


if __name__ == "__main__":
    main()
