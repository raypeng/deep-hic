import numpy as np


# Splits the data into training and validation data, keeping training_frac of
# the input samples in the training set and the rest for validation
def split_train_and_val_data_hg19(X1_train, X2_train, dist_train, y_train, training_frac):

    n_train = int(training_frac * np.shape(y_train)[0]) # number of training samples

    X1_val = X1_train[n_train:, :]
    X1_train = X1_train[:n_train, :]

    X2_val = X2_train[n_train:, :]
    X2_train = X2_train[:n_train, :]

    dist_val = dist_train[n_train:, :]
    dist_train = dist_train[:n_train, :]

    y_val = y_train[n_train:]
    y_train = y_train[:n_train]

    return X1_train, X2_train, dist_train, y_train, X1_val, X2_val, dist_val, y_val
