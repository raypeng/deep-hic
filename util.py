import h5py
import cPickle
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


def split_train_and_val_data_hg19_fat(X1_train, X2_train, X3_train, dist_train, y_train, training_frac):
    n_train = int(training_frac * np.shape(y_train)[0]) # number of training samples

    X1_val = X1_train[n_train:, :]
    X1_train = X1_train[:n_train, :]

    X2_val = X2_train[n_train:, :]
    X2_train = X2_train[:n_train, :]

    X3_val = X3_train[n_train:, :]
    X3_train = X3_train[:n_train, :]

    dist_val = dist_train[n_train:, :]
    dist_train = dist_train[:n_train, :]

    y_val = y_train[n_train:]
    y_train = y_train[:n_train]

    return X1_train, X2_train, X3_train, dist_train, y_train, X1_val, X2_val, X3_val, dist_val, y_val


def load_hdf5_hg19(path, downsample=False):
    with h5py.File(path,'r') as hf:
        x1 = np.array(hf.get('b1'))
        x2 = np.array(hf.get('b2'))
        dist = np.array(hf.get('dist'))
        y = np.array(hf.get('val'))
        indices = list(hf.get('indices'))
        print 'histogram of dist'
        print np.histogram(dist)
        print 'histogram of y'
        print np.histogram(y)
        print 'histogram of log(y)'
        print np.histogram(np.log(y))
        assert len(indices) == y.shape[0]
        if downsample:
            return x1[::2], x2[::2], dist[::2], y[::2], indices[::2]
        else:
            return x1, x2, dist, y, indices

def load_hdf5_hg19_fat(path, downsample=False):
    with h5py.File(path,'r') as hf:
        x1 = np.array(hf.get('b1'))
        x2 = np.array(hf.get('b2'))
        x3 = np.array(hf.get('b3'))
        dist = np.array(hf.get('dist'))
        y = np.array(hf.get('val'))
        indices = list(hf.get('indices'))
        print 'histogram of dist'
        print np.histogram(dist)
        print 'histogram of y'
        print np.histogram(y)
        print 'histogram of log(y)'
        print np.histogram(np.log(y))
        assert len(indices) == y.shape[0]
        if downsample:
            return x1[::2], x2[::2], x3[::2], dist[::2], y[::2], indices[::2]
        else:
            return x1, x2, x3, dist, y, indices

def load_ctcf_counts(ctcf_file, indices, res=5000):
    prefix_sum = cPickle.load(open(ctcf_file, 'rb'))
    n = len(indices)
    counts = np.zeros((n, 1))
    for i in range(n):
        a, b = indices[i]
        try:
            count = prefix_sum[b - res] - prefix_sum[a]
        except:
            print i, a, b
            count = 0
        counts[i] = max(count, 0)
    print 'histogram of ctcf counts'
    print np.histogram(counts.ravel())
    return counts
