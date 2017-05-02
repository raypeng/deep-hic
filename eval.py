import sys

assert len(sys.argv) == 2, 'weight_file'

from sklearn.metrics import r2_score, mean_squared_error

import numpy as np
np.set_printoptions(suppress=True) # Suppress scientific notation when printing small
import h5py
import scipy.io
from datetime import datetime
import load_data_pairs as ld # my own scripts for loading data

import build_model

import util
# np.random.seed(1337) # for reproducibility

# Keras imports
from keras.layers import Input, Convolution1D, MaxPooling1D, Merge, Dropout, Flatten, Dense, BatchNormalization, LSTM, Activation
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, ReduceLROnPlateau
from keras.models import Sequential
from seq2seq.layers.bidirectional import Bidirectional

batch_size = 50
training_frac = 0.9 # use 90% of data for training, 10% for testing/validation
t = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
lr = 1e-5
# opt = RMSprop(lr=lr)
opt = Adam(lr=lr)

chromosome = 22
# n_samples = 321812
n_samples = 100000
resolution = 5000
mode_str = 'linear'
print chromosome, n_samples, resolution, mode_str

# Load data and split into training and validation sets
data_path = 'data/chr{0}_{1}k_kr_pairs_{2}_{3}.h5'.format(chromosome, resolution / 1000, n_samples, mode_str)
print 'Loading data from ' + data_path

X1_train, X2_train, dist_train, y_train, _ = ld.load_hdf5_hg19(data_path)
# y_train = np.clip(y_train, 0, 100)
y_train = np.log(y_train)
X1_train, X2_train, dist_train, y_train, X1_val, X2_val, dist_val, y_val = util.split_train_and_val_data_hg19(X1_train, X2_train, dist_train, y_train, training_frac)

X1_length = X1_train.shape[1]
X2_length = X2_train.shape[1]

print 'Building model...'
model = build_model.build_model_v1()

model.load_weights(sys.argv[1].strip())

print 'Data sizes: '
print 'X1_train, X2_train, dist_train:', [np.shape(X1_train), np.shape(X2_train), np.shape(dist_train)]
print 'y_train:', np.shape(y_train)
print 'y_train.mean():', y_train.mean()

y_pred = model.predict([X1_val, X2_val, dist_val], batch_size=batch_size)

print 'median of y_true', np.median(y_val.ravel())
print 'median of y_pred', np.median(y_pred.ravel())

print 'r2 score', r2_score(y_val, y_pred + delta)
print 'mse     ', mean_squared_error(y_val, y_pred)

np.save('chr{0}_5k_{1}y_pred'.format(chromosome, mode_str), y_pred)
np.save('chr{0}_5k_{1}y_true'.format(chromosome, mode_str), y_val)
np.save('chr{0}_5k_{1}dist'.format(chromosome, mode_str), dist_val)
