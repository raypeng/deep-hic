import sys
assert len(sys.argv) == 3, 'model, chr'

model_file = sys.argv[1]
chromosome = int(sys.argv[2])
start_pos = 27000000
end_pos = 32000000

import numpy as np
np.set_printoptions(suppress=True) # Suppress scientific notation when printing small
import h5py
import scipy.io
from datetime import datetime

import build_model
import util

# Keras imports
from keras.layers import Input, Convolution1D, MaxPooling1D, Merge, Dropout, Flatten, Dense, BatchNormalization, LSTM, Activation
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, ReduceLROnPlateau
from keras.models import Sequential
from seq2seq.layers.bidirectional import Bidirectional

# training parameters
batch_size = 50
n_samples = 100000
# n_samples = 321812
res = 5000
mode_str = 'linear'

# # Load data and split into training and validation sets
data_path = 'data/chr{0}_{1}k_kr_mat_{2}_{3}x.h5'.format(chromosome, res / 1000, start_pos, end_pos)
print 'Loading data from ' + data_path

print 'Building model...'
model = build_model.build_model_v1()

# Print a summary of the model
model.summary()

X1, X2, dist, y, indices = util.load_hdf5_hg19(data_path)
y = np.log(y)

train_data_path = 'data/chr{0}_{1}k_kr_pairs_{2}_{3}.h5'.format(chromosome, res / 1000, n_samples, mode_str)
_, _, _, _, train_indices = util.load_hdf5_hg19(train_data_path)
train_indices = [tuple(arr) for arr in train_indices]
print 'train_indices', len(train_indices), train_indices[:3]

model.load_weights(model_file)

print 'num values to predict:', len(dist)
y_pred = model.predict([X1, X2, dist], batch_size=batch_size).ravel()
y = y.ravel()

d = (end_pos - start_pos) / res
pred_mat = np.zeros((d,d))
true_mat = np.zeros((d,d))
count = 0
for i in range(y.shape[0]):
    x1_pos, x2_pos = indices[i]
    if (x1_pos, x2_pos) in train_indices:
        count += 1
    else:
        pred_mat[(x1_pos - start_pos) / res][(x2_pos - start_pos) / res] = y_pred[i]
        pred_mat[(x2_pos - start_pos) / res][(x1_pos - start_pos) / res] = y_pred[i]
        true_mat[(x1_pos - start_pos) / res][(x2_pos - start_pos) / res] = y[i]
        true_mat[(x2_pos - start_pos) / res][(x1_pos - start_pos) / res] = y[i]
print 'fraction in training data', count, '/', y.shape[0]

np.save('matgen/chr{0}_{1}k_{2}{3}_{4}_pred_mat'.format(chromosome, res / 1000, mode_str, start_pos, end_pos), pred_mat)
np.save('matgen/chr{0}_{1}k_{2}{3}_{4}_true_mat'.format(chromosome, res / 1000, mode_str, start_pos, end_pos), true_mat)
print 'mat saved'
