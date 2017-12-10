import sys
import build_model

# assert len(sys.argv) == 3, 'train_data_path, model_file'
# train_data_path = sys.argv[1]
# model_file = sys.argv[2]

train_data_path = 'data/chr22_5k_kr_pairs_50000_banded-fat_40000000-45000000.h5'
# train_data_path = 'data/chr22_5k_kr_pairs_300000_banded-mid_25000000-50000000.h5'
assert len(sys.argv) == 3, 'model_file, arch_ver'

model_file = sys.argv[1]
arch_ver = sys.argv[2]

model_ver = int(model_file.split('/')[-1].split('.')[0])
build_model_str = 'build_model.build_model_mid_v' + arch_ver + '(use_ctcf=True)'
model = eval(build_model_str)
print 'running model', model_ver, 'with model', build_model_str

train_chr = 22
test_chr = 22

# test_chr = 20
test_chr = 21

# visualization area
# start_pos = 27000000
# end_pos = 32000000
# start_pos = 25000000
# end_pos = 30000000
start_pos = 40000000
end_pos = 45000000
# start_pos = 45000000
# end_pos = 50000000
start_pos = 35000000
end_pos = 40000000
# start_pos = 20000000
# end_pos = 25000000
# start_pos = 10000000
# end_pos = 15000000
# start_pos = 25000000
# end_pos = 35000000


# training parameters
batch_size = 256 # 128 # 64 # 20
res = 5000

import numpy as np
np.set_printoptions(suppress=True) # Suppress scientific notation when printing small
import h5py
import scipy.io
from datetime import datetime

import util

# if test_chr == train_chr, exclude training pairs from vis
print 'train_chr', train_chr
print 'test_chr', test_chr
if train_chr == test_chr:
    _, _, _, _, train_indices = util.load_hdf5_hg19(train_data_path)
    train_indices = set([tuple(arr) for arr in train_indices])
    # print 'train_indices', len(train_indices), train_indices[:3]
else:
    print 'predicting on different chr'
    train_indices = []    

# Keras imports
from keras.layers import Input, Convolution1D, MaxPooling1D, Merge, Dropout, Flatten, Dense, BatchNormalization, LSTM, Activation
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, ReduceLROnPlateau
from keras.models import Sequential
from seq2seq.layers.bidirectional import Bidirectional

print 'Building model...'
# model = build_model.build_model_v1()
# model = build_model.build_model_v2(use_ctcf=use_ctcf)
# model = build_model.build_model_v3()
# model = build_model.build_model_v4()
# model = build_model.build_model_v5()
# model = build_model.build_model_v52()

# Print a summary of the model
model.summary()

# Load data and split into training and validation sets
data_path = 'data/chr{0}_{1}k_kr_mat-mid_{2}_{3}.h5'.format(test_chr, res / 1000, start_pos, end_pos)
print 'Loading data from ' + data_path

X1, X2, X3, dist, y, indices = util.load_hdf5_hg19_fat(data_path)
y = np.log(y)

ctcf_path = 'data/fimo-output/chr{}-5000-ctcf/psum.pkl'.format(test_chr)
ctcf = util.load_ctcf_counts(ctcf_path, indices)

def get_random_x3(x3):
    mat = np.zeros(x3.shape)
    n_sample, n_dim = x3.shape[0], x3.shape[1]
    seq = (np.random.random((n_sample, n_dim))/ 0.25).astype(int)
    for sample_idx in range(n_sample):
        for dim_idx in range(n_dim):
            ch = seq[sample_idx, dim_idx]
            mat[sample_idx, dim_idx, ch] = 1
    print np.histogram(seq.ravel())
    print mat.ravel().sum(), n_sample * n_dim
    return mat

def get_avg(x):
    avg = x.ravel().mean()
    mat = np.ones(x.shape) * avg
    print mat.ravel().mean()
    return mat

inspect_mode = ''

# X3 = get_random_x3(X3)
# dist = get_avg(dist)
# ctcf = get_avg(ctcf)
# inspect_mode = '_random-x3'

model.load_weights(model_file)

num_predict = y.shape[0]
print 'num values to predict:', num_predict
y_pred = model.predict([X1, X3, X2, dist, ctcf], batch_size=batch_size).ravel()
y = y.ravel()

d = (end_pos - start_pos) / res
pred_mat = np.zeros((d,d))
true_mat = np.zeros((d,d))
count = 0
for i in range(y.shape[0]):
    if i % (num_predict / 10) == 0:
        print 'progress\t', i / (num_predict / 10) * 10, '%'
    x1_pos, x2_pos = indices[i]
    if (x1_pos, x2_pos) in train_indices:
        count += 1
    else:
        pred_mat[(x1_pos - start_pos) / res][(x2_pos - start_pos) / res] = y_pred[i]
        pred_mat[(x2_pos - start_pos) / res][(x1_pos - start_pos) / res] = y_pred[i]
        true_mat[(x1_pos - start_pos) / res][(x2_pos - start_pos) / res] = y[i]
        true_mat[(x2_pos - start_pos) / res][(x1_pos - start_pos) / res] = y[i]
print 'fraction in training data', count, '/', y.shape[0]

np.save('matrices/m{0}_testchr{1}{2}_{3}_{4}_pred_mat'.format(model_ver, test_chr, inspect_mode, start_pos, end_pos), pred_mat)
np.save('matrices/m{0}_testchr{1}{2}_{3}_{4}_true_mat'.format(model_ver, test_chr, inspect_mode, start_pos, end_pos), true_mat)
print 'mat saved'
