import sys
import build_model

assert len(sys.argv) == 3, 'model_file, arch_ver'
model_number = sys.argv[1]
arch_ver = sys.argv[2]
build_model_str = 'build_model.build_model_mid_v' + arch_ver + '(use_ctcf=True)'
model = eval(build_model_str)
print 'training model', model_number, 'with model', build_model_str

import numpy as np
import re
import os
import sys
np.set_printoptions(suppress=True) # Suppress scientific notation when printing small
import h5py
import scipy.io
from datetime import datetime

import util

# Keras imports
from keras.layers import Input, Convolution1D, MaxPooling1D, Merge, Dropout, Flatten, Dense, BatchNormalization, LSTM, Activation
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, ReduceLROnPlateau
from keras.models import Sequential
from seq2seq.layers.bidirectional import Bidirectional

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.externals import joblib


# training parameters
num_epochs = 100
batch_size = 128
training_frac = 0.95 # use 90% of data for training, 10% for testing/validation
t = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
lr = 1e-5
# opt = RMSprop(lr=lr)
opt = Adam(lr=lr)

checkpoint_path = 'weights/{}.hdf5'.format(model_number)
assert not os.path.exists(checkpoint_path)
print 'checkpoint', checkpoint_path

# # Load data and split into training and validation sets
data_path = 'data/chr22_5k_kr_pairs_100000_banded-mid-new_40000000-45000000.h5'
# data_path = 'data/chr22_5k_kr_pairs_10000_banded-mid-new_40000000-50000000.h5'
# data_path = 'data/chr22_5k_kr_pairs_100000_banded-mid-new_0-50000000.h5'
# data_path = 'data/chr22_5k_kr_pairs_50000_banded-fat_40000000-45000000.h5'
# data_path = 'data/chr22_5k_kr_pairs_300000_banded-mid_25000000-50000000.h5'
# data_path = 'data/chr22_5k_kr_pairs_300000_banded-mid_0-50000000.h5'
print 'Loading data from ' + data_path

# TODO: Resample 10 times and do cross-validation
X1_train, X2_train, X3_train, dist_train, y_train, indices = util.load_hdf5_hg19_fat(data_path)
res = 5000
# X3_train = X3_train[:,res:res*2,:]
y_train = np.log(y_train)
# y_train = np.clip(y_train, None, 300) / 300.
X1_train, X2_train, X3_train, dist_train, y_train, X1_val, X2_val, X3_val, dist_val, y_val = util.split_train_and_val_data_hg19_fat(X1_train, X2_train, X3_train, dist_train, y_train, training_frac)

ctcf_path = 'data/fimo-output/chr22-5000-ctcf/psum.pkl'
ctcf_data = util.load_ctcf_counts(ctcf_path, indices)
n_train = len(y_train)
ctcf_train, ctcf_val = ctcf_data[:n_train], ctcf_data[n_train:]

X1_length = X1_train.shape[1]
X2_length = X2_train.shape[1]

print 'Compiling model...'
model.compile(loss = 'mean_squared_error',
              optimizer = opt)

# Print a summary of the model
model.summary()

# Define custom callback that prints/plots performance at end of each epoch
class ConfusionMatrix(Callback):
    def on_train_begin(self, logs = {}):
        self.epoch = 0
        self.train_mse = []
        self.val_mse = []

    def on_epoch_end(self, batch, logs = {}):
        self.train_mse.append(logs.get('loss'))
        self.val_mse.append(logs.get('val_loss'))
        self.epoch += 1
        print
        print 'epoch', self.epoch
        # print 'train'
        # train_pred = model.predict([X1_train, X2_train, dist_train], batch_size=batch_size)
        # for i in range(10):
        #     print train_pred[i], y_train[i]
        print 'val'
        val_pred = model.predict([X1_val[::10], X3_val[::10], X2_val[::10], dist_val[::10], ctcf_val[::10]], batch_size=batch_size)
        for i in range(10):
            print val_pred[i], y_val[i]
        print 'r2 score', r2_score(y_val[::10], val_pred)
        print 'mse     ', mean_squared_error(y_val[::10], val_pred)


print 'Data sizes: '
print 'X1_train, X2_train, dist_train:', [np.shape(X1_train), np.shape(X2_train), np.shape(dist_train)]
print 'y_train:', np.shape(y_train)
print 'y_train.mean():', y_train.mean()

# Instantiate callbacks
confusionMatrix = ConfusionMatrix()
checkpointer = ModelCheckpoint(filepath=checkpoint_path,
                               verbose=1,
                               save_best_only=True)

# Reduce learning rate by 1/5 if val_loss is stagnant for 5 epochs
reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=3)

# earlystopper = EarlyStopping(monitor='val_loss', patience=20, verbose=1)

print 'Running fully trainable model for exactly', num_epochs, 'epochs...'
print X1_train.shape, X2_train.shape, dist_train.shape, y_train.shape
print X1_val.shape, X2_val.shape, dist_val.shape, y_val.shape

train_x, train_y = [X1_train, X3_train, X2_train, dist_train, ctcf_train], [y_train]
val_x, val_y = [X1_val, X3_val, X2_val, dist_val, ctcf_val], [y_val]

# train_x, train_y = [X1_train[::1000], X3_train[::1000], X2_train[::1000], dist_train[::1000], ctcf_train[::1000]], [y_train[::1000]]
# val_x, val_y = [X1_val[::1000], X3_val[::1000], X2_val[::1000], dist_val[::1000], ctcf_val[::1000]], [y_val[::1000]]

model.fit(train_x,
          train_y,
          validation_data = (val_x, val_y),
          batch_size = batch_size,
          nb_epoch = num_epochs,
          shuffle = True,
          callbacks=[confusionMatrix, checkpointer, reduce_lr]
)
