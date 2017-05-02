import numpy as np
import re
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
num_epochs = 100
batch_size = 50
training_frac = 0.9 # use 90% of data for training, 10% for testing/validation
t = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
lr = 1e-5
# opt = RMSprop(lr=lr)
opt = Adam(lr=lr)

chromosome = 21
# n_samples = 321812
n_samples = 10000
resolution = 5000
mode_str = 'uniform_500000'

checkpoint_path = 'weights/chr{0}_kr_{1}{2}k_{3}_{4}_logy.hdf5'.format(chromosome, mode_str, resolution / 1000, n_samples, t)
print n_samples, resolution, checkpoint_path

# # Load data and split into training and validation sets
data_path = 'data/chr{0}_{1}k_kr_pairs_{2}_{3}.h5'.format(chromosome, resolution / 1000, n_samples, mode_str)
print 'Loading data from ' + data_path

# TODO: Resample 10 times and do cross-validation
X1_train, X2_train, dist_train, y_train, indices = util.load_hdf5_hg19(data_path)
y_train = np.log(y_train)
X1_train, X2_train, dist_train, y_train, X1_val, X2_val, dist_val, y_val = util.split_train_and_val_data_hg19(X1_train, X2_train, dist_train, y_train, training_frac)

X1_length = X1_train.shape[1]
X2_length = X2_train.shape[1]

print 'Building model...'
model = build_model.build_model_v1()

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
        print 'train'
        train_pred = model.predict([X1_train, X2_train, dist_train], batch_size=batch_size)
        for i in range(10):
            print train_pred[i], y_train[i]
        print 'val'
        val_pred = model.predict([X1_val, X2_val, dist_val], batch_size=batch_size)
        for i in range(10):
            print val_pred[i], y_val[i]


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
reduce_lr = ReduceLROnPlateau(monitor='loss',
                              factor=0.2,
                              patience=5)

# earlystopper = EarlyStopping(monitor='val_loss', patience=20, verbose=1)

print 'Running fully trainable model for exactly', num_epochs, 'epochs...'
print X1_train.shape, X2_train.shape, dist_train.shape, y_train.shape
print X1_val.shape, X2_val.shape, dist_val.shape, y_val.shape
model.fit([X1_train, X2_train, dist_train],
          [y_train],
          validation_data = ([X1_val, X2_val, dist_val], [y_val]),
          batch_size = batch_size,
          nb_epoch = num_epochs,
          shuffle = True,
          callbacks=[confusionMatrix, checkpointer, reduce_lr]
)
