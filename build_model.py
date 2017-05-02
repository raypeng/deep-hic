import util

# Keras imports
from keras.layers import Input, Convolution1D, MaxPooling1D, Merge, Dropout, Flatten, Dense, BatchNormalization, LSTM, Activation, Bidirectional, SimpleRNN
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.models import Sequential
from keras.regularizers import l2, activity_l2

import keras
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
log_device_placement=True
_sess = tf.Session(config=config)
keras.backend.set_session(_sess)


def build_model_v1():
  # model parameters
  seq_length = 5000
  n_kernels = 1024 # Number of kernels; used to be 1024
  filter_length = 40 # Length of each kernel
  LSTM_out_dim = 100 # Output direction of ONE DIRECTION of LSTM; used to be 512
  l2_penalty = 1e-5

  seq1_branch = Sequential()
  seq1_branch.add(
    Convolution1D(input_dim=4,
                  input_length=seq_length,
                  nb_filter=n_kernels,
                  filter_length=filter_length,
                  border_mode="valid",
                  subsample_length=1,
                  W_regularizer=l2(l2_penalty))
  )
  seq1_branch.add(Activation('relu'))
  seq1_branch.add(
    MaxPooling1D(pool_length=filter_length/2, stride=filter_length/2)
  )

  seq2_branch = Sequential()
  seq2_branch.add(
    Convolution1D(input_dim=4,
                  input_length=seq_length,
                  nb_filter=n_kernels,
                  filter_length=filter_length,
                  border_mode="valid",
                  subsample_length=1,
                  W_regularizer=l2(l2_penalty))
  )
  seq2_branch.add(Activation('relu'))
  seq2_branch.add(
    MaxPooling1D(pool_length=filter_length/2, stride=filter_length/2)
  )

  merge1 = Merge([seq1_branch, seq2_branch],
                 mode='concat',
                 concat_axis=1)

  biLSTM_layer = Bidirectional(LSTM(input_dim=n_kernels,
                                    output_dim=LSTM_out_dim,
                                    return_sequences=True))

  model_prev = Sequential()
  model_prev.add(merge1)
  model_prev.add(BatchNormalization())
  model_prev.add(Dropout(0.25))
  model_prev.add(biLSTM_layer)
  model_prev.add(BatchNormalization())
  model_prev.add(Dropout(0.5))
  model_prev.add(Flatten())

  dist_branch = Sequential()
  dist_branch.add(
    Dense(1, input_dim=1, init='identity')
  )

  merge_with_dist_layer = Merge([model_prev, dist_branch],
                                mode='concat',
                                concat_axis=1)

  dense_layer = Dense(output_dim=1000,
                      init="glorot_uniform",
                      W_regularizer=l2(l2_penalty))
  output_layer = Dense(output_dim=1)

  model = Sequential()
  model.add(merge_with_dist_layer)
  model.add(dense_layer)
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  model.add(output_layer)
  return model
