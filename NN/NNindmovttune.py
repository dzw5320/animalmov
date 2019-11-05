


from __future__ import print_function
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.layers import Input, Embedding, Dense, Dropout
from keras.models import Model

from keras.layers.core import Dense, Dropout, Activation

from keras.models import Sequential
from keras.utils import np_utils


from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import talos as ta
from talos.model.normalizers import lr_normalizer



filename='/storage/home/d/dzw5320/MachineLearning/Data/indants2.csv'
data=pd.read_csv(filename, sep=',',header=0)
data.columns

#Scale data
cols_to_norm = [  
       'vx','vy','x-1', 'x-2', 'x-3', 'x-4', 'x-5', 'y-1', 'y-2', 'y-3', 'y-4',
       'y-5', 'vx-1', 'vx-2', 'vx-3', 'vx-4', 'vx-5', 'vy-1', 'vy-2',
       'vy-3', 'vy-4', 'vy-5', 'chamber', 'distind', 'stattime',
       'nwalldist', 'swalldist', 'wwalldist', 'ewalldist',
       'nndist', 'nnxlag1','nnylag1', 'nnvxlag1', 'nnvylag1', 'Q1',
       'Q2', 'Q3', 'Q4','nnmove', 'nnstill','distqueen']

data[cols_to_norm] = data[cols_to_norm].apply(lambda x: (x - x.mean()) / (x.std()))

train1=data[data['t']<=11515]
test1=data[(data['t']>11515) ]

train_in=train1.iloc[:, np.r_[9:36,37:train1.shape[1]]]
train_in=train_in.values
train_out=train1.iloc[:, 36]
train_out=train_out.values

test_in=test1.iloc[:, np.r_[9:36,37:test1.shape[1]]]
test_in=test_in.values
test_out=test1.iloc[:, 36]
test_out=test_out.values

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(train_out)
train_out = encoder.transform(train_out)
# convert integers to dummy variables (i.e. one hot encoded)
train_out = np_utils.to_categorical(train_out)

test_out = encoder.transform(test_out)
# convert integers to dummy variables (i.e. one hot encoded)
test_out = np_utils.to_categorical(test_out)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, Nadam, Adam, RMSprop


import tensorflow as tf

def create_model(train_in, train_out, test_in, test_out, params):
  model = Sequential()
  model.add(Dense(params['first_neuron'], input_dim=39, init='uniform', activation=params['first_activation']))
  model.add(Dropout(params['first_dropout']))
  model.add(Dense(params['second_neuron'], init='uniform', activation=params['second_activation']))
  model.add(Dropout(params['second_dropout']))
  model.add(Dense(3, activation='softmax'))
  model.compile(loss='categorical_crossentropy',metrics=['accuracy'],
              optimizer=params['optimizer'](lr=lr_normalizer(params['lr'], params['optimizer'])))
  out=model.fit(train_in, train_out,
          epochs=params['epochs'],validation_data=[test_in, test_out],batch_size=params['batch_size'],
          verbose=0)
  return out, model

from keras.optimizers import Adam, Nadam, RMSprop, Adadelta

p = {'lr': (0.1, 10, 10),
     'first_neuron':[32, 64, 128, 256],
     'first_dropout': (0, 0.40, 10),
     'first_activation': ['relu', 'elu', 'tanh'],
     'second_neuron':[32, 64, 128, 256],
     'second_dropout': (0, 0.40, 10),
     'second_activation': ['relu', 'elu', 'tanh'],
     'optimizer': [Adam, Nadam, RMSprop, Adadelta],
     'batch_size': [219, 657, 1279, 3837],
     'epochs': (10, 100, 10)}

h = ta.Scan( x=train_in, y=train_out,params=p,experiment_no='2',
            model=create_model,disable_progress_bar=True, grid_downsample=0.0005)

# accessing the results data frame
results=h.data.values
np.save('tunekerasmovind.npy', results)
h.data.to_csv('tunekerasmovind.csv')