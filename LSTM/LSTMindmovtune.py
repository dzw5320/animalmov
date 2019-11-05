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
from sklearn.decomposition import PCA



from talos.model.normalizers import lr_normalizer





filename='/storage/home/d/dzw5320/MachineLearning/Data/indants2.csv'
data=pd.read_csv(filename, sep=',',header=0)
#train_in=data_in[data_in['t']<=11515]
#test_in=data_in[data_in['t']>11515]

vxmean=data['vx'].mean()
vxstd=data['vx'].std()
vymean=data['vy'].mean()
vystd=data['vy'].std()


cols_to_norm = [ 
       'vx', 'vy','x-1', 'x-2', 'x-3', 'x-4', 'x-5', 'y-1', 'y-2', 'y-3', 'y-4',
       'y-5', 'vx-1', 'vx-2', 'vx-3', 'vx-4', 'vx-5', 'vy-1', 'vy-2',
       'vy-3', 'vy-4', 'vy-5', 'chamber', 'distind', 'stattime',
       'nwalldist', 'swalldist', 'wwalldist', 'ewalldist',
       'nndist', 'nnxlag1','nnylag1', 'nnvxlag1', 'nnvylag1', 'Q1',
       'Q2', 'Q3', 'Q4','nnmove', 'nnstill','distqueen']
#trainmov[cols_to_norm] = trainmov[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max()-x.min()))
data[cols_to_norm] = data[cols_to_norm].apply(lambda x: (x - x.mean()) / (x.std()))

data.loc[data['movt'] == 'out', 'movt'] = 1
data.loc[data['movt'] == 'no', 'movt'] = 0
data.loc[data['movt'] == 'yes', 'movt'] = 2

trainmov=data[data['t']<=11515]
testmov=data[(data['t']>11515) ]

train_in=trainmov.iloc[:, np.r_[9:36,37:data.shape[1]]]
train_in=train_in.values
train_out=trainmov.iloc[:, 36]
train_out=train_out.values

test_in=testmov.iloc[:, np.r_[9:36,37:data.shape[1]]]
test_in=test_in.values
test_out=testmov.iloc[:, 36]
test_out=test_out.values

train_in=np.reshape(train_in, ( 73,11511, 39))
test_in=np.reshape(test_in, ( 73,2883, 39))




encoder = LabelEncoder()
encoder.fit(train_out)
train_out = encoder.transform(train_out)
# convert integers to dummy variables (i.e. one hot encoded)
train_out = np_utils.to_categorical(train_out)

test_out = encoder.transform(test_out)
# convert integers to dummy variables (i.e. one hot encoded)
test_out = np_utils.to_categorical(test_out)

train_out=np.reshape(train_out, (73, 11511, 3))
test_out=np.reshape(test_out, (73, 2883, 3))



train_out=train_out.astype(np.int32)


test_out=test_out.astype(np.int32)


train_in=train_in.astype(np.float32)


test_in=test_in.astype(np.float32)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM
from keras.optimizers import SGD, Nadam, Adam, RMSprop, Adadelta


import tensorflow as tf

#N = 52   # number of images in a minibatch.
D = 39  # number of features.
K = 1  
L=75
#n_batch=int(num_examples/N)
#n_epoch=10
#num_iters=n_batch*n_epoch
num_iters=500
loss_history = np.zeros([num_iters])
n_steps=11511


def create_model(train_in, train_out, test_in, test_out, params):

    model = Sequential()
    model.add(LSTM(params['first_neuron'], activation=params['recc_activation'],return_sequences=True,dropout=params['first_dropout'], recurrent_dropout=params['recc_dropout'], input_shape=(None, D)))
    model.add(Dense(params['second_neuron'], activation=params['second_activation']))
    model.add(Dropout(params['second_dropout']))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy',metrics=['accuracy'],
              optimizer=params['optimizer'](lr=lr_normalizer(params['lr'], params['optimizer'])))
    out= model.fit(train_in, train_out, epochs=params['epochs'], validation_data=[test_in, test_out])
    return out, model


p = {'lr': (0.1, 10, 10),
     'first_neuron':[32, 64, 128, 256],
     'first_dropout': (0, 0.40, 10),
     'recc_dropout': (0, 0.40, 10),
     'recc_activation': ['relu', 'tanh'],
     'second_neuron':[32, 64, 128, 256],
     'second_dropout': (0, 0.40, 10),
     'second_activation': ['relu', 'elu', 'tanh'],
     'optimizer': [Adam, Nadam, RMSprop, Adadelta],
     'epochs': (10, 100, 10)}

h = ta.Scan( x=train_in, y=train_out,params=p,
            model=create_model, x_val=test_in, y_val=test_out,disable_progress_bar=True, grid_downsample=0.0005)

# accessing the results data frame
results=h.data.values
np.save('tunekeraslstmmovind.npy', results)
h.data.to_csv('tunekeraslstmmovind.csv')