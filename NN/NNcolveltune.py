

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Input, LSTM, TimeDistributed, SimpleRNN
from keras.optimizers import SGD, Adam, Adamax
from keras.models import Model
from keras import regularizers
from keras.constraints import max_norm


import tensorflow as tf

filename='/storage/home/d/dzw5320/MachineLearning/Data/Col_inall.csv'
data_in=pd.read_csv(filename, sep=',',header=0)
train_in=data_in[data_in['t']<=11515]
test_in=data_in[data_in['t']>11515]
train_in=train_in.iloc[:, np.r_[2:train_in.shape[1]]]
train_in=train_in.values



train_in=train_in.astype(np.float32)
test_in=test_in.iloc[:, np.r_[2:test_in.shape[1]]]
test_in=test_in.values



test_in=test_in.astype(np.float32)

xnumbers1=np.arange(0, 73, 1)
xnumbers2=np.arange(365, 438, 1)
xnumbers=np.concatenate((xnumbers1, xnumbers2))
xnumbers=np.reshape(xnumbers, (146))


train_xy_1=train_in[:,xnumbers]
test_xy_1=test_in[:,xnumbers]


filename='/storage/home/d/dzw5320/MachineLearning/Data/Col_inall.csv'
data_in=pd.read_csv(filename, sep=',',header=0)
train_in=data_in[data_in['t']<=11515]
test_in=data_in[data_in['t']>11515]

filename='/storage/home/d/dzw5320/MachineLearning/Data/Col_out.csv'
data_out=pd.read_csv(filename, sep=',',header=0)

data_out=data_out.iloc[:, np.r_[1, 75:221]]

train_out=data_out[data_out['t']<=11515]

train_out=train_out.iloc[:, np.r_[1:train_out.shape[1]]]

train_out=train_out.values

from sklearn import preprocessing

scaler1 = preprocessing.StandardScaler()
scaler1.fit(train_out)
train_out=scaler1.transform(train_out)
train_out=train_out.astype(np.float32)

test_out=data_out[data_out['t']>11515]

test_out=test_out.iloc[:, np.r_[1:test_out.shape[1]]]

test_out=test_out.values
test_out=scaler1.transform(test_out)

test_out=test_out.astype(np.float32)
train_in=train_in.iloc[:, np.r_[2:(train_in.shape[1]-1)]]
train_std=train_in.std()
criteria=train_std>0.0
train_in=train_in[criteria.index[criteria]]
train_in=train_in.values

scaler2 = preprocessing.StandardScaler()
scaler2.fit(train_in)
train_in=scaler2.transform(train_in)

train_in=train_in.astype(np.float32)
test_in=test_in.iloc[:, np.r_[2:(test_in.shape[1]-1)]]

test_in=test_in[criteria.index[criteria]]
test_in=test_in.values

test_in=scaler2.transform(test_in)

test_in=test_in.astype(np.float32)
  
train_in=np.reshape(train_in, ( 11511, 2831))
test_in=np.reshape(test_in, ( 2883, 2831))
train_out=np.reshape(train_out, ( 11511, 146))
test_out=np.reshape(test_out, ( 2883, 146))

from sklearn.decomposition import PCA
pca = PCA(n_components=1000)
train_in=pca.fit_transform(train_in)
test_in=pca.transform(test_in)

drop=np.array([0.1, 0.2])
L1=np.array([200, 500, 800])
L2=np.array([200, 500, 800])
L3=np.array([0, 250, 500])
LR=np.array([ 0.005, 0.001, 0.0005 ])
eps=np.array([50, 100 ])


drop_Series=np.repeat(drop, 162)
L1_Series=np.tile(np.repeat(L1, 54), 2)
L2_Series=np.tile(np.repeat(L2, 18), 6)
L3_Series=np.tile(np.repeat(L3, 6), 18)
LR_Series=np.tile(np.repeat(LR,2), 54)
eps_Series=np.tile(eps, 162)
MSPEtrain=np.zeros(324)
MSPEtest=np.zeros(324)

tune=np.column_stack((drop_Series,L1_Series,L2_Series,L3_Series,LR_Series, eps_Series, MSPEtrain,  MSPEtest  ))

for r in range(0,324,1):
  
  L1=tune[r,1]
  L1=L1.astype(np.int32)
  L2=tune[r,2]
  L2=L2.astype(np.int32)
  L3=tune[r,3]
  L3=L3.astype(np.int32)
  epochs=tune[r,5].astype(np.int32)
  
  

  ####MODEL

  main_input=Input(shape=(1000,), name='main_input',dtype='float32')
  x = Dropout(tune[r,0])(main_input, training=True)
  x=Dense(L1, activation='relu')(x)
  x = Dropout(tune[r,0])(x, training=True)
  x=Dense(L2, activation='relu')(x)
  x = Dropout(tune[r,0])(x, training=True)
  if(L3>0):
    x=Dense(L3, activation='relu')(x)
    x = Dropout(tune[r,0])(x, training=True)
  out=Dense(146, activation='linear')(x)


  #sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
  model=Model(inputs=main_input, outputs=out)
  optimizer=Adam(lr=tune[r,4])
  model.compile(optimizer=optimizer, loss='mse')
  model.fit(train_in, train_out,verbose=0,
            epochs=epochs)

  pred_train=np.zeros(( 500, 11511, 146))
  pred_test=np.zeros(( 500, 2883, 146))

  for j in range(500):
    xpred=model.predict(train_in)
    xpredtest=model.predict(test_in)
    xpred1=np.reshape(xpred, ( 11511, 146))
    xpred1=scaler1.inverse_transform(xpred1)
    xpred1test=np.reshape(xpredtest, ( 2883, 146))
    xpred1test=scaler1.inverse_transform(xpred1test)
    pred_train[j,:,:]=xpred1
    pred_test[j,:,:]=xpred1test
  

  pred_train=np.mean(pred_train, axis=0)
  pred_test=np.mean(pred_test, axis=0)
  act_train=scaler1.inverse_transform(train_out)
  act_test=scaler1.inverse_transform(test_out)
  pred_train=pred_train+train_xy_1
  pred_test=pred_test+test_xy_1
  

  act_train=act_train+train_xy_1
  act_test=act_test+test_xy_1
  
  unstd_pred_train_mean_reshape=pred_train.reshape((11511, 2, 73))
  train_out_reshape=act_train.reshape((11511, 2, 73))

  unstd_pred_test_mean_reshape=pred_test.reshape((2883, 2, 73))
  test_out_reshape=act_test.reshape((2883, 2, 73))
  
  check=np.zeros((11511, 73))


  check=((unstd_pred_train_mean_reshape[:, 0, :]-train_out_reshape[:, 0,:])**2+(unstd_pred_train_mean_reshape[:, 1, :]-train_out_reshape[:, 1,:])**2)**(0.5)

  tune[r, 6]=check.sum(axis=0).sum()/(11511*73)

  #Calculating MSPE
  check=np.zeros((2883, 73))


  check=((unstd_pred_test_mean_reshape[:, 0, :]-test_out_reshape[:, 0,:])**2+(unstd_pred_test_mean_reshape[:, 1, :]-test_out_reshape[:, 1,:])**2)**(0.5)

  tune[r, 7]=check.sum(axis=0).sum()/(2883*73)
  np.save('NN1colveltune.npy', tune)