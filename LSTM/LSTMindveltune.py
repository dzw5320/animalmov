
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Input, LSTM, TimeDistributed, SimpleRNN
from keras.optimizers import SGD, Adam, Adamax
from keras.models import Model
from keras import regularizers
from keras.constraints import max_norm


import tensorflow as tf

filename='/storage/home/d/dzw5320/MachineLearning/Data/indants2.csv'
data_in=pd.read_csv(filename, sep=',',header=0)
train_in=data_in[data_in['t']<=11515]
test_in=data_in[data_in['t']>11515]

train_xy_1=train_in.iloc[:, np.r_[9,14]].values
test_xy_1=test_in.iloc[:, np.r_[9,14]].values

train_out_act=train_in.iloc[:, np.r_[3,4]].values
test_out_act=test_in.iloc[:, np.r_[3,4]].values

train_out=train_in.iloc[:, np.r_[5,6]].values
test_out=test_in.iloc[:, np.r_[5,6]].values

train_in=train_in.iloc[:, np.r_[9:36,37:train_in.shape[1]]]
train_in=train_in.values



train_in=train_in.astype(np.float32)
test_in=test_in.iloc[:, np.r_[9:36,37:test_in.shape[1]]]
test_in=test_in.values



test_in=test_in.astype(np.float32)

from sklearn import preprocessing

scaler1 = preprocessing.StandardScaler()
scaler1.fit(train_in)
train_in=scaler1.transform(train_in)
train_in=train_in.astype(np.float32)

test_in=scaler1.transform(test_in)
test_in=test_in.astype(np.float32)

scaler2 = preprocessing.StandardScaler()
scaler2.fit(train_out)
train_out=scaler2.transform(train_out)
train_out=train_out.astype(np.float32)

test_out=scaler2.transform(test_out)
test_out=test_out.astype(np.float32)

train_in=np.reshape(train_in, (73, 11511, 39))
test_in=np.reshape(test_in, (73, 2883, 39))
train_out=np.reshape(train_out, (73, 11511, 2))
test_out=np.reshape(test_out, (73, 2883, 2))

drop=np.array([0.1, 0.2])
recc_drop=np.array([0.1, 0.2])
L1=np.array([50, 75, 100])
L2=np.array([50, 75, 100])
L3=np.array([0, 50, 100])
LR=np.array([0.001, 0.0005])
eps=np.array([25, 50 ])


drop_Series=np.repeat(drop, 216)
recc_drop_Series=np.tile(np.repeat(recc_drop, 108), 2)
L1_Series=np.tile(np.repeat(L1, 36), 4)
L2_Series=np.tile(np.repeat(L2, 12), 12)
L3_Series=np.tile(np.repeat(L3, 4), 36)
LR_Series=np.tile(np.repeat(LR, 2), 108)
eps_Series=np.tile(eps, 216)
MSPEtrain=np.zeros(432)
MSPEtest=np.zeros(432)

tune=np.column_stack((drop_Series,recc_drop_Series,L1_Series,L2_Series,L3_Series,LR_Series, eps_Series, MSPEtrain,  MSPEtest  ))

for r in range(0,433,1):
  
  L1=tune[r,2]
  L1=L1.astype(np.int32)
  L2=tune[r,3]
  L2=L2.astype(np.int32)
  L3=tune[r,4]
  L3=L3.astype(np.int32)
  epochs=tune[r,6].astype(np.int32)
  
  

  ####MODEL

  main_input=Input(shape=(None,39), name='main_input',dtype='float32')
  x=LSTM(L1, activation='tanh',recurrent_dropout=tune[r,1], dropout=tune[r,0], return_sequences=True )(main_input, training=True)
  x = Dropout(tune[r,0])(x, training=True)
  x=TimeDistributed(Dense(L2, activation='relu'))(x)
  x = Dropout(tune[r,0])(x, training=True)
  if(L3>0):
    x=TimeDistributed(Dense(L3, activation='relu'))(x)
    x = Dropout(tune[r,0])(x, training=True)
  out=TimeDistributed(Dense(2, activation='linear'))(x)


 
  model=Model(inputs=main_input, outputs=out)
  optimizer=Adam(lr=tune[r,5])
  model.compile(optimizer=optimizer, loss='mse')
  model.fit(train_in, train_out,verbose=0,
            epochs=epochs, batch_size=35)

  pred_train=np.zeros(( 500, 840303, 2))
  pred_test=np.zeros(( 500, 210459, 2))

  for j in range(500):
    xpred=model.predict(train_in)
    xpredtest=model.predict(test_in)
    xpred=np.reshape(xpred, (840303, 2))
    xpredtest=np.reshape(xpredtest, (210459, 2))
    xpred1=scaler2.inverse_transform(xpred)
    xpred1test=scaler2.inverse_transform(xpredtest)
    pred_train[j,:,:]=xpred1
    pred_test[j,:,:]=xpred1test
  

  pred_train=np.mean(pred_train, axis=0)
  pred_test=np.mean(pred_test, axis=0)
  pred_train=pred_train+train_xy_1
  pred_test=pred_test+test_xy_1
  
  tune[r, 7]=np.mean(((pred_train[:,0]-train_out_act[:,0])**2+(pred_train[:,1]-train_out_act[:,1])**2)**(0.5))

  #Calculating MSPE
  tune[r, 8]=np.mean(((pred_test[:,0]-test_out_act[:,0])**2+(pred_test[:,1]-test_out_act[:,1])**2)**(0.5))
  np.save('LSTM1indveltune.npy', tune)