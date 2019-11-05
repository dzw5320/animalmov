
from __future__ import print_function
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.layers import Input, Embedding, Dense, Dropout, LSTM
from keras.models import Model
from keras.activations import softmax
from keras.losses import categorical_crossentropy

from keras.layers.core import Dense, Dropout, Activation

from keras.models import Sequential
from keras.utils import np_utils
from keras.optimizers import Adam


from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
#import talos as ta
from sklearn.decomposition import PCA



#from talos.model.normalizers import lr_normalizer

filename='/storage/home/d/dzw5320/MachineLearning/Data/Col_inall.csv'
data_in=pd.read_csv(filename, sep=',',header=0)
train_in=data_in[data_in['t']<=11515]
test_in=data_in[data_in['t']>11515]

filename='/storage/home/d/dzw5320/MachineLearning/Data/Col_out.csv'
data_out=pd.read_csv(filename, sep=',',header=0)

data_out=data_out.iloc[:, 0:75]

train_out=data_out[data_out['t']<=11515]

train_out=train_out.iloc[:, np.r_[2:train_out.shape[1]]]

train_out=train_out.values

train_out[train_out=="out"]=1

train_out[train_out=="no"]=0

train_out[train_out=="yes"]=2

train_out=train_out.astype(np.int32)

test_out=data_out[data_out['t']>11515]

test_out=test_out.iloc[:, np.r_[2:test_out.shape[1]]]

test_out=test_out.values

test_out[test_out=="out"]=1

test_out[test_out=="no"]=0

test_out[test_out=="yes"]=2
test_out=test_out.astype(np.int32)

train_in=train_in.iloc[:, np.r_[2:(train_in.shape[1]-1)]]
train_mean=train_in.mean()
train_std=train_in.std()
train_in=(train_in-train_in.mean())/train_in.std()
criteria=train_std>0.0
train_in=train_in[criteria.index[criteria]]
train_in=train_in.values



train_in=train_in.astype(np.float32)

test_in=test_in.iloc[:, np.r_[2:(test_in.shape[1]-1)]]
test_in=(test_in-train_in.mean())/train_in.std()
test_in=test_in[criteria.index[criteria]]
test_in=test_in.values



test_in=test_in.astype(np.float32)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Fit on training set only.
scaler.fit(train_in)
# Apply transform to both the training set and the test set.
train_in = scaler.transform(train_in)
test_in = scaler.transform(test_in)

from sklearn.decomposition import PCA
pca = PCA(n_components=1000)
train_in=pca.fit_transform(train_in)
test_in=pca.transform(test_in)

#do one hot encoding 

from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(train_out[:,0])

# convert integers to dummy variables (i.e. one hot encoded)
train_out_hot=np.zeros((11511, 73, 3))

for i in range(73):
    tr = encoder.transform(train_out[:,i])
    train_out_hot[:,i,:] = np_utils.to_categorical(tr)


test_out_hot=np.zeros((2883, 73, 3))

for i in range(73):
    tr = encoder.transform(test_out[:,i])
    test_out_hot[:,i,:] = np_utils.to_categorical(tr)

train_in=np.reshape(train_in, (1, 11511,1000))
train_out_hot=np.reshape(train_out_hot, (1, 11511, 73, 3))
test_out_hot=np.reshape(test_out_hot, (1, 2883, 73, 3))

test_in=np.reshape(test_in, (1, 2883,1000))

drop=np.array([0.1, 0.2])
recc_drop=np.array([0.1, 0.2])
L1=np.array([200, 500, 700])
L2=np.array([ 200, 500, 700])
L3=np.array([0, 200, 400])
LR=np.array([0.001, 0.0005])
eps=np.array([50, 100 ])


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

for r in range(0, 432, 1):
  
  L1=tune[r,2]
  L1=L1.astype(np.int32)
  L2=tune[r,3]
  L2=L2.astype(np.int32)
  L3=tune[r,4]
  L3=L3.astype(np.int32)
  epochs=tune[r,6].astype(np.int32)
  
  
  main_input=Input(shape=(None, 1000), name='main_input',dtype='float32')
  x=LSTM(L1, activation='tanh',dropout=tune[r,0], recurrent_dropout=tune[r, 1], return_sequences=True)(main_input)
  x=Dense(L2, activation='relu')(x)
  x = Dropout(tune[r,0])(x)
  if(L3>0):
    x=Dense(L3, activation='relu')(x)
    x = Dropout(tune[r,0])(x)


  out1=Dense(3, activation='softmax', name='out1')(x)
  out2=Dense(3, activation='softmax', name='out2')(x)
  out3=Dense(3, activation='softmax', name='out3')(x)
  out4=Dense(3, activation='softmax', name='out4')(x)
  out5=Dense(3, activation='softmax', name='out5')(x)
  out6=Dense(3, activation='softmax', name='out6')(x)
  out7=Dense(3, activation='softmax', name='out7')(x)
  out8=Dense(3, activation='softmax', name='out8')(x)
  out9=Dense(3, activation='softmax', name='out9')(x)
  out10=Dense(3, activation='softmax', name='out10')(x)

  out11=Dense(3, activation='softmax', name='out11')(x)
  out12=Dense(3, activation='softmax', name='out12')(x)
  out13=Dense(3, activation='softmax', name='out13')(x)
  out14=Dense(3, activation='softmax', name='out14')(x)
  out15=Dense(3, activation='softmax', name='out15')(x)
  out16=Dense(3, activation='softmax', name='out16')(x)
  out17=Dense(3, activation='softmax', name='out17')(x)
  out18=Dense(3, activation='softmax', name='out18')(x)
  out19=Dense(3, activation='softmax', name='out19')(x)
  out20=Dense(3, activation='softmax', name='out20')(x)

  out21=Dense(3, activation='softmax', name='out21')(x)
  out22=Dense(3, activation='softmax', name='out22')(x)
  out23=Dense(3, activation='softmax', name='out23')(x)
  out24=Dense(3, activation='softmax', name='out24')(x)
  out25=Dense(3, activation='softmax', name='out25')(x)
  out26=Dense(3, activation='softmax', name='out26')(x)
  out27=Dense(3, activation='softmax', name='out27')(x)
  out28=Dense(3, activation='softmax', name='out28')(x)
  out29=Dense(3, activation='softmax', name='out29')(x)
  out30=Dense(3, activation='softmax', name='out30')(x)

  out31=Dense(3, activation='softmax', name='out31')(x)
  out32=Dense(3, activation='softmax', name='out32')(x)
  out33=Dense(3, activation='softmax', name='out33')(x)
  out34=Dense(3, activation='softmax', name='out34')(x)
  out35=Dense(3, activation='softmax', name='out35')(x)
  out36=Dense(3, activation='softmax', name='out36')(x)
  out37=Dense(3, activation='softmax', name='out37')(x)
  out38=Dense(3, activation='softmax', name='out38')(x)
  out39=Dense(3, activation='softmax', name='out39')(x)
  out40=Dense(3, activation='softmax', name='out40')(x)

  out41=Dense(3, activation='softmax', name='out41')(x)
  out42=Dense(3, activation='softmax', name='out42')(x)
  out43=Dense(3, activation='softmax', name='out43')(x)
  out44=Dense(3, activation='softmax', name='out44')(x)
  out45=Dense(3, activation='softmax', name='out45')(x)
  out46=Dense(3, activation='softmax', name='out46')(x)
  out47=Dense(3, activation='softmax', name='out47')(x)
  out48=Dense(3, activation='softmax', name='out48')(x)
  out49=Dense(3, activation='softmax', name='out49')(x)
  out50=Dense(3, activation='softmax', name='out50')(x)

  out51=Dense(3, activation='softmax', name='out51')(x)
  out52=Dense(3, activation='softmax', name='out52')(x)
  out53=Dense(3, activation='softmax', name='out53')(x)
  out54=Dense(3, activation='softmax', name='out54')(x)
  out55=Dense(3, activation='softmax', name='out55')(x)
  out56=Dense(3, activation='softmax', name='out56')(x)
  out57=Dense(3, activation='softmax', name='out57')(x)
  out58=Dense(3, activation='softmax', name='out58')(x)
  out59=Dense(3, activation='softmax', name='out59')(x)
  out60=Dense(3, activation='softmax', name='out60')(x)

  out61=Dense(3, activation='softmax', name='out61')(x)
  out62=Dense(3, activation='softmax', name='out62')(x)
  out63=Dense(3, activation='softmax', name='out63')(x)
  out64=Dense(3, activation='softmax', name='out64')(x)
  out65=Dense(3, activation='softmax', name='out65')(x)
  out66=Dense(3, activation='softmax', name='out66')(x)
  out67=Dense(3, activation='softmax', name='out67')(x)
  out68=Dense(3, activation='softmax', name='out68')(x)
  out69=Dense(3, activation='softmax', name='out69')(x)
  out70=Dense(3, activation='softmax', name='out70')(x)

  out71=Dense(3, activation='softmax', name='out71')(x)
  out72=Dense(3, activation='softmax', name='out72')(x)
  out73=Dense(3, activation='softmax', name='out73')(x)

  model=Model(inputs=main_input, outputs=[out1, out2, out3, out4, out5, out6, out7, out8, out9, out10,
                                       out11, out12, out13, out14, out15, out16, out17, out18, out19, out20,
                                       out21, out22, out23, out24, out25, out26, out27, out28, out29, out30,
                                       out31, out32, out33, out34, out35, out36, out37, out38, out39, out40,
                                       out41, out42, out43, out44, out45, out46, out47, out48, out49, out50,
                                       out51, out52, out53, out54, out55, out56, out57, out58, out59, out60,
                                       out61, out62, out63, out64, out65, out66, out67, out68, out69, out70,
                                       out71, out72, out73])
  optimizer=Adam(lr=tune[r,5])
  model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
  out=model.fit(train_in, [train_out_hot[:,:,0,:], train_out_hot[:,:,1,:],  train_out_hot[:, :,2,:],  train_out_hot[:, :,3,:],  train_out_hot[:, :,4,:],  train_out_hot[:, :,5,:],  train_out_hot[:, :,6,:],  train_out_hot[:, :,7,:],  train_out_hot[:, :,8,:],  train_out_hot[:, :,9,:],
                      train_out_hot[:, :,10,:], train_out_hot[:, :,11,:],  train_out_hot[:, :,12,:],  train_out_hot[:, :,13,:],  train_out_hot[:, :,14,:],  train_out_hot[:, :,15,:],  train_out_hot[:, :,16,:],  train_out_hot[:, :,17,:],  train_out_hot[:, :,18,:],  train_out_hot[:, :,19,:],
                      train_out_hot[:, :,20,:], train_out_hot[:, :,21,:],  train_out_hot[:, :,22,:],  train_out_hot[:, :,23,:],  train_out_hot[:, :,24,:],  train_out_hot[:, :,25,:],  train_out_hot[:, :,26,:],  train_out_hot[:, :,27,:],  train_out_hot[:, :,28,:],  train_out_hot[:, :,29,:],
                      train_out_hot[:, :,30,:], train_out_hot[:, :,31,:],  train_out_hot[:, :,32,:],  train_out_hot[:, :,33,:],  train_out_hot[:, :,34,:],  train_out_hot[:, :,35,:],  train_out_hot[:, :,36,:],  train_out_hot[:, :,37,:],  train_out_hot[:, :,38,:],  train_out_hot[:, :,39,:],
                      train_out_hot[:, :,40,:], train_out_hot[:, :,41,:],  train_out_hot[:, :,42,:],  train_out_hot[:, :,43,:],  train_out_hot[:, :,44,:],  train_out_hot[:, :,45,:],  train_out_hot[:, :,46,:],  train_out_hot[:, :,47,:],  train_out_hot[:, :,48,:],  train_out_hot[:, :,49,:],
                      train_out_hot[:, :,50,:], train_out_hot[:, :,51,:],  train_out_hot[:, :,52,:],  train_out_hot[:, :,53,:],  train_out_hot[:, :,54,:],  train_out_hot[:, :,55,:],  train_out_hot[:, :,56,:],  train_out_hot[:, :,57,:],  train_out_hot[:, :,58,:],  train_out_hot[:, :,59,:],
                      train_out_hot[:, :,60,:], train_out_hot[:, :,61,:],  train_out_hot[:, :,62,:],  train_out_hot[:, :,63,:],  train_out_hot[:, :,64,:],  train_out_hot[:, :,65,:],  train_out_hot[:, :,66,:],  train_out_hot[:, :,67,:],  train_out_hot[:, :,68,:],  train_out_hot[:, :,69,:],
                      train_out_hot[:, :,70,:], train_out_hot[:, :,71,:],  train_out_hot[:, :,72,:]], verbose=0,epochs=epochs)
  scorestrain=model.evaluate(train_in, [train_out_hot[:,:,0,:], train_out_hot[:,:,1,:],  train_out_hot[:, :,2,:],  train_out_hot[:, :,3,:],  train_out_hot[:, :,4,:],  train_out_hot[:, :,5,:],  train_out_hot[:, :,6,:],  train_out_hot[:, :,7,:],  train_out_hot[:, :,8,:],  train_out_hot[:, :,9,:],
                      train_out_hot[:, :,10,:], train_out_hot[:, :,11,:],  train_out_hot[:, :,12,:],  train_out_hot[:, :,13,:],  train_out_hot[:, :,14,:],  train_out_hot[:, :,15,:],  train_out_hot[:, :,16,:],  train_out_hot[:, :,17,:],  train_out_hot[:, :,18,:],  train_out_hot[:, :,19,:],
                      train_out_hot[:, :,20,:], train_out_hot[:, :,21,:],  train_out_hot[:, :,22,:],  train_out_hot[:, :,23,:],  train_out_hot[:, :,24,:],  train_out_hot[:, :,25,:],  train_out_hot[:, :,26,:],  train_out_hot[:, :,27,:],  train_out_hot[:, :,28,:],  train_out_hot[:, :,29,:],
                      train_out_hot[:, :,30,:], train_out_hot[:, :,31,:],  train_out_hot[:, :,32,:],  train_out_hot[:, :,33,:],  train_out_hot[:, :,34,:],  train_out_hot[:, :,35,:],  train_out_hot[:, :,36,:],  train_out_hot[:, :,37,:],  train_out_hot[:, :,38,:],  train_out_hot[:, :,39,:],
                      train_out_hot[:, :,40,:], train_out_hot[:, :,41,:],  train_out_hot[:, :,42,:],  train_out_hot[:, :,43,:],  train_out_hot[:, :,44,:],  train_out_hot[:, :,45,:],  train_out_hot[:, :,46,:],  train_out_hot[:, :,47,:],  train_out_hot[:, :,48,:],  train_out_hot[:, :,49,:],
                      train_out_hot[:, :,50,:], train_out_hot[:, :,51,:],  train_out_hot[:, :,52,:],  train_out_hot[:, :,53,:],  train_out_hot[:, :,54,:],  train_out_hot[:, :,55,:],  train_out_hot[:, :,56,:],  train_out_hot[:, :,57,:],  train_out_hot[:, :,58,:],  train_out_hot[:, :,59,:],
                      train_out_hot[:, :,60,:], train_out_hot[:, :,61,:],  train_out_hot[:, :,62,:],  train_out_hot[:, :,63,:],  train_out_hot[:, :,64,:],  train_out_hot[:, :,65,:],  train_out_hot[:, :,66,:],  train_out_hot[:, :,67,:],  train_out_hot[:, :,68,:],  train_out_hot[:, :,69,:],
                      train_out_hot[:, :,70,:], train_out_hot[:, :,71,:],  train_out_hot[:, :,72,:]], verbose=0)
  tune[r, 7]=np.mean(scorestrain[74:147])
  scorestest = model.evaluate(test_in, [test_out_hot[:,:,0,:], test_out_hot[:,:,1,:],  test_out_hot[:, :,2,:],  test_out_hot[:, :,3,:],  test_out_hot[:,:,4,:],  test_out_hot[:,:,5,:],  test_out_hot[:,:,6,:],  test_out_hot[:,:,7,:],  test_out_hot[:,:,8,:],  test_out_hot[:,:,9,:],
                      test_out_hot[:,:,10,:], test_out_hot[:,:,11,:],  test_out_hot[:,:,12,:],  test_out_hot[:,:,13,:],  test_out_hot[:,:,14,:],  test_out_hot[:,:,15,:],  test_out_hot[:,:,16,:],  test_out_hot[:,:,17,:],  test_out_hot[:,:,18,:],  test_out_hot[:,:,19,:],
                      test_out_hot[:,:,20,:], test_out_hot[:,:,21,:],  test_out_hot[:,:,22,:],  test_out_hot[:,:,23,:],  test_out_hot[:,:,24,:],  test_out_hot[:,:,25,:],  test_out_hot[:,:,26,:],  test_out_hot[:,:,27,:],  test_out_hot[:,:,28,:],  test_out_hot[:,:,29,:],
                      test_out_hot[:,:,30,:], test_out_hot[:,:,31,:],  test_out_hot[:,:,32,:],  test_out_hot[:,:,33,:],  test_out_hot[:,:,34,:],  test_out_hot[:,:,35,:],  test_out_hot[:,:,36,:],  test_out_hot[:,:,37,:],  test_out_hot[:,:,38,:],  test_out_hot[:,:,39,:],
                      test_out_hot[:,:,40,:], test_out_hot[:,:,41,:],  test_out_hot[:,:,42,:],  test_out_hot[:,:,43,:],  test_out_hot[:,:,44,:],  test_out_hot[:,:,45,:],  test_out_hot[:,:,46,:],  test_out_hot[:,:,47,:],  test_out_hot[:,:,48,:],  test_out_hot[:,:,49,:],
                      test_out_hot[:,:,50,:], test_out_hot[:,:,51,:],  test_out_hot[:,:,52,:],  test_out_hot[:,:,53,:],  test_out_hot[:,:,54,:],  test_out_hot[:,:,55,:],  test_out_hot[:,:,56,:],  test_out_hot[:,:,57,:],  test_out_hot[:,:,58,:],  test_out_hot[:,:,59,:],
                      test_out_hot[:,:,60,:], test_out_hot[:,:,61,:],  test_out_hot[:,:,62,:],  test_out_hot[:,:,63,:],  test_out_hot[:,:,64,:],  test_out_hot[:,:,65,:],  test_out_hot[:,:,66,:],  test_out_hot[:,:,67,:],  test_out_hot[:,:,68,:],  test_out_hot[:,:,69,:],
                      test_out_hot[:,:,70,:], test_out_hot[:,:,71,:],  test_out_hot[:,:,72,:]], verbose=0)
  tune[r, 8]=np.mean(scorestest[74:147])
  np.save('LSTM1colmovtune.npy', tune)