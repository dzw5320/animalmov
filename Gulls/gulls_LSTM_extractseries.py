
from __future__ import print_function
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.layers import Input, Embedding, Dense, Dropout, LSTM
from keras.models import Model
from keras.activations import softmax
from keras.losses import categorical_crossentropy

from keras.layers.core import Dense, Dropout, Activation, Masking

from keras.models import Sequential
from keras.utils import np_utils
from keras.optimizers import Adam

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import mean_squared_error



filename='gulls40.csv'
data_in=pd.read_csv(filename, sep=',',header=0)
data_in.columns

data_std=data_in

data_transform=data_std.iloc[:, np.r_[10:173]]

# Step 1: Classification

#Normalize the input variables

from sklearn import preprocessing

scaler_in = preprocessing.StandardScaler()

scaler_in.fit(data_transform)

scaler_out = preprocessing.StandardScaler()
scaler_out.fit(data_in[['vx','vy']])

idx_BY=[]
#len_BY=[]
fold_BY=[]
#common_BY=[]

for i in range(1,16,1):

  idx_=np.where((data_std['BirdYear']==i))[0]
  idx_BY.append(idx_)
  #len_BY.append(len(idx_))
  fold_BY.append(data_std['fold'][idx_].unique())
  #data_fold=data_std.iloc[idx_,:]
  #common_BY.append(np.where(data_fold['common']==1)[0])

BY1=data_std.iloc[idx_BY[0]].astype(np.float32)
BY2=data_std.iloc[idx_BY[1]].astype(np.float32)
BY3=data_std.iloc[idx_BY[2]].astype(np.float32)
BY4=data_std.iloc[idx_BY[3]].astype(np.float32)
BY5=data_std.iloc[idx_BY[4]].astype(np.float32)

BY6=data_std.iloc[idx_BY[5]].astype(np.float32)
BY7=data_std.iloc[idx_BY[6]].astype(np.float32)
BY8=data_std.iloc[idx_BY[7]].astype(np.float32)
BY9=data_std.iloc[idx_BY[8]].astype(np.float32)
BY10=data_std.iloc[idx_BY[9]].astype(np.float32)

BY11=data_std.iloc[idx_BY[10]].astype(np.float32)
BY12=data_std.iloc[idx_BY[11]].astype(np.float32)
BY13=data_std.iloc[idx_BY[12]].astype(np.float32)
BY14=data_std.iloc[idx_BY[13]].astype(np.float32)
BY15=data_std.iloc[idx_BY[14]].astype(np.float32)

input_seq=[BY1, BY2, BY3, BY4, BY5, BY6, BY7, BY8, BY9, BY10, BY11, BY12, BY13, BY14, BY15]
fold_BY=np.array(fold_BY)

#there are 4 states 1,2,4,5

def getdata(state,lag,end,fold,maxlen):
  state1_input=[]
  #print(f)
  fold_id=np.where(fold_BY==fold)[0]
  fold_id=fold_id.astype(np.int32)
  for Y in fold_id:
   #print(Y)
   BY_data=input_seq[Y]
   for r in range(BY_data.shape[0]):
    if(BY_data['state'].iloc[r]==state):
     #print(r)
     extr_times=BY_data.iloc[0:(r+1),:]
     state1_input.append(extr_times)
  input_seq_=[]
  for i in range(len(state1_input)):
   input_seq_.append(scaler_in.transform(state1_input[i].iloc[:,10:end]))
  input_seq_pad=pad_sequences(input_seq_, padding='pre', dtype='float32', maxlen=maxlen )
  metadata=[]
  for i in range(len(state1_input)):
   metadata.append(state1_input[i].iloc[-1,np.r_[0:10,12,(12+lag)]])
  metadata=np.asarray(metadata)
  output=scaler_out.transform(metadata[:,7:9])
  end_pt=input_seq_pad.shape[1]
  start_pt=end_pt-1000
  input_seq_pad=input_seq_pad[:,start_pt:end_pt,:]
  return input_seq_pad,metadata,output



for f in range(1,6,1):
 input_seq_pad,metadata,output=getdata(5,40,173,f,14958)
 np.savez_compressed('input_seq_pad40_5_fold'+ str(f), a=input_seq_pad, b=metadata,c=output)
 del input_seq_pad
 del metadata
 del output


for f in range(1,6,1):
 input_seq_pad,metadata,output=getdata(2,40,173,f,9884)
 np.savez_compressed('input_seq_pad40_2_fold'+ str(f), a=input_seq_pad, b=metadata,c=output)
 del input_seq_pad
 del metadata
 del output




for f in range(1,6,1):
 input_seq_pad,metadata,output=getdata(1,40,173,f, 17526)
 np.savez_compressed('input_seq_pad40_1_fold'+ str(f), a=input_seq_pad, b=metadata,c=output)
 del input_seq_pad
 del metadata
 del output

for f in range(1,6,1):
 input_seq_pad,metadata,output=getdata(4,40,173,f,14448)
 np.savez_compressed('input_seq_pad40_4_fold'+ str(f), a=input_seq_pad, b=metadata,c=output)
 del input_seq_pad
 del metadata
 del output

