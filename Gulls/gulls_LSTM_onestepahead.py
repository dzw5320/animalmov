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
data_transform=scaler_in.transform(data_transform)

scaler_out = preprocessing.StandardScaler()
scaler_out.fit(data_in[['vx','vy']])

encoder = LabelEncoder()
encoder.fit(data_std['state'])
out_ = encoder.transform(data_std['state'])
# convert integers to dummy variables (i.e. one hot encoded)
out_cat = np_utils.to_categorical(out_)
del data_in

idx_BY=[]
len_BY=[]
fold_BY=[]
common_BY=[]

for i in range(1,16,1):

  idx_=np.where((data_std['BirdYear']==i))[0]
  idx_BY.append(idx_)
  len_BY.append(len(idx_))
  fold_BY.append(data_std['fold'][idx_].unique())
  data_fold=data_std.iloc[idx_,:]
  common_BY.append(np.where(data_fold['common']==1)[0])

len_BY=np.array(len_BY)
fold_BY=np.array(fold_BY)



BY1=data_transform[idx_BY[0]].astype(np.float32)
BY2=data_transform[idx_BY[1]].astype(np.float32)
BY3=data_transform[idx_BY[2]].astype(np.float32)
BY4=data_transform[idx_BY[3]].astype(np.float32)
BY5=data_transform[idx_BY[4]].astype(np.float32)

BY6=data_transform[idx_BY[5]].astype(np.float32)
BY7=data_transform[idx_BY[6]].astype(np.float32)
BY8=data_transform[idx_BY[7]].astype(np.float32)
BY9=data_transform[idx_BY[8]].astype(np.float32)
BY10=data_transform[idx_BY[9]].astype(np.float32)

BY11=data_transform[idx_BY[10]].astype(np.float32)
BY12=data_transform[idx_BY[11]].astype(np.float32)
BY13=data_transform[idx_BY[12]].astype(np.float32)
BY14=data_transform[idx_BY[13]].astype(np.float32)
BY15=data_transform[idx_BY[14]].astype(np.float32)

#one hot encoded

out_1=out_cat[idx_BY[0],:]
out_2=out_cat[idx_BY[1],:]
out_3=out_cat[idx_BY[2],:]
out_4=out_cat[idx_BY[3],:]
out_5=out_cat[idx_BY[4],:]

out_6=out_cat[idx_BY[5],:]
out_7=out_cat[idx_BY[6],:]
out_8=out_cat[idx_BY[7],:]
out_9=out_cat[idx_BY[8],:]
out_10=out_cat[idx_BY[9],:]

out_11=out_cat[idx_BY[10],:]
out_12=out_cat[idx_BY[11],:]
out_13=out_cat[idx_BY[12],:]
out_14=out_cat[idx_BY[13],:]
out_15=out_cat[idx_BY[14],:]

#state now coded as 0,1,2,3 not 1,2,4,5

out1=out_[idx_BY[0]]
out2=out_[idx_BY[1]]
out3=out_[idx_BY[2]]
out4=out_[idx_BY[3]]
out5=out_[idx_BY[4]]

out6=out_[idx_BY[5]]
out7=out_[idx_BY[6]]
out8=out_[idx_BY[7]]
out9=out_[idx_BY[8]]
out10=out_[idx_BY[9]]

out11=out_[idx_BY[10]]
out12=out_[idx_BY[11]]
out13=out_[idx_BY[12]]
out14=out_[idx_BY[13]]
out15=out_[idx_BY[14]]

out_1dim=np.array([out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15])

input_seq=[BY1, BY2, BY3, BY4, BY5, BY6, BY7, BY8, BY9, BY10, BY11, BY12, BY13, BY14, BY15]
#Padded
input_seq_pad=pad_sequences([BY1, BY2, BY3, BY4, BY5, BY6, BY7, BY8, BY9, BY10, BY11, BY12, BY13, BY14, BY15], padding='post', dtype='float32' )

output_seq_pad=pad_sequences([out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8, out_9, out_10, out_11, out_12, out_13, out_14, out_15], padding='post', dtype='int32' )


#compiling classification model
main_input=Input(shape=(None, 163), name='main_input', dtype='float32')
mask = Masking(mask_value=0.)(main_input)
x=LSTM(250, activation='tanh',return_sequences=True,dropout=0.2, recurrent_dropout=0.1)(mask)
x=Dense(200, activation='relu')(x)
x=Dropout(0.2)(x)
x=Dense(50, activation='relu')(x)
x = Dropout(0.2)(x)
out= Dense(4, activation='softmax')(x)
model_class=Model(inputs=main_input,outputs=out)
optimizer=Adam(lr=0.001)
model_class.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

#compiling vel state 1 model
main_input1=Input(shape=(None, 163), name='main_input1', dtype='float32')
mask1 = Masking(mask_value=0.)(main_input1)
x1=LSTM(200, activation='tanh',return_sequences=False,dropout=0.1, recurrent_dropout=0.1)(mask1)
x1=Dense(200, activation='relu')(x1)
x1=Dropout(0.1)(x1)
out1= Dense(2, activation='linear')(x1)
model1=Model(inputs=main_input1,outputs=out1)
optimizer1=Adam(lr=0.001)
model1.compile(optimizer=optimizer1, loss='mse')

#compiling vel state 2 model
main_input2=Input(shape=(None, 163), name='main_input2', dtype='float32')
mask2 = Masking(mask_value=0.)(main_input2)
x2=LSTM(300, activation='tanh',return_sequences=False,dropout=0.2, recurrent_dropout=0.2)(mask2)
x2=Dense(250, activation='relu')(x2)
x2=Dropout(0.2)(x2)
x2=Dense(50, activation='relu')(x2)
x2=Dropout(0.2)(x2)
out2= Dense(2, activation='linear')(x2)
model2=Model(inputs=main_input2,outputs=out2)
optimizer2=Adam(lr=0.0005)
model2.compile(optimizer=optimizer2, loss='mse')

#compiling vel state 4 model
main_input4=Input(shape=(None, 163), name='main_input4', dtype='float32')
mask4 = Masking(mask_value=0.)(main_input4)
x4=LSTM(250, activation='tanh',return_sequences=False,dropout=0.1, recurrent_dropout=0.1)(mask4)
x4=Dense(200, activation='relu')(x4)
x4=Dropout(0.1)(x4)
out4= Dense(2, activation='linear')(x4)
model4=Model(inputs=main_input4,outputs=out4)
optimizer4=Adam(lr=0.001)
model4.compile(optimizer=optimizer4, loss='mse')

#compiling vel state 5 model
main_input5=Input(shape=(None, 163), name='main_input5', dtype='float32')
mask5 = Masking(mask_value=0.)(main_input5)
x5=LSTM(200, activation='tanh',return_sequences=False,dropout=0.1, recurrent_dropout=0.1)(mask5)
x5=Dense(200, activation='relu')(x5)
x5=Dropout(0.1)(x5)
out5= Dense(2, activation='linear')(x5)
model5=Model(inputs=main_input5,outputs=out5)
optimizer5=Adam(lr=0.001)
model5.compile(optimizer=optimizer5, loss='mse')




fold_mspe=np.zeros(5)


for k in range(1,6,1):

 #Fit models using the data not in fold
 #Classification model for state
 
 fit_class=model_class.fit(input_seq_pad[np.where(fold_BY!=k)[0],:,:], output_seq_pad[np.where(fold_BY!=k)[0],:,:], epochs=50, verbose=0, batch_size=128)
 #Velocity model for state 1
 data1 = np.load('input_seq_pad40_1_fold1.npz')
 data2 = np.load('input_seq_pad40_1_fold2.npz')
 data3 = np.load('input_seq_pad40_1_fold3.npz')
 data4 = np.load('input_seq_pad40_1_fold4.npz')
 data5 = np.load('input_seq_pad40_1_fold5.npz')
 input_seq_pad1 =np.concatenate((data1['a'],data2['a'],data3['a'],data4['a'],data5['a'] ), axis=0)
 metadata=np.concatenate((data1['b'],data2['b'],data3['b'],data4['b'],data5['b'] ), axis=0)
 output=np.concatenate((data1['c'],data2['c'],data3['c'],data4['c'],data5['c'] ), axis=0)
 del data1,data2,data3,data4,data5
 fit1=model1.fit(input_seq_pad1[np.where(metadata[:,2]!=k)[0],:,:], output[np.where(metadata[:,2]!=k)[0],:], epochs=8,verbose=0, batch_size=128)
 del input_seq_pad1,metadata,output
 #Velocity model for state 2
 data1 = np.load('input_seq_pad40_2_fold1.npz')
 data2 = np.load('input_seq_pad40_2_fold2.npz')
 data3 = np.load('input_seq_pad40_2_fold3.npz')
 data4 = np.load('input_seq_pad40_2_fold4.npz')
 data5 = np.load('input_seq_pad40_2_fold5.npz')
 input_seq_pad2 =np.concatenate((data1['a'],data2['a'],data3['a'],data4['a'],data5['a'] ), axis=0)
 metadata=np.concatenate((data1['b'],data2['b'],data3['b'],data4['b'],data5['b'] ), axis=0)
 output=np.concatenate((data1['c'],data2['c'],data3['c'],data4['c'],data5['c'] ), axis=0)
 del data1,data2,data3,data4,data5
 fit2=model2.fit(input_seq_pad2[np.where(metadata[:,2]!=k)[0],:,:], output[np.where(metadata[:,2]!=k)[0],:], epochs=15,verbose=0, batch_size=128)
 del input_seq_pad2,metadata,output
 #Velocity model for state 4
 data1 = np.load('input_seq_pad40_4_fold1.npz')
 data2 = np.load('input_seq_pad40_4_fold2.npz')
 data3 = np.load('input_seq_pad40_4_fold3.npz')
 data4 = np.load('input_seq_pad40_4_fold4.npz')
 data5 = np.load('input_seq_pad40_4_fold5.npz')
 input_seq_pad4 =np.concatenate((data1['a'],data2['a'],data3['a'],data4['a'],data5['a'] ), axis=0)
 metadata=np.concatenate((data1['b'],data2['b'],data3['b'],data4['b'],data5['b'] ), axis=0)
 output=np.concatenate((data1['c'],data2['c'],data3['c'],data4['c'],data5['c'] ), axis=0)
 del data1,data2,data3,data4,data5
 fit4=model4.fit(input_seq_pad4[np.where(metadata[:,2]!=k)[0],:,:], output[np.where(metadata[:,2]!=k)[0],:], epochs=10, batch_size=128)
 del input_seq_pad4,metadata,output
 #Velocity model for state 5
 data1 = np.load('input_seq_pad40_5_fold1.npz')
 data2 = np.load('input_seq_pad40_5_fold2.npz')
 data3 = np.load('input_seq_pad40_5_fold3.npz')
 data4 = np.load('input_seq_pad40_5_fold4.npz')
 data5 = np.load('input_seq_pad40_5_fold5.npz')
 input_seq_pad5 =np.concatenate((data1['a'],data2['a'],data3['a'],data4['a'],data5['a'] ), axis=0)
 metadata=np.concatenate((data1['b'],data2['b'],data3['b'],data4['b'],data5['b'] ), axis=0)
 output=np.concatenate((data1['c'],data2['c'],data3['c'],data4['c'],data5['c'] ), axis=0)
 del data1,data2,data3,data4,data5
 fit5=model5.fit(input_seq_pad5[np.where(metadata[:,2]!=k)[0],:,:], output[np.where(metadata[:,2]!=k)[0],:], epochs=15,verbose=0, batch_size=128)
 del input_seq_pad5,metadata,output
 #Predict state for all in fold==f
 id_fold=np.where(fold_BY==k)[0]
 getclass1=model_class.predict(input_seq[id_fold[0]].reshape(1,len_BY[id_fold[0]],163))
 getclass1=np.argmax(getclass1, axis=2).reshape([-1])
 getclass2=model_class.predict(input_seq[id_fold[1]].reshape(1,len_BY[id_fold[1]],163))
 getclass2=np.argmax(getclass2, axis=2).reshape([-1])
 getclass3=model_class.predict(input_seq[id_fold[2]].reshape(1,len_BY[id_fold[2]],163))
 getclass3=np.argmax(getclass3, axis=2).reshape([-1])
 scores1=np.zeros((getclass1.shape[0],2))
 scores2=np.zeros((getclass2.shape[0],2))
 scores3=np.zeros((getclass3.shape[0],2))
 for j in range(getclass1.shape[0]):
  if j<999:
   pred_data1=input_seq[id_fold[0]]
   pred_data1=pred_data1[0:(j+1),:]
   pred_data1=pred_data1.reshape(1,(j+1),163)
  else:
   pred_data1=input_seq[id_fold[0]]
   pred_data1=pred_data1[(j-999):(j+1),:]
   pred_data1=pred_data1.reshape(1,1000,163)
  if getclass1[j]==0:
      scores1[j,:]=scaler_out.inverse_transform(model1.predict(pred_data1))
  elif getclass1[j]==1:
       scores1[j,:]=scaler_out.inverse_transform(model2.predict(pred_data1))
  elif getclass1[j]==2:
      scores1[j,:]=scaler_out.inverse_transform(model4.predict(pred_data1))
  else:
      scores1[j,:]=scaler_out.inverse_transform(model5.predict(pred_data1))
 for j in range(getclass2.shape[0]):
  if j<999:
   pred_data1=input_seq[id_fold[1]]
   pred_data1=pred_data1[0:(j+1),:]
   pred_data1=pred_data1.reshape(1,(j+1),163)
  else:
   pred_data1=input_seq[id_fold[1]]
   pred_data1=pred_data1[(j-999):(j+1),:]
   pred_data1=pred_data1.reshape(1,1000,163)
  if getclass2[j]==0:
      scores2[j,:]=scaler_out.inverse_transform(model1.predict(pred_data1))
  elif getclass2[j]==1:
      scores2[j,:]=scaler_out.inverse_transform(model2.predict(pred_data1))
  elif getclass2[j]==2:
      scores2[j,:]=scaler_out.inverse_transform(model4.predict(pred_data1))
  else:
      scores2[j,:]=scaler_out.inverse_transform(model5.predict(pred_data1))
 for j in range(getclass3.shape[0]):
  if j<999:
   pred_data1=input_seq[id_fold[2]]
   pred_data1=pred_data1[0:(j+1),:]
   pred_data1=pred_data1.reshape(1,(j+1),163)
  else:
   pred_data1=input_seq[id_fold[2]]
   pred_data1=pred_data1[(j-999):(j+1),:]
   pred_data1=pred_data1.reshape(1,1000,163)
  if getclass3[j]==0:
      scores3[j,:]=scaler_out.inverse_transform(model1.predict(pred_data1))
  elif getclass3[j]==1:
      scores3[j,:]=scaler_out.inverse_transform(model2.predict(pred_data1))
  elif getclass3[j]==2:
      scores3[j,:]=scaler_out.inverse_transform(model4.predict(pred_data1))
  else:
      scores3[j,:]=scaler_out.inverse_transform(model5.predict(pred_data1))
 prev1=data_std.iloc[idx_BY[id_fold[0]],[12,52]]
 prev2=data_std.iloc[idx_BY[id_fold[1]],[12,52]]
 prev3=data_std.iloc[idx_BY[id_fold[2]],[12,52]]
 act1=data_std.iloc[idx_BY[id_fold[0]],[5,6]]
 act2=data_std.iloc[idx_BY[id_fold[1]],[5,6]]
 act3=data_std.iloc[idx_BY[id_fold[2]],[5,6]]
 scores1=scores1+prev1
 scores2=scores2+prev2
 scores3=scores3+prev3
 mspe1=np.mean(((scores1.iloc[:,0]-act1.iloc[:,0])**2+(scores1.iloc[:,1]-act1.iloc[:,1])**2)**0.5)
 mspe2=np.mean(((scores2.iloc[:,0]-act2.iloc[:,0])**2+(scores2.iloc[:,1]-act2.iloc[:,1])**2)**0.5)
 mspe3=np.mean(((scores3.iloc[:,0]-act3.iloc[:,0])**2+(scores3.iloc[:,1]-act3.iloc[:,1])**2)**0.5)
 fold_mspe[(k-1)]=np.mean((mspe1, mspe2,mspe3))
 np.save('LSTMlag40best.npy', fold_mspe)

