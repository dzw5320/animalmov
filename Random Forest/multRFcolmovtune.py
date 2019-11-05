

from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd


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

m_depth=np.linspace(2, 100, num=25).astype(int)
m_samples_split=np.linspace(2, 15, num=5).astype(int)
m_samples_leaf=np.linspace(1, 10, num=5).astype(int)
m_depth_series=np.repeat(m_depth, 25)
m_samples_split_series=np.tile(np.repeat(m_samples_split, 5),25)
m_samples_leaf_series=np.tile(m_samples_leaf, 125)
MSPEtrain=np.zeros(625)
MSPEtest=np.zeros(625)
tune=np.column_stack((m_depth_series,m_samples_split_series,m_samples_leaf_series,MSPEtrain,  MSPEtest  ))

for r in range(0, 626, 1):

  regr = RandomForestClassifier( random_state=0,max_depth=tune[r, 0].astype(int), min_samples_split=tune[r, 1].astype(int), min_samples_leaf=tune[r,2].astype(int),
                              n_estimators=500)
  regr.fit(train_in, train_out)
  test_pred=regr.predict(test_in)
  train_pred=regr.predict(train_in)
  checktrain=(train_pred==train_out)
  tune[r,3 ]=np.sum(checktrain)/(checktrain.shape[0]*checktrain.shape[1])
  checktest=(test_pred==test_out)
  tune[r,4]=np.sum(checktest)/(checktest.shape[0]*checktest.shape[1])
  np.save('RFcolmovtune.npy', tune)



