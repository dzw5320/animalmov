

from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd


filename='/storage/home/d/dzw5320/MachineLearning/Data/Col_inall.csv'
data_in=pd.read_csv(filename, sep=',',header=0)
train_in_=data_in[data_in['t']<=11515]
test_in_=data_in[data_in['t']>11515]



filename='/storage/home/d/dzw5320/MachineLearning/Data/Col_out.csv'
data_out=pd.read_csv(filename, sep=',',header=0)

data_out=data_out.iloc[:, np.r_[1, 75:221]]



train_out=data_out[data_out['t']<=11515]

train_out=train_out.iloc[:, np.r_[1:train_out.shape[1]]]
train_out_mean=train_out.mean()
train_out_std=train_out.std()
train_out=(train_out-train_out.mean())/train_out.std()
train_out=train_out.values

train_out=train_out.astype(np.float32)


test_out=data_out[data_out['t']>11515]

test_out=test_out.iloc[:, np.r_[1:test_out.shape[1]]]

test_out=test_out.values

test_out=test_out.astype(np.float32)

xnumbers1=np.arange(0, 73, 1)
xnumbers2=np.arange(365, 438, 1)
xnumbers=np.concatenate((xnumbers1, xnumbers2))
xnumbers=np.reshape(xnumbers, (146))    

train_in=train_in_
train_in=train_in.iloc[:, np.r_[2:(train_in.shape[1]-1)]]
train_xy_1=train_in.values[:,xnumbers]
train_mean=train_in.mean()
train_std=train_in.std()
train_in=(train_in-train_in.mean())/train_in.std()
criteria=train_std>0.0
train_in=train_in[criteria.index[criteria]]
train_in=train_in.values
train_in=train_in.astype(np.float32)

test_in=test_in_
test_in=test_in.iloc[:, np.r_[2:(test_in.shape[1]-1)]]
test_xy_1=test_in.values[:,xnumbers]
test_in=(test_in-train_mean)/train_std
test_in=test_in[criteria.index[criteria]]
test_in=test_in.values




test_in=test_in.astype(np.float32)

train_out_unstd=np.zeros(( 11511, 146))
for i in range(146):
  train_out_unstd[:,i]=train_out[:, i]*train_out_std[i]+train_out_mean[i]

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

  regr = RandomForestRegressor( random_state=0,max_depth=tune[r, 0].astype(int), min_samples_split=tune[r, 1].astype(int), min_samples_leaf=tune[r,2].astype(int),
                              n_estimators=75)
  regr.fit(train_in, train_out)
  test_pred=regr.predict(test_in)
  train_pred=regr.predict(train_in)
  samplepredictions_unstd=np.zeros(( 11511, 146))
  for i in range(146):
    samplepredictions_unstd[:,i]=train_pred[:, i]*train_out_std[i]+train_out_mean[i]
  samplepredictions_unstd_test=np.zeros(( 2883, 146))
  for i in range(146):
    samplepredictions_unstd_test[:,i]=test_pred[:, i]*train_out_std[i]+train_out_mean[i]
  restrain=np.zeros([train_out.shape[0], 146])
  restrain=samplepredictions_unstd+train_xy_1
  restest=np.zeros([test_out.shape[0], 146])
  restest=samplepredictions_unstd_test+test_xy_1
  act_train=train_out_unstd+train_xy_1
  restrain_reshape=restrain.reshape((train_out.shape[0], 2, 73))
  train_out_reshape=act_train.reshape((train_out.shape[0], 2, 73))
  restest_reshape=restest.reshape((test_in.shape[0], 2, 73))
  act_test=test_out+test_xy_1
  test_out_reshape=act_test.reshape((test_in.shape[0], 2, 73))
  checktrain=np.zeros((11511, 73))
  checktrain=((restrain_reshape[:, 0, :]-train_out_reshape[:, 0,:])**2+(restrain_reshape[:, 1, :]-train_out_reshape[:, 1,:])**2)**(0.5)
  tune[r,3 ]=checktrain.sum(axis=0).sum()/(11511*73)
  checktest=np.zeros((2883, 73))
  checktest=((restest_reshape[:, 0, :]-test_out_reshape[:, 0,:])**2+(restest_reshape[:, 1, :]-test_out_reshape[:, 1,:])**2)**(0.5)
  tune[r,4]=checktest.sum(axis=0).sum()/(test_in.shape[0]*73)
  np.save('RFcolveltune.npy', tune)

