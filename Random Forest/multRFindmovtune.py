
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix



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
#data[cols_to_norm] = data[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max()-x.min()))
data[cols_to_norm] = data[cols_to_norm].apply(lambda x: (x - x.mean()) / (x.std()))

train1=data[data['t']<=11515]
test1=data[(data['t']>11515) ]

train_in=train1.iloc[:, np.r_[9:36,37:train1.shape[1]]]
train_out=train1.iloc[:, 36]

test_in=test1.iloc[:, np.r_[9:36,37:test1.shape[1]]]
test_out=test1.iloc[:, 36]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(train_out)
train_out = encoder.transform(train_out)

test_out = encoder.transform(test_out)

train_in=train_in.values



train_in=train_in.astype(np.float32)

test_in=test_in.values



test_in=test_in.astype(np.float32)


m_depth=np.linspace(2, 50, num=20).astype(int)
m_samples_split=np.linspace(2, 10, num=5).astype(int)
m_samples_leaf=np.linspace(1, 10, num=5).astype(int)
m_depth_series=np.repeat(m_depth, 25)
m_samples_split_series=np.tile(np.repeat(m_samples_split, 5),20)
m_samples_leaf_series=np.tile(m_samples_leaf, 100)
MSPEtrain=np.zeros(500)
MSPEtest=np.zeros(500)
tune=np.column_stack((m_depth_series,m_samples_split_series,m_samples_leaf_series,MSPEtrain,  MSPEtest  ))

for r in range(tune.shape[0]):
    regr = RandomForestClassifier( random_state=0,max_depth=tune[r, 0].astype(int), min_samples_split=tune[r, 1].astype(int), min_samples_leaf=tune[r,2].astype(int),
                              n_estimators=500, n_jobs=-2)
    regr.fit(train_in, train_out)

    test_pred=regr.predict(test_in)
    train_pred=regr.predict(train_in)
    cm = confusion_matrix(test_out, test_pred)
    tune[r,4]=(cm[0,0]+cm[1,1]+cm[2,2])/(sum(cm[0, ])+sum( cm[1,])+sum(cm[2,]))
    cm1 = confusion_matrix(train_out, train_pred)
    tune[r,3]=(cm1[0,0]+cm1[1,1]+cm1[2,2])/(sum(cm1[0, ])+sum( cm1[1,])+sum(cm1[2,]))
    np.save('RFindmovttune.npy', tune)





