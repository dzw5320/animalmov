from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix



filename='gulls50.csv'
data_in=pd.read_csv(filename, sep=',',header=0)
data_in.columns

data_std=data_in

data_transform=data_std.iloc[:, np.r_[10:213]]

# Step 1: Classification

#Normalize the input variables

from sklearn import preprocessing

scaler_in = preprocessing.StandardScaler()

scaler_in.fit(data_transform)

data_transform=scaler_in.transform(data_transform)

scaler_out = preprocessing.StandardScaler()
scaler_out.fit(data_in[['vx','vy']])
data_out_transform=scaler_out.transform(data_in[['vx','vy']])

id1=np.where(data_std['state']==1)[0]
id2=np.where(data_std['state']==2)[0]
id4=np.where(data_std['state']==4)[0]
id5=np.where(data_std['state']==5)[0]

data_std1=data_std[data_std['state']==1]
data_std2=data_std[data_std['state']==2]
data_std4=data_std[data_std['state']==4]
data_std5=data_std[data_std['state']==5]

data_transform1=data_transform[id1]
data_transform2=data_transform[id2]
data_transform4=data_transform[id4]
data_transform5=data_transform[id5]

data_out_transform1=data_out_transform[id1]
data_out_transform2=data_out_transform[id2]
data_out_transform4=data_out_transform[id4]
data_out_transform5=data_out_transform[id5]

fold_mspe=np.zeros(5)


for f in range(1,6,1):

 #Fit models using the data not in fold
 #Classification model for state
 idx_nofold=np.where((data_std['fold']==f) & (data_std['common']==1))[0]
 idx_fold=np.where(data_std['fold']!=f)[0]
 fold_vel_act=data_in[['vx','vy']].values[idx_fold]
 fold_vel_pred=np.zeros((fold_vel_act.shape[0], 2))
 fold_in=data_transform[idx_fold]
 fold_out=data_transform[idx_nofold]
 classfr = RandomForestClassifier( max_depth=9, min_samples_split=2, min_samples_leaf=3,
                              n_estimators=5, n_jobs=-2)
 classfr.fit(fold_in, data_in['state'].values[idx_fold])
 #Velocity model for state 1
 idx_nofold1=np.where((data_std1['fold']==f) & (data_std1['common']==1))[0]
 idx_fold1=np.where(data_std1['fold']!=f)[0]
 fold_in1=data_transform1[idx_fold1]
 fold_out1=data_transform1[idx_nofold1]
 regr1 = RandomForestRegressor( max_depth=9, min_samples_split=10, min_samples_leaf=3,
                              n_estimators=5, n_jobs=-1)
 regr1.fit(fold_in1, data_out_transform1[idx_fold1])
 #Velocity model for state 2
 idx_nofold2=np.where((data_std2['fold']==f) & (data_std2['common']==1))[0]
 idx_fold2=np.where(data_std2['fold']!=f)[0]
 fold_in2=data_transform2[idx_fold2]
 fold_out2=data_transform2[idx_nofold2]
 regr2 = RandomForestRegressor( max_depth=2, min_samples_split=2, min_samples_leaf=5,
                              n_estimators=5, n_jobs=-1)
 regr2.fit(fold_in2, data_out_transform2[idx_fold2])
 #Velocity model for state 4
 idx_nofold4=np.where((data_std4['fold']==f) & (data_std4['common']==1))[0]
 idx_fold4=np.where(data_std4['fold']!=f)[0]
 fold_in4=data_transform4[idx_fold4]
 fold_out4=data_transform4[idx_nofold4]
 regr4 = RandomForestRegressor( max_depth=7, min_samples_split=4, min_samples_leaf=1,
                              n_estimators=5, n_jobs=-1)
 regr4.fit(fold_in4, data_out_transform4[idx_fold4])
 #Velocity model for state 5
 idx_nofold5=np.where((data_std5['fold']==f) & (data_std5['common']==1))[0]
 idx_fold5=np.where(data_std5['fold']!=f)[0]
 fold_in5=data_transform5[idx_fold5]
 fold_out5=data_transform5[idx_nofold5]
 regr5 = RandomForestRegressor( max_depth=4, min_samples_split=8, min_samples_leaf=1,
                              n_estimators=5, n_jobs=-1)
 regr5.fit(fold_in5, data_out_transform5[idx_fold5])
 #Predict state for all in fold==f
 test_pred=classfr.predict(fold_out)
 for j in range(test_pred.shape[0]):
    if test_pred[j]==1:
        fold_vel_pred[j,:]=scaler_out.inverse_transform(regr1.predict(fold_out[j:(j+1),:]))
    elif test_pred[j]==2:
        fold_vel_pred[j,:]=scaler_out.inverse_transform(regr2.predict(fold_out[j:(j+1),:]))
    elif test_pred[j]==4:
        fold_vel_pred[j,:]=scaler_out.inverse_transform(regr4.predict(fold_out[j:(j+1),:]))
    else:
        fold_vel_pred[j,:]=scaler_out.inverse_transform(regr5.predict(fold_out[j:(j+1),:]))
 fold_mspe[f]=np.mean(((fold_vel_pred.iloc[:,0]-fold_vel_act.iloc[:,0])**2+(fold_vel_pred.iloc[:,1]-fold_vel_act.iloc[:,1])**2)**0.5)
 np.save('RFlag50best.npy', fold_mspe)





