
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix
import random


filename='/storage/work/d/dzw5320/Gulls/Data/gulls50.csv'
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

#Partition training data. Simulate bird years 1,7 and 12. Use all others years as training data.



idx_train=np.where( (data_std['BirdYear']!=7) )[0]
train_in=data_transform[idx_train]

# Classification model
classfr = RandomForestClassifier( max_depth=9, min_samples_split=6, min_samples_leaf=10,
                              n_estimators=250, n_jobs=-1)
classfr.fit(train_in, data_in['state'].values[idx_train])

# Velocity model for state 1
id_train1=np.where((data_std1['BirdYear']!=7) )[0]
train_in1=data_transform1[id_train1]

regr1 = RandomForestRegressor( max_depth=9, min_samples_split=4, min_samples_leaf=5,
                              n_estimators=250, n_jobs=-1)
regr1.fit(train_in1, data_out_transform1[id_train1])

# Velocity model for state 2
id_train2=np.where((data_std2['BirdYear']!=7) )[0]
train_in2=data_transform2[id_train2]

regr2 = RandomForestRegressor(  max_depth=7, min_samples_split=2, min_samples_leaf=1,
                              n_estimators=250, n_jobs=-1)

regr2.fit(train_in2, data_out_transform2[id_train2])

# Velocity model for state 4

id_train4=np.where((data_std4['BirdYear']!=7) )[0]
train_in4=data_transform4[id_train4]

regr4 = RandomForestRegressor( max_depth=7, min_samples_split=8, min_samples_leaf=7,
                              n_estimators=250, n_jobs=-1)

regr4.fit(train_in4, data_out_transform4[id_train4])

# Velocity model for state 5

id_train5=np.where((data_std5['BirdYear']!=7) )[0]
train_in5=data_transform5[id_train5]

regr5 = RandomForestRegressor( max_depth=7, min_samples_split=4, min_samples_leaf=1,
                              n_estimators=250, n_jobs=-1)


regr5.fit(train_in5, data_out_transform5[id_train5])

def getClass(parray):
  onehot=np.random.multinomial(1, parray)
  state=np.array([1,2,4,5])[np.argmax(onehot, axis=0)]
  return state

def euc_dist(x1, x2, y1, y2):
  
  dist=((x1-x2)**2+(y1-y2)**2)**(0.5)
  return dist

veuc_dist = np.vectorize(euc_dist)

idx_sim7=np.where( (data_std['BirdYear']==7)   )[0]

sim7=data_std.loc[idx_sim7]

#Simulate state 1 to 2

for sim in range(5):
 sim7=data_std.loc[idx_sim7]
 for r in range(2696, 2996, 1):
  #Let's start with sim
  sim_step_in=sim7.iloc[r:(r+1), np.r_[10:213]]
  sim_step_in=scaler_in.transform(sim_step_in)
  predstate=classfr.predict_proba(sim_step_in)
  sim7['state'][r]=getClass(predstate[-1])
  if sim7['state'][r]==1:
   tree=random.randint(0,249)#change to 249
   vxvy_invtransform=scaler_out.inverse_transform(regr1.estimators_[tree].predict(sim_step_in))
  elif sim7['state'][r]==2:
   tree=random.randint(0,249)#change to 249
   vxvy_invtransform=scaler_out.inverse_transform(regr2.estimators_[tree].predict(sim_step_in))
  elif sim7['state'][r]==4:
   tree=random.randint(0,249)#change to 249
   vxvy_invtransform=scaler_out.inverse_transform(regr4.estimators_[tree].predict(sim_step_in))
  else:
   tree=random.randint(0,249)#change to 249
   vxvy_invtransform=scaler_out.inverse_transform(regr5.estimators_[tree].predict(sim_step_in)) 
  sim7.iloc[r:(r+1), np.r_[7:9]]=vxvy_invtransform[-1]
  sim7['x'][r:(r+1)]=sim7['x-1'][r:(r+1)]+sim7['vx'][r:(r+1)]
  sim7['y'][r:(r+1)]=sim7['y-1'][r:(r+1)]+sim7['vy'][r:(r+1)]
  sim7['x-1'][(r+1):(r+2)]=sim7['x'][r:(r+1)]
  sim7['y-1'][(r+1):(r+2)]=sim7['y'][r:(r+1)]
  sim7['vx-1'][(r+1):(r+2)]=sim7['vx'][r:(r+1)]
  sim7['vy-1'][(r+1):(r+2)]=sim7['vy'][r:(r+1)]
  #update lagged x,y
  sim7.iloc[(r+1):(r+2),np.r_[13:62]]=sim7.iloc[r:(r+1),np.r_[12:61]].values#update x-2:x-50
  sim7.iloc[(r+1):(r+2),63:112]=sim7.iloc[r:(r+1),np.r_[62:111]].values#update y-2:y-50
  #update lagged vx, vy
  sim7.iloc[(r+1):(r+2),113:162]=sim7.iloc[r:(r+1),np.r_[112:161]].values#update vx-2:vx-50
  sim7.iloc[(r+1):(r+2),163:212]=sim7.iloc[r:(r+1),np.r_[162:211]].values#update vy-2:vy-50
  #update dist
  sim7['dist'][(r+1):(r+2)]=veuc_dist(sim7['x-1'][(r+1):(r+2)],sim7['x-2'][(r+1):(r+2)],sim7['y-1'][(r+1):(r+2)],sim7['y-2'][(r+1):(r+2)])
 get_sim=sim7.to_numpy()
 filename='/gpfs/group/emh30/default/dzw5320/sims/sim7/'+str(sim)+'RF7_50_1_2.npy'
 np.save(filename, get_sim)

#simulate from state 4 to 5

for sim in range(5):
 sim7=data_std.loc[idx_sim7]
 for r in range(13762, 14062, 1):
  #Let's start with sim
  sim_step_in=sim7.iloc[r:(r+1), np.r_[10:213]]
  sim_step_in=scaler_in.transform(sim_step_in)
  predstate=classfr.predict_proba(sim_step_in)
  sim7['state'][r]=getClass(predstate[-1])
  if sim7['state'][r]==1:
   tree=random.randint(0,249)#change to 249
   vxvy_invtransform=scaler_out.inverse_transform(regr1.estimators_[tree].predict(sim_step_in))
  elif sim7['state'][r]==2:
   tree=random.randint(0,249)#change to 249
   vxvy_invtransform=scaler_out.inverse_transform(regr2.estimators_[tree].predict(sim_step_in))
  elif sim7['state'][r]==4:
   tree=random.randint(0,249)#change to 249
   vxvy_invtransform=scaler_out.inverse_transform(regr4.estimators_[tree].predict(sim_step_in))
  else:
   tree=random.randint(0,249)#change to 249
   vxvy_invtransform=scaler_out.inverse_transform(regr5.estimators_[tree].predict(sim_step_in)) 
  sim7.iloc[r:(r+1), np.r_[7:9]]=vxvy_invtransform[-1]
  sim7['x'][r:(r+1)]=sim7['x-1'][r:(r+1)]+sim7['vx'][r:(r+1)]
  sim7['y'][r:(r+1)]=sim7['y-1'][r:(r+1)]+sim7['vy'][r:(r+1)]
  sim7['x-1'][(r+1):(r+2)]=sim7['x'][r:(r+1)]
  sim7['y-1'][(r+1):(r+2)]=sim7['y'][r:(r+1)]
  sim7['vx-1'][(r+1):(r+2)]=sim7['vx'][r:(r+1)]
  sim7['vy-1'][(r+1):(r+2)]=sim7['vy'][r:(r+1)]
  #update lagged x,y
  sim7.iloc[(r+1):(r+2),np.r_[13:62]]=sim7.iloc[r:(r+1),np.r_[12:61]].values#update x-2:x-50
  sim7.iloc[(r+1):(r+2),63:112]=sim7.iloc[r:(r+1),np.r_[62:111]].values#update y-2:y-50
  #update lagged vx, vy
  sim7.iloc[(r+1):(r+2),113:162]=sim7.iloc[r:(r+1),np.r_[112:161]].values#update vx-2:vx-50
  sim7.iloc[(r+1):(r+2),163:212]=sim7.iloc[r:(r+1),np.r_[162:211]].values#update vy-2:vy-50
  #update dist
  sim7['dist'][(r+1):(r+2)]=veuc_dist(sim7['x-1'][(r+1):(r+2)],sim7['x-2'][(r+1):(r+2)],sim7['y-1'][(r+1):(r+2)],sim7['y-2'][(r+1):(r+2)])
 get_sim=sim7.to_numpy()
 filename='/gpfs/group/emh30/default/dzw5320/sims/sim7/'+str(sim)+'RF7_50_4_5.npy'
 np.save(filename, get_sim)

#simulate during state 2

for sim in range(5):
 sim7=data_std.loc[idx_sim7]
 for r in range(2880, 3180, 1):
  #Let's start with sim
  sim_step_in=sim7.iloc[r:(r+1), np.r_[10:213]]
  sim_step_in=scaler_in.transform(sim_step_in)
  predstate=classfr.predict_proba(sim_step_in)
  sim7['state'][r]=getClass(predstate[-1])
  if sim7['state'][r]==1:
   tree=random.randint(0,249)#change to 249
   vxvy_invtransform=scaler_out.inverse_transform(regr1.estimators_[tree].predict(sim_step_in))
  elif sim7['state'][r]==2:
   tree=random.randint(0,249)#change to 249
   vxvy_invtransform=scaler_out.inverse_transform(regr2.estimators_[tree].predict(sim_step_in))
  elif sim7['state'][r]==4:
   tree=random.randint(0,249)#change to 249
   vxvy_invtransform=scaler_out.inverse_transform(regr4.estimators_[tree].predict(sim_step_in))
  else:
   tree=random.randint(0,249)#change to 249
   vxvy_invtransform=scaler_out.inverse_transform(regr5.estimators_[tree].predict(sim_step_in)) 
  sim7.iloc[r:(r+1), np.r_[7:9]]=vxvy_invtransform[-1]
  sim7['x'][r:(r+1)]=sim7['x-1'][r:(r+1)]+sim7['vx'][r:(r+1)]
  sim7['y'][r:(r+1)]=sim7['y-1'][r:(r+1)]+sim7['vy'][r:(r+1)]
  sim7['x-1'][(r+1):(r+2)]=sim7['x'][r:(r+1)]
  sim7['y-1'][(r+1):(r+2)]=sim7['y'][r:(r+1)]
  sim7['vx-1'][(r+1):(r+2)]=sim7['vx'][r:(r+1)]
  sim7['vy-1'][(r+1):(r+2)]=sim7['vy'][r:(r+1)]
  #update lagged x,y
  sim7.iloc[(r+1):(r+2),np.r_[13:62]]=sim7.iloc[r:(r+1),np.r_[12:61]].values#update x-2:x-50
  sim7.iloc[(r+1):(r+2),63:112]=sim7.iloc[r:(r+1),np.r_[62:111]].values#update y-2:y-50
  #update lagged vx, vy
  sim7.iloc[(r+1):(r+2),113:162]=sim7.iloc[r:(r+1),np.r_[112:161]].values#update vx-2:vx-50
  sim7.iloc[(r+1):(r+2),163:212]=sim7.iloc[r:(r+1),np.r_[162:211]].values#update vy-2:vy-50
  #update dist
  sim7['dist'][(r+1):(r+2)]=veuc_dist(sim7['x-1'][(r+1):(r+2)],sim7['x-2'][(r+1):(r+2)],sim7['y-1'][(r+1):(r+2)],sim7['y-2'][(r+1):(r+2)])
 get_sim=sim7.to_numpy()
 filename='/gpfs/group/emh30/default/dzw5320/sims/sim7/'+str(sim)+'RF7_50_2.npy'
 np.save(filename, get_sim)


#simulate during state 5

for sim in range(5):
 sim7=data_std.loc[idx_sim7]
 for r in range(13814, 14114, 1):
  #Let's start with sim
  sim_step_in=sim7.iloc[r:(r+1), np.r_[10:213]]
  sim_step_in=scaler_in.transform(sim_step_in)
  predstate=classfr.predict_proba(sim_step_in)
  sim7['state'][r]=getClass(predstate[-1])
  if sim7['state'][r]==1:
   tree=random.randint(0,249)#change to 249
   vxvy_invtransform=scaler_out.inverse_transform(regr1.estimators_[tree].predict(sim_step_in))
  elif sim7['state'][r]==2:
   tree=random.randint(0,249)#change to 249
   vxvy_invtransform=scaler_out.inverse_transform(regr2.estimators_[tree].predict(sim_step_in))
  elif sim7['state'][r]==4:
   tree=random.randint(0,249)#change to 249
   vxvy_invtransform=scaler_out.inverse_transform(regr4.estimators_[tree].predict(sim_step_in))
  else:
   tree=random.randint(0,249)#change to 249
   vxvy_invtransform=scaler_out.inverse_transform(regr5.estimators_[tree].predict(sim_step_in)) 
  sim7.iloc[r:(r+1), np.r_[7:9]]=vxvy_invtransform[-1]
  sim7['x'][r:(r+1)]=sim7['x-1'][r:(r+1)]+sim7['vx'][r:(r+1)]
  sim7['y'][r:(r+1)]=sim7['y-1'][r:(r+1)]+sim7['vy'][r:(r+1)]
  sim7['x-1'][(r+1):(r+2)]=sim7['x'][r:(r+1)]
  sim7['y-1'][(r+1):(r+2)]=sim7['y'][r:(r+1)]
  sim7['vx-1'][(r+1):(r+2)]=sim7['vx'][r:(r+1)]
  sim7['vy-1'][(r+1):(r+2)]=sim7['vy'][r:(r+1)]
  #update lagged x,y
  sim7.iloc[(r+1):(r+2),np.r_[13:62]]=sim7.iloc[r:(r+1),np.r_[12:61]].values#update x-2:x-50
  sim7.iloc[(r+1):(r+2),63:112]=sim7.iloc[r:(r+1),np.r_[62:111]].values#update y-2:y-50
  #update lagged vx, vy
  sim7.iloc[(r+1):(r+2),113:162]=sim7.iloc[r:(r+1),np.r_[112:161]].values#update vx-2:vx-50
  sim7.iloc[(r+1):(r+2),163:212]=sim7.iloc[r:(r+1),np.r_[162:211]].values#update vy-2:vy-50
  #update dist
  sim7['dist'][(r+1):(r+2)]=veuc_dist(sim7['x-1'][(r+1):(r+2)],sim7['x-2'][(r+1):(r+2)],sim7['y-1'][(r+1):(r+2)],sim7['y-2'][(r+1):(r+2)])
 get_sim=sim7.to_numpy()
 filename='/gpfs/group/emh30/default/dzw5320/sims/sim7/'+str(sim)+'RF7_50_5.npy'
 np.save(filename, get_sim)


#simulate from state 2 to 4

for sim in range(5):
 sim7=data_std.loc[idx_sim7]
 for r in range(3003, 3303, 1):
  #Let's start with sim
  sim_step_in=sim7.iloc[r:(r+1), np.r_[10:213]]
  sim_step_in=scaler_in.transform(sim_step_in)
  predstate=classfr.predict_proba(sim_step_in)
  sim7['state'][r]=getClass(predstate[-1])
  if sim7['state'][r]==1:
   tree=random.randint(0,249)#change to 249
   vxvy_invtransform=scaler_out.inverse_transform(regr1.estimators_[tree].predict(sim_step_in))
  elif sim7['state'][r]==2:
   tree=random.randint(0,249)#change to 249
   vxvy_invtransform=scaler_out.inverse_transform(regr2.estimators_[tree].predict(sim_step_in))
  elif sim7['state'][r]==4:
   tree=random.randint(0,249)#change to 249
   vxvy_invtransform=scaler_out.inverse_transform(regr4.estimators_[tree].predict(sim_step_in))
  else:
   tree=random.randint(0,249)#change to 249
   vxvy_invtransform=scaler_out.inverse_transform(regr5.estimators_[tree].predict(sim_step_in)) 
  sim7.iloc[r:(r+1), np.r_[7:9]]=vxvy_invtransform[-1]
  sim7['x'][r:(r+1)]=sim7['x-1'][r:(r+1)]+sim7['vx'][r:(r+1)]
  sim7['y'][r:(r+1)]=sim7['y-1'][r:(r+1)]+sim7['vy'][r:(r+1)]
  sim7['x-1'][(r+1):(r+2)]=sim7['x'][r:(r+1)]
  sim7['y-1'][(r+1):(r+2)]=sim7['y'][r:(r+1)]
  sim7['vx-1'][(r+1):(r+2)]=sim7['vx'][r:(r+1)]
  sim7['vy-1'][(r+1):(r+2)]=sim7['vy'][r:(r+1)]
  #update lagged x,y
  sim7.iloc[(r+1):(r+2),np.r_[13:62]]=sim7.iloc[r:(r+1),np.r_[12:61]].values#update x-2:x-50
  sim7.iloc[(r+1):(r+2),63:112]=sim7.iloc[r:(r+1),np.r_[62:111]].values#update y-2:y-50
  #update lagged vx, vy
  sim7.iloc[(r+1):(r+2),113:162]=sim7.iloc[r:(r+1),np.r_[112:161]].values#update vx-2:vx-50
  sim7.iloc[(r+1):(r+2),163:212]=sim7.iloc[r:(r+1),np.r_[162:211]].values#update vy-2:vy-50
  #update dist
  sim7['dist'][(r+1):(r+2)]=veuc_dist(sim7['x-1'][(r+1):(r+2)],sim7['x-2'][(r+1):(r+2)],sim7['y-1'][(r+1):(r+2)],sim7['y-2'][(r+1):(r+2)])
 get_sim=sim7.to_numpy()
 filename='/gpfs/group/emh30/default/dzw5320/sims/sim7/'+str(sim)+'RF7_50_2_4.npy'
 np.save(filename, get_sim)


#simulate from state 5 to 1

for sim in range(5):
 sim7=data_std.loc[idx_sim7]
 for r in range(14639, 14939, 1):
  #Let's start with sim
  sim_step_in=sim7.iloc[r:(r+1), np.r_[10:213]]
  sim_step_in=scaler_in.transform(sim_step_in)
  predstate=classfr.predict_proba(sim_step_in)
  sim7['state'][r]=getClass(predstate[-1])
  if sim7['state'][r]==1:
   tree=random.randint(0,249)#change to 249
   vxvy_invtransform=scaler_out.inverse_transform(regr1.estimators_[tree].predict(sim_step_in))
  elif sim7['state'][r]==2:
   tree=random.randint(0,249)#change to 249
   vxvy_invtransform=scaler_out.inverse_transform(regr2.estimators_[tree].predict(sim_step_in))
  elif sim7['state'][r]==4:
   tree=random.randint(0,249)#change to 249
   vxvy_invtransform=scaler_out.inverse_transform(regr4.estimators_[tree].predict(sim_step_in))
  else:
   tree=random.randint(0,249)#change to 249
   vxvy_invtransform=scaler_out.inverse_transform(regr5.estimators_[tree].predict(sim_step_in)) 
  sim7.iloc[r:(r+1), np.r_[7:9]]=vxvy_invtransform[-1]
  sim7['x'][r:(r+1)]=sim7['x-1'][r:(r+1)]+sim7['vx'][r:(r+1)]
  sim7['y'][r:(r+1)]=sim7['y-1'][r:(r+1)]+sim7['vy'][r:(r+1)]
  sim7['x-1'][(r+1):(r+2)]=sim7['x'][r:(r+1)]
  sim7['y-1'][(r+1):(r+2)]=sim7['y'][r:(r+1)]
  sim7['vx-1'][(r+1):(r+2)]=sim7['vx'][r:(r+1)]
  sim7['vy-1'][(r+1):(r+2)]=sim7['vy'][r:(r+1)]
  #update lagged x,y
  sim7.iloc[(r+1):(r+2),np.r_[13:62]]=sim7.iloc[r:(r+1),np.r_[12:61]].values#update x-2:x-50
  sim7.iloc[(r+1):(r+2),63:112]=sim7.iloc[r:(r+1),np.r_[62:111]].values#update y-2:y-50
  #update lagged vx, vy
  sim7.iloc[(r+1):(r+2),113:162]=sim7.iloc[r:(r+1),np.r_[112:161]].values#update vx-2:vx-50
  sim7.iloc[(r+1):(r+2),163:212]=sim7.iloc[r:(r+1),np.r_[162:211]].values#update vy-2:vy-50
  #update dist
  sim7['dist'][(r+1):(r+2)]=veuc_dist(sim7['x-1'][(r+1):(r+2)],sim7['x-2'][(r+1):(r+2)],sim7['y-1'][(r+1):(r+2)],sim7['y-2'][(r+1):(r+2)])
 get_sim=sim7.to_numpy()
 filename='/gpfs/group/emh30/default/dzw5320/sims/sim7/'+str(sim)+'RF7_50_5_1.npy'
 np.save(filename, get_sim)


