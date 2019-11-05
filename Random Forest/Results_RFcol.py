from __future__ import print_function
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.layers import Input, Embedding, Dense, Dropout
from keras.models import Model

from keras.layers.core import Dense, Dropout, Activation

from keras.models import Sequential
from keras.utils import np_utils


from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import talos as ta
from talos.model.normalizers import lr_normalizer
from keras.models import model_from_json

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, Nadam, Adam, RMSprop


import tensorflow as tf



from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np
import pandas as pd
#from joblib import dump, load
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix






#Projection function
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely.geometry.polygon import LinearRing, Polygon


def isPointInPath(x, y, poly):
    """
    x, y -- x and y coordinates of point
    poly -- a list of tuples [(x, y), (x, y), ...]
    """
    num = len(poly)
    i = 0
    j = num - 1
    c = False
    for i in range(num):
        if ((poly[i][1] > y) != (poly[j][1] > y)) and \
                (x < poly[i][0] + (poly[j][0] - poly[i][0]) * (y - poly[i][1]) /
                                  (poly[j][1] - poly[i][1])):
            c = not c
        j = i
    return c
  
  
  
  
#Create function that takes in points and poly and gives array of T/F

import shapely
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

poly=[(0.0, 0),(18.5, 0), (18.5, 53), (21.5, 53), (21.5, 0), (70.5, 0), (70.5, 53),
      (73.5,53),(73.5, 0),(122.5, 0), (122.5, 53), ( 125.5, 53), (125.5,0),( 174.5, 0),
      (174.5, 53), ( 177.5, 53), (177.5, 0), (199.01, 0), (199.01, 6),(196.0, 6), 
      (196.0, 65), (156.0, 65), (156.0, 6), (144.0, 6), (144.0, 65), (104.0, 65),
      (104.0, 6), (92.0, 6), (92.0, 65), (52.0, 65),(52.0, 6),(40.0, 6),(40.0, 65),(0.0, 65),(0.0, 0)]


  
#This considers points on the nest boundary as outside of the nest
def InPoly(parray, poly):
  polygon=Polygon(poly)
  T=len(parray.shape)
  if T>1:
    c = [Point(coord[0], coord[1]) for coord in parray]
    getin=[polygon.contains(each) for each in c]
  else:
    point1=Point(parray[0], parray[1])
    getin=polygon.contains(point1)
      
  return getin

#This considers points on the nest boundary as inside of the nest
def InPoly2(parray, poly):
  
  T=len(parray.shape)
  if T>1:
    
    getin=[isPointInPath(parray[each,0], parray[each,1], poly) for each in np.arange(parray.shape[0])]
  else:
    getin=isPointInPath(parray[0], parray[1], poly)
      
  return getin



#Now for the ones that are outside project onto the wall

def projpoly(parray, poly, index, eps=1.0e-6):
  l=np.array(poly[index+1])-np.array(poly[index])
  w=min(1, np.dot(parray-np.array(poly[index]), l)/np.dot(l, l))
  w=max(0, w)
  projection=np.array(poly[index])+w*l
  #to get prjection slightly in
  projection=projection+(projection-parray)/np.sqrt(np.dot(projection-parray,projection-parray))*eps
  return projection


def project(parray1, poly, eps=1.0e-6):
  parray=parray1
  T=parray.shape[0]
  nvert=len(poly)
  for i in np.arange(T):
    d=float('Inf')
    inp=InPoly2(parray[i], poly)
    if(inp==False):
      parrayOut=parray[i]
      parrayTmp=parray[i]
      for j in np.arange(nvert-1):
        parrayProj=projpoly(parrayOut, poly, j, eps)
        dtmp=(parrayOut[0]-parrayProj[0])**2+(parrayOut[1]-parrayProj[1])**2
        if dtmp<d:
          d=dtmp
          parrayTmp=parrayProj  
      parray[i]=parrayTmp 
    inpp=InPoly(parray[i],poly)
    if(inpp==False):
      parrayTmp1=parray[i].copy()
      parrayTmp2=parray[i].copy()
      parrayTmp3=parray[i].copy()
      parrayTmp4=parray[i].copy()
      parrayTmp1[0]=parrayTmp1[0]+eps
      parrayTmp1[1]=parrayTmp1[1]+eps
      parrayTmp2[0]=parrayTmp2[0]+eps
      parrayTmp2[1]=parrayTmp2[1]-eps
      parrayTmp3[0]=parrayTmp3[0]-eps
      parrayTmp3[1]=parrayTmp3[1]-eps
      parrayTmp4[0]=parrayTmp4[0]-eps
      parrayTmp4[1]=parrayTmp4[1]+eps
      inp1=InPoly(parrayTmp1,poly)
      inp2=InPoly(parrayTmp2,poly)
      inp3=InPoly(parrayTmp3,poly)
      inp4=InPoly(parrayTmp4,poly)
      if(inp1==True):
        parray[i]=parrayTmp1
      elif(inp2==True):
        parray[i]=parrayTmp2
      elif(inp3==True):
        parray[i]=parrayTmp3
      elif(inp4==True):
        parray[i]=parrayTmp4
  
  return parray

def getChamNo(x):
  cham=0  
  if(x<18.5):
    cham=1
  elif(x<46):
    cham=2
  elif(x<70.5):
    cham=3
  elif(x<98):
    cham=4
  elif(x<122.5):
    cham=5
  elif(x<150):
    cham=6
  elif(x<174.5):
    cham=7
  else:
    cham=8
  return cham

def euc_dist(x1, x2, y1, y2):
  
  dist=((x1-x2)**2+(y1-y2)**2)**(0.5)
  return dist

def getstattime(distindt,stattime_t ):
  if(distindt==0.0):
    stattime=stattime_t+1
  else:
    stattime=0
  return stattime

def distwalln(x, y):
  
  if((x<40) or (x>52 and x<92) or (x>104 and x<144) or (x>156 and x<196)):
    nwall=65
  else:
    nwall=6
  return nwall-y

def distwalls(x, y):
  
  if( (x>18.5 and x<21.5)or (x>70.5 and x<73.5) or(x>122.5 and x<125.5) or (x>174.5 and x<177.5) ):
    swall=53
  else:
    swall=0
  return y-swall

def distwallw(x, y, chamber):
  
  if((chamber==1) or (chamber==2 and  y>=53)):
    wwall=0
  elif((chamber==3 and y>=6) or (chamber==4 and y>=53) ):
    wwall=52
  elif((chamber==5 and y>=6) or (chamber==6 and y>=53)):
    wwall=104
  elif((chamber==7 and y>=6) or (chamber==8 and y>=53)):
    wwall=156
  elif((chamber==2 and y<53) or (chamber==3 and y<6)):
    wwall=21.5
  elif((chamber==4 and y<53) or (chamber==5 and y<6)):
    wwall=73.5
  elif((chamber==6 and y<53) or (chamber==7 and y<6)):
    wwall=125.5
  elif(chamber==8 and y<53):
    wwall=177.5
  else:
    wwall=NA
  
  return x-wwall

def distwalle(x, y, chamber):
  
  if(chamber==1 and y<53):
    ewall=18.5
  elif((chamber==1 and y>=53) or (chamber==2 and y>=6)):
    ewall=40
  elif((chamber==2 and y<6) or (chamber==3 and y<53)):
    ewall=70.5
  elif((chamber==3 and y>=53) or (chamber==4 and y>=6)):
    ewall=92
  elif((chamber==4 and y<6) or (chamber==5 and y<53)):
    ewall=122.5
  elif((chamber==5 and y>=53) or (chamber==6 and y>=6)):
    ewall=144
  elif((chamber==6 and y<6) or (chamber==7 and y<53)):
    ewall=174.5
  elif((chamber==7 and y>=53) or (chamber==8 and y>=6)):
    ewall=196
  elif((chamber==8 and y<6) or (x>=196)):
    ewall=199
  else:
    ewall=NA
  
  return ewall-x

import numpy as np
vgetChamNo = np.vectorize(getChamNo)
veuc_dist = np.vectorize(euc_dist)
vdistwalln = np.vectorize(distwalln)
vdistwalls = np.vectorize(distwalls)
vdistwalle = np.vectorize(distwalle)
vdistwallw = np.vectorize(distwallw)

vgetstattime=np.vectorize(getstattime)



def getQ1(iant, rest):
  
  
  Q1sub=rest[(rest['x-1'].values>(iant['x-1'].values-8)) & (rest['x-1'].values<(iant['x-1'].values)) & 
                  (rest['y-1'].values>(iant['y-1'].values)) & (rest['y-1'].values<(iant['y-1'].values+8))]
  
  return Q1sub.shape[0]

def getQ2(iant, rest):
  
  
  Q2sub=rest[(rest['x-1'].values>(iant['x-1'].values)) & (rest['x-1'].values<(iant['x-1'].values+8)) & 
                  (rest['y-1'].values>(iant['y-1'].values)) & (rest['y-1'].values<(iant['y-1'].values+8))]
  
  return Q2sub.shape[0]

def getQ3(iant, rest):
  
  
  Q3sub=rest[ (rest['x-1'].values>(iant['x-1'].values-8)) & (rest['x-1'].values<(iant['x-1'].values)) & 
                  (rest['y-1'].values>(iant['y-1'].values-8)) & (rest['y-1'].values<(iant['y-1'].values))]
  
  return Q3sub.shape[0]

def getQ4(iant, rest):
  
  
  Q4sub=rest[(rest['x-1'].values>(iant['x-1'].values)) & (rest['x-1'].values<(iant['x-1'].values+8)) & 
                  (rest['y-1'].values>(iant['y-1'].values-8)) & (rest['y-1'].values<(iant['y-1'].values))]
  
  return Q4sub.shape[0]

def getnndist(iant, rest):
  
  
  nndist=min(veuc_dist(x1 = iant['x-1'], y1 = iant['y-1'], x2 = rest['x-1'], y2 = rest['y-1']))
  return nndist

def getnnxlag1(iant, rest):
  
  dists=veuc_dist(x1 = iant['x-1'], y1 = iant['y-1'], x2 = rest['x-1'], y2 = rest['y-1'])
  id=np.where(dists == np.amin(dists))[0][0]
  return rest['x-1'].values[id]

def getnnylag1(iant, rest):
  
  dists=veuc_dist(x1 = iant['x-1'], y1 = iant['y-1'], x2 = rest['x-1'], y2 = rest['y-1'])
  id=np.where(dists==np.amin(dists))[0][0]
  return rest['y-1'].values[id]

def getnnvxlag1(iant, rest):
  
  dists=veuc_dist(x1 = iant['x-1'], y1 = iant['y-1'], x2 = rest['x-1'], y2 = rest['y-1'])
  id=np.where(dists==np.amin(dists))[0][0]
  return rest['vx-1'].values[id]

def getnnvylag1(iant, rest):
  
  dists=veuc_dist(x1 = iant['x-1'], y1 = iant['y-1'], x2 = rest['x-1'], y2 = rest['y-1'])
  id=np.where(dists==np.amin(dists))[0][0]
  return rest['vy-1'].values[id]

def getnnmove(iant, rest):
  
  dists=veuc_dist(x1 = iant['x-1'], y1 = iant['y-1'], x2 = rest['x-1'], y2 = rest['y-1'])
  idx=np.where(dists<12)[0][:]
  rest1=rest.iloc[idx]
  return sum(rest1['movt']==2)

def getnnstill(iant, rest):
  
  dists=veuc_dist(x1 = iant['x-1'], y1 = iant['y-1'], x2 = rest['x-1'], y2 = rest['y-1'])
  idx=np.where(dists<10)[0][:]
  rest1=rest.iloc[idx]
  return sum(rest1['movt']==0)

def getdistqueen(iant, queen):
  
  if((iant['id']!="Que").bool()):
    distqueen=veuc_dist(x1 = iant['x-1'], y1 = iant['y-1'], x2 = queen['x-1'], y2 = queen['y-1'])
  else:
    distqueen=[[0]]
    
  return distqueen[0]

def getClass(parray):
  parray=parray.astype(np.float64)
  parray=parray/sum(sum(parray))
  onehot=np.random.multinomial(1, parray[0])
  if(onehot.shape[0]==3):
    return np.argmax(onehot, axis=0).astype(int)
  elif(onehot.shape[0]==2):
    if(np.argmax(onehot, axis=0).astype(int)==0):
      return 0
    else:
      return 2

filename='/storage/home/d/dzw5320/MachineLearning/Data/Col_inall.csv'
data_in=pd.read_csv(filename, sep=',',header=0)
train_in_=data_in[data_in['t']<=11515]
test_in_=data_in[data_in['t']>11515]

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



from sklearn import preprocessing

xnumbers1=np.arange(0, 73, 1)
xnumbers2=np.arange(365, 438, 1)
xnumbers=np.concatenate((xnumbers1, xnumbers2))
xnumbers=np.reshape(xnumbers, (146))    

train_in=train_in_
train_in=train_in.iloc[:, np.r_[2:(train_in.shape[1]-1)]]
train_xy_1=train_in.values[:,xnumbers]

test_in=test_in_
test_in=test_in.iloc[:, np.r_[2:(test_in.shape[1]-1)]]
test_xy_1=test_in.values[:,xnumbers]


train_std=train_in.std()
criteria=train_std>0.0
train_in=train_in[criteria.index[criteria]]
train_in=train_in.values

scaler2 = preprocessing.StandardScaler()
scaler2.fit(train_in)
train_in=scaler2.transform(train_in)

train_in=train_in.astype(np.float32)

test_in=scaler2.transform(test_in)

test_in=test_in.astype(np.float32)



RFmov = RandomForestClassifier( random_state=0,max_depth=10, min_samples_split=11, min_samples_leaf=5,
                              n_estimators=500,  n_jobs=-2)

RFmov.fit(train_in, train_out)



test_pred=RFmov.predict_proba(test_in)

train_pred=RFmov.predict_proba(train_in)


test_col_mov=test_pred
train_col_mov=train_pred



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


RFreg = RandomForestRegressor( random_state=0,max_depth=2, min_samples_split=2, min_samples_leaf=1,
                              n_estimators=500, n_jobs=-2)

RFreg.fit(train_in, train_out)

test_pred=RFreg.predict(test_in)
train_pred=RFreg.predict(train_in)
samplepredictions_unstd=np.zeros(( 11511, 146))
samplepredictions_unstd=scaler2.transform(train_pred)
samplepredictions_unstd_test=np.zeros(( 2883, 146))
samplepredictions_unstd_test=scaler2.transform(test_pred)
restrain=np.zeros([train_out.shape[0], 146])
restrain=samplepredictions_unstd+train_xy_1
restest=np.zeros([test_out.shape[0], 146])
restest=samplepredictions_unstd_test+test_xy_1
act_train=train_out_unstd+train_xy_1
act_test=test_out+test_xy_1

test_col_vel=samplepredictions_unstd_test
train_col_vel=samplepredictions_unstd

test_col_velact=act_test
train_col_velact=act_train

########################################
####One step ahead predictions##########


for t in range(train_col_mov.shape[0]):
  
  
  
  for i in range(train_col_mov.shape[1]):
    
    if train_col_mov[t,i]==0:
      train_col_vel[t, i]<-train_xy_1[t, i]
      train_col_vel[t, (i+73)]<-train_xy_1[t, (i+73)]
    elif(train_col_mov[t,i]==1):
      train_col_vel[t, i]<-199
      train_col_vel[t, (i+73)]<-0
    else:
      pj=project(np.reshape(train_col_vel[i, (i, (i+73))], (1,2)), poly=poly, eps=1e-06)
      train_col_vel[t, i]<-pj[0, 0]
      train_col_vel[t, (i+73)]<-pj[0, 1]


for t in range(test_col_mov.shape[0]):
  
  
  
  for i in range(test_col_mov.shape[1]):
    
    if test_col_mov[t,i]==0:
      test_col_vel[t, i]<-test_xy_1[t, i]
      test_col_vel[t, (i+73)]<-test_xy_1[t, (i+73)]
    elif(test_col_mov[t,i]==1):
      test_col_vel[t, i]<-199
      test_col_vel[t, (i+73)]<-0
    else:
      pj=project(np.reshape(test_col_vel[i, (i, (i+73))], (1,2)), poly=poly, eps=1e-06)
      test_col_vel[t, i]<-pj[0, 0]
      test_col_vel[t, (i+73)]<-pj[0, 1]


#########################
###MSPE training set#####


train_col_vel_reshape=train_col_vel.reshape((11511, 2, 73))
train_act_reshape=train_col_velact.reshape((11511, 2, 73))

test_col_vel_reshape=test_col_vel.reshape((2883, 2, 73))
test_act_reshape=test_col_velact.reshape((2883, 2, 73))
  
check=np.zeros((11511, 73))


check=((train_col_vel_reshape[:, 0, :]-train_act_reshape[:, 0,:])**2+(train_col_vel_reshape[:, 1, :]-train_act_reshape[:, 1,:])**2)**(0.5)
check.sum(axis=0).sum()/(11511*73)

###########################
#####MSPE testing set######

check=np.zeros((2883, 73))


check=((test_col_vel_reshape[:, 0, :]-test_act_reshape[:, 0,:])**2+(test_col_vel_reshape[:, 1, :]-test_act_reshape[:, 1,:])**2)**(0.5)
check.sum(axis=0).sum()/(2883*73)



##########################################
###1000 steps ahead simulations###########

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np
import pandas as pd
#from joblib import dump, load
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix



filename='/storage/home/d/dzw5320/MachineLearning/Data/indants2.csv'


data=pd.read_csv(filename, sep=',',header=0)

cond=data['id']=="823"
data['id'][cond]=823

import random

for sim in np.arange(0, 100, 1):
  predseries=data[data['t']==11516].copy()
  for t in range(1000):
    pred=predseries[predseries['t']==(11516+t)].copy()
    predpivot=pred.pivot(index='t', columns='id', values=['x-1', 'x-2','x-3', 'x-4','x-5', 'y-1', 'y-2','y-3','y-4','y-5',  'vx-1', 'vx-2','vx-3', 'vx-4','vx-5',  'vy-1', 'vy-2','vy-3', 'vy-4','vy-5',
        'chamber', 'distind', 'stattime',
       'nwalldist', 'swalldist', 'wwalldist', 'ewalldist',
       'nndist', 'nnxlag1','nnylag1', 'nnvxlag1', 'nnvylag1', 'Q1',
       'Q2', 'Q3', 'Q4','nnmove', 'nnstill','distqueen'])
    predpivot.columns =[s1 + '_'+ str(s2) for (s1,s2) in predpivot.columns.tolist()]
    predpivot=predpivot[criteria.index[criteria]]
    predpivot=predpivot.values
    getdatastdmov=scaler2.transform(predpivot)
    getdatastdreg=scaler2.transform(predpivot)
    predmov=RFmov.predict_proba(getdatastdmov)
    for i in np.arange(73):
      pred.iloc[i, 36]=getClass(predmov[i])
    tree=random.randint(0,499)
    predvel=RFreg.estimators_[tree].predict(getdatastdreg)
    predvel=scaler1.inverse_transform(predvel)
  
    for i in range(73):
      if(pred.iloc[i,36]==2):
        
        pred.iloc[i,5:7 ]=predvel[0, np.array([i, i+73])]#vx, vy
        pred.iloc[i, 3]=pred.iloc[i, 5]+pred.iloc[i, 9]
        pred.iloc[i, 4]=pred.iloc[i, 6]+pred.iloc[i, 14]
        pred.iloc[i, 3:5]=project(np.reshape(pred.iloc[i, 3:5].values, (1, 2)), poly=poly, eps=1e-06)[0]
      elif(pred.iloc[i,36]==0):
        pred.iloc[i,5:7 ]=np.array([0, 0])
        pred.iloc[i, 3:5]=np.array([pred.iloc[i, 9],pred.iloc[i, 14] ])
      elif(pred.iloc[i,36]==1):
        pred.iloc[i, 3:5]=np.array([199, 0])
        pred.iloc[i, 5]=pred.iloc[i, 3]-pred.iloc[i, 9]
        pred.iloc[i, 6]=pred.iloc[i, 4]-pred.iloc[i, 14]
      else:
        print('error')

    predseries[predseries['t']==(11516+t)]=pred
    pred1=pred.copy()
    pred1.iloc[:,np.r_[10:14]]=pred.iloc[:,np.r_[9:13]].values#update x-2:x-5
    pred1.iloc[:,15:19]=pred.iloc[:,np.r_[14:18]].values#update y-2:y-5
    pred1.iloc[:,20:24]=pred.iloc[:,np.r_[19:23]].values#update vx-2:vx-5
    pred1.iloc[:,25:29]=pred.iloc[:,np.r_[24:28]].values#update vy-2:vy-5
    pred1.iloc[:,9]=pred.iloc[:,3].values
    pred1.iloc[:,14]=pred.iloc[:,4].values
    pred1.iloc[:,19]=pred.iloc[:,5].values
    pred1.iloc[:,24]=pred.iloc[:,6].values
    pred1['t']=11516+t+1
    #update other derived variables
    pred1['chamber']=vgetChamNo(pred1['x'].values)
    pred1['chamber']=vgetChamNo(pred1['x'].values)
    pred1['distind']=veuc_dist(pred1['x-1'].values, pred1['x-2'].values, pred1['y-1'].values, pred1['y-2'].values)
    pred1['stattime']=vgetstattime(pred1['distind'].values,pred1['stattime'].values )
    pred1['nwalldist']=vdistwalln(pred1['x'].values, pred1['y'].values)
    pred1['swalldist']=vdistwalls(pred1['x'].values, pred1['y'].values)
    pred1['wwalldist']=vdistwallw(pred1['x'].values, pred1['y'].values, pred1['chamber'].values)
    pred1['ewalldist']=vdistwalle(pred1['x'].values, pred1['y'].values, pred1['chamber'].values)
    for i in range(73):
      bad_df = pred1.index.isin([pred1.index[i]])
      iant=pred1[bad_df]
      rest=pred1[~bad_df]
      queen=pred1.iloc[72, :]
      pred1.iloc[i,42]=getQ1(iant, rest)
      pred1.iloc[i,43]=getQ2(iant, rest)
      pred1.iloc[i,44]=getQ3(iant, rest)
      pred1.iloc[i,45]=getQ4(iant, rest)
      pred1.iloc[i,37]=getnndist(iant, rest)
      pred1.iloc[i,38]=getnnxlag1(iant, rest)
      pred1.iloc[i,39]=getnnylag1(iant, rest)
      pred1.iloc[i,40]=getnnvxlag1(iant, rest)
      pred1.iloc[i,41]=getnnvylag1(iant, rest)
      pred1.iloc[i,46]=getnnmove(iant, rest)
      pred1.iloc[i,47]=getnnstill(iant, rest)
      pred1.iloc[i,48]=getdistqueen(iant, rest)
    
    predseries=predseries.append(pred1)
  filename='/storage/home/d/dzw5320/MachineLearning/Results3/sims/RFcol/'+str(sim)+'RFcol.csv'
  predseries.to_csv(filename, encoding='utf-8', index=False)