from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np
import pandas as pd
#from joblib import dump, load
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix



filename='/storage/home/d/dzw5320/MachineLearning/Data/indants2.csv'
data=pd.read_csv(filename, sep=',',header=0)

###standardize
data_in=data.iloc[:, np.r_[9:36,37:data.shape[1]]]
data_in=data_in.values
data_in=data_in.astype(np.float32)
from sklearn import preprocessing

scaler1 = preprocessing.StandardScaler()

scaler1.fit(data_in)

####


train1=data[data['t']<=11515]
test1=data[(data['t']>11515) ]

train_in=train1.iloc[:, np.r_[9:36,37:train1.shape[1]]]
train_out=train1.iloc[:, 36]

test_in=test1.iloc[:, np.r_[9:36,37:test1.shape[1]]]
test_out=test1.iloc[:, 36]

################
#Scale the input
train_in=train_in.values
train_in=train_in.astype(np.float32)

test_in=test_in.values
test_in=test_in.astype(np.float32)


train_in=scaler1.transform(train_in)
test_in=scaler1.transform(test_in)

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(train_out)
train_out = encoder.transform(train_out)

test_out = encoder.transform(test_out)

RFmov = RandomForestClassifier( random_state=0,max_depth=34, min_samples_split=10, min_samples_leaf=10,
                              n_estimators=500, n_jobs=-2)
RFmov.fit(train_in, train_out)

test_ind_mov=RFmov.predict(test_in)
train_ind_mov=RFmov.predict(train_in)

#Variable Importance for classification model

feature_importances = pd.DataFrame(RFmov.feature_importances_,
                                   index = data.iloc[:, np.r_[9:36,37:data.shape[1]]].columns,
                                    columns=['importance']).sort_values('importance',ascending=False)

filename='/storage/home/d/dzw5320/MachineLearning/VIRFindmov.csv'
feature_importances.to_csv(filename, encoding='utf-8', index=True)

#################################################

filename='/storage/home/d/dzw5320/MachineLearning/Data/indants2.csv'


data=pd.read_csv(filename, sep=',',header=0)
data=data[data['movt']=='yes']

vxmean=data['vx'].mean()
vxstd=data['vx'].std()
vymean=data['vy'].mean()
vystd=data['vy'].std()

###standardize data
data_in=data.iloc[:, np.r_[9:36,37:data.shape[1]]]
data_in=data_in.values
scaler2 = preprocessing.StandardScaler()
scaler2.fit(data_in)
data_out=data.iloc[:, 5:7]
data_out=data_out.values
scaler3 = preprocessing.StandardScaler()
scaler3.fit(data_out)



trainmov=data[data['t']<=11515]
testmov=data[(data['t']>11515) ]



trainmovvxvy=trainmov.iloc[:, np.r_[5:7]]
testmovvxvy=testmov.iloc[:, np.r_[5:7]]
trainmovvxvy_1=trainmov[['x-1', 'y-1']].values
testmovvxvy_1=testmov[['x-1', 'y-1']].values
trainmovxy_1=trainmov[['x', 'y']].values
testmovxy_1=testmov[['x', 'y']].values






train_in_mov=trainmov.iloc[:, np.r_[9:36,37:trainmov.shape[1]]]
train_in_mov=train_in_mov.values
train_out_mov=trainmov.iloc[:, 5:7]
train_out_mov=train_out_mov.values

test_in_mov=testmov.iloc[:, np.r_[9:36,37:testmov.shape[1]]]
test_in_mov=test_in_mov.values
test_out_mov=testmov.iloc[:, 5:7]
test_out_mov=test_out_mov.values


train_in_mov=train_in_mov.astype(np.float32)
train_out_mov=train_out_mov.astype(np.float32)

test_in_mov=test_in_mov.astype(np.float32)
test_out_mov=test_out_mov.astype(np.float32)

#scale train data
train_in_mov=scaler2.transform(train_in_mov)
test_in_mov=scaler2.transform(test_in_mov)

#scale test data
train_out_mov=scaler3.transform(train_out_mov)
test_out_mov=scaler3.transform(test_out_mov)





RFreg = RandomForestRegressor( random_state=0,max_depth=19, min_samples_split=10, min_samples_leaf=7,
                              n_estimators=500, n_jobs=-2)

RFreg.fit(train_in_mov, train_out_mov)


###Variable Importance

feature_importances = pd.DataFrame(RFreg.feature_importances_,
                                   index = data.iloc[:, np.r_[9:36,37:data.shape[1]]].columns,
                                    columns=['importance']).sort_values('importance',ascending=False)

filename='/storage/home/d/dzw5320/MachineLearning/VIRFindreg.csv'
feature_importances.to_csv(filename, encoding='utf-8', index=True)

#####################################
##We want predictions for all data
filename='/storage/home/d/dzw5320/MachineLearning/Data/indants2.csv'
data=pd.read_csv(filename, sep=',',header=0)
#data.columns


#data[cols_to_norm] = data[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max()-x.min()))


train1=data[data['t']<=11515]
test1=data[(data['t']>11515) ]

train_in=train1.iloc[:, np.r_[9:36,37:train1.shape[1]]]
train_out=train1.iloc[:, 5:7]

test_in=test1.iloc[:, np.r_[9:36,37:test1.shape[1]]]
test_out=test1.iloc[:, 5:7]

trainvxvy=train1.iloc[:, np.r_[5:7]]
testvxvy=test1.iloc[:, np.r_[5:7]]
trainvxvy_1=train1[['x-1', 'y-1']].values
testvxvy_1=test1[['x-1', 'y-1']].values
trainxy_1=train1[['x', 'y']].values
testxy_1=test1[['x', 'y']].values

################
#Scale the input
train_in=train_in.values
train_in=train_in.astype(np.float32)

test_in=test_in.values
test_in=test_in.astype(np.float32)

#standardize data
train_in=scaler2.transform(train_in)
test_in=scaler2.transform(test_in)

#Get predictions
test_pred=regr.predict(test_in)
train_pred=regr.predict(train_in)

#inverse transform output
unstdpred_train=scaler3.inverse_transform(train_pred)
unstdpred_test=scaler3.inverse_transform(test_pred)

unstdpred_test[:,0]=unstdpred_test[:,0]+testvxvy_1[:,0]
unstdpred_test[:,1]=unstdpred_test[:,1]+testvxvy_1[:,1]
unstdpred_train[:,0]=unstdpred_train[:,0]+trainvxvy_1[:,0]
unstdpred_train[:,1]=unstdpred_train[:,1]+trainvxvy_1[:,1]
unstdpred_train=np.hstack((unstdpred_train,trainxy_1))
unstdpred_test=np.hstack((unstdpred_test,testxy_1))

train_ind_vel=unstdpred_train
test_ind_vel=unstdpred_test

train_ind_vel=train_ind_vel[:,0:2]
test_ind_vel=test_ind_vel[:,0:2]

filename='drive/My Drive/Colab Notebooks/indants2.csv'
data_in=pd.read_csv(filename, sep=',',header=0)

train_in=data_in[data_in['t']<=11515]
test_in=data_in[data_in['t']>11515]

train_xy_1=train_in.iloc[:, np.r_[9,14]].values
test_xy_1=test_in.iloc[:, np.r_[9,14]].values

train_out_act=train_in.iloc[:, np.r_[3,4]].values
test_out_act=test_in.iloc[:, np.r_[3,4]].values

###############################################
##########1000 steps ahead simulation##########

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

###############################################
##########Calculate MSPE for training data#####

for i in range(train_ind_vel.shape[0]):
  
  
  if train_ind_mov[i]==0:
    train_ind_vel[i, 0]=train_xy_1[i, 0]
    train_ind_vel[i, 1]=train_xy_1[i, 1]
  elif(train_ind_mov[i]==1):
    train_ind_vel[i, 0]=199.0
    train_ind_vel[i, 1]=0.0
  else:
    train_ind_vel[i, :]=project(np.reshape(train_ind_vel[i, :], (1, 2)), poly=poly, eps=1e-06)


MSPEtrain=np.mean(((train_ind_vel[:,0]-train_out_act[:,0])**2+(train_ind_vel[:,1]-train_out_act[:,1])**2)**(0.5))

MSPEtrain

###############################################
##########Calculate MSPE for testing data######

for i in range(test_ind_vel.shape[0]):
  
  
  if test_ind_mov[i]==0:
    test_ind_vel[i, 0]=test_xy_1[i, 0]
    test_ind_vel[i, 1]=test_xy_1[i, 1]
  elif(test_ind_mov[i]==1):
    test_ind_vel[i, 0]=199.0
    test_ind_vel[i, 1]=0.0
  else:
    test_ind_vel[i, :]=project(np.reshape(test_ind_vel[i, :], (1, 2)), poly=poly, eps=1e-06)


MSPEtest=np.mean(((test_ind_vel[:,0]-test_out_act[:,0])**2+(test_ind_vel[:,1]-test_out_act[:,1])**2)**(0.5))

MSPEtest

#################################################

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
  onehot=np.random.multinomial(1, parray)
  return np.argmax(onehot, axis=0)


from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np
import pandas as pd
#from joblib import dump, load
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix



filename='/storage/home/d/dzw5320/MachineLearning/Data/indants2.csv'
data=pd.read_csv(filename, sep=',',header=0)


train1=data[data['t']<=11515]
test1=data[(data['t']>11515) ]

train_in=train1.iloc[:, np.r_[9:36,37:train1.shape[1]]]
train_out=train1.iloc[:, 36]

test_in=test1.iloc[:, np.r_[9:36,37:test1.shape[1]]]
test_out=test1.iloc[:, 36]

################
#Scale the input
train_in=train_in.values
train_in=train_in.astype(np.float32)

test_in=test_in.values
test_in=test_in.astype(np.float32)

from sklearn import preprocessing

scaler1 = preprocessing.StandardScaler()
scaler1.fit(train_in)
train_in=scaler1.transform(train_in)

filename='/storage/home/d/dzw5320/MachineLearning/Data/indants2.csv'


data=pd.read_csv(filename, sep=',',header=0)
data=data[data['movt']=='yes']

vxmean=data['vx'].mean()
vxstd=data['vx'].std()
vymean=data['vy'].mean()
vystd=data['vy'].std()


trainmov=data[data['t']<=11515]
testmov=data[(data['t']>11515) ]



trainmovvxvy=trainmov.iloc[:, np.r_[5:7]]
testmovvxvy=testmov.iloc[:, np.r_[5:7]]
trainmovvxvy_1=trainmov[['x-1', 'y-1']].values
testmovvxvy_1=testmov[['x-1', 'y-1']].values
trainmovxy_1=trainmov[['x', 'y']].values
testmovxy_1=testmov[['x', 'y']].values






train_in_mov=trainmov.iloc[:, np.r_[9:36,37:trainmov.shape[1]]]
train_in_mov=train_in_mov.values
train_out_mov=trainmov.iloc[:, 5:7]
train_out_mov=train_out_mov.values

test_in_mov=testmov.iloc[:, np.r_[9:36,37:testmov.shape[1]]]
test_in_mov=test_in_mov.values
test_out_mov=testmov.iloc[:, 5:7]
test_out_mov=test_out_mov.values


train_in_mov=train_in_mov.astype(np.float32)
train_out_mov=train_out_mov.astype(np.float32)

test_in_mov=test_in_mov.astype(np.float32)
test_out_mov=test_out_mov.astype(np.float32)

#scale train data
scaler2 = preprocessing.StandardScaler()
scaler2.fit(train_in_mov)

scaler3=preprocessing.StandardScaler()
scaler3.fit(train_out_mov)
train_out_mov=scaler3.transform(train_out_mov)
test_out_mov=scaler3.transform(test_out_mov)

filename='/storage/home/d/dzw5320/MachineLearning/Data/indants2.csv'


data=pd.read_csv(filename, sep=',',header=0)

import random



for sim in range(0,100,1):
  predseries=data[data['t']==11516].copy()
  for t in range(1000):
    print(t)
    pred=predseries[predseries['t']==(11516+t)].copy()
    getdata=pred.iloc[:, np.r_[9:36,37:pred.shape[1]]].values
    getdatastdmov=scaler1.transform(getdata)
    getdatastdreg=scaler2.transform(getdata)
    predmov=RFmov.predict_proba(getdatastdmov)
    #pred['movt']=predmov.copy()
    #if t==0:
    #  predseries['movt']=pred['movt']
  
    for i in range(73):
      pred.iloc[i,36]=getClass(predmov[i])
      if(pred.iloc[i,36]==2):
        tree=random.randint(0,499)#change to 500
        vxpred=RFreg.estimators_[tree].predict(getdatastdreg[i, :].reshape((1, 39)))
        pred.iloc[i,5:7 ]=scaler3.inverse_transform(vxpred)[0]
        pred.iloc[i, 3]=pred.iloc[i, 5]+pred.iloc[i, 9]
        pred.iloc[i, 4]=pred.iloc[i, 6]+pred.iloc[i, 14]
        pred.iloc[i, 3:5]=project(np.reshape(pred.iloc[i, 3:5].values, (1, 2)), poly=poly, eps=1e-06)[0]
      elif(pred.iloc[i,36]==0):
        pred.iloc[i,5:7 ]=np.array([0, 0])
        pred.iloc[i, 3:5]=np.array([pred.iloc[i, 9],pred.iloc[i, 14] ])
      else:
        pred.iloc[i, 3:5]=np.array([199, 0])
        pred.iloc[i, 5]=pred.iloc[i, 3]-pred.iloc[i, 9]
        pred.iloc[i, 6]=pred.iloc[i, 4]-pred.iloc[i, 14]
    
    if t==0:
      predseries['movt']=pred['movt']
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
  print(predseries.shape)
  filename=str(sim)+'RFind.csv'
  predseries.to_csv(filename, encoding='utf-8', index=False)
  del predseries
  