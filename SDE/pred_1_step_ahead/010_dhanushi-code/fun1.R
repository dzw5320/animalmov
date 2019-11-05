getChamNo<-function(x){
  
  if(x<18.5){1}else if(x<46){2}else if(x<70.5){3}else if(x<98){4}else if(x<122.5){5}else if(x<150){6}else if(x<174.5){7}else{8}
  
}



euc_dist<-function(x1, x2, y1, y2){
  
  sqrt((x1-x2)^2+(y1-y2)^2)
}

getstattime<-function(x){
  statvec<-rep(NA, length(x))
  counter<-0
  for(i in c(1:length(x))){
    if(x[i]==0){
      counter=counter+1
      
    }else{
      
      counter<-0
      
    }
    statvec[i]<-counter
    
  }
  statvec
}

distwalln<-function(x, y){
  
  nwall<-if((x<40) || (x>52 && x<92)|| (x>104 && x<144) ||(x>156 && x<196)){65}else{6}
  nwall-y
  
}

distwalls<-function(x, y){
  
  swall<-if( (x>18.5 && x<21.5)|| (x>70.5 && x<73.5) ||(x>122.5 && x<125.5) || (x>174.5 && x<177.5) ){53}else{0}
  y-swall
  
}

distwallw<-function(x, y, chamber){
  
  wwall<-if((chamber==1)||(chamber==2 && y>=53)){0}else if((chamber==3 && y>=6)||(chamber==4 && y>=53) ){52}else if((chamber==5 && y>=6)||(chamber==6 && y>=53)){104}else if((chamber==7 && y>=6)||(chamber==8 && y>=53)){156}else 
    if((chamber==2 && y<53)|| (chamber==3 && y<6)){21.5}else if((chamber==4 && y<53)||(chamber==5 && y<6)){73.5}else
      if((chamber==6 && y<53)||(chamber==7 && y<6)){125.5}else if(chamber==8 && y<53){177.5}else{NA}
  
  x-wwall
}

distwalle<-function(x, y, chamber){
  
  ewall<-if(chamber==1 && y<53){18.5}else if((chamber==1 && y>=53) || (chamber==2 && y>=6)){40}else if((chamber==2 && y<6)||(chamber==3 && y<53)){70.5}else
    if((chamber==3 && y>=53)||(chamber==4 && y>=6)){92}else if((chamber==4 && y<6)||(chamber==5 && y<53)){122.5}else
      if((chamber==5 && y>=53)||(chamber==6 && y>=6)){144}else if((chamber==6 && y<6)||(chamber==7 && y<53)){174.5}else
        if((chamber==7 && y>=53)||(chamber==8 && y>=6)){196}else if((chamber==8 && y<6)||(x>=196)){199}else{NA}
  
  ewall-x
}


vgetChamNo <- Vectorize(getChamNo)
veuc_dist<-Vectorize(euc_dist)
vdistwalln<-Vectorize(distwalln)
vdistwalls<-Vectorize(distwalls)
vdistwalle<-Vectorize(distwalle)
vdistwallw<-Vectorize(distwallw)



getQ1<-function(iant, rest){
  
  
  Q1sub<-subset(rest, rest$`x-1`>((iant$`x-1`)-8) & rest$`x-1`<(iant$`x-1`) & 
                  rest$`y-1`>(iant$`y-1`) & rest$`y-1`<(iant$`y-1`+8))
  
  nrow(Q1sub)
  
}

getQ2<-function(iant, rest){
  
  
  Q2sub<-subset(rest, rest$`x-1`>((iant$`x-1`)) & rest$`x-1`<(iant$`x-1`+8) & 
                  rest$`y-1`>(iant$`y-1`) & rest$`y-1`<(iant$`y-1`+8))
  
  nrow(Q2sub)
  
}


getQ3<-function(iant, rest){
  
  
  Q3sub<-subset(rest, rest$`x-1`>((iant$`x-1`-8)) & rest$`x-1`<(iant$`x-1`) & 
                  rest$`y-1`>(iant$`y-1`-8) & rest$`y-1`<(iant$`y-1`))
  
  nrow(Q3sub)
  
}

getQ4<-function(iant, rest){
  
  
  Q4sub<-subset(rest, rest$`x-1`>((iant$`x-1`)) & rest$`x-1`<(iant$`x-1`+8) & 
                  rest$`y-1`>(iant$`y-1`-8) & rest$`y-1`<(iant$`y-1`))
  
  nrow(Q4sub)
  
}


getnndist<-function(iant, rest){
  
  
  min(veuc_dist(x1 = iant$`x-1`, y1 = iant$`y-1`, x2 = rest$`x-1`, y2 = rest$`y-1`))
  
}

getnnxlag1<-function(iant, rest){
  
  id<-which.min(veuc_dist(x1 = iant$`x-1`, y1 = iant$`y-1`, x2 = rest$`x-1`, y2 = rest$`y-1`))
  rest$`x-1`[id]
  
  
}

getnnylag1<-function(iant, rest){
  
  id<-which.min(veuc_dist(x1 = iant$`x-1`, y1 = iant$`y-1`, x2 = rest$`x-1`, y2 = rest$`y-1`))
  rest$`y-1`[id]
  
  
}

getnnvxlag1<-function(iant, rest){
  
  id<-which.min(veuc_dist(x1 = iant$`x-1`, y1 = iant$`y-1`, x2 = rest$`x-1`, y2 = rest$`y-1`))
  rest$`vx-1`[id]
  
  
}

getnnvylag1<-function(iant, rest){
  
  id<-which.min(veuc_dist(x1 = iant$`x-1`, y1 = iant$`y-1`, x2 = rest$`x-1`, y2 = rest$`y-1`))
  rest$`vy-1`[id]
  
  
}

getnnmove<-function(iant, rest){
  
  
  idx<-which((veuc_dist(x1 = iant$`x-1`, y1 = iant$`y-1`, x2 = rest$`x-1`, y2 = rest$`y-1`))<12)
  rest1<-rest[idx,]
  sum(rest1$movt=="yes")
  
}

getnnstill<-function(iant, rest){
  
  
  idx<-which((veuc_dist(x1 = iant$`x-1`, y1 = iant$`y-1`, x2 = rest$`x-1`, y2 = rest$`y-1`))<10)
  rest1<-rest[idx,]
  sum(rest1$movt=="no") 
}

getdistqueen<-function(iant, queen){
  
  if(iant$id!="Que"){
    distqueen=veuc_dist(x1 = iant$`x-1`, y1 = iant$`y-1`, x2 = queen$`x-1`, y2 = queen$`y-1`)
    
  }else{
   distqueen=0
    }
  
   distqueen
}

