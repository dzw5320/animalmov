load("ants.Rdata")
library(DataCombine)
source(file = "fun1.R")

allants<-antdata
ant_ids<-as.character(unique(allants$id))

# getChamNo<-function(x){
#   
#   if(x<18.5){1}else if(x<46){2}else if(x<70.5){3}else if(x<98){4}else if(x<122.5){5}else if(x<150){6}else if(x<174.5){7}else{8}
#   
# }
# 
# 
# 
# euc_dist<-function(x1, x2, y1, y2){
#   
#   sqrt((x1-x2)^2+(y1-y2)^2)
# }
# 
# getstattime<-function(x){
#   statvec<-rep(NA, length(x))
#   counter<-0
#   for(i in c(1:length(x))){
#     if(x[i]==0){
#       counter=counter+1
#       
#     }else{
#       
#       counter<-0
#       
#     }
#     statvec[i]<-counter
#     
#   }
#   statvec
# }
# 
# distwalln<-function(x, y){
#   
#   nwall<-if((x<40) || (x>52 && x<92)|| (x>104 && x<144) ||(x>156 && x<196)){65}else{6}
#   nwall-y
#   
# }
# 
# distwalls<-function(x, y){
#   
#   swall<-if( (x>18.5 && x<21.5)|| (x>70.5 && x<73.5) ||(x>122.5 && x<125.5) || (x>174.5 && x<177.5) ){53}else{0}
#   y-swall
#   
# }
# 
# distwallw<-function(x, y, chamber){
#   
#   wwall<-if((chamber==1)||(chamber==2 && y>=53)){0}else if((chamber==3 && y>=6)||(chamber==4 && y>=53) ){52}else if((chamber==5 && y>=6)||(chamber==6 && y>=53)){104}else if((chamber==7 && y>=6)||(chamber==8 && y>=53)){156}else 
#     if((chamber==2 && y<53)|| (chamber==3 && y<6)){21.5}else if((chamber==4 && y<53)||(chamber==5 && y<6)){73.5}else
#       if((chamber==6 && y<53)||(chamber==7 && y<6)){125.5}else if(chamber==8 && y<53){177.5}else{NA}
#   
#   x-wwall
# }
# 
# distwalle<-function(x, y, chamber){
#   
#   ewall<-if(chamber==1 && y<53){18.5}else if((chamber==1 && y>=53) || (chamber==2 && y>6)){40}else if((chamber==2 && y<6)||(chamber==3 && y<53)){70.5}else
#     if((chamber==3 && y>53)||(chamber==4 && y>6)){92}else if((chamber==4 && y<6)||(chamber==5 && y<53)){122.5}else
#       if((chamber==5 && y>53)||(chamber==6 && y>6)){144}else if((chamber==6 && y<6)||(chamber==7 && y<53)){174.5}else
#         if((chamber==7 && y>53)||(chamber==8 && y>6)){196}else if((chamber==8 && y<6)||(x>196)){199}else{NA}
#   
#   ewall-x
# }
# 
# 
# vgetChamNo <- Vectorize(getChamNo)
# veuc_dist<-Vectorize(euc_dist)
# vdistwalln<-Vectorize(distwalln)
# vdistwalls<-Vectorize(distwalls)
# vdistwalle<-Vectorize(distwalle)
# vdistwallw<-Vectorize(distwallw)
# 
# antdata$wwalldist<-vdistwallw(antdata$`x-1`,antdata$`y-1`, antdata$chamber)
# 
# antdata$ewalldist<-vdistwalle(antdata$`x-1`,antdata$`y-1`, antdata$chamber)
# 



for(i in 1:length(ant_ids)){
  
  cat(i, "\n")
  ind_ant<-allants[allants$id==ant_ids[i],]
  test<-slide(ind_ant, Var="x", slideBy=-c(1:5))
  test<-slide(test, Var="y", slideBy=-c(1:5))
  test<-slide(test, Var="vx", slideBy=-c(1:5))
  test<-slide(test, Var="vy", slideBy=-c(1:5))
  ind_ant<-test
  ind_ant<-na.omit(ind_ant)
  ind_ant$chamber<-vgetChamNo(ind_ant$`x-1`)
  ind_ant$distind<-veuc_dist(ind_ant$`x-1`, ind_ant$`x-2`, ind_ant$`y-1`, ind_ant$`y-2`)
  ind_ant$stattime<-getstattime(ind_ant$distind)
  ind_ant$nwalldist<-vdistwalln(ind_ant$`x-1`, ind_ant$`y-1`)
  ind_ant$swalldist<-vdistwalls(ind_ant$`x-1`, ind_ant$`y-1`)
  ind_ant$wwalldist<-vdistwallw(ind_ant$`x-1`, ind_ant$`y-1`, ind_ant$chamber)
  ind_ant$ewalldist<-vdistwalle(ind_ant$`x-1`, ind_ant$`y-1`, ind_ant$chamber)
  if(i==1){
  antdata<-ind_ant
  }else{
    antdata<-rbind(antdata, ind_ant)
  }

}

antdata$movt<-unlist(lapply(c(1:nrow(antdata)), function(x){if(antdata$vx[x]!=0 || antdata$vy[x]!=0){"yes"}else{"no"}}))

save(antdata, file="C:\\Users\\dhanu\\OneDrive\\Machine Learning\\Results8\\antdata.Rdata")

