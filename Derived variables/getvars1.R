load("ants.Rdata")
library(DataCombine)
source(file = "fun1.R")

allants<-antdata
ant_ids<-as.character(unique(allants$id))




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

