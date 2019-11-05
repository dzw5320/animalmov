 load("antdata.Rdata")
# library(DataCombine)
# 
# allants<-ants.c1low

source("fun1.R")
ant_ids<-as.character(unique(antdata$id))




for(i in 1:length(ant_ids)){
  
  cat(i, "\n")
  ind_ant<-antdata[antdata$id==ant_ids[i],]
  
  for(t in 1:length(ind_ant$t)){
    cat(t, "\n")
    ind_ant_t<-ind_ant[t,]
    rest<- antdata[(antdata$id!=ant_ids[i] & antdata$t==ind_ant_t$t),]
    queen<-antdata[antdata$id=="Que" & antdata$t==ind_ant_t$t, ]
    
    ind_ant$nndist[t]<-getnndist(ind_ant_t, rest)
    ind_ant$nnxlag1[t]<-getnnxlag1(ind_ant_t, rest)
    ind_ant$nnylag1[t]<-getnnylag1(ind_ant_t, rest)
    ind_ant$nnvxlag1[t]<-getnnvxlag1(ind_ant_t, rest)
    ind_ant$nnvylag1[t]<-getnnvylag1(ind_ant_t, rest)
    ind_ant$Q1[t]<-getQ1(ind_ant_t,rest)
    ind_ant$Q2[t]<-getQ2(ind_ant_t,rest)
    ind_ant$Q3[t]<-getQ3(ind_ant_t,rest)
    ind_ant$Q4[t]<-getQ4(ind_ant_t,rest)
    ind_ant$nnmove[t]<-getnnmove(ind_ant_t,rest)
    ind_ant$nnstill[t]<-getnnstill(ind_ant_t,rest)
    ind_ant$distqueen[t]<-getdistqueen(ind_ant_t,queen)
    
  
   
    
  }
  if(i==1){
    indants1<-ind_ant
  }else{
    indants1<-rbind(indants1, ind_ant)
  }
  
}

# save(indants, file="C:\\Users\\dhanu\\OneDrive\\Machine Learning\\Results6\\indants.Rdata")

save(indants1, file="indants1.Rdata")

