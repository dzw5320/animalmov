dir.create(Sys.getenv("R_LIBS_USER"), showWarnings = FALSE, recursive = TRUE)
install.packages("foreach", Sys.getenv("R_LIBS_USER"), repos = "http://cran.case.edu" )
install.packages("doParallel", Sys.getenv("R_LIBS_USER"), repos = "http://cran.case.edu" )


library(parallel)
library(foreach)
library(doParallel)

load("//storage//home//d//dzw5320//MachineLearning//Data//indants2.Rdata")


#Set the number of cores

no_cores<-detectCores()-1

#Create a cluster
cl<-makeCluster(no_cores)


########################################################
########Computing metrics for long run simulations######

###Functions to compute metrics for an individual simulation
###Assumes that the prediction at time t and the predictor variables used to predict are all in the same row

#####Derived funs###########
euc_dist<-function(x1, x2, y1, y2){
  
  sqrt((x1-x2)^2+(y1-y2)^2)
}

veuc_dist<-Vectorize(euc_dist)


#Total distance moved by all ants in 1000 seconds


TotDist<-function(sim, start, finish){#11517 12515
  
  sum(sim$distind[sim$t>=start & sim$t<=finish] )+sum(veuc_dist(sim$x[sim$t==finish], sim$`x-1`[sim$t==finish],sim$y[sim$t==finish], sim$`y-1`[sim$t==finish]  ))
  
}


#Percentage of ant seconds stationary

Percstat<-function(sim, start, finish){
  
  movt=sim$movt[sim$t>=start & sim$t<=finish]
  movt_ind=mean(unlist(lapply(c(1:length(movt)), function(x){movt[x]=="no" || movt[x]==0})))
  movt_ind
}


#Percentage of ant seconds out of the nest

Percout<-function(sim, start, finish){
  
  movt=sim$movt[sim$t>=start & sim$t<=finish]
  movt_ind=mean(unlist(lapply(c(1:length(movt)), function(x){movt[x]=="out" || movt[x]==1})))
  movt_ind
}

#Number of ant seconds in each sub chamber

Nosubcham1<-function(sim, start, finish){
  
  cham=sim$chamber[sim$t>=start & sim$t<=finish]
  mean(cham==1)
}

Nosubcham2<-function(sim, start, finish){
  
  cham=sim$chamber[sim$t>=start & sim$t<=finish]
  mean(cham==2)
}

Nosubcham3<-function(sim, start, finish){
  
  cham=sim$chamber[sim$t>=start & sim$t<=finish]
  mean(cham==3)
}

Nosubcham4<-function(sim, start, finish){
  
  cham=sim$chamber[sim$t>=start & sim$t<=finish]
  mean(cham==4)
}

Nosubcham5<-function(sim, start, finish){
  
  cham=sim$chamber[sim$t>=start & sim$t<=finish]
  mean(cham==5)
}

Nosubcham6<-function(sim, start, finish){
  
  cham=sim$chamber[sim$t>=start & sim$t<=finish]
  mean(cham==6)
}

Nosubcham7<-function(sim, start, finish){
  
  cham=sim$chamber[sim$t>=start & sim$t<=finish]
  mean(cham==7)
}

Nosubcham8<-function(sim, start, finish){
  
  cham=sim$chamber[sim$t>=start & sim$t<=finish]
  mean(cham==8)
}

#Average number of ants around each ant at a radius of 12mm

avgindant<-function(simt, ind){
  
  iant=simt[simt$id==ind,]
  rest=simt[simt$id!=ind,]
  idx=which((veuc_dist(x1 = iant$`x-1`, y1 = iant$`y-1`, x2 = rest$`x-1`, y2 = rest$`y-1`))<12)
  length(idx)
}

Avgnearants<-function(sim, start, finish){
  listants={}
  sim1=sim[sim$t>=start & sim$t<=finish, ]
  tids<-unique(sim1$t)
  antids<-unique(sim1$id)
  for (t in tids){
    simt=sim1[sim1$t==t,]
    for(ind in antids ){
      listants=c(listants, avgindant(simt, ind))
      
    }
  }
  mean(listants)
}

#Export data to cluster
clusterExport(cl=cl, varlist=c("euc_dist","veuc_dist", "TotDist","Percstat", "Percout", "Nosubcham1", "Nosubcham2", "Nosubcham3", "Nosubcham4", "Nosubcham5", "Nosubcham6", "Nosubcham7", "Nosubcham8", "avgindant", "Avgnearants"), envir = environment())


#Calculate each metric for a 1000 time step moving window

M1_dist_real=unlist(parLapply(cl, c(5:13400), function(x){TotDist(indants2, x, (x+998))}))

M1_stat_real=unlist(parLapply(cl, c(5:13400), function(x){Percstat(indants2, x, (x+998))}))

M1_out_real=unlist(parLapply(cl, c(5:13400), function(x){Percout(indants2, x, (x+998))}))

M1_sub1_real=unlist(parLapply(cl, c(5:13400), function(x){Nosubcham1(indants2, x, (x+998))}))

M1_sub2_real=unlist(parLapply(cl, c(5:13400), function(x){Nosubcham2(indants2, x, (x+998))}))

M1_sub3_real=unlist(parLapply(cl, c(5:13400), function(x){Nosubcham3(indants2, x, (x+998))}))

M1_sub4_real=unlist(parLapply(cl, c(5:13400), function(x){Nosubcham4(indants2, x, (x+998))}))

M1_sub5_real=unlist(parLapply(cl, c(5:13400), function(x){Nosubcham5(indants2, x, (x+998))}))

M1_sub6_real=unlist(parLapply(cl, c(5:13400), function(x){Nosubcham6(indants2, x, (x+998))}))

M1_sub7_real=unlist(parLapply(cl, c(5:13400), function(x){Nosubcham7(indants2, x, (x+998))}))

M1_sub8_real=unlist(parLapply(cl, c(5:13400), function(x){Nosubcham8(indants2, x, (x+998))}))

M1_near_real=unlist(parLapply(cl, c(5:13400), function(x){Avgnearants(indants2, x, (x+998))}))

save(M1_dist_real,M1_stat_real, M1_out_real, M1_sub1_real, M1_sub2_real, M1_sub3_real, M1_sub4_real, M1_sub5_real, M1_sub6_real, 
     M1_sub7_real, M1_sub8_real, M1_near_real, file="metric_real.Rdata")

