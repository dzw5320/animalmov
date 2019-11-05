dir.create(Sys.getenv("R_LIBS_USER"), showWarnings = FALSE, recursive = TRUE)
nstall.packages("foreach", Sys.getenv("R_LIBS_USER"), repos = "http://cran.case.edu" )
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
###Order of variables and variable names should be the same as in indants2
###Convert csv files of long run simulations to .rda format for all models



#####Derived funs###########
euc_dist<-function(x1, x2, y1, y2){
  
  sqrt((x1-x2)^2+(y1-y2)^2)
}

veuc_dist<-Vectorize(euc_dist)


#Total distance moved by all ants in 1000 seconds


TotDist<-function(sim){
  
  sum(sim$distind[sim$t>=11517 & sim$t<=12515] )+sum(veuc_dist(sim$x[sim$t==12515], sim$`x-1`[sim$t==12515],sim$y[sim$t==12515], sim$`y-1`[sim$t==12515]  ))
  
}


#Percentage of ant seconds stationary

Percstat<-function(sim){
  
  movt=sim$movt[sim$t>=11517 & sim$t<=12515]
  movt_ind=mean(unlist(lapply(c(1:length(movt)), function(x){movt[x]=="no" || movt[x]==0})))
  movt_ind
}


#Percentage of ant seconds out of the nest

Percout<-function(sim){
  
  movt=sim$movt[sim$t>=11517 & sim$t<=12515]
  movt_ind=mean(unlist(lapply(c(1:length(movt)), function(x){movt[x]=="out" || movt[x]==1})))
  movt_ind
}

#Number of ant seconds in each sub chamber

Nosubcham1<-function(sim){
  
  cham=sim$chamber[sim$t>=11517 & sim$t<=12515]
  mean(cham==1)
}

Nosubcham2<-function(sim){
  
  cham=sim$chamber[sim$t>=11517 & sim$t<=12515]
  mean(cham==2)
}

Nosubcham3<-function(sim){
  
  cham=sim$chamber[sim$t>=11517 & sim$t<=12515]
  mean(cham==3)
}

Nosubcham4<-function(sim){
  
  cham=sim$chamber[sim$t>=11517 & sim$t<=12515]
  mean(cham==4)
}

Nosubcham5<-function(sim){
  
  cham=sim$chamber[sim$t>=11517 & sim$t<=12515]
  mean(cham==5)
}

Nosubcham6<-function(sim){
  
  cham=sim$chamber[sim$t>=11517 & sim$t<=12515]
  mean(cham==6)
}

Nosubcham7<-function(sim){
  
  cham=sim$chamber[sim$t>=11517 & sim$t<=12515]
  mean(cham==7)
}

Nosubcham8<-function(sim){
  
  cham=sim$chamber[sim$t>=11517 & sim$t<=12515]
  mean(cham==8)
}

#Average number of ants around each ant at a radius of 12mm

avgindant<-function(simt, ind){
  
  iant=simt[simt$id==ind,]
  rest=simt[simt$id!=ind,]
  idx=which((veuc_dist(x1 = iant$`x-1`, y1 = iant$`y-1`, x2 = rest$`x-1`, y2 = rest$`y-1`))<12)
  length(idx)
}

Avgnearants<-function(sim){
  listants={}
  sim1=sim[sim$t>=11517 & sim$t<=12515, ]
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


####################################

M1_dist<-data.frame(Model=rep(NA, 900), Metric=rep(NA, 900))

M1_dist$Model[1:100]<-"SDE"
M1_dist$Metric[1:100]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//SDE//pred_", x, ".rda", sep=""))
  TotDist(findata)}))

M1_dist$Model[101:200]<-"RF ind"
M1_dist$Metric[101:200]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//RF//", x, "RFind.rda", sep=""))
  TotDist(findata)}))

M1_dist$Model[201:300]<-"BNN ind"
M1_dist$Metric[201:300]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//BNN//", x, "NNind.rda", sep=""))
  TotDist(findata)}))

M1_dist$Model[301:400]<-"RNN ind"
M1_dist$Metric[301:400]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//RNN//", x, "RNNind.rda", sep=""))
  TotDist(findata)}))

M1_dist$Model[401:500]<-"LSTM ind"
M1_dist$Metric[401:500]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//LSTM//", x, "LSTMind.rda", sep=""))
  TotDist(findata)}))

M1_dist$Model[501:600]<-"RF col"
M1_dist$Metric[501:600]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//RF//colony//", x, "RFcol.rda", sep=""))
  TotDist(findata)}))

M1_dist$Model[601:700]<-"BNN col"
M1_dist$Metric[601:700]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//BNN//colony//", x, "NNcol.rda", sep=""))
  TotDist(findata)}))

M1_dist$Model[701:800]<-"RNN col"
M1_dist$Metric[701:800]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//RNN//colony//", x, "RNNcol.rda", sep=""))
  TotDist(findata)}))

M1_dist$Model[801:900]<-"LSTM col"
M1_dist$Metric[801:900]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//LSTM//colony//", x, "LSTMcol.rda", sep=""))
  TotDist(findata)}))

save(M1_dist, file = "M1_dist.Rdata")




#######
# Metric2

M1_stat<-data.frame(Model=rep(NA, 900), Metric=rep(NA, 900))

M1_stat$Model[1:100]<-"SDE"
M1_stat$Metric[1:100]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//SDE//pred_", x, ".rda", sep=""))
  Percstat(findata)}))

M1_stat$Model[101:200]<-"RF ind"
M1_stat$Metric[101:200]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//RF//", x, "RFind.rda", sep=""))
  Percstat(findata)}))

M1_stat$Model[201:300]<-"BNN ind"
M1_stat$Metric[201:300]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//BNN//", x, "NNind.rda", sep=""))
  Percstat(findata)}))

M1_stat$Model[301:400]<-"RNN ind"
M1_stat$Metric[301:400]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//RNN//", x, "RNNind.rda", sep=""))
  Percstat(findata)}))

M1_stat$Model[401:500]<-"LSTM ind"
M1_stat$Metric[401:500]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//LSTM//", x, "LSTMind.rda", sep=""))
  Percstat(findata)}))

M1_stat$Model[501:600]<-"RF col"
M1_stat$Metric[501:600]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//RF//colony//", x, "RFcol.rda", sep=""))
  Percstat(findata)}))

M1_stat$Model[601:700]<-"BNN col"
M1_stat$Metric[601:700]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//BNN//colony//", x, "NNcol.rda", sep=""))
  Percstat(findata)}))

M1_stat$Model[701:800]<-"RNN col"
M1_stat$Metric[701:800]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//RNN//colony//", x, "RNNcol.rda", sep=""))
  Percstat(findata)}))

M1_stat$Model[801:900]<-"LSTM col"
M1_stat$Metric[801:900]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//LSTM//colony//", x, "LSTMcol.rda", sep=""))
  Percstat(findata)}))


save(M1_stat, file = "M1_stat.Rdata")



##############################
#Metric 3


M1_out<-data.frame(Model=rep(NA, 900), Metric=rep(NA, 900))

M1_out$Model[1:100]<-"SDE"
M1_out$Metric[1:100]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//SDE//pred_", x, ".rda", sep=""))
  Percout(findata)}))

M1_out$Model[101:200]<-"RF ind"
M1_out$Metric[101:200]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//RF//", x, "RFind.rda", sep=""))
  Percout(findata)}))

M1_out$Model[201:300]<-"BNN ind"
M1_out$Metric[201:300]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//BNN//", x, "NNind.rda", sep=""))
  Percout(findata)}))

M1_out$Model[301:400]<-"RNN ind"
M1_out$Metric[301:400]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//RNN//", x, "RNNind.rda", sep=""))
  Percout(findata)}))

M1_out$Model[401:500]<-"LSTM ind"
M1_out$Metric[401:500]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//LSTM//", x, "LSTMind.rda", sep=""))
  Percout(findata)}))

M1_out$Model[501:600]<-"RF col"
M1_out$Metric[501:600]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//RF//colony//", x, "RFcol.rda", sep=""))
  Percout(findata)}))

M1_out$Model[601:700]<-"BNN col"
M1_out$Metric[601:700]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//BNN//colony//", x, "NNcol.rda", sep=""))
  Percout(findata)}))

M1_out$Model[701:800]<-"RNN col"
M1_out$Metric[701:800]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//RNN//colony//", x, "RNNcol.rda", sep=""))
  Percout(findata)}))

M1_out$Model[801:900]<-"LSTM col"
M1_out$Metric[801:900]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//LSTM//colony//", x, "LSTMcol.rda", sep=""))
  Percout(findata)}))


save(M1_out, file="M1_out.Rdata")


##################
# Metric sub1

M1_sub1<-data.frame(Model=rep(NA, 900), Metric=rep(NA, 900))

M1_sub1$Model[1:100]<-"SDE"
M1_sub1$Metric[1:100]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//SDE//pred_", x, ".rda", sep=""))
  Nosubcham1(findata)}))

M1_sub1$Model[101:200]<-"RF ind"
M1_sub1$Metric[101:200]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//RF//", x, "RFind.rda", sep=""))
  Nosubcham1(findata)}))

M1_sub1$Model[201:300]<-"BNN ind"
M1_sub1$Metric[201:300]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//BNN//", x, "NNind.rda", sep=""))
  Nosubcham1(findata)}))

M1_sub1$Model[301:400]<-"RNN ind"
M1_sub1$Metric[301:400]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//RNN//", x, "RNNind.rda", sep=""))
  Nosubcham1(findata)}))

M1_sub1$Model[401:500]<-"LSTM ind"
M1_sub1$Metric[401:500]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//LSTM//", x, "LSTMind.rda", sep=""))
  Nosubcham1(findata)}))

M1_sub1$Model[501:600]<-"RF col"
M1_sub1$Metric[501:600]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//RF//colony//", x, "RFcol.rda", sep=""))
  Nosubcham1(findata)}))

M1_sub1$Model[601:700]<-"BNN col"
M1_sub1$Metric[601:700]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//BNN//colony//", x, "NNcol.rda", sep=""))
  Nosubcham1(findata)}))

M1_sub1$Model[701:800]<-"RNN col"
M1_sub1$Metric[701:800]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//RNN//colony//", x, "RNNcol.rda", sep=""))
  Nosubcham1(findata)}))

M1_sub1$Model[801:900]<-"LSTM col"
M1_sub1$Metric[801:900]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//LSTM//colony//", x, "LSTMcol.rda", sep=""))
  Nosubcham1(findata)}))


save(M1_sub1, file="M1_sub1.Rdata")


##################
# Metric sub2

M1_sub2<-data.frame(Model=rep(NA, 900), Metric=rep(NA, 900))

M1_sub2$Model[1:100]<-"SDE"
M1_sub2$Metric[1:100]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//SDE//pred_", x, ".rda", sep=""))
  Nosubcham2(findata)}))

M1_sub2$Model[101:200]<-"RF ind"
M1_sub2$Metric[101:200]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//RF//", x, "RFind.rda", sep=""))
  Nosubcham2(findata)}))

M1_sub2$Model[201:300]<-"BNN ind"
M1_sub2$Metric[201:300]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//BNN//", x, "NNind.rda", sep=""))
  Nosubcham2(findata)}))

M1_sub2$Model[301:400]<-"RNN ind"
M1_sub2$Metric[301:400]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//RNN//", x, "RNNind.rda", sep=""))
  Nosubcham2(findata)}))

M1_sub2$Model[401:500]<-"LSTM ind"
M1_sub2$Metric[401:500]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//LSTM//", x, "LSTMind.rda", sep=""))
  Nosubcham2(findata)}))

M1_sub2$Model[501:600]<-"RF col"
M1_sub2$Metric[501:600]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//RF//colony//", x, "RFcol.rda", sep=""))
  Nosubcham2(findata)}))

M1_sub2$Model[601:700]<-"BNN col"
M1_sub2$Metric[601:700]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//BNN//colony//", x, "NNcol.rda", sep=""))
  Nosubcham2(findata)}))

M1_sub2$Model[701:800]<-"RNN col"
M1_sub2$Metric[701:800]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//RNN//colony//", x, "RNNcol.rda", sep=""))
  Nosubcham2(findata)}))

M1_sub2$Model[801:900]<-"LSTM col"
M1_sub2$Metric[801:900]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//LSTM//colony//", x, "LSTMcol.rda", sep=""))
  Nosubcham2(findata)}))



save(M1_sub2, file="M1_sub2.Rdata")


##################
# Metric sub3

M1_sub3<-data.frame(Model=rep(NA, 900), Metric=rep(NA, 900))

M1_sub3$Model[1:100]<-"SDE"
M1_sub3$Metric[1:100]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//SDE//pred_", x, ".rda", sep=""))
  Nosubcham3(findata)}))

M1_sub3$Model[101:200]<-"RF ind"
M1_sub3$Metric[101:200]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//RF//", x, "RFind.rda", sep=""))
  Nosubcham3(findata)}))

M1_sub3$Model[201:300]<-"BNN ind"
M1_sub3$Metric[201:300]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//BNN//", x, "NNind.rda", sep=""))
  Nosubcham3(findata)}))

M1_sub3$Model[301:400]<-"RNN ind"
M1_sub3$Metric[301:400]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//RNN//", x, "RNNind.rda", sep=""))
  Nosubcham3(findata)}))

M1_sub3$Model[401:500]<-"LSTM ind"
M1_sub3$Metric[401:500]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//LSTM//", x, "LSTMind.rda", sep=""))
  Nosubcham3(findata)}))

M1_sub3$Model[501:600]<-"RF col"
M1_sub3$Metric[501:600]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//RF//colony//", x, "RFcol.rda", sep=""))
  Nosubcham3(findata)}))

M1_sub3$Model[601:700]<-"BNN col"
M1_sub3$Metric[601:700]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//BNN//colony//", x, "NNcol.rda", sep=""))
  Nosubcham3(findata)}))

M1_sub3$Model[701:800]<-"RNN col"
M1_sub3$Metric[701:800]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//RNN//colony//", x, "RNNcol.rda", sep=""))
  Nosubcham3(findata)}))

M1_sub3$Model[801:900]<-"LSTM col"
M1_sub3$Metric[801:900]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//LSTM//colony//", x, "LSTMcol.rda", sep=""))
  Nosubcham3(findata)}))


save(M1_sub3, file="M1_sub3.Rdata")

##################
# Metric sub4

M1_sub4<-data.frame(Model=rep(NA, 900), Metric=rep(NA, 900))

M1_sub4$Model[1:100]<-"SDE"
M1_sub4$Metric[1:100]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//SDE//pred_", x, ".rda", sep=""))
  Nosubcham4(findata)}))

M1_sub4$Model[101:200]<-"RF ind"
M1_sub4$Metric[101:200]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//RF//", x, "RFind.rda", sep=""))
  Nosubcham4(findata)}))

M1_sub4$Model[201:300]<-"BNN ind"
M1_sub4$Metric[201:300]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//BNN//", x, "NNind.rda", sep=""))
  Nosubcham4(findata)}))

M1_sub4$Model[301:400]<-"RNN ind"
M1_sub4$Metric[301:400]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//RNN//", x, "RNNind.rda", sep=""))
  Nosubcham4(findata)}))

M1_sub4$Model[401:500]<-"LSTM ind"
M1_sub4$Metric[401:500]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//LSTM//", x, "LSTMind.rda", sep=""))
  Nosubcham4(findata)}))

M1_sub4$Model[501:600]<-"RF col"
M1_sub4$Metric[501:600]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//RF//colony//", x, "RFcol.rda", sep=""))
  Nosubcham4(findata)}))

M1_sub4$Model[601:700]<-"BNN col"
M1_sub4$Metric[601:700]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//BNN//colony//", x, "NNcol.rda", sep=""))
  Nosubcham4(findata)}))

M1_sub4$Model[701:800]<-"RNN col"
M1_sub4$Metric[701:800]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//RNN//colony//", x, "RNNcol.rda", sep=""))
  Nosubcham4(findata)}))

M1_sub4$Model[801:900]<-"LSTM col"
M1_sub4$Metric[801:900]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//LSTM//colony//", x, "LSTMcol.rda", sep=""))
  Nosubcham4(findata)}))



save(M1_sub4, file="M1_sub4.Rdata")



##################
# Metric sub5

M1_sub5<-data.frame(Model=rep(NA, 900), Metric=rep(NA, 900))

M1_sub5$Model[1:100]<-"SDE"
M1_sub5$Metric[1:100]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//SDE//pred_", x, ".rda", sep=""))
  Nosubcham5(findata)}))

M1_sub5$Model[101:200]<-"RF ind"
M1_sub5$Metric[101:200]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//RF//", x, "RFind.rda", sep=""))
  Nosubcham5(findata)}))

M1_sub5$Model[201:300]<-"BNN ind"
M1_sub5$Metric[201:300]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//BNN//", x, "NNind.rda", sep=""))
  Nosubcham5(findata)}))

M1_sub5$Model[301:400]<-"RNN ind"
M1_sub5$Metric[301:400]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//RNN//", x, "RNNind.rda", sep=""))
  Nosubcham5(findata)}))

M1_sub5$Model[401:500]<-"LSTM ind"
M1_sub5$Metric[401:500]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//LSTM//", x, "LSTMind.rda", sep=""))
  Nosubcham5(findata)}))

M1_sub5$Model[501:600]<-"RF col"
M1_sub5$Metric[501:600]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//RF//colony//", x, "RFcol.rda", sep=""))
  Nosubcham5(findata)}))

M1_sub5$Model[601:700]<-"BNN col"
M1_sub5$Metric[601:700]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//BNN//colony//", x, "NNcol.rda", sep=""))
  Nosubcham5(findata)}))

M1_sub5$Model[701:800]<-"RNN col"
M1_sub5$Metric[701:800]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//RNN//colony//", x, "RNNcol.rda", sep=""))
  Nosubcham5(findata)}))

M1_sub5$Model[801:900]<-"LSTM col"
M1_sub5$Metric[801:900]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//LSTM//colony//", x, "LSTMcol.rda", sep=""))
  Nosubcham5(findata)}))


save(M1_sub5, file="M1_sub5.Rdata")



# Metric sub6

M1_sub6<-data.frame(Model=rep(NA, 900), Metric=rep(NA, 900))

M1_sub6$Model[1:100]<-"SDE"
M1_sub6$Metric[1:100]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//SDE//pred_", x, ".rda", sep=""))
  Nosubcham6(findata)}))

M1_sub6$Model[101:200]<-"RF ind"
M1_sub6$Metric[101:200]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//RF//", x, "RFind.rda", sep=""))
  Nosubcham6(findata)}))

M1_sub6$Model[201:300]<-"BNN ind"
M1_sub6$Metric[201:300]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//BNN//", x, "NNind.rda", sep=""))
  Nosubcham6(findata)}))

M1_sub6$Model[301:400]<-"RNN ind"
M1_sub6$Metric[301:400]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//RNN//", x, "RNNind.rda", sep=""))
  Nosubcham6(findata)}))

M1_sub6$Model[401:500]<-"LSTM ind"
M1_sub6$Metric[401:500]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//LSTM//", x, "LSTMind.rda", sep=""))
  Nosubcham6(findata)}))

M1_sub6$Model[501:600]<-"RF col"
M1_sub6$Metric[501:600]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//RF//colony//", x, "RFcol.rda", sep=""))
  Nosubcham6(findata)}))

M1_sub6$Model[601:700]<-"BNN col"
M1_sub6$Metric[601:700]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//BNN//colony//", x, "NNcol.rda", sep=""))
  Nosubcham6(findata)}))

M1_sub6$Model[701:800]<-"RNN col"
M1_sub6$Metric[701:800]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//RNN//colony//", x, "RNNcol.rda", sep=""))
  Nosubcham6(findata)}))

M1_sub6$Model[801:900]<-"LSTM col"
M1_sub6$Metric[801:900]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//LSTM//colony//", x, "LSTMcol.rda", sep=""))
  Nosubcham6(findata)}))


save(M1_sub16, file="M1_sub6.Rdata")



# Metric sub7

M1_sub7<-data.frame(Model=rep(NA, 900), Metric=rep(NA, 900))

M1_sub7$Model[1:100]<-"SDE"
M1_sub7$Metric[1:100]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//SDE//pred_", x, ".rda", sep=""))
  Nosubcham7(findata)}))

M1_sub7$Model[101:200]<-"RF ind"
M1_sub7$Metric[101:200]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//RF//", x, "RFind.rda", sep=""))
  Nosubcham7(findata)}))

M1_sub7$Model[201:300]<-"BNN ind"
M1_sub7$Metric[201:300]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//BNN//", x, "NNind.rda", sep=""))
  Nosubcham7(findata)}))

M1_sub7$Model[301:400]<-"RNN ind"
M1_sub7$Metric[301:400]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//RNN//", x, "RNNind.rda", sep=""))
  Nosubcham7(findata)}))

M1_sub7$Model[401:500]<-"LSTM ind"
M1_sub7$Metric[401:500]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//LSTM//", x, "LSTMind.rda", sep=""))
  Nosubcham7(findata)}))

M1_sub7$Model[501:600]<-"RF col"
M1_sub7$Metric[501:600]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//RF//colony//", x, "RFcol.rda", sep=""))
  Nosubcham7(findata)}))

M1_sub7$Model[601:700]<-"BNN col"
M1_sub7$Metric[601:700]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//BNN//colony//", x, "NNcol.rda", sep=""))
  Nosubcham7(findata)}))

M1_sub7$Model[701:800]<-"RNN col"
M1_sub7$Metric[701:800]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//RNN//colony//", x, "RNNcol.rda", sep=""))
  Nosubcham7(findata)}))

M1_sub7$Model[801:900]<-"LSTM col"
M1_sub7$Metric[801:900]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//LSTM//colony//", x, "LSTMcol.rda", sep=""))
  Nosubcham7(findata)}))

save(M1_sub7, file="M1_sub7.Rdata")




# Metric sub8

M1_sub8<-data.frame(Model=rep(NA, 900), Metric=rep(NA, 900))

M1_sub8$Model[1:100]<-"SDE"
M1_sub8$Metric[1:100]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//SDE//pred_", x, ".rda", sep=""))
  Nosubcham8(findata)}))

M1_sub8$Model[101:200]<-"RF ind"
M1_sub8$Metric[101:200]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//RF//", x, "RFind.rda", sep=""))
  Nosubcham8(findata)}))

M1_sub8$Model[201:300]<-"BNN ind"
M1_sub8$Metric[201:300]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//BNN//", x, "NNind.rda", sep=""))
  Nosubcham8(findata)}))

M1_sub8$Model[301:400]<-"RNN ind"
M1_sub8$Metric[301:400]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//RNN//", x, "RNNind.rda", sep=""))
  Nosubcham8(findata)}))

M1_sub8$Model[401:500]<-"LSTM ind"
M1_sub8$Metric[401:500]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//LSTM//", x, "LSTMind.rda", sep=""))
  Nosubcham8(findata)}))

M1_sub8$Model[501:600]<-"RF col"
M1_sub8$Metric[501:600]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//RF//colony//", x, "RFcol.rda", sep=""))
  Nosubcham8(findata)}))

M1_sub8$Model[601:700]<-"BNN col"
M1_sub8$Metric[601:700]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//BNN//colony//", x, "NNcol.rda", sep=""))
  Nosubcham8(findata)}))

M1_sub8$Model[701:800]<-"RNN col"
M1_sub8$Metric[701:800]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//RNN//colony//", x, "RNNcol.rda", sep=""))
  Nosubcham8(findata)}))

M1_sub8$Model[801:900]<-"LSTM col"
M1_sub8$Metric[801:900]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//LSTM//colony//", x, "LSTMcol.rda", sep=""))
  Nosubcham8(findata)}))


save(M1_sub8, file="M1_sub8.Rdata")

# Metric Avgnearants

M1_near<-data.frame(Model=rep(NA, 900), Metric=rep(NA, 900))

M1_near$Model[1:100]<-"SDE"
M1_near$Metric[1:100]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//SDE//pred_", x, ".rda", sep=""))
  Avgnearants(findata)}))

M1_near$Model[101:200]<-"RF ind"
M1_near$Metric[101:200]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//RF//", x, "RFind.rda", sep=""))
  Avgnearants(findata)}))

M1_near$Model[201:300]<-"BNN ind"
M1_near$Metric[201:300]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//BNN//", x, "NNind.rda", sep=""))
  Avgnearants(findata)}))

M1_near$Model[301:400]<-"RNN ind"
M1_near$Metric[301:400]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//RNN//", x, "RNNind.rda", sep=""))
  Avgnearants(findata)}))

M1_near$Model[401:500]<-"LSTM ind"
M1_near$Metric[401:500]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//LSTM//", x, "LSTMind.rda", sep=""))
  Avgnearants(findata)}))

M1_near$Model[501:600]<-"RF col"
M1_near$Metric[501:600]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//RF//colony//", x, "RFcol.rda", sep=""))
  Avgnearants(findata)}))

M1_near$Model[601:700]<-"BNN col"
M1_near$Metric[601:700]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//BNN//colony//", x, "NNcol.rda", sep=""))
  Avgnearants(findata)}))

M1_near$Model[701:800]<-"RNN col"
M1_near$Metric[701:800]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//RNN//colony//", x, "RNNcol.rda", sep=""))
  Avgnearants(findata)}))

M1_near$Model[801:900]<-"LSTM col"
M1_near$Metric[801:900]<-unlist(parLapply(cl, c(1:100), function(x){ load(paste("//storage//work//d//dzw5320//sims//LSTM//colony//", x, "LSTMcol.rda", sep=""))
  Avgnearants(findata)}))


save(M1_near, file="M1_near.Rdata")







