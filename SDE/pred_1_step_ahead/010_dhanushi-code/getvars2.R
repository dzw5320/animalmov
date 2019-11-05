 load("antdata.Rdata")
# library(DataCombine)
# 
# allants<-ants.c1low

source("fun1.R")
ant_ids<-as.character(unique(antdata$id))

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
#   wwall<-if(chamber==1){0}else if(chamber==3){52}else if(chamber==5){104}else if(chamber==7){156}else 
#     if((chamber==2 && y<53)|| (chamber==3 && y<6)){21.5}else if((chamber==4 && y<53)||(chamber==5 && y<6)){73.5}else
#       if((chamber==6 && y<53)||(chamber==7 && y<6)){125.5}else{NA}
#   
#   x-wwall
# }
# 
# distwalle<-function(x, y, chamber){
#   
#   ewall<-if(chamber==1 && y<53){18.5}else if(chamber==2 && y>6){40}else if((chamber==2 && y<6)||(chamber==3 && y<53)){70.5}else
#     if((chamber==3 && y>53)||(chamber==4 && y>6)){92}else if((chamber==4 && y<6)||(chamber==5 && y<53)){122.5}else
#       if((chamber==5 && y>53)||(chamber==6 && y>6)){144}else if((chamber==6 && y<6)||(chamber==7 && y<53)){174.5}else
#         if((chamber==7 && y>53)||(chamber==8 && y>6)){196}else if(x>196){199}else{NA}
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
# 
# getQ1<-function(iant, rest){
#   
#   
#   Q1sub<-subset(rest, rest$`x-1`>((iant$`x-1`)-8) & rest$`x-1`<(iant$`x-1`) & 
#                   rest$`y-1`>(iant$`y-1`) & rest$`y-1`<(iant$`y-1`+8))
#   
#   l<-nrow(Q1sub)
#   if(l>1){
#     d<-veuc_dist(x1 = iant$`x-1`, y1 = iant$`y-1`, x2 = Q1sub$`x-1`, y2 = Q1sub$`y-1`)
#     mind<-min(d)
#     id<-which.min(d)
#     xlag1<-Q1sub$`x-1`[id]
#     ylag1<-Q1sub$`y-1`[id]
#     vxlag1<-Q1sub$`vx-1`[id]
#     vylag1<-Q1sub$`vy-1`[id]
#     xlag2<-Q1sub$`x-2`[id]
#     ylag2<-Q1sub$`y-2`[id]
#     vxlag2<-Q1sub$`vx-2`[id]
#     vylag2<-Q1sub$`vy-2`[id]
#     c(l, mind, xlag1, xlag2, ylag1, ylag2, vxlag1, vxlag2, vylag1, vylag2)
#   }else{
#     c(0, 0, 0, 0, 0, 0, 0, 0, 0, 0) 
#     
#   }
#   
#   
# }
# 
# 
# 
# 
# 
# getQ2<-function(iant, rest){
#   
#   
#   Q2sub<-subset(rest, rest$`x-1`>((iant$`x-1`)) & rest$`x-1`<(iant$`x-1`+8) & 
#                   rest$`y-1`>(iant$`y-1`) & rest$`y-1`<(iant$`y-1`+8))
#   
#   l<-nrow(Q2sub)
#   if(l>1){
#     d<-veuc_dist(x1 = iant$`x-1`, y1 = iant$`y-1`, x2 = Q2sub$`x-1`, y2 = Q2sub$`y-1`)
#     mind<-min(d)
#     id<-which.min(d)
#     xlag1<-Q2sub$`x-1`[id]
#     ylag1<-Q2sub$`y-1`[id]
#     vxlag1<-Q2sub$`vx-1`[id]
#     vylag1<-Q2sub$`vy-1`[id]
#     xlag2<-Q2sub$`x-2`[id]
#     ylag2<-Q2sub$`y-2`[id]
#     vxlag2<-Q2sub$`vx-2`[id]
#     vylag2<-Q2sub$`vy-2`[id]
#     c(l, mind, xlag1, xlag2, ylag1, ylag2, vxlag1, vxlag2, vylag1, vylag2)
#   }else{
#     c(0, 0, 0, 0, 0, 0, 0, 0, 0, 0) 
#     
#   }
#   
# }
# 
# 
# 
# 
# getQ3<-function(iant, rest){
#   
#   
#   Q3sub<-subset(rest, rest$`x-1`>((iant$`x-1`-8)) & rest$`x-1`<(iant$`x-1`) & 
#                   rest$`y-1`>(iant$`y-1`-8) & rest$`y-1`<(iant$`y-1`))
#   
#   l<-nrow(Q3sub)
#   if(l>1){
#     d<-veuc_dist(x1 = iant$`x-1`, y1 = iant$`y-1`, x2 = Q3sub$`x-1`, y2 = Q3sub$`y-1`)
#     mind<-min(d)
#     id<-which.min(d)
#     xlag1<-Q3sub$`x-1`[id]
#     ylag1<-Q3sub$`y-1`[id]
#     vxlag1<-Q3sub$`vx-1`[id]
#     vylag1<-Q3sub$`vy-1`[id]
#     xlag2<-Q3sub$`x-2`[id]
#     ylag2<-Q3sub$`y-2`[id]
#     vxlag2<-Q3sub$`vx-2`[id]
#     vylag2<-Q3sub$`vy-2`[id]
#     c(l, mind, xlag1, xlag2, ylag1, ylag2, vxlag1, vxlag2, vylag1, vylag2)
#   }else{
#     c(0, 0, 0, 0, 0, 0, 0, 0, 0, 0) 
#     
#   }
#   
# }
# 
# 
# 
# getQ4<-function(iant, rest){
#   
#   
#   Q4sub<-subset(rest, rest$`x-1`>((iant$`x-1`)) & rest$`x-1`<(iant$`x-1`+8) & 
#                   rest$`y-1`>(iant$`y-1`-8) & rest$`y-1`<(iant$`y-1`))
#   
#   l<-nrow(Q4sub)
#   if(l>1){
#     d<-veuc_dist(x1 = iant$`x-1`, y1 = iant$`y-1`, x2 = Q4sub$`x-1`, y2 = Q4sub$`y-1`)
#     mind<-min(d)
#     id<-which.min(d)
#     xlag1<-Q4sub$`x-1`[id]
#     ylag1<-Q4sub$`y-1`[id]
#     vxlag1<-Q4sub$`vx-1`[id]
#     vylag1<-Q4sub$`vy-1`[id]
#     xlag2<-Q4sub$`x-2`[id]
#     ylag2<-Q4sub$`y-2`[id]
#     vxlag2<-Q4sub$`vx-2`[id]
#     vylag2<-Q4sub$`vy-2`[id]
#     c(l, mind, xlag1, xlag2, ylag1, ylag2, vxlag1, vxlag2, vylag1, vylag2)
#   }else{
#     c(0, 0, 0, 0, 0, 0, 0, 0, 0, 0) 
#     
#   }
#   
# }
# 
# 
# 
# 
# # getnndist<-function(iant, rest){
# #   
# #   
# #   min(veuc_dist(x1 = iant$`x-1`, y1 = iant$`y-1`, x2 = rest$`x-1`, y2 = rest$`y-1`))
# #   
# # }
# # 
# # getnnxlag1<-function(iant, rest){
# #   
# #   id<-which.min(veuc_dist(x1 = iant$`x-1`, y1 = iant$`y-1`, x2 = rest$`x-1`, y2 = rest$`y-1`))
# #   rest$`x-1`[id]
# #   
# #   
# # }
# # 
# # getnnylag1<-function(iant, rest){
# #   
# #   id<-which.min(veuc_dist(x1 = iant$`x-1`, y1 = iant$`y-1`, x2 = rest$`x-1`, y2 = rest$`y-1`))
# #   rest$`y-1`[id]
# #   
# #   
# # }
# # 
# # getnnvxlag1<-function(iant, rest){
# #   
# #   id<-which.min(veuc_dist(x1 = iant$`x-1`, y1 = iant$`y-1`, x2 = rest$`x-1`, y2 = rest$`y-1`))
# #   rest$`vx-1`[id]
# #   
# #   
# # }
# # 
# # getnnvylag1<-function(iant, rest){
# #   
# #   id<-which.min(veuc_dist(x1 = iant$`x-1`, y1 = iant$`y-1`, x2 = rest$`x-1`, y2 = rest$`y-1`))
# #   rest$`vy-1`[id]
# #   
# #   
# # }
# # 


# indants<-{}
# 
# for(i in 1:length(ant_ids)){
#   
#   indants[[i]]<-allants[allants$id==ant_ids[i],]
#   
#   
# }  




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

