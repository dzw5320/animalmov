#Get colony level data
library(dplyr)
load("indants2.Rdata")
data1<-indants2


data_in<-data1[,c(1:2, 9:35, 37:48)]
data_out<-data1[,c(1:2, 5,6,36)]


##Now reshape these so that each observation is a time point.

library(reshape)
library(data.table)

data_in=as.data.table(data_in)
data_out=as.data.table(data_out)
data_in_reshape<-dcast(data_in, t~id, value.var=c("x-1" ,      "x-2"  ,     "x-3",       "x-4",      
                                                    "x-5"  ,     "y-1"  ,     "y-2" ,      "y-3"  ,     "y-4" ,      "y-5"  ,    
                                                    "vx-1"  ,    "vx-2" ,     "vx-3"  ,    "vx-4" ,     "vx-5"  ,    "vy-1" ,    
                                                    "vy-2"  ,    "vy-3" ,     "vy-4"  ,    "vy-5" ,     "chamber" ,  "distind"  ,
                                                    "stattime"  ,"nwalldist", "swalldist" ,"wwalldist" ,"ewalldist" ,"nndist" ,  
                                                    "nnxlag1" ,  "nnylag1",   "nnvxlag1"  ,"nnvylag1" , "Q1"  ,      "Q2" ,      
                                                    "Q3"      ,  "Q4"   ,     "nnmove"  ,  "nnstill"  , "distqueen") )

data_out_reshape<-dcast(data_out, t~id, value.var=c("movt", "vx", "vy"))



write.csv(data_in_reshape, file = "C:\\Users\\dhanu\\OneDrive\\Machine Learning\\Python\\Recurrent\\Col_inall.csv")
write.csv(data_out_reshape, file = "C:\\Users\\dhanu\\OneDrive\\Machine Learning\\Python\\Recurrent\\Col_out.csv")

