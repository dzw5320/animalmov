
################################################################
##
## Rcpp libraries
##
################################################################

library(inline)
library(Rcpp)
library(RcppArmadillo)

v = "010"

################################################################
##
## Project C code
##
## project(xy.mat,P,index) - projects xy.mat onto P[index,]
##
################################################################

##
## Polygon constraint
##
P=matrix(c(0,-.2,
           .1,0,
           .2,-.15,
           .4,-.05,
           .5,0,
           .6,.4,
           .5,.5,
           .4,.6,
           .3,.4,
           .2,.2,
           0,.2,
           -.1,.4,
           -.2,.3,
           -.3,.3,
           -.4,.1,
           -.3,-.1,
           -.2,.1,
           -.1,0,
           0,-.2)
              ,byrow=TRUE,ncol=2)
plot(P,type="b")



library(here)

sourceCpp(here(paste0(v, "_ephraim_projection_code"),"project.cpp"))

plot(P,type="b")
xy.noproj=matrix(runif(100*2,-.5,.5),nrow=100)
points(xy.noproj,col="red")
## project onto P
xy.proj=project(xy.noproj,P) ### ??
points(xy.proj,col="blue",pch=3)
arrows(xy.noproj[,1],xy.noproj[,2],xy.proj[,1],xy.proj[,2],length=.1)

