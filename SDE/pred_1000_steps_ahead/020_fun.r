
# Subroutines -------------------------------------------------------------


get.motpot.data <- function(ani.list, # mucst be list of data frames, not tibbles
                            boundary.poly, res = 1, xcolname = "x", 
                            ycolname = "y", tcolname = "t") {
  ## Get raster and adjacency objects -----------------------------------
  R <- raster(xmn = min(boundary.poly[, 1]), xmx = max(boundary.poly[, 1]), 
              ymn = min(boundary.poly[, 2]), ymx = max(boundary.poly[, 2]), 
              res = res)
  values(R) <- NA
  xy <- xyFromCell(R, 1:ncell(R))
  idx.in.rast <- inpip(xy, boundary.poly)
  ##
  values(R)[idx.in.rast] <- 1
  vals <- values(R)
  ##
  adj <- adjacent(R, idx.in.rast, target = idx.in.rast)
  ncell <- ncell(R)
  Q <- Matrix(0, ncell, ncell, sparse = TRUE)
  Q[adj] <- 1
  ## Q=Q+t(Q)
  one <- Matrix(1, nrow = ncell, ncol = 1)
  diag(Q) <- -Q %*% one
  Q <- -Q[idx.in.rast, idx.in.rast]
  n.alpha <- nrow(Q)
  ## Make data for motility / potential analysis -----------------------
  ts.df <- data.frame()
  ants.df <- data.frame()
  cat("Individual Ants: ")
  for (i in 1:length(ani.list)) { # for each ant..
    cat(i, " ")
    x <- ani.list[[i]][, xcolname] # x-positions of ant i
    y <- ani.list[[i]][, ycolname] # y-positions of ant i
    t <- ani.list[[i]][, tcolname]
    T <- length(x) 
    # h=1
    vx <- (x[1:(T - 2)] - x[2:(T - 1)]) ## [x(t)-x(t+h)]
    wx <- (x[3:T] - x[2:(T - 1)]) / (t[3:(T)] - t[2:(T - 1)]) - (x[2:(T - 1)] - x[1:(T - 2)]) / (t[2:(T - 1)] - t[1:(T - 2)]) ## [ x(t+2h)-2x(t+h)+x(t) ]/h if time steps are regular
    vy <- (y[1:(T - 2)] - y[2:(T - 1)]) ## y(t)-y(t+h)
    wy <- (y[3:T] - y[2:(T - 1)]) / (t[3:(T)] - t[2:(T - 1)]) - (y[2:(T - 1)] - y[1:(T - 2)]) / (t[2:(T - 1)] - t[1:(T - 2)]) ## [ y(t+2h)-2y(t+h)+y(t) ]/h if time steps are regular
    x <- x[-(T:(T - 1))]
    y <- y[-(T:(T - 1))]
    t <- t[-(T:(T - 1))]
    h <- t[2:(T - 1)] - t[1:(T - 2)]
    ## making "ants.df"
    ants.df <- rbind(ants.df, data.frame(id = names(ani.list)[i], t = t, h = h, 
                                         x = x, y = y, vx = vx, vy = vy, wx = wx, 
                                         wy = wy))
    ## making "ts.df"  -- we remove obs with NA or not moving
    na.idx <- which(is.na(x + y + t + vx + vy + wx + wy + h))
    nomove.idx <- which(vx == 0 & vy == 0)
    x <- x[-unique(c(na.idx, nomove.idx))]
    y <- y[-unique(c(na.idx, nomove.idx))]
    t <- t[-unique(c(na.idx, nomove.idx))]
    h <- h[-unique(c(na.idx, nomove.idx))]
    wx <- wx[-unique(c(na.idx, nomove.idx))]
    vx <- vx[-unique(c(na.idx, nomove.idx))]
    wy <- wy[-unique(c(na.idx, nomove.idx))]
    vy <- vy[-unique(c(na.idx, nomove.idx))]
    ##
    ## Get difference matrix for x-directional derivative of H
    ##
    rx <- res(R)[1]
    ry <- res(R)[2]
    cell.locs <- cellFromXY(R, cbind(x, y))
    cell.locs.up <- cellFromXY(R, cbind(x, y + ry))
    cell.locs.down <- cellFromXY(R, cbind(x, y - ry))
    cell.locs.right <- cellFromXY(R, cbind(x + rx, y))
    cell.locs.left <- cellFromXY(R, cbind(x - rx, y))
    idx.x.cent.diff <- which(vals[cell.locs.right] == 1 & vals[cell.locs.left] == 1)
    ## Note: A has dims = (time points , total num cells in R)
    Ax <- Matrix(0, length(x), ncell, sparse = TRUE)
    Ax[cbind(idx.x.cent.diff, cell.locs.right[idx.x.cent.diff])] <- 1 / 2 / rx
    Ax[cbind(idx.x.cent.diff, cell.locs.left[idx.x.cent.diff])] <- -1 / 2 / rx
    ##
    idx.y.cent.diff <- which(vals[cell.locs.up] == 1 & vals[cell.locs.down] == 1)
    Ay <- Matrix(0, length(y), ncell, sparse = TRUE)
    Ay[cbind(idx.y.cent.diff, cell.locs.up[idx.y.cent.diff])] <- 1 / 2 / rx
    Ay[cbind(idx.y.cent.diff, cell.locs.down[idx.y.cent.diff])] <- -1 / 2 / rx
    ## remove columns of A that are out of the nest polygon
    idx.x <- idx.x.cent.diff
    idx.y <- idx.y.cent.diff
    Ax <- Ax[idx.x, idx.in.rast]
    Ay <- Ay[idx.y, idx.in.rast]
    ## compile data frame
    if (length(idx.x > 0)) {
      ts.df <- rbind(ts.df, data.frame(id = names(ani.list)[i], t = t[c(idx.x, idx.y)], h = h[c(idx.x, idx.y)], x = x[c(idx.x, idx.y)], y = y[c(idx.x, idx.y)], w = c(wx[idx.x], wy[idx.y]), v = c(vx[idx.x], vy[idx.y]), y.idx = c(rep(0, length(idx.x)), rep(1, length(idx.y))))) ## ,cam=cam[c(idx.x,idx.y)])
      ##
      if (i == 1) {
        A <- rbind(Ax, Ay)
      } else {
        A <- rbind(A, Ax, Ay)
      }
    }
  }
  ## output
  list(ants.df = ants.df, ts.df = ts.df, A = A, Q = Q, R = R, idx.in.rast = idx.in.rast)
}


motpot.estim <- function(loglam, Q = Q, R = R, ts.df = ts.df, A = A, 
                         holdout.idx = integer(), idx.in.rast, cl = NA) {
  lambda <- exp(loglam)
  QQ <- Q
  QQplus <- cbind(0, QQ)
  QQplus <- rbind(0, QQplus)
  out <- list()
  mspe <- rep(NA, length(lambda))
  ho <- 0
  if (length(holdout.idx) > 0) {
    ho <- 1
    train.idx <- (1:nrow(ts.df))[-holdout.idx]
  } else {
    train.idx <- 1:nrow(ts.df)
  }
  cat("lambda ")
  for (i in length(lambda):1) {
    cat(i, " ")
    ## get beta.hat and alpha.hat
    X <- cbind(ts.df$v[train.idx], Diagonal(length(ts.df$h[train.idx]), ts.df$h[train.idx]) %*% A[train.idx, ]) #
    ab.hat <- solve(t(X) %*% X + lambda[i] * QQplus, t(X) %*% (ts.df$w[train.idx]))
    ## get resids
    eps.hat <- ts.df$w[train.idx] - X %*% ab.hat
    if (is.na(cl)[1]) {
      ef <- bam(log((as.numeric(eps.hat))^2) - log(ts.df$h[train.idx]) ~ s(x, y), data = ts.df[train.idx, ]) #-3*log(ts.df$h[train.idx])
    } else {
      ef <- bam(log((as.numeric(eps.hat))^2) - log(ts.df$h[train.idx]) ~ s(x, y), data = ts.df[train.idx, ], cluster = cl) #-3*log(ts.df$h[train.idx])
    }
    msq <- exp(fitted(ef)) # motility surface squared, evaluated at data points
    wt <- sqrt(msq) * sqrt(ts.df$h[train.idx]) # weights
    ## re-fit with weighted least squares
    X <- cbind(ts.df$v[train.idx] / wt, Diagonal(length(ts.df$h[train.idx]), sqrt(ts.df$h[train.idx])) %*% A[train.idx, ])
    ab.hat <- solve(t(X) %*% X + lambda[i] * QQplus, t(X) %*% (ts.df$w[train.idx] / wt))
    eps.hat <- ts.df$w[train.idx] - wt * X %*% ab.hat
    H.hat <- R
    values(H.hat)[idx.in.rast] <- -ab.hat[-1] / ab.hat[1]
    mrastvals <- (1 / 2) * predict(ef, newdata = data.frame(x = xyFromCell(R, idx.in.rast)[, 1], y = xyFromCell(R, idx.in.rast)[, 2]))
    M <- R
    values(M)[idx.in.rast] <- mrastvals
    ## predict to get mspe to decide on which tuning param lambda to use
    if (ho == 1) {
      newm <- exp((1 / 2) * predict(ef, newdata = data.frame(x = ts.df$x[-train.idx], y = ts.df$y[-train.idx])))
      Xpred <- cbind(ts.df$v[-train.idx], Diagonal(length(newm), newm * ts.df$h[-train.idx]) %*% A[-train.idx, ])
      wpred <- Xpred %*% ab.hat
      mspe[i] <- mean((wpred - ts.df$w[-train.idx])^2)
    } else {
      mspe[i] <- mean(eps.hat^2)
    }
    out[[i]] <- list(ab.hat = ab.hat, H.hat = H.hat, M.hat = M)
  }
  list(out = out, mspe = mspe)
}

motpot.bestlambda <- function(eee, loglamseq, c1l.mp) {
  ## pick the best value of the tuning parameter
  idx.min <- which.min(eee$mspe)
  ## add best tuning parameter value to c1l.mp
  c1l.mp$lam.best <- exp(loglamseq[idx.min])
  ## add the estimated potential surface to c1l.mp
  c1l.mp$H.hat <- eee$out[[idx.min]]$H.hat
  crs(c1l.mp$H.hat) <- crs("+proj=utm")
  ## add the estimated motility surface to c1l.mp
  c1l.mp$M <- eee$out[[idx.min]]$M
  c1l.mp$ab.hat <- eee$out[[idx.min]]$ab.hat
  return(c1l.mp)
}



pred.neg.grad.potential <- function (pot_surface, predict_mat, direction) {
  if (direction == "x") {
    dif <- 0.5 * 
               (values(pot_surface)[cellFromXY(pot_surface, 
                                               cbind(predict_mat$`x-2` + 0.5, 
                                                     predict_mat$`y-2`))] - 
                  values(pot_surface)[cellFromXY(pot_surface, 
                                                 cbind(predict_mat$`x-2` - 0.5, 
                                                       predict_mat$`y-2`))])
    if (length(is.na(dif)) > 0) {
      dif[is.na(dif)] <- 0.5 * 
        (values(pot_surface)[cellFromXY(pot_surface, 
                                        cbind(predict_mat$`x-2` + 1, 
                                              predict_mat$`y-2`))] - 
           values(pot_surface)[cellFromXY(pot_surface, 
                                          cbind(predict_mat$`x-2`, 
                                                predict_mat$`y-2`))])[is.na(dif)]
    } 
    if (length(is.na(dif)) > 0) {
      dif[is.na(dif)] <- 0.5 * 
        (values(pot_surface)[cellFromXY(pot_surface, 
                                        cbind(predict_mat$`x-2`, 
                                              predict_mat$`y-2`))] - 
           values(pot_surface)[cellFromXY(pot_surface, 
                                          cbind(predict_mat$`x-2` - 1, 
                                                predict_mat$`y-2`))])[is.na(dif)]
    }
    return(dif)
  } 
  else if (direction == "y") {
    dif <- 0.5 * 
      (values(pot_surface)[cellFromXY(pot_surface, 
                                      cbind(predict_mat$`x-2`, 
                                            predict_mat$`y-2` + 0.5))] - 
         values(pot_surface)[cellFromXY(pot_surface, 
                                        cbind(predict_mat$`x-2`, 
                                              predict_mat$`y-2` - 0.5))])
    if (length(is.na(dif)) > 0) {
      dif[is.na(dif)] <- 0.5 * 
        (values(pot_surface)[cellFromXY(pot_surface, 
                                        cbind(predict_mat$`x-2`, 
                                              predict_mat$`y-2` + 1))] - 
           values(pot_surface)[cellFromXY(pot_surface, 
                                          cbind(predict_mat$`x-2`, 
                                                predict_mat$`y-2`))])[is.na(dif)]
    } 
    if (length(is.na(dif)) > 0) {
      dif[is.na(dif)] <- 0.5 * 
        (values(pot_surface)[cellFromXY(pot_surface, 
                                        cbind(predict_mat$`x-2`, 
                                              predict_mat$`y-2`))] - 
           values(pot_surface)[cellFromXY(pot_surface, 
                                          cbind(predict_mat$`x-2`, 
                                                predict_mat$`y-2` - 1))])[is.na(dif)]
    }
    return(dif)
  }
}



# Get new row of derived variables ----------------------------------------

get_new_row <- function(all_ids, ids_c, t_c, xlag, ylag, vxlag, vylag, data_c,
                        rest_t, queen_t) {
  out <- tibble(
    id = all_ids[[ids_c]],
    t = t_c,
    `x-1` = xlag[[1]],
    `x-2` = xlag[[2]],
    `x-3` = xlag[[3]],
    `x-4` = xlag[[4]],
    `x-5` = xlag[[5]],
    `y-1` = ylag[[1]],
    `y-2` = ylag[[2]],
    `y-3` = ylag[[3]],
    `y-4` = ylag[[4]],
    `y-5` = ylag[[5]],
    `vx-1` = vxlag[[1]],
    `vx-2` = vxlag[[2]],
    `vx-3` = vxlag[[3]],
    `vx-4` = vxlag[[4]],
    `vx-5` = vxlag[[5]],
    `vy-1` = vylag[[1]],
    `vy-2` = vylag[[2]],
    `vy-3` = vylag[[3]],
    `vy-4` = vylag[[4]],
    `vy-5` = vylag[[5]],
    chamber = getChamNo(xlag[[1]]) %>% as.factor(),
    distind = euc_dist(xlag[[1]], xlag[[2]], ylag[[1]], ylag[[2]]),
    stattime = getstattime(euc_dist(xlag[[1]], xlag[[2]], 
                                    ylag[[1]], ylag[[2]]),
                           data_c[which(data_c$t == (t_c - 1)), 
                                  which(names(data_c) == "stattime")]),
    nwalldist = distwalln(xlag[[1]], ylag[[1]]),
    swalldist = distwalls(xlag[[1]], ylag[[1]]),
    wwalldist = distwallw(xlag[[1]], ylag[[1]], getChamNo(xlag[[1]])),
    ewalldist = distwalle(xlag[[1]], ylag[[1]], getChamNo(xlag[[1]])),
    nndist = getnndist(xlag[[1]], ylag[[1]], rest_t),
    nnxlag1 = getnnxlag1(xlag[[1]], ylag[[1]], rest_t),
    nnylag1 = getnnylag1(xlag[[1]], ylag[[1]], rest_t),
    nnvxlag1 = getnnvxlag1(xlag[[1]], ylag[[1]], rest_t),
    nnvylag1 = getnnvylag1(xlag[[1]], ylag[[1]], rest_t),
    Q1 = getQ1(xlag[[1]], ylag[[1]], rest_t),
    Q2 = getQ2(xlag[[1]], ylag[[1]], rest_t),
    Q3 = getQ3(xlag[[1]], ylag[[1]], rest_t),
    Q4 = getQ4(xlag[[1]], ylag[[1]], rest_t),
    nnmove = getnnmove(xlag[[1]], ylag[[1]], rest_t),
    nnstill = getnnstill(xlag[[1]], ylag[[1]], rest_t),
    distqueen = getdistqueen(xlag[[1]], ylag[[1]], 
                             id = all_ids[ids_c], 
                             queen_t),
    observed = 0
    
  )
  return(out)
}


# Derive variables --------------------------------------------------------
# Most functions by Dhanushi

getChamNo<-function(x){
  
  if(x<18.5){1}else if(x<46){2}else if(x<70.5){3}else if(x<98){4}else 
    if(x<122.5){5}else if(x<150){6}else if(x<174.5){7}else{8}
  
}

euc_dist<-function(x1, x2, y1, y2){
  
  sqrt((x1-x2)^2+(y1-y2)^2)
}

getstattime<-function(distindt,stattime_t ){
  if(distindt==0){
    stattime=stattime_t+1
  }else{
    stattime=0
  }
  stattime
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



getnndist<-function(x, y, rest){
  
  
  min(veuc_dist(x1 = x, y1 = y, x2 = rest$x, y2 = rest$y))
  
}

getnnxlag1<-function(x, y, rest){
  
  id<-which.min(veuc_dist(x1 = x , y1 = y, x2 = rest$x, y2 = rest$y))
  rest$x[id]
  
  
}

getnnylag1<-function(x, y, rest){
  
  id<-which.min(veuc_dist(x1 = x, y1 = y, x2 = rest$x, y2 = rest$y))
  rest$y[id]
  
  
}

getnnvxlag1<-function(x, y, rest){
  
  id<-which.min(veuc_dist(x1 = x, y1 = y, x2 = rest$x, y2 = rest$y))
  rest$`vx-1`[id]
  
  
}

getnnvylag1<-function(x, y, rest){
  
  id<-which.min(veuc_dist(x1 = x, y1 = y, x2 = rest$x, y2 = rest$y))
  rest$`vy-1`[id]
  
  
}

getQ1<-function(x_c, y_c, rest){
  
  
  Q1sub<-subset(rest, rest$x>(x_c-8) & rest$x<(x_c) & 
                  rest$y>(y_c) & rest$y<(y_c+8))
  
  nrow(Q1sub)
}

  
getQ2<-function(x_c, y_c, rest){
  
  
  Q2sub<-subset(rest, rest$x>(x_c) & rest$x<(x_c+8) & 
                  rest$y>(y_c) & rest$y<(y_c+8))
  
  nrow(Q2sub)
  
}


getQ3<-function(x_c, y_c, rest){
  
  
  Q3sub<-subset(rest, rest$x>(x_c-8) & rest$x<(x_c) & 
                  rest$y>(y_c-8) & rest$y<(y_c))
  
  nrow(Q3sub)
  
}

getQ4<-function(x_c, y_c, rest){
  
  
  Q4sub<-subset(rest, rest$x>(x_c) & rest$x<(x_c+8) & 
                  rest$y>(y_c-8) & rest$y<(y_c))
  
  nrow(Q4sub)
  
}

getnnmove<-function(x, y, rest){
  
  
  idx<-which((euc_dist(x1 = x, y1 = y, x2 = rest$x, 
                        y2 = rest$y))<12)
  rest1<-rest[idx,]
  sum(rest1$movt2=="yes")
  
}

getnnstill<-function(x, y, rest){
  
  
  idx<-which((euc_dist(x1 = x, y1 = y, x2 = rest$x, 
                        y2 = rest$y))<10)
  rest1<-rest[idx,]
  sum(rest1$movt2=="no") 
  
}

getdistqueen<-function(x, y, id, queen){
  
  if(id != "Que"){
    distqueen=euc_dist(x1 = x, y1 = y, x2 = queen$x, y2 = queen$y)
    
  }else{
    distqueen=0
  }
  
  distqueen
}





# plots -------------------------------------------------------------------

quiver <- function(rast, spacing = 1, scaling = 1, ...) {
  ##
  ## Function to plot "force" arrows pointing in the
  ## direction of steepest descent of a potential surface
  ##
  ## Author: Ephraim M. Hanks (hanks@psu.edu)
  ## Last Update: 20170914
  ##
  rast <- aggregate(rast, spacing)
  R <- rast.grad(rast)
  R2 <- rast
  xy <- xyFromCell(R2, cell = 1:ncell(R2))
  cells <- cellFromXY(rast, xy)
  idx.in <- which(values(rast)[cells] > -Inf)
  cells <- cells[idx.in]
  xy.start <- xy[idx.in, ]
  xy.end <- xy.start
  xy.end[, 1] <- xy.end[, 1] + R$rast.grad.x[cells] * scaling
  xy.end[, 2] <- xy.end[, 2] + R$rast.grad.y[cells] * scaling
  length.arrow <- apply(abs(xy.end - xy.start), 1, sum)
  idx.nonzero <- which(length.arrow > 0)
  arrows(xy.start[idx.nonzero, 1], xy.start[idx.nonzero, 2], xy.end[idx.nonzero, 1], xy.end[idx.nonzero, 2], ...)
}

quiverror <- function(rast, spacing = 1, scaling = 1, ...) {
  ## function to get ends of arrows.. to be used to compare gradients of potential surfaces
  R <- rast.grad(rast)
  R2 <- rast
  xy <- xyFromCell(R2, cell = 1:ncell(R2))
  cells <- cellFromXY(rast, xy)
  idx.in <- which(values(rast)[cells] > -Inf)
  cells <- cells[idx.in]
  xy.start <- xy[idx.in, ]
  xy.end <- xy.start
  xy.end[, 1] <- xy.end[, 1] + R$rast.grad.x[cells] * scaling
  xy.end[, 2] <- xy.end[, 2] + R$rast.grad.y[cells] * scaling
  out <- list(xy.start, xy.end)
  names(out) <- c("start", "end")
  out
}

quiverrorplot <- function(truerast, rast, spacing = 1, scaling = 1, ...) {
  rast <- aggregate(rast, spacing)
  truerast <- aggregate(truerast, spacing)
  R <- rast.grad(rast)
  trueR <- rast.grad(truerast)
  R2 <- rast
  xy <- xyFromCell(R2, cell = 1:ncell(R2))
  cells <- cellFromXY(rast, xy)
  idx.in <- which(values(rast)[cells] > -Inf)
  cells <- cells[idx.in]
  xy.start <- xy[idx.in, ]
  xy.end <- xy.start
  xy.end[, 1] <- xy.end[, 1] + trueR$rast.grad.x[cells] * scaling - R$rast.grad.x[cells] * scaling
  xy.end[, 2] <- xy.end[, 2] + trueR$rast.grad.y[cells] * scaling - R$rast.grad.y[cells] * scaling
  length.arrow <- apply(abs(xy.end - xy.start), 1, sum)
  idx.nonzero <- which(length.arrow > 0)
  arrows(xy.start[idx.nonzero, 1], xy.start[idx.nonzero, 2], xy.end[idx.nonzero, 1], xy.end[idx.nonzero, 2], ...)
}
