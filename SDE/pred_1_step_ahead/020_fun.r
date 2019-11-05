
# Subroutines -------------------------------------------------------------


get.motpot.data <- function(ani.list, boundary.poly, res = 1, xcolname = "x", ycolname = "y", tcolname = "t") {
  ## Get raster and adjacency objects -----------------------------------
  R <- raster(xmn = min(boundary.poly[, 1]), xmx = max(boundary.poly[, 1]), ymn = min(boundary.poly[, 2]), ymx = max(boundary.poly[, 2]), res = res)
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
    ants.df <- rbind(ants.df, data.frame(id = names(ani.list)[i], t = t, h = h, x = x, y = y, vx = vx, vy = vy, wx = wx, wy = wy))
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


motpot.estim <- function(loglam, Q = Q, R = R, ts.df = ts.df, A = A, holdout.idx = integer(), idx.in.rast, cl = NA) {
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


####################### plots ############################

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
