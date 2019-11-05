##
## code.r
##
## Code to replicate the estimation of motility and potential
##  surfaces from ant movement data as in Modlmeier et al. (2017+)
##  and also to predict values to compare with Dhanushi's results.
##
## Author: Ephraim M. Hanks (hanks@psu.edu) and Elizabeth Eisenhauer
## Last Update:
##

# Setup ------------------------------------------------------------------------

# version
v <- "020"

set.seed(710)

# The data

# - boundary lines for nest polygon
library(here)
load(here("pred_1_step_ahead", paste0(v, "_data"), "nest.poly.ld.Rdata"))

# - use Dhanushi's ant data, generated with getvars1.R and getvars2.R
#   using functions in fun1.R
load(here("pred_1_step_ahead", paste0(v, "_data"), "indants2.Rdata"))

# libraries
library(tidyverse)
library(raster)
library(splancs)
library(Matrix)
library(parallel)
library(mgcv)
library(fields)
library(ctmcmove)
library(inline)
library(Rcpp)
library(RcppArmadillo)
library(dplyr)
library(ggthemes)
library(foreign)
library(nnet)
library(reshape2)

# files to source
source(here("pred_1_step_ahead", paste0(v, "_fun.r")))

# Train / testing sets ---------------------------------------------------------

train <- indants2 %>%
  mutate(
    movt2 = relevel(as.factor(indants2$movt), ref = "yes"),
    chamber = as.factor(chamber)
  ) %>%
  filter(t <= 11515)
test <- indants2 %>%
  mutate(
    movt2 = relevel(as.factor(indants2$movt), ref = "yes"),
    chamber = as.factor(chamber)
  ) %>%
  filter(t > 11515)
both <- indants2 %>%
  mutate(
    movt2 = relevel(as.factor(indants2$movt), ref = "yes"),
    chamber = as.factor(chamber)
  )

# what position is assigned for ants outside of the nest?
test %>% filter(movt2 == "out") %>%
  dplyr::select(x, y) %>%
  summarise(min_x = min(x),
            max_x = max(x),
            min_y = min(y),
            max_y = max (y)) # so always at (199, 0)

# Multinomial logistic regression for movt -------------------------------------

logit_train <- train %>%
  dplyr::select(
    -c("id", "x", "y", "t", "wx", "wy", "distind", "stattime", "movt"),
    -starts_with("v")
  )

logit_model <- multinom(movt2 ~ ., data = logit_train)

logit_step_model <- step(logit_model, direction = "backward") # takes a while to run

# Save the logistic regression model
save(logit_model, logit_step_model, file = here("pred_1_step_ahead", paste0(v, "_data"), 
                                                "logit_step_model.rda"))

# Use model to predict whether moving or not moving
# One step ahead predictions

vars<- gsub("`|8", "", logit_step_model$coefnames[-c(1, 11:16)]) ## must change each time you run


# * Predict for test and training data ------------------------------------

data_to_predict <- both

## predict for both the test and training data
logit_test <- data_to_predict %>%
  dplyr::select(vars)

predict_move <- predict(
  object = logit_step_model,
  newdata = logit_test,
  type = "probs"
  ) %>%
  as.data.frame() %>%
  mutate(movt_pred = case_when(
    yes > no & yes > out ~ "yes",
    no > yes & no > out ~ "no",
    out > yes & out > no ~ "out"
  )) %>%
  cbind(data_to_predict[, 1:4], data_to_predict$`x-1`, data_to_predict$`y-1`, 
        data_to_predict$`x-2`, data_to_predict$`y-2`) %>%
  rename(
    "x-1" = "data_to_predict$`x-1`",
    "y-1" = "data_to_predict$`y-1`",
    "x-2" = "data_to_predict$`x-2`",
    "y-2" = "data_to_predict$`y-2`"
  )

# if not moving, predicted point is previous point
predict_move_no <- predict_move %>%
  filter(movt_pred == "no") %>%
  mutate(
    x_pred = `x-1`,
    y_pred = `y-1`
  )

# if outside nest, set to point (0, 199)
predict_move_out <- predict_move %>%
  filter(movt_pred == "out") %>%
  mutate(
    x_pred = 199,
    y_pred = 0
  )

# check classification accuracy for test data
check_test <- inner_join(dplyr::select(test, t, id, movt2), 
                    dplyr::select(predict_move, t, id, movt_pred),
                    by = c("id", "t")) %>%
  group_by(movt2, movt_pred) %>%
  summarize(count =n()) %>%
  ungroup() %>%
  mutate(prop = count / sum(count),
         correct = ifelse(movt2 == movt_pred, TRUE, FALSE))

# accuracy table for test data
check_test %>%
  group_by(correct) %>%
  summarise(accuracy_rate = sum(prop))

# check classification accuracy for training data
check_train <- inner_join(dplyr::select(train, t, id, movt2), 
                    dplyr::select(predict_move, t, id, movt_pred),
                    by = c("id", "t")) %>%
  group_by(movt2, movt_pred) %>%
  summarize(count =n()) %>%
  ungroup() %>%
  mutate(prop = count / sum(count),
         correct = ifelse(movt2 == movt_pred, TRUE, FALSE))

# accuracy table for test data
check_train %>%
  group_by(correct) %>%
  summarise(accuracy_rate = sum(prop))


# Now we will fit the training data with the SDE model to predict the positions
# of the ants at points where logistic regression predicted they were moving.

# Transform training data ------------------------------------------------------
# - Transform data to a list with one entry per ant
# - Each list item is a data.frame with "x", "y", and "t" columns of locations
#   for each ant

train_in_nest <- train %>%
  dplyr::filter(movt2 != "out")

train_list <- train_in_nest %>%
  dplyr::select(x, y, t) %>%
  split(as.factor(train_in_nest$id))

# Get data objects - training data ---------------------------------------------
# - Making data objects and rasters for estimation of Motility and Potential
#   Surfaces

train_mp <- get.motpot.data(
  ani.list = train_list,
  boundary.poly = nest.poly.ld,
  res = 1,
  xcolname = "x",
  ycolname = "y",
  tcolname = "t"
)

## elements of train_mp are:
##
## ants.df = data frame with all ants together.  One data point for each ant and each time point.
## ts.df = same data, but now I have removed all time points where the ants don't move
##         and also, now each row corresponds to either an "x" or a "y" coordinate
##         so each ant has 2 rows for each time point in which it is moving
## A = Matrix linking rows of ts.df to the motility and potential surfaces
## Q = Matrix to make motility and potential surfaces smooth (penalty matrix)
## R = Base raster on which the motility and potential surfaces will be estimated
## idx.in.rast = index of which time points the ants are in the boundaries of boundary.poly

# * Estimate Motility and Potential Surfaces - training data -------------------

# Make cluster
cl <- makeCluster(10)

# Separate out a holdout set for cross validation to choose lambda.
# Here I'm holding out 80% of 200000 ant-time points
holdout_idx <- sample(1:nrow(train_mp$ts.df), 
                      size = round(.2 * nrow(train_mp$ts.df)))

# Specify a series of "penalty" tuning parameters.  Larger values
# make motility and potential surfaces more smooth
#
# We'll use the holdout set to choose this tuning parameter
loglamseq <- seq(5, 8, by = 1)

#
# Estimate motility / potential surfaces for different values of the tuning parameter
#
# This takes about 5 minutes on my computer, and may take much longer on slower computers
#
eee <- motpot.estim(
  loglam = loglamseq, ## sequence of tuning parameters
  Q = train_mp$Q, ## from motpot object
  R = train_mp$R, ## from motpot object
  ts.df = train_mp$ts.df, ## from motpot object
  A = train_mp$A, ## from motpot object
  holdout.idx = holdout_idx, ## index of hold-out set
  idx.in.rast = train_mp$idx.in.rast ## from motpot object
)

#
# Plot the mean-squared prediction error on the hold-out set for different
#  values of the tuning parameter (lower is better)
#
plot(loglamseq, eee$mspe)

# pick the best value of the tuning parameter
idx_min <- which.min(eee$mspe)
# add best tuning parameter value to train_mp
train_mp$lam.best <- exp(loglamseq[idx_min])
# add the estimated potential surface to train_mp
train_mp$H.hat <- eee$out[[idx_min]]$H.hat
# set coordinate reference system
crs(train_mp$H.hat) <- crs("+proj=utm")
# add the estimated motility surface to train_mp
train_mp$M <- eee$out[[idx_min]]$M
# add model parameters to train_mp
train_mp$ab.hat <- eee$out[[idx_min]]$ab.hat

# save params
save(train_mp, file = here("pred_1_step_ahead", paste0(v, "_data"), "train_mp.Rdata"))

# * Plots of motility and potential surfaces - training data -------------------

# color assignment for gradient in plots
yp.colors <- colorRampPalette(c("yellow", "purple"))

# potential surface plot with arrows
image.plot(train_mp$H.hat,
  col = (yp.colors(99)),
  breaks = seq(min(minValue(train_mp$H.hat)),
    max(maxValue(train_mp$H.hat)),
    length.out = 100
  ),
  bty = "n", xlab = "", ylab = "", axes = FALSE, asp = 1,
  main = "C1 Low Density Potential - Training Data",
  sub = paste("log(lambda)=", train_mp$lam.best, sep = " ")
)
quiver(train_mp$H.hat, spacing = 2, scaling = -3, length = .04, lwd = .6)

# Motility Surface Plot
image.plot(train_mp$M,
  col = (yp.colors(99)),
  breaks = seq(min(minValue(train_mp$M)),
    max(maxValue(train_mp$M)),
    length.out = 100
  ),
  bty = "n", xlab = "", ylab = "", axes = FALSE, asp = 1,
  main = "C1 Low Density Motility - Full Data",
  sub = paste("log(lambda)=", loglamseq[idx_min], sep = " ")
)


# Predict test points where ant was predicted to be moving ---------------------

predict_move_yes <- predict_move %>%
  filter(movt_pred == "yes")

# get motility for prediction equation
predict_move_yes$mot <- values(train_mp$M)[cellFromXY(
  train_mp$M,
  cbind(
    predict_move_yes$`x-2`,
    predict_move_yes$`y-2`
  )
)]


# change in potential in x direction
predict_move_yes$delta_x_pot <- pred.neg.grad.potential(
  pot_surface = train_mp$H.hat,
  predict_mat = predict_move_yes,
  direction = "x"
)

# change in potential in y direction
predict_move_yes$delta_y_pot <- pred.neg.grad.potential(
  pot_surface = train_mp$H.hat,
  predict_mat = predict_move_yes,
  direction = "y"
)

# predict positions
predict_move_yes <- predict_move_yes %>%
  mutate(
    x_pred = `x-1` + (`x-1` - `x-2`) + train_mp$ab.hat[1] *
      (mot * (-delta_x_pot) - (`x-1` - `x-2`)),
    y_pred = `y-1` + (`y-1` - `y-2`) + train_mp$ab.hat[1] *
      (mot * (-delta_y_pot) - (`y-1` - `y-2`))
  )


# Project onto boundary if not already inside ----------------------------------

# boundary polygon
P <- nest.poly.ld

# source projection code
sourceCpp(here("pred_1_step_ahead", paste0(v, "_ephraim_projection_code"), "project.cpp"))

# get points to project
xy.noproj <- predict_move_yes %>%
  dplyr::select("x_pred", "y_pred")

# project onto P
xy.proj <- project(as.matrix(xy.noproj), as.matrix(P))

# plot
plot(P, type = "b")
points(xy.noproj, col = "red")
points(xy.proj, col = "blue", pch = 3)
arrows(xy.noproj[, 1], xy.noproj[, 2], xy.proj[, 1], xy.proj[, 2], length = .1)


# Summarize predictions --------------------------------------------------------

pred_yes_tmp <- predict_move_yes %>%
  mutate(
    x_pred = xy.proj[, 1],
    y_pred = xy.proj[, 2]
  ) %>%
  dplyr::select(-c("mot", starts_with("delta")))

predictions_movt_only <- inner_join(dplyr::select(data_to_predict, t, id, movt2), 
                   dplyr::select(predict_move, t, id, movt_pred),
                   by = c("id", "t")) %>%
  rename("movt" = "movt2")

predictions <- rbind(pred_yes_tmp, predict_move_no, predict_move_out) %>%
  dplyr::select(id, t, x, y, x_pred, y_pred) %>%
  inner_join(predictions_movt_only, by = c("id", "t"))

# save predictions
save(predictions, file = here("pred_1_step_ahead", paste0(v, "_data"), "predict_1_step_ahead_both.Rdata"))

# check if NAs
table(is.na(predictions))

# Prediction plots -------------------------------------------------------------

predictions %>%
  filter(id %in% head(unique(id), 5)) %>%
  gather(key = "key", value = "value", x, x_pred) %>%
  ggplot(aes(x = t, y = value, col = key)) +
  geom_line() +
  facet_wrap(~id, nrow = 5)

# not easy to see because completely covered by x_pred

predictions %>%
  mutate(
    x_dif = x_pred - x,
    y_dif = y_pred - y
  ) %>%
  gather(key = "xory", value = "dif", x_dif, y_dif) %>%
  ggplot(aes(x = xory, y = dif)) +
  geom_boxplot() +
  scale_x_discrete(labels = expression(hat("x") - "x", hat("y") - "y")) +
  coord_flip() +
  theme_fivethirtyeight() +
  theme(legend.position = "none") +
  labs(title = "Prediction errors in the x and y directions")


predictions %>%
  mutate(
    x_dif = x_pred - x,
    y_dif = y_pred - y
  ) %>%
  ggplot(aes(x = x_dif)) +
  geom_histogram() +
  xlim(c(-2.5, 2.5))

## look at the extreme point
hh <- predictions %>%
  mutate(
    x_dif = x_pred - x,
    y_dif = y_pred - y
  )
hh[hh$x_dif == max(hh$x_dif),]
