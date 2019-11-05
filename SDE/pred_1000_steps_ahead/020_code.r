##
## code.r
##
## Code to replicate the estimation of motility and potential
##  surfaces from ant movement data as in Modlmeier et al. (2017+)
##  and also to predict values to compare with Dhanushi's results.
##  - This code does predictions for the last 1000 data points
##
## Author: Ephraim M. Hanks (hanks@psu.edu) and Elizabeth Eisenhauer
## Last Update:
##

# Setup ------------------------------------------------------------------------

# version
v <- "020"

set.seed(710)

# packages
library(here)
source(here("pred_1000_steps_ahead", paste0(v, "_packages.R")))

# data
# - boundary lines for nest polygon and lines to determine if ant jumped a wall
load(here("pred_1000_steps_ahead", paste0(v, "_data"), "P_list_lines.rda"))
# - use Dhanushi's ant data, generated with getvars1.R and getvars2.R
#   using functions in 010_fun.R
load(here("pred_1000_steps_ahead", paste0(v, "_data"), "indants2.Rdata"))
indants3 <- indants2 %>%
  mutate_if(is.factor, as.character) %>%
  mutate(
    x = replace(x,
                movt == "out",
                198.99),
    y = replace(y,
                movt == "out",
                0.01)
  )

# source projection code
sourceCpp(here("pred_1000_steps_ahead", paste0(v, "_ephraim_projection_code"), 
               "projectFCNs.cpp"))
sourceCpp(here("pred_1000_steps_ahead", paste0(v, "_ephraim_projection_code"), 
               "project.cpp"))

# functions
source(here("pred_1000_steps_ahead", paste0(v, "_fun.r")))

# model
redo_step_model <- FALSE
redo_eee <- FALSE



# Train / testing sets ---------------------------------------------------------
  
train <- indants3 %>%
  mutate(
    movt2 = relevel(as.factor(indants3$movt), ref = "yes"),
    chamber = as.factor(chamber),
    observed = 1
  ) %>%
  filter(t <= 11515) %>%
  dplyr::select(-"movt")

# Multinomial logistic regression for movt2 ------------------------------------

logit_train <- train %>%
  dplyr::select(
    -c("id", "x", "y", "t", "wx", "wy", "stattime", "distind"),
    -starts_with("v")
  )

if (redo_step_model == FALSE) {
  load(here("pred_1000_steps_ahead", paste0(v, "_data"), "logit_step_model.rda"))
} else {
  logit_model <- multinom(movt2 ~ ., data = logit_train)
  logit_step_model <- step(logit_model, direction = "backward") # takes a while to run
  save(logit_model, logit_step_model, file = here("pred_1000_steps_ahead", 
                                                  paste0(v, "_data"), 
                                                  "logit_step_model.rda"))
}

# get vars in step model
vars <- gsub("`|8", "", logit_step_model$coefnames[-c(1, 11:16)]) ## must change each time you run


# Fit SDE model -----------------------------------------------------------

# * Transform training data ----------------------------------------------------
# - Transform data to a list with one entry per ant
# - Each list item is a data.frame with "x", "y", and "t" columns of locations
#   for each ant

train_in_nest <- train %>%
  dplyr::filter(movt2 != "out")

train_list <- train_in_nest %>%
  dplyr::select(x, y, t) %>%
  split(as.factor(train_in_nest$id))

# * Get data objects - training data -------------------------------------------
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

# * Estimate Motility and Potential Surfaces - training data -------------------

if (redo_eee == TRUE) {
  # Make cluster
  cl <- makeCluster(10)
  
  # Separate out a holdout set for cross validation to choose lambda
  holdout_idx <- sample(1:nrow(train_mp$ts.df), 
                        size = round(.2 * nrow(train_mp$ts.df)))
  
  # Specify a series of "penalty" tuning parameters.  Larger values
  # make motility and potential surfaces more smooth
  
  # We'll use the holdout set to choose this tuning parameter
  loglamseq <- seq(2, 8, by = 1)
  
  # Estimate motility / potential surfaces for different values of the tuning parameter
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
  train_mp$lam.best <- loglamseq[idx_min]
  # add the estimated potential surface to train_mp
  train_mp$H.hat <- eee$out[[idx_min]]$H.hat
  # set coordinate reference system
  crs(train_mp$H.hat) <- crs("+proj=utm")
  # add the estimated motility surface to train_mp
  train_mp$M <- eee$out[[idx_min]]$M
  # add model parameters to train_mp
  train_mp$ab.hat <- eee$out[[idx_min]]$ab.hat
  
  # save params
  save(train_mp, file = here("pred_1000_steps_ahead", paste0(v, "_data"), 
                             "train_mp.Rdata"))
  
} else {
  load(here("pred_1000_steps_ahead", paste0(v, "_data"), 
            "train_mp.Rdata"))
}


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
           sub = paste("log(lambda)=", train_mp$lam.best, sep = " ")
)


# Loop --------------------------------------------------------------------

{ # begin chunk
all_ids <- unique(train$id)

train_and_pred <- train
train_and_pred[(nrow(train_and_pred) + 1):
                 (nrow(train_and_pred) + 1000*length(all_ids)), ] <- NA
row_index <- nrow(train) + 1
tictoc::tic()
for (t_c in c(11516:(11516+999))) { #1000 time points starting at time 11515
  cat("t: ", t_c - 11515, "\n")
  queen_t <- train_and_pred %>%
    dplyr::filter(id == "Que",
                  t == t_c - 1)
  for (ids_c in seq_along(all_ids)) {
    # get variables for time t_c
    data_c <- train_and_pred %>%
      dplyr::filter(id == all_ids[ids_c])
    rest_t <- train_and_pred %>%
      dplyr::filter(id != all_ids[ids_c],
                    t == t_c - 1)
    xlag <- data_c[data_c$t %in% (t_c - c(1:6)), ] %>%
      arrange(desc(t)) %>%
      dplyr::select(x) %>%
      t() %>%
      c()
    if (sum(is.na(xlag)) > 0) {
      browser()
    }
    ylag <- data_c[data_c$t %in% (t_c - c(1:6)), ] %>%
      arrange(desc(t)) %>%
      dplyr::select(y) %>%
      t() %>%
      c()
    vxlag <- xlag[1:5] - xlag[2:6]
    vylag <- ylag[1:5] - ylag[2:6]   
    new_row <- get_new_row(all_ids, id_c, t_c, xlag, ylag, vxlag, vylag, data_c,
                           rest_t, queen_t)
    # now predict movt2, x, y
    probs_c <- stats::predict(object = logit_step_model,
                             newdata = new_row, "probs")
    new_row$movt2 <- base::sample(names(probs_c), size = 1, prob = probs_c)
    if (new_row$movt2 == "out" & 
        new_row$`x-1` > 177 & 
        new_row$`y-1` < 40) {
      new_row$x <- 198.99
      new_row$y <- 0.01
    } # if far from exit and "out", resample til "yes" or "no"
    else if (new_row$movt2 == "out" & 
               (new_row$`x-1` <= 177 | 
               new_row$`y-1` >= 40)) { 
      while (new_row$movt2 == "out") {
        new_row$movt2 <- base::sample(names(probs_c), size = 1, prob = probs_c)
      }
    }
    if (new_row$movt2 == "yes") {
      # get motility for prediction equation
      mot <- values(train_mp$M)[cellFromXY(
        train_mp$M,
        cbind(
          new_row$`x-2`,
          new_row$`y-2`
        )
      )]
      # change in potential in x direction
      delta_x_pot <- pred.neg.grad.potential(
        pot_surface = train_mp$H.hat,
        predict_mat = new_row,
        direction = "x"
      )
      # change in potential in y direction
      delta_y_pot <- pred.neg.grad.potential(
        pot_surface = train_mp$H.hat,
        predict_mat = new_row,
        direction = "y"
      )
      done <- FALSE
      while (done == FALSE) {
        # predict positions
        new_row <- new_row %>%
          dplyr::mutate(
            x = `x-1` + (`x-1` - `x-2`) + train_mp$ab.hat[1] *
              (mot * (-delta_x_pot) - (`x-1` - `x-2`)) + 
              rnorm(n = 1, mean = 0, sd = abs(mot)),
            y = `y-1` + (`y-1` - `y-2`) + train_mp$ab.hat[1] *
              (mot * (-delta_y_pot) - (`y-1` - `y-2`)) + 
              rnorm(n = 1, mean = 0, sd = abs(mot))
          )
        if (pracma::inpolygon(x = new_row$x,
                             y = new_row$y,
                             xp = P[,1],
                             yp = P[,2],
                             boundary = TRUE) == FALSE){
          # project onto P
          xy_proj <- project(as.matrix(dplyr::select(new_row, "x", "y")), 
                             as.matrix(P))
          # add projected x and y to new_row
          new_row <- new_row %>%
            dplyr::mutate(
              x = xy_proj[1,1],
              y = xy_proj[1,2]
            )
        }
        # check that the ant didn't jump over a wall
        yn <- vector()
        for (l in seq_along(P_list_lines)) {
          line_c <- P_list_lines[[l]]
          yn[l] <- lineIntersection(Ax = new_row$x,
                                    Ay = new_row$y,
                                    Bx = new_row$`x-1`,
                                    By = new_row$`y-1`,
                                    Cx = line_c[1,1],
                                    Cy = line_c[1,2],
                                    Dx = line_c[2,1],
                                    Dy = line_c[2,2])
        }
        # if it jumped over wall, try again
        done <- ifelse(sum(yn) > 0, FALSE, TRUE)
        cat(done)
      }
    } 
    else if (new_row$movt2 == "no") {
      new_row$x <- new_row$`x-1`
      new_row$y <- new_row$`y-1`
    }
    # get vx, vy, wx, wy
    new_row <- new_row %>%
      dplyr::mutate(
        vx = x - `x-1`,
        vy = y - `y-1`,
        wx = vx - `vx-1`,
        wy = vy - `vy-1`
      )
    if (sum(is.na(new_row)) > 0) {
      browser()
    }
    # add new row to data
    train_and_pred[row_index, ] <- new_row %>%
      dplyr::select(names(train_and_pred))
    row_index <- row_index + 1
  } 
}
tictoc::toc()
# check tail
train_and_pred %>% tidyr::drop_na() %>% dplyr::filter(id == all_ids[ids_c]) %>% tail(5)

# save
save(train_and_pred, file = here("pred_1000_steps_ahead", paste0(v, "_data"), 
                           "train_and_pred.Rdata"))
} # end chunk
## time elapsed: 6.6 hrs with going through walls.. 
##               10 hrs elapsed without going through walls
##               26 hrs without going through walls & without going "out" unless 
##                  near exit (but closed computer for a long time so idk)


# Get predicted xyt form -------------------------------------------------------

# get just xyt and id for predicted
predictions <- train_and_pred %>%
  filter(observed == 0) %>%
  dplyr::select(id, x, y, t, movt2) %>%
  rename(x_pred = x,
         y_pred = y,
         movt_pred = movt2) %>%
  arrange(id) %>%
  inner_join(indants3, by = c("id", "t")) %>%
  dplyr::select(id, t, x, y, movt, x_pred, y_pred, movt_pred)

# save
save(predictions, file = here("pred_1000_steps_ahead", paste0(v, "_data"), 
                                 "predictions.Rdata"))


# Prediction plots -------------------------------------------------------------

hh <- predictions %>%
  mutate(
    x_dif = x_pred - x,
    y_dif = y_pred - y
  )

hh %>%
  tidyr::gather(key = "xory", value = "dif", x_dif, y_dif) %>%
  ggplot(aes(x = xory, y = dif)) +
  geom_boxplot() +
  scale_x_discrete(labels = expression(hat("x") - "x", hat("y") - "y")) +
  coord_flip() +
  theme_fivethirtyeight() +
  theme(legend.position = "none") +
  labs(title = "Prediction errors in the x and y directions")


hh %>%
  ggplot(aes(x = x_dif)) +
  geom_histogram()

## look at the most extreme point
hh[hh$x_dif == max(hh$x_dif),]

## look at one simulated path
nest_poly <- as_tibble(P)
dat <- subset(predictions, id == all_ids[2])
ggplot(dat, aes(x = x, y = y)) + geom_path(col = "grey") + 
  theme(plot.margin = unit(c(1,1,1,1), "lines"),
        legend.position = "none") + 
  theme_classic() + 
  coord_fixed() + 
  xlim(-1, 200) +
  ylim(-1, 70) + 
  geom_point(size = 0.4, col = "black") +
  geom_path(aes(x = x_pred, y = y_pred), col = "orange", alpha = 0.3) +
  geom_point(aes(x = x_pred, y = y_pred), size = 0.4, col = "orange") +
  geom_point(data = subset(dat, t == max(t)), aes(x = x_pred, y = y_pred), 
             size = 3, col = "red") +
  geom_point(data = subset(dat, t == min(t)), aes(x = x_pred, y = y_pred), 
             size = 3, col = "green") +
  geom_path(data = nest_poly, aes(x = V1, y = V2), col = "lightblue") +
  # geom_path(data = P_bigger, aes(x = long, y = lat), col = "lightblue") +
  labs(title = paste("Ant ID: ", dat$id[1])) +
  theme(
    axis.title = element_blank(),
    axis.ticks = element_blank(),
    axis.text = element_blank(),
    axis.line = element_blank()
  )

## look at final positions
last <- subset(predictions, t == max(t))
ggplot(last, aes(x = x, y = y)) + 
  theme(plot.margin = unit(c(1,1,1,1), "lines"),
        legend.position = "none") + 
  theme_classic() + 
  coord_fixed() + 
  xlim(0, 200) +
  ylim(0, 70) + 
  geom_point(size = 0.1, col = "black") +
  geom_path(data = nest_poly, aes(x = V1, y = V2), col = "lightblue")
  






