## Goal: get lines to tell if the ant jumped a wall

# version
v <- "020"

# packages
library(here)
source(here("pred_1000_steps_ahead", paste0(v, "_packages.r")))

# boundary lines for nest polygon
load(here("pred_1000_steps_ahead", paste0(v, "_data"), "nest.poly.ld.Rdata"))
P <- nest.poly.ld %>% round()

P_sp <- SpatialPolygons(list(Polygons(list(Polygon(P)), "ID")))
P_bigger <- P_sp %>%
  gBuffer(width = 1) %>%
  fortify() %>%
  dplyr::select(long, lat)
P_internal <- list(matrix(c(41,65,41,7), nrow = 2, byrow = TRUE),
                    matrix(c(51,65,51,7), nrow = 2, byrow = TRUE),
                    matrix(c(93,65,93,7), nrow = 2, byrow = TRUE),
                    matrix(c(103,65,103,7), nrow = 2, byrow = TRUE),
                    matrix(c(145,65,145,7), nrow = 2, byrow = TRUE),
                    matrix(c(155,65,155,7), nrow = 2, byrow = TRUE),
                    matrix(c(177,52,177,0), nrow = 2, byrow = TRUE),
                    matrix(c(175,52,175,0), nrow = 2, byrow = TRUE),
                    matrix(c(125,52,125,0), nrow = 2, byrow = TRUE),
                    matrix(c(123,52,123,0), nrow = 2, byrow = TRUE),
                    matrix(c(73,52,73,0), nrow = 2, byrow = TRUE),
                    matrix(c(71,52,71,0), nrow = 2, byrow = TRUE),
                    matrix(c(21,52,21,0), nrow = 2, byrow = TRUE),
                    matrix(c(19,52,19,0), nrow = 2, byrow = TRUE))

P_list_lines <- list(matrix(c(41,65,41,7), nrow = 2, byrow = TRUE),
                    matrix(c(51,65,51,7), nrow = 2, byrow = TRUE),
                    matrix(c(93,65,93,7), nrow = 2, byrow = TRUE),
                    matrix(c(103,65,103,7), nrow = 2, byrow = TRUE),
                    matrix(c(145,65,145,7), nrow = 2, byrow = TRUE),
                    matrix(c(155,65,155,7), nrow = 2, byrow = TRUE),
                    matrix(c(176,52,176,0), nrow = 2, byrow = TRUE),
                    matrix(c(124,52,124,0), nrow = 2, byrow = TRUE),
                    matrix(c(72,52,72,0), nrow = 2, byrow = TRUE),
                    matrix(c(20,52,20,0), nrow = 2, byrow = TRUE))

save(P_list_lines, P, nest.poly.ld,  file = here("pred_1000_steps_ahead", 
                         paste0(v, "_data"), 
                         "P_list_lines.rda"))



