This project is to fit the SDE model with the 2 step approach (described in LARI paper) but using multinomial regression to model the states as well (moving in nest, not moving in nest, outside of nest). Then we did some simulations from the fitted model.

***

# pred_1_step_ahead

The pred_1_step_ahead file contains code to predict the position at the next time point using the previous (true) time point for each ant. We start doing this at time t=11516.

***

# pred_1000_step_ahead

This file contains code to predict 1000 steps starting at 11516. Each step uses the prediction at the previous time point (taken as truth) to get its prediction at the current time point.

## version 010

Predict 1000 steps ahead one time.

## version 020

Predict 1000 steps ahead 100 times separately. ex5version works the best. Load 020_packages_parallelize.R, 020_ex5version.PBS, 020_ex5version.bashrc, 020_ex5version.R, 020_data (file), and 020_fun.r into the cluster. Run the command line prompts in 020_parallelize_terminal_code.txt to run the simulation 100 different times and save the data into 020_data file.

020_make_boundary_lines.R was used to make P_list_lines.rda and doesn't need to be run again except if you want to change P_list_lines.rda.

ex5version files send the PBS file 100 times separately, so while it is not exactly parallel, they all finish in about 4 hours and they start within a couple hours of each other. Also, it can be run on the open queue.

ex4version files try to run it in parallel but I couldn't get it to work on Ephraim's allocation and you can't run it on the open queue.

I also kept the files needed just for running 1000 simulations just one time. They are 020_code.r and 020_packages.R.

020_dhanushi_code is not used anywhere because I took her functions and rewrote them in 020_fun.r.

020_ephraim_projection_code uses c++ to project onto a boundary but since this movement looked unnatural I don't use this code in version 020. Instead, I just predict a bunch of times until the prediction is in the boundary.

***














