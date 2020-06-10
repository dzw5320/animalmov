* This folder contains code used to fit the best Random Forest and LSTM models to obtain one step ahead predictions and MSPE as well as longer run simulations. 

* Code used for tuning the models are not included as they are very similar to the code used to tune the ant movement models except that 5 fold cross validation was used for the gulls. 

* Raw data for this analysis can be found at https://doi.org/10.5061/dryad.4271s. Data post-processed that were used to fit the models can be found [here](https://drive.google.com/drive/folders/1JfnJ4L9nQ7m8ltV-XTL_mGUI-Ot4t7EK?usp=sharing). Note that Longitude and Latitude coordinates in the original data were converted to UTM zone 30 in the analysis.

* The files gulls_RF_onestepahead.py and gulls_RF_longsims.py contain python code to run the one step ahead predictions and long run simulations for the Random Forest model. 

* The file gulls_LSTM_extractseries.py contains code that was used to extract series ending in a particular state with a 1000 step moving window. This data was used to fit the 4 velocity models for the 4 states. 

* The files gulls_LSTM_onestepahead.py and gulls_LSTM_longsims.py contain python code to run the one step ahead predictions and long run simulations for the best LSTM model. 
