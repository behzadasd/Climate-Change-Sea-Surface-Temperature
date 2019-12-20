# Climate Change impact on global Sea Surface Temperatures

Main code: CMIP5_SST_xarray.py
* The code reads 4-Dimensional sea surface temperature (SST) data from 14 CMIP5 climate models for two periods of historical 1980-1999 and future 2080-2099 (under RCP8.5 scenario). It regrids them into 1 degree by 1 degree fields, and plots all models' average SST averages for each period. It also plots the difference between 2080-2099 and 1980-1999 which shows the impact of climate change on global SSTs under RCP8.5 scenario. It also plots the global average SST of each model versus the changes in SSTs between two periods as a scatter plot. The averages are grid-cell-area-weighted meaning the lager tropical gridcells have higher impact on global averages than the smaller polar cells

* The climate model data are stored at UPenn's local server

Functions code: Behzadlib.py
* This code contains various analysis/plotting functions that are imported in the main code as needed

Main code #2: CMIP5_SST.py
* Does same job as the CMIP5_SST_xarray.py code, but reads .nc files using a different method by listing all file names and reading the files one by one (instead of loading them all using MFDATSET) - This is useful when the data are huge and the computer memory is low

Final plotting products:

Fig_CMIP5_SST_hist_1980_1999.png   = Global SST maps of 14 CMIP5 models for the historical 1980-1999 period
Fig_CMIP5_SST_rcp8p5_2080_2099.png = Global SST maps of 14 CMIP5 models for the 2080-2099 under RCP8.5 scenario
Fig_CMIP5_SST_climate_change_Impact_RCP8p5.png = Climate Change impact - 2080-2099 average minus 1980-1999 average
Fig_CMIP5_SST_climate_change_Impact_RCP8p5_scatter.png = Scatter plot of global average SST versus the changes in SSTs between two periods, for each model
