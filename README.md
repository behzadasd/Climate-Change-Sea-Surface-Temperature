# Climate Change impact on global Sea Surface Temperatures

Main code: CMIP5_SST_xarray.py
* The code reads 4-Dimensional sea surface temperature (SST) data from 14 CMIP5 climate models for two periods of historical 1980-1999 and future 2080-2099 (under RCP8.5 scenario). It regrids them into 1 degree by 1 degree fields, and plots all models' average SST averages for each period. It also plots the difference between 2080-2099 and 1980-1999 which shows the impact of climate change on global SSTs under RCP8.5 scenario

* The climate model data are stored at UPenn's local server

Functions code: Behzadlib.py
* This code contains various analysis/plotting functions that are imported in the main code as needed

Main code #2: CMIP5_SST.py
* Does same job as the CMIP5_SST_xarray.py code, but reads .nc files using a different method by listing all file names and reading the files one by one (instead of loading them all using MFDATSET) - This is useful when the data are huge and the computer memory is low

Example plotting product: thetao_AllGCMs_surface_hist-1980-1999.png
