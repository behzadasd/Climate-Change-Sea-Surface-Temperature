# Climate-Change-Sea-Surface-Temperature

Reading 4-Dimensional sea surface temperature data from multiple CMIP5 climate models, regridding and plotting them

Main code: CMIP5_SST_xarray.py
* The code reads multiple climate data, regrids them into 1 degree by 1 degree fields, and plots them all together
* The climate model data are stored at UPenn's local server

Main code #2: CMIP5_SST.py
* Does same job as the CMIP5_SST_xarray.py code, but reads .nc files using a different method by listing all file names and reading the files one by one (instead of loading them all using MFDATSET) - This is useful when the data are huge and the computer memory is low

Functions code: Behzadlib.py
* This code contains various analysis/plotting functions that are imported in the main code as needed

Example plotting product: thetao_AllGCMs_surface_hist-1980-1999.png
