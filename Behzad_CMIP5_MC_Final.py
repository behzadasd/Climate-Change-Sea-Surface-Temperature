#############################################
####     Behzad Asadieh, Ph.D.      ####
####  University of Pennsylvania    ####
####    basadieh@sas.upenn.edu      ####
####     github.com/behzadasd       ####
########################################
import numpy as np
import xarray as xr
import numpy.ma as ma
import os
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm, maskoceans
from scipy.interpolate import griddata
import math
import copy
########################################
from dask.distributed import Client
from dask_kubernetes import KubeCluster

cluster = KubeCluster()
cluster.adapt(minimum=1, maximum=10, interval='2s')
client = Client(cluster)
client
########################################
def func_latlon_regrid(lat_n_regrid, lon_n_regrid, lat_min_regrid, lat_max_regrid, lon_min_regrid, lon_max_regrid): 
    ### creating arrays of regridded lats and lons ###
    ### Latitude Bounds ###
    Lat_regrid_1D=np.zeros ((lat_n_regrid));
    Lat_bound_regrid = np.zeros ((lat_n_regrid,2)); Lat_bound_regrid[0,0]=-90;  Lat_bound_regrid[0,1]=Lat_bound_regrid[0,0] + (180/lat_n_regrid); Lat_regrid_1D[0]=(Lat_bound_regrid[0,0]+Lat_bound_regrid[0,1])/2
    for ii in range(1,lat_n_regrid):
        Lat_bound_regrid[ii,0]=Lat_bound_regrid[ii-1,1]
        Lat_bound_regrid[ii,1]=Lat_bound_regrid[ii,0] +  (180/lat_n_regrid)
        Lat_regrid_1D[ii]=(Lat_bound_regrid[ii,0]+Lat_bound_regrid[ii,1])/2
    ### Longitude Bounds ####
    Lon_regrid_1D=np.zeros ((lon_n_regrid));
    Lon_bound_regrid = np.zeros ((lon_n_regrid,2)); Lon_bound_regrid[0,0]=0;  Lon_bound_regrid[0,1]=Lon_bound_regrid[0,0] + (360/lon_n_regrid); Lon_regrid_1D[0]=(Lon_bound_regrid[0,0]+Lon_bound_regrid[0,1])/2
    for ii in range(1,lon_n_regrid):
        Lon_bound_regrid[ii,0]=Lon_bound_regrid[ii-1,1]
        Lon_bound_regrid[ii,1]=Lon_bound_regrid[ii,0] +  (360/lon_n_regrid)
        Lon_regrid_1D[ii]=(Lon_bound_regrid[ii,0]+Lon_bound_regrid[ii,1])/2
    
    return Lat_regrid_1D, Lon_regrid_1D, Lat_bound_regrid, Lon_bound_regrid

def func_regrid(Data_orig, Lat_orig, Lon_orig, Lat_regrid_2D, Lon_regrid_2D):    
    ### Regridding GCM data from original coordinates to a new coordinates
    Lon_orig[Lon_orig < 0] +=360
    if np.ndim(Lon_orig)==1: # If the GCM grid is not curvlinear
        Lon_orig,Lat_orig=np.meshgrid(Lon_orig, Lat_orig)
        
    coords=np.squeeze(np.dstack((np.asarray(Lon_orig).flatten(), np.asarray(Lat_orig).flatten())))

    Data_orig=np.squeeze(Data_orig)
    if Data_orig.ndim==2:#this is for 2d regridding
        data_vec = np.asarray(Data_orig)
        if np.ndim(data_vec)>1:
            data_vec = data_vec.flatten()
        Data_regrid = griddata(coords, data_vec, (Lon_regrid_2D, Lat_regrid_2D), method='nearest')
        return np.asarray(Data_regrid)
    if Data_orig.ndim==3:#this is for 3d regridding - (lat-lon-depth)
        Data_regrid=[]
        for d in range(len(Data_orig)):
            z = np.asarray(Data_orig[d,:,:])
            if np.ndim(z)>1:
                z = z.flatten()
            zi = griddata(coords, z, (Lon_regrid_2D, Lat_regrid_2D), method='nearest')
            Data_regrid.append(zi)
        return np.asarray(Data_regrid)

### Regrdridding calculations - creating new grid coordinates, which will be used in regridding all models to the same resolution
lat_n_regrid, lon_n_regrid = 180, 360 # Number of Lat and Lon elements in the regridded data
lon_min_regrid, lon_max_regrid = 0, 360 # Min and Max value of Lon in the regridded data
lat_min_regrid, lat_max_regrid = -90, 90 # Min and Max value of Lat in the regridded data

Lat_regrid_1D, Lon_regrid_1D, Lat_bound_regrid, Lon_bound_regrid = func_latlon_regrid(lat_n_regrid, lon_n_regrid, lat_min_regrid, lat_max_regrid, lon_min_regrid, lon_max_regrid)
Lon_regrid_2D, Lat_regrid_2D = np.meshgrid(Lon_regrid_1D, Lat_regrid_1D)

######################################
GCM_Names = ['GFDL-ESM2M', 'GFDL-ESM2G', 'IPSL-CM5A-MR', 'IPSL-CM5A-LR', 'MIROC-ESM', 'MIROC-ESM-CHEM', 'CESM1-BGC', 'CMCC-CESM', 'CanESM2', 'GISS-E2-H-CC', 'GISS-E2-R-CC', 'MPI-ESM-MR', 'MPI-ESM-LR', 'NorESM1-ME']

dir_data = ('/data2/scratch/cabre/CMIP5/CMIP5_models/ocean_physics/') # Directory to read raw data from - Data saved on UPenn HPC cluster
dir_figs = (os.getcwd() + '/Figures/') # Directory to save figures

dates = [1980, 1999, 2080, 2099]
Var_name='thetao' # vriable name to be read from .nc files

Multimodel_Ave_hist = np.zeros((len(GCM_Names), lat_n_regrid, lon_n_regrid)) # Multimodel surface average of specified variable, regridded - Historical
Multimodel_Ave_rcp8p5 = np.zeros((len(GCM_Names), lat_n_regrid, lon_n_regrid)) # Multimodel surface average of specified variable, regridded - RCP8.5

for M_i in range(len(GCM_Names)):

    # Historical Period Calculations
    dset = xr.open_mfdataset(dir_data+ GCM_Names[M_i] + '/historical/mo/'+Var_name+'*.nc') # a 4-D datastet (time, depths/lev, lat, lon)
    if 'lEV' in dset.coords: # coordinates for some models are upper case ans some lower case
        lat_t='LAT'
        lon_t='LON'
        time_t='TIME'
        Data_all = dset['THETAO'].sel(LEV=0,method='nearest') # data at lev=0               
    else:
        lat_t='lat'
        lon_t='lon'
        time_t='time'
        Data_all = dset['thetao'].sel(lev=0,method='nearest') # data at lev=0     
             
    Data_regrid = np.nanmean( Data_all[ (Data_all[time_t].dt.year >= dates[0] ) & (Data_all[time_t].dt.year <= dates[1])].values ,axis=0)
    Data_regrid [ Data_regrid > 1e19 ] = np.nan
    dset.close()
    
    # Regriding data to 1degree by 1degree fields
    Data_regrid = func_regrid(Data_regrid, dset[lat_t].values, dset[lon_t].values, Lat_regrid_2D, Lon_regrid_2D)      
    Multimodel_Ave_hist[M_i,:,:]=Data_regrid
    
    # 21st century (RCP8.5) Calculations
    dset = xr.open_mfdataset(dir_data+ GCM_Names[M_i] + '/rcp85/mo/'+Var_name+'*.nc') # a 4-D datastet (time, depths/lev, lat, lon)
    if 'lEV' in dset.coords: # coordinates for some models are upper case ans some lower case
        Data_all = dset['THETAO'].sel(LEV=0,method='nearest') # data at lev=0               
    else:
        Data_all = dset['thetao'].sel(lev=0,method='nearest') # data at lev=0         
    
    Data_regrid = np.nanmean( Data_all[ (Data_all[time_t].dt.year >= dates[2] ) & (Data_all[time_t].dt.year <= dates[3])].values ,axis=0)
    Data_regrid [ Data_regrid > 1e19 ] = np.nan
    dset.close()
    
    # Regriding data to 1degree by 1degree fields
    Data_regrid = func_regrid(Data_regrid, dset[lat_t].values, dset[lon_t].values, Lat_regrid_2D, Lon_regrid_2D)  
    Multimodel_Ave_rcp8p5[M_i,:,:]=Data_regrid    
    
### Ploting change in SST - rcp85 minus hist ###
bounds_max=6
bound_ranges=bounds_max/20
bounds = np.arange(-1*bounds_max, bounds_max+bound_ranges, bound_ranges)
norm = matplotlib.colors.BoundaryNorm(boundaries=bounds, ncolors=256)

fig=plt.figure()
for ii in list(range(len(GCM_Names))):
    ax = fig.add_subplot(4,4,ii+1)
    Plot_Var=Multimodel_Ave_rcp8p5[ii,:,:] - Multimodel_Ave_hist[ii,:,:]
    m = Basemap(projection='cyl', lat_0=0, lon_0=0)
    m.drawcoastlines(linewidth=1.25)
    m.fillcontinents(color='0.95')
    m.drawparallels(np.arange(-90.,90.001,30.),labels=[True,False,False,False], linewidth=0.01) # labels = [left,right,top,bottom]
    if ii+1 >= list(range(len(GCM_Names)))[-4+1]: # Adds longitude ranges only to the last subplots that appear at the bottom of plot
        m.drawmeridians(np.arange(0.,360.,60.),labels=[False,False,False,True], linewidth=0.01) # labels = [left,right,top,bottom]
    im1 = m.pcolormesh(Lon_regrid_2D, Lat_regrid_2D, Plot_Var, norm=norm, shading='flat', cmap=plt.cm.RdBu_r, latlon=True) # Choose colormap: https://matplotlib.org/users/colormaps.html
    
    plt.title(GCM_Names[ii])
plt.suptitle( ('Climate Change impact on sea surface temperature under RCP8.5 Scenario - '+str(dates[2])+'-'+str(dates[3])+' minus '+ str(dates[0])+'-'+str(dates[1])), fontsize=18)
cbar = plt.colorbar(cax=plt.axes([0.93, 0.1, 0.015, 0.8]), extend='both') # cax = [left position, bottom postion, width, height] 
cbar.set_label('Unit = °C')
plt.show()
mng = plt.get_current_fig_manager()
mng.window.showMaximized() # Maximizes the plot window to save figures in full
fig.savefig(dir_figs +'/Fig_CMIP5_SST_climate_change_Impact_RCP8p5.png', format='png', dpi=300, transparent=True, bbox_inches='tight')
plt.close()

######################################################################
def func_gridcell_area(Lat_bound_regrid, Lon_bound_regrid): 
    ### Calculate Grid Cell areas, based on given coordinates [output units: million km2] ###
    earth_R = 6378 # Earth Radius - Unit is kilometer (km)
    GridCell_Area = np.empty((Lat_bound_regrid.shape[0], Lon_bound_regrid.shape[0] )) *np.nan
    for ii in range(Lat_bound_regrid.shape[0]):
        for jj in range(Lon_bound_regrid.shape[0]):
            GridCell_Area [ii,jj] = math.fabs( (earth_R**2) * (math.pi/180) * (Lon_bound_regrid[jj,1] - Lon_bound_regrid[jj,0])  * ( math.sin(math.radians(Lat_bound_regrid[ii,1])) - math.sin(math.radians(Lat_bound_regrid[ii,0]))) )
    GridCell_Area = GridCell_Area / 1e6 # to convert the area to million km2
    
    return GridCell_Area

### Change in global SST in 21st century vs 20st centiru average ###
# Calculate Grid Cell areas in million km2 - This function is saved in Behzadlib code in this directory - imported at the begenning
GridCell_Area = func_gridcell_area(Lat_bound_regrid, Lon_bound_regrid)

# Calculating grid-area-weighted global averages in SST in historical period and 21st century period, and their difference
SST_hist=np.zeros((len(GCM_Names))) # Historical average of SST
Delta_SST=np.zeros((len(GCM_Names))) # Historical average of SST minus 21st century average of SST
for ii in range(len(GCM_Names)):
    SST_hist[ii]= np.nansum( np.multiply(Multimodel_Ave_hist[ii,:,:]- 273.15, GridCell_Area) ) / np.nansum(GridCell_Area)
    Delta_SST[ii]= (np.nansum( np.multiply(Multimodel_Ave_rcp8p5[ii,:,:], GridCell_Area) ) / np.nansum(GridCell_Area)) - (np.nansum( np.multiply(Multimodel_Ave_hist[ii,:,:], GridCell_Area) ) / np.nansum(GridCell_Area))

fig, ax = plt.subplots()
ax.scatter(SST_hist, Delta_SST, s=200, marker='d', c='r')
ax.scatter(SST_hist, Delta_SST, s=20, marker='d', c='b')
for ii, txt in enumerate(GCM_Names):
    ax.annotate(txt, (SST_hist[ii],Delta_SST[ii]), fontsize=14)
plt.xlabel('Global SST (hist ave) [°C]', fontsize=18)
plt.xlim(17, 19.5)
plt.xticks( fontsize = 18)
plt.ylabel('Δ SST (rcp8.5 minus hist) [°C]', fontsize=18)
plt.ylim(1.5, 4)
plt.yticks( fontsize = 18)
plt.title( ('Global Change in SST under climate change (rcp8.5 '+str(dates[2])+'-'+str(dates[3])+' minus '+ str(dates[0])+'-'+str(dates[1]))+') VS. '+str(dates[0])+'-'+str(dates[1])+ ' average', fontsize=18)
mng = plt.get_current_fig_manager()
mng.window.showMaximized() # Maximizes the plot window to save figures in full
fig.savefig(dir_figs+'/Fig_CMIP5_SST_climate_change_Impact_RCP8p5_scatter.png', format='png', dpi=300, transparent=True, bbox_inches='tight')
plt.close()

###############################################################
def func_EOF (Calc_Var, Calc_Lat): # Empirical Orthogonal Functions maps and indices
    EOF_all=[]
    for i in range(Calc_Var.shape[0]):
        print ('EOF calc - Year: ', i)
        data_i=np.squeeze(Calc_Var[i,:,:])       
        data_EOF=[]
        if i==0:
            [lat_ii,lon_jj] = np.where(~np.isnan(data_i))

        for kk in range(len(lat_ii)):
            data_EOF.append( data_i[lat_ii[kk],lon_jj[kk]]*np.sqrt(np.cos(np.deg2rad(Calc_Lat[lat_ii[kk],lon_jj[kk]]))) )
        EOF_all.append(data_EOF)    
    EOF_all=np.asarray(EOF_all)
    
    C=np.cov(np.transpose(EOF_all))
    eigval,eigvec=np.linalg.eig(C)
    eigval=np.real(eigval)
    eigvec=np.real(eigvec)
    
    EOF_spatial_pattern = np.empty((10,Calc_Var.shape[1],Calc_Var.shape[2]))*np.nan # Stores first 10 EOFs for spatial pattern map
    for ss in range(EOF_spatial_pattern.shape[0]):
        for kk in range(len(lat_ii)):
            EOF_spatial_pattern[ss,lat_ii[kk],lon_jj[kk]] = eigvec[kk,ss]

    EOF_time_series = np.empty((10,Calc_Var.shape[0]))*np.nan # Stores first 10 EOFs times series
    for ss in range(EOF_time_series.shape[0]):
        EOF_time_series[ss,:] = np.dot(np.transpose(eigvec[:,ss]),np.transpose(EOF_all))
        
    EOF_variance_prcnt = np.empty((10))*np.nan # Stores first 10 EOFs variance percentage
    for ss in range(EOF_variance_prcnt.shape[0]):
        EOF_variance_prcnt[ss]=( eigval[ss]/np.nansum(eigval,axis=0) ) * 100        

    return EOF_spatial_pattern, EOF_time_series, EOF_variance_prcnt
    
def func_butterworth(var, CutOff_T, n_order):
    ### ButterWorth Filtering of high frequency signals
    # CutOff_T = Cut-off period , n_order = Order of filtering
    from scipy import signal
    fs = 1  # Sampling frequency, equal to 1 year in our case
    fc = 1/CutOff_T  # Cut-off frequency of the filter
    ww = fc / (fs / 2) # Normalize the frequency
    bb, aa = signal.butter(n_order, ww, 'low')
    return signal.filtfilt(bb, aa, var)

def func_plotmap_contourf(P_Var, P_Lon, P_Lat, P_range, P_title, P_unit, P_cmap, P_proj, P_lon0, P_latN, P_latS, P_c_fill):
### P_Var= Plotting variable, 2D(lat,lon) || P_Lon=Longitude, 2D || P_range=range of plotted values, can be vector or number || P_title=Plot title || P_unit=Plot colorbar unit
### P_cmap= plt.cm.seismic , plt.cm.jet || P_proj= 'cyl', 'npstere', 'spstere' || P_lon0=middle longitude of plot || P_latN=upper lat bound of plot || P_latS=lower lat bound of plot || P_c_fill= 'fill' fills the continets with grey color  
    fig=plt.figure()
    
    if P_proj=='npstere':
        m = Basemap( projection='npstere',lon_0=0, boundinglat=30)
        m.drawparallels(np.arange(-90,90,20), labels=[1,1,0,1], linewidth=1, color='k', fontsize=20)
        m.drawmeridians(np.arange(0,360,30), labels=[1,1,0,1], linewidth=1, color='k', fontsize=20)
    elif P_proj=='spstere':
        m = Basemap( projection='spstere',lon_0=180, boundinglat=-30)
        m.drawparallels(np.arange(-90,90,20), labels=[1,1,0,1], linewidth=1, color='k', fontsize=20)
        m.drawmeridians(np.arange(0,360,30), labels=[1,1,0,1], linewidth=1, color='k', fontsize=20)        
    else:
         m = Basemap( projection=P_proj, lon_0=P_lon0, llcrnrlon=P_lon0-180, llcrnrlat=P_latS, urcrnrlon=P_lon0+180, urcrnrlat=P_latN)
         m.drawparallels(np.arange(P_latS, P_latN+0.001, 40.),labels=[True,False,False,False], linewidth=0.01, color='k', fontsize=20) # labels = [left,right,top,bottom] # Latitutes
         m.drawmeridians(np.arange(P_lon0-180,P_lon0+180,60.),labels=[False,False,False,True], linewidth=0.01, color='k', fontsize=20) # labels = [left,right,top,bottom] # Longitudes        
    if P_c_fill=='fill':
        m.fillcontinents(color='0.8')
    m.drawcoastlines(linewidth=1.0, linestyle='solid', color='k', antialiased=1, ax=None, zorder=None)
    im=m.contourf(P_Lon, P_Lat, P_Var,P_range,latlon=True, cmap=P_cmap, extend='both')
    if P_proj=='npstere' or P_proj=='spstere':
        cbar = m.colorbar(im,"right", size="4%", pad="14%")
    else:
        cbar = m.colorbar(im,"right", size="3%", pad="2%")
    cbar.ax.tick_params(labelsize=20) 
    cbar.set_label(P_unit)
    plt.show()
    plt.title(P_title, fontsize=18)
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    return fig, m
  
#################################################################################################
###### North Atlantic Oscillation (NAO) calculated as the 1st EOF of Sea-Level Air Presure ######
#################################################################################################
GCM = 'GFDL-ESM2G' # The rest of calculations are continued for GFDL-ESM2G model only
dir_data_in=('/data2/scratch/cabre/CMIP5/CMIP5_models/atmosphere_physics/'+ GCM + '/historical/mo/') # Directory to raed raw data from 

dates = [1901, 2000]
Var_name='psl' # The variable name to be read from .nc files
dset = xr.open_mfdataset(dir_data_in+Var_name+'*12.nc')
Data_all = dset[Var_name]

Data_NAO = Data_all[ (Data_all.time.dt.year >= dates[0] ) & (Data_all.time.dt.year <= dates[1])].sel(lat=np.arange(10,85,2), lon=np.arange(280,360,2), method='nearest').values
Data_NAO [ Data_NAO > 1e19 ] = np.nan # eliminating missing data saved as 1e20
Data_NAO_rannual = np.nanmean(Data_NAO.reshape( np.int(Data_NAO.shape[0]/12) ,12,Data_NAO.shape[1],Data_NAO.shape[2]),axis=1) # Calculating annual average values, from monthly data

Lat_NAO = dset['lat'].sel(lat=np.arange(10,85,2), method='nearest').values # Latitude range [10,85]
Lon_NAO = dset['lon'].sel(lon=np.arange(280,360,2), method='nearest').values # Longitude range [280,360]
Lon_NAO_2D, Lat_NAO_2D = np.meshgrid(Lon_NAO, Lat_NAO)
dset.close()

EOF_spatial_pattern, EOF_time_series, EOF_variance_prcnt = func_EOF (Data_NAO_rannual, Lat_NAO_2D)
# NAO calculat as the 1st EOF of Sea-Level Air Presure
NAO_spatial_pattern = EOF_spatial_pattern[0,:,:]
NAO_index = EOF_time_series[0,:]

Plot_Var = NAO_spatial_pattern
cmap_limit=np.nanmax(np.abs( np.nanpercentile(Plot_Var, 99)))
Plot_range=np.linspace(-cmap_limit,cmap_limit,27)

fig=plt.figure()
P_lon0=360.
m = Basemap( projection='cyl', lon_0=P_lon0, llcrnrlon=P_lon0-120, llcrnrlat=-10., urcrnrlon=P_lon0+60, urcrnrlat=80.)
m.drawparallels(np.arange(-20., 80.+0.001, 20.),labels=[True,False,False,False], linewidth=0.01, color='k', fontsize=20) # labels = [left,right,top,bottom] # Latitutes
m.drawmeridians(np.arange(-180,+180,30.),labels=[False,False,False,True], linewidth=0.01, color='k', fontsize=20) # labels = [left,right,top,bottom] # Longitudes        
m.drawcoastlines(linewidth=1.0, linestyle='solid', color='k', antialiased=1, ax=None, zorder=None)
im=m.contourf(Lon_NAO_2D, Lat_NAO_2D, Plot_Var,Plot_range,latlon=True, cmap=plt.cm.seismic, extend='both')
cbar = m.colorbar(im,"right", size="3%", pad="2%")
cbar.ax.tick_params(labelsize=20) 
plt.show()
plt.title('North Atlantic Oscillation (NAO) calculated as the 1st EOF of Sea-Level Air Presure'+'\n'+'Spatial Pattern map - '+str(dates[0])+'-'+str(dates[1])+' - '+str(GCM), fontsize=18)
mng = plt.get_current_fig_manager()
mng.window.showMaximized() # Maximizes the plot window to save figures in full
fig.savefig(dir_figs+'/Fig_NAO_SpatialPattern_'+str(GCM)+'.png', format='png', dpi=300, transparent=True, bbox_inches='tight')


Plot_Var = (NAO_index - np.nanmean(NAO_index))/np.std(NAO_index) # NAO index normalized   

fig=plt.figure()
n_l=Plot_Var.shape[0]
years=np.linspace(dates[0], dates[0]+n_l-1, n_l)
plt.plot(years,Plot_Var, 'k') 
y2=np.zeros(len(Plot_Var))
plt.fill_between(years, Plot_Var, y2, where=Plot_Var >= y2, color = 'r', interpolate=True)
plt.fill_between(years, Plot_Var, y2, where=Plot_Var <= y2, color = 'b', interpolate=True)
plt.axhline(linewidth=1, color='k')
plt.xticks(fontsize = 18); plt.yticks(fontsize = 18)
plt.title('North Atlantic Oscillation (NAO) calculated as the 1st EOF of Sea-Level Air Presure'+'\n'+'Time Series - '+str(dates[0])+'-'+str(dates[1])+' - '+str(GCM), fontsize=20)    
mng = plt.get_current_fig_manager()
mng.window.showMaximized() # Maximizes the plot window to save figures in full
fig.savefig(dir_figs+'/Fig_NAO_Indices_'+str(GCM)+'.png', format='png', dpi=300, transparent=True, bbox_inches='tight')

#############################################################################################
###  Empirical Orthogonal Functions (EOFs) of Pacific Ocean Sea Surface Temperature  ########
#############################################################################################
dir_data_in=('/data2/scratch/cabre/CMIP5/CMIP5_models/ocean_physics/'+ GCM + '/historical/mo/')

dates = [1901, 2000]
Var_name='thetao' # The variable name to be read from .nc files
dset = xr.open_mfdataset(dir_data_in+Var_name+'*.nc')
Data_all = dset[Var_name].sel(lev=0,method='nearest') # data at lev=0   

Data_monthly = Data_all[ (Data_all.time.dt.year >= dates[0] ) & (Data_all.time.dt.year <= dates[1])].sel(rlat=np.arange(-90,66,2), rlon=np.arange(100,290,2), method='nearest').values      
Data_monthly [ Data_monthly > 1e19 ] = np.nan
Data_rannual = np.nanmean(Data_monthly.reshape( np.int(Data_monthly.shape[0]/12) ,12,Data_monthly.shape[1],Data_monthly.shape[2]),axis=1)

# Regriding data to 1degree by 1degree fields
SST_annual = func_regrid(Data_rannual, dset['lat'].values, dset['lon'].values, Lat_regrid_2D, Lon_regrid_2D)
dset.close()

# Empirical Orthogonal Functions (EOFs)
EOF_spatial_pattern, EOF_time_series, EOF_variance_prcnt = func_EOF (SST_annual, Lat_regrid_2D)

## ButterWorth Filtering of high frequency (less than 3 yr time-period) signals in the EOF time series
EOF_time_series_BWfilt = copy.deepcopy(EOF_time_series)
for ii in range(EOF_time_series_BWfilt.shape[0]):
    EOF_time_series_BWfilt[ii,:] = func_butterworth(EOF_time_series_BWfilt[ii,:],3,4)

### Ploting the Spatial Patterns and Indices
Plot_Var = EOF_spatial_pattern
Plot_prcnt=EOF_variance_prcnt
P_cmap=plt.cm.seismic; P_proj='cyl'; P_lon0=210.; P_latN=90.; P_latS=-90.; P_range=np.linspace(-0.08,0.08,41); P_Lon=Lon_regrid_2D; P_Lat=Lat_regrid_2D;
n_r=2 ; n_c=2 ; n_t=4
fig=plt.figure()
for M_i in range(n_t):
    ax = fig.add_subplot(n_r,n_c,M_i+1)     
    Var_plot_ii=Plot_Var[M_i,:,:]    
    
    m = Basemap( projection=P_proj, lon_0=P_lon0, llcrnrlon=P_lon0-180, llcrnrlat=P_latS, urcrnrlon=P_lon0+180, urcrnrlat=P_latN)    
    if M_i == (n_c*(n_r-1)): # Adds longitude ranges only to the last subplots that appear at the bottom of plot
        m.drawparallels(np.arange(P_latS, P_latN+0.001, 30.),labels=[True,False,False,False], linewidth=0.01, color='k', fontsize=16) # labels = [left,right,top,bottom] # Latitutes
        m.drawmeridians(np.arange(0,360,90.),labels=[False,False,False,True], linewidth=0.01, color='k', fontsize=16) # labels = [left,right,top,bottom] # Longitudes    
    elif M_i==0 or M_i==n_c or M_i==n_c*2 or M_i==n_c*3 or M_i==n_c*4 or M_i==n_c*5 or M_i==n_c*6 or M_i==n_c*7 or M_i==n_c*8:
        m.drawparallels(np.arange(P_latS, P_latN+0.001, 30.),labels=[True,False,False,False], linewidth=0.01, color='k', fontsize=16) # labels = [left,right,top,bottom] # Latitutes
    elif M_i >= n_t-n_c and M_i != (n_c*(n_r-1)): # Adds longitude ranges only to the last subplots that appear at the bottom of plot
        m.drawmeridians(np.arange(0,360,90.),labels=[False,False,False,True], linewidth=0.01, color='k', fontsize=16) # labels = [left,right,top,bottom] # Longitudes
    else:
        m.drawparallels(np.arange(P_latS, P_latN+0.001, 30.),labels=[False,False,False,False], linewidth=0.01, color='k', fontsize=16) # labels = [left,right,top,bottom] # Latitutes
        m.drawmeridians(np.arange(0,360,90.),labels=[False,False,False,False], linewidth=0.01, color='k', fontsize=16) # labels = [left,right,top,bottom] # Longitudes

    m.fillcontinents(color='0')
    m.drawcoastlines(linewidth=1.0, linestyle='solid', antialiased=1, ax=None, zorder=None)
    m.fillcontinents(color='0.95')
    im=m.contourf(P_Lon, P_Lat, Var_plot_ii, P_range, latlon=True, cmap=P_cmap, extend='both')
    plt.title('EOF #'+str(M_i+1)+'  ,  '+str(round(Plot_prcnt[M_i], 2))+' % of the variance', fontsize=14)
        
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, hspace=0.05, wspace=0.1) # the amount of height/width reserved for space between subplots
cbar_ax = fig.add_axes([0.93, 0.1, 0.015, 0.8]) # [right,bottom,width,height] 
fig.colorbar(im, cax=cbar_ax)
plt.suptitle('Empirical Orthogonal Functions (EOFs) of Pacific Ocean sea surface temperature'+'\n'+'Spatial Pattern maps - '+str(dates[0])+'-'+str(dates[1])+' - '+str(GCM), fontsize=20)    
plt.show()
mng = plt.get_current_fig_manager()
mng.window.showMaximized() # Maximizes the plot window to save figures in full   
fig.savefig(dir_figs+'/Fig_EOF_SST_SpatialPattern_'+str(GCM)+'.png', format='png', dpi=300, transparent=True, bbox_inches='tight')


Plot_Var = EOF_time_series_BWfilt
Plot_prcnt=EOF_variance_prcnt
fig, ax = plt.subplots(nrows=2, ncols=2)
for ii in range(4):
    EOF_time_series_plot_norm=(Plot_Var[ii,:]-np.nanmean(Plot_Var[ii,:]))/np.std(Plot_Var[ii,:])    
    #EOF_time_series_plot_norm_rm=runningMeanFast(EOF_time_series_plot_norm, 10)
    EOF_time_series_plot_norm_rm=EOF_time_series_plot_norm
    plt.subplot(2, 2, ii+1)
    n_l=EOF_time_series_plot_norm.shape[0]
    years=np.linspace(0, n_l, n_l)
    plt.plot(years,EOF_time_series_plot_norm_rm, 'k') 
    y2=np.zeros(len(EOF_time_series_plot_norm_rm))
    plt.fill_between(years, EOF_time_series_plot_norm_rm, y2, where=EOF_time_series_plot_norm_rm >= y2, color = 'r', interpolate=True)
    plt.fill_between(years, EOF_time_series_plot_norm_rm, y2, where=EOF_time_series_plot_norm_rm <= y2, color = 'b', interpolate=True)
    plt.axhline(linewidth=1, color='k')
    plt.title('EOF # '+str(ii+1)+'  ,  '+str(round(Plot_prcnt[ii], 2))+' % of the variance', fontsize=18)
    plt.xticks(fontsize = 18); plt.yticks(fontsize = 18)
plt.suptitle('Empirical Orthogonal Functions (EOFs) of Pacific Ocean sea surface temperature'+'\n'+'Time Series (Butterworth filtered, Normalized) - '+str(dates[0])+'-'+str(dates[1])+' - '+str(GCM), fontsize=20)    
mng = plt.get_current_fig_manager()
mng.window.showMaximized() # Maximizes the plot window to save figures in full
fig.savefig(dir_figs+'/Fig_EOF_SST_Indices_'+str(GCM)+'.png', format='png', dpi=300, transparent=True, bbox_inches='tight')


##################################################################################################
###  Calculating Wind Curls using wind stress in latitudinal and longitudinal directions  ########
##################################################################################################
dir_data_in1 = ('/data2/scratch/cabre/CMIP5/CMIP5_models/atmosphere_physics/') # Directory to raed raw data from
dir_data_in2=(dir_data_in1+ GCM + '/historical/mo/')

year_start=1901
year_end=2000

Var_name='tauu' # The variable name to be read from .nc files
dset = xr.open_mfdataset(dir_data_in2+Var_name+'*.nc')
Data_all = dset[Var_name] 
Data_monthly = Data_all[ (Data_all.time.dt.year >= dates[0] ) & (Data_all.time.dt.year <= dates[1])].values
Data_rannual = np.nanmean(Data_monthly.reshape( np.int(Data_monthly.shape[0]/12) ,12,Data_monthly.shape[1],Data_monthly.shape[2]),axis=1)
Tau_X = func_regrid(Data_rannual, dset['lat'].values, dset['lon'].values, Lat_regrid_2D, Lon_regrid_2D) # Regriding data to 1degree by 1degree fields

Var_name='tauv' # The variable name to be read from .nc files
dset = xr.open_mfdataset(dir_data_in2+Var_name+'*.nc')
Data_all = dset[Var_name] 
Data_monthly = Data_all[ (Data_all.time.dt.year >= year_start ) & (Data_all.time.dt.year <= year_end)].values
Data_rannual = np.nanmean(Data_monthly.reshape( np.int(Data_monthly.shape[0]/12) ,12,Data_monthly.shape[1],Data_monthly.shape[2]),axis=1)
Tau_Y = func_regrid(Data_rannual, dset['lat'].values, dset['lon'].values, Lat_regrid_2D, Lon_regrid_2D)

# Wind_Curl = ( D_Tau_Y / D_X ) - ( D_Tau_X / D_Y ) # D_X = (Lon_1 - Lon_2) * COS(Lat)
Wind_Curl = np.zeros(( Tau_X.shape[0], Tau_X.shape[1], Tau_X.shape[2]))  
for tt in range (0,Tau_X.shape[0]):  
    for ii in range (1,Tau_X.shape[1]-1):
        for jj in range (1,Tau_X.shape[2]-1): # Wind_Curl = ( D_Tau_Y / D_X ) - ( D_Tau_X / D_Y ) # D_X = (Lon_1 - Lon_2) * COS(Lat)
            Wind_Curl[tt,ii,jj] = (  ( Tau_Y[tt, ii,jj+1] - Tau_Y[tt, ii,jj-1] ) /  np.absolute(  ( ( Lon_regrid_2D[ii,jj+1] -  Lon_regrid_2D[ii,jj-1] ) * 111321  *  math.cos(math.radians( ( Lat_regrid_2D[ii,jj])))   )  )     )   -   (  ( Tau_X[tt, ii+1,jj] - Tau_X[tt, ii-1,jj] ) / np.absolute( ( ( Lat_regrid_2D[ii+1,jj] -  Lat_regrid_2D[ii-1,jj] ) * 111321 ) )  )

        Wind_Curl[tt,ii,0] = (  ( Tau_Y[tt, ii,1] - Tau_Y[tt, ii,-1] ) /  np.absolute(  ( ( Lon_regrid_2D[ii,1] -  Lon_regrid_2D[ii,-1] ) * 111321  *  math.cos(math.radians( ( Lat_regrid_2D[ii,0])))   )  )     )   -   (  ( Tau_X[tt, ii+1,0] - Tau_X[tt, ii-1,0] ) / np.absolute( ( ( Lat_regrid_2D[ii+1,0] -  Lat_regrid_2D[ii-1,0] ) * 111321 ) )  )
        Wind_Curl[tt,ii,-1] = (  ( Tau_Y[tt, ii,0] - Tau_Y[tt, ii,-2] ) /  np.absolute(  ( ( Lon_regrid_2D[ii,0] -  Lon_regrid_2D[ii,-2] ) * 111321  *  math.cos(math.radians( ( Lat_regrid_2D[ii,-1])))   )  )     )   -   (  ( Tau_X[tt, ii+1,-1] - Tau_X[tt, ii-1,-1] ) / np.absolute( ( ( Lat_regrid_2D[ii+1,-1] -  Lat_regrid_2D[ii-1,-1] ) * 111321 ) )  )

# Wind_Crul / f # f = coriolis parameter = 2Wsin(LAT) , W = 7.292E-5 rad/s
Wind_Curl_f = np.zeros(( Tau_X.shape[0], Tau_X.shape[1], Tau_X.shape[2])) 
for tt in range (0,Tau_X.shape[0]):   
    for ii in range (1,Tau_X.shape[1]-1):
        if np.absolute( Lat_regrid_2D[ii,0] ) >= 5: # Only calulate for Lats > 5N and Lats < 5S, to avoid infinit numbers in equator where f is zero
            for jj in range (1,Tau_X.shape[2]-1): # Wind_Curl = ( D_Tau_Y / D_X ) - ( D_Tau_X / D_Y ) # D_X = (Lon_1 - Lon_2) * COS(Lat)
                Wind_Curl_f[tt,ii,jj] = (  ( ( Tau_Y[tt, ii,jj+1] - Tau_Y[tt, ii,jj-1] ) / ( 2*7.292E-5 *  math.sin(math.radians( ( Lat_regrid_2D[ii,jj]))) ) ) /  np.absolute(  ( ( Lon_regrid_2D[ii,jj+1] -  Lon_regrid_2D[ii,jj-1] ) * 111321  *  math.cos(math.radians( ( Lat_regrid_2D[ii,jj])))   )  )     )   -   (  ( ( Tau_X[tt, ii+1,jj] - Tau_X[tt, ii-1,jj] ) / ( 2*7.292E-5 *  math.sin(math.radians( ( Lat_regrid_2D[ii,jj]))) ) ) / np.absolute( ( ( Lat_regrid_2D[ii+1,jj] -  Lat_regrid_2D[ii-1,jj] ) * 111321 ) )  )

            Wind_Curl_f[tt,ii,0] = (  ( ( Tau_Y[tt, ii,1] - Tau_Y[tt, ii,-1] ) / ( 2*7.292E-5 *  math.sin(math.radians( ( Lat_regrid_2D[ii,0]))) ) ) /  np.absolute(  ( ( Lon_regrid_2D[ii,1] -  Lon_regrid_2D[ii,-1] ) * 111321  *  math.cos(math.radians( ( Lat_regrid_2D[ii,0])))   )  )     )   -   (  ( ( Tau_X[tt, ii+1,jj] - Tau_X[tt, ii-1,0] ) / ( 2*7.292E-5 *  math.sin(math.radians( ( Lat_regrid_2D[ii,0]))) ) ) / np.absolute( ( ( Lat_regrid_2D[ii+1,0] -  Lat_regrid_2D[ii-1,0] ) * 111321 ) )  )
            Wind_Curl_f[tt,ii,-1] = (  ( ( Tau_Y[tt, ii,0] - Tau_Y[tt, ii,-2] ) / ( 2*7.292E-5 *  math.sin(math.radians( ( Lat_regrid_2D[ii,-1]))) ) ) /  np.absolute(  ( ( Lon_regrid_2D[ii,0] -  Lon_regrid_2D[ii,-2] ) * 111321  *  math.cos(math.radians( ( Lat_regrid_2D[ii,-1])))   )  )     )   -   (  ( ( Tau_X[tt, ii+1,-1] - Tau_X[tt, ii-1,-1] ) / ( 2*7.292E-5 *  math.sin(math.radians( ( Lat_regrid_2D[ii,-1]))) ) ) / np.absolute( ( ( Lat_regrid_2D[ii+1,-1] -  Lat_regrid_2D[ii-1,-1] ) * 111321 ) )  )

###############################################################################
Lat_regrid_1D_4, Lon_regrid_1D_4, Lat_bound_regrid_4, Lon_bound_regrid_4 = func_latlon_regrid(45, 90, lat_min_regrid, lat_max_regrid, lon_min_regrid, lon_max_regrid)
Lon_regrid_2D_4, Lat_regrid_2D_4 = np.meshgrid(Lon_regrid_1D_4, Lat_regrid_1D_4)
Tau_X_4 = func_regrid(np.nanmean(Tau_X,axis=0), Lat_regrid_2D, Lon_regrid_2D, Lat_regrid_2D_4, Lon_regrid_2D_4)
Tau_Y_4 = func_regrid(np.nanmean(Tau_Y,axis=0), Lat_regrid_2D, Lon_regrid_2D, Lat_regrid_2D_4, Lon_regrid_2D_4)

Plot_Var_f = np.nanmean(Wind_Curl_f,axis=0) * 1E3

cmap_limit=np.nanmax(np.abs( np.nanpercentile(Plot_Var_f, 99))) # Scale the colorbar to 99th percentile
Plot_range=np.linspace(-cmap_limit,cmap_limit,27)
Plot_unit='(1E-3 N.S/m3.rad)'; Plot_title= 'Curl of (Wind/f) (Ekman upwelling) (1E-3 N.S/m3.rad) - '+str(year_start)+'-'+str(year_end)+' - '+str(GCM)+'\n(Arrows: wind direction) (contour line: Curl(wind/f)=0)'

fig, m = func_plotmap_contourf(Plot_Var_f, Lon_regrid_2D, Lat_regrid_2D, Plot_range, Plot_title, Plot_unit, plt.cm.seismic, 'cyl', 210., 80., -80., '-')
im2=m.quiver(Lon_regrid_2D_4, Lat_regrid_2D_4, Tau_X_4, Tau_Y_4, latlon=True, pivot='middle')
plt.show()
im3=m.contour(Lon_regrid_2D[25:50,:], Lat_regrid_2D[25:50,:],Plot_Var_f[25:50,:], levels = [0], latlon=True, colors='darkgreen')
plt.clabel(im3, fontsize=8, inline=1)
fig.savefig(dir_figs+'/'+'Fig_Wind_Curl_f_WQuiver_'+str(GCM)+'.png', format='png', dpi=300, transparent=True, bbox_inches='tight')
plt.close()
