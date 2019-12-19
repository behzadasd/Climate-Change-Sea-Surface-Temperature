############################################
####   Data Analysis/Plotting Library   ####
############################################
####     Behzad Asadieh, Ph.D.      ####
####  University of Pennsylvania    ####
####    basadieh@sas.upenn.edu      ####
####     github.com/behzadasd       ####
########################################
import numpy as np
from numpy import zeros, ones, empty, nan, shape
from numpy import isnan, nanmean, nanmax, nanmin
import numpy.ma as ma
from netCDF4 import MFDataset, Dataset, num2date, date2num, date2index
import os
import matplotlib
import matplotlib.mlab as ml
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm, maskoceans
from scipy.interpolate import griddata
import math
import copy


def func_latlon_regrid(lat_n_regrid, lon_n_regrid, lat_min_regrid, lat_max_regrid, lon_min_regrid, lon_max_regrid): 
    # This function centers lat at [... -1.5, -0.5, +0.5, +1.5 ...] - No specific equator lat cell
    ###lat_n_regrid, lon_n_regrid = 180, 360 # Number of Lat and Lon elements in the regridded data
    ###lon_min_regrid, lon_max_regrid = 0, 360 # Min and Max value of Lon in the regridded data
    ###lat_min_regrid, lat_max_regrid = -90, 90 # Min and Max value of Lat in the regridded data
    ####creating arrays of regridded lats and lons ###
    #### Latitude Bounds ####
    Lat_regrid_1D=zeros ((lat_n_regrid));
    Lat_bound_regrid = zeros ((lat_n_regrid,2)); Lat_bound_regrid[0,0]=-90;  Lat_bound_regrid[0,1]=Lat_bound_regrid[0,0] + (180/lat_n_regrid); Lat_regrid_1D[0]=(Lat_bound_regrid[0,0]+Lat_bound_regrid[0,1])/2
    for ii in range(1,lat_n_regrid):
        Lat_bound_regrid[ii,0]=Lat_bound_regrid[ii-1,1]
        Lat_bound_regrid[ii,1]=Lat_bound_regrid[ii,0] +  (180/lat_n_regrid)
        Lat_regrid_1D[ii]=(Lat_bound_regrid[ii,0]+Lat_bound_regrid[ii,1])/2
    #### Longitude Bounds ####
    Lon_regrid_1D=zeros ((lon_n_regrid));
    Lon_bound_regrid = zeros ((lon_n_regrid,2)); Lon_bound_regrid[0,0]=0;  Lon_bound_regrid[0,1]=Lon_bound_regrid[0,0] + (360/lon_n_regrid); Lon_regrid_1D[0]=(Lon_bound_regrid[0,0]+Lon_bound_regrid[0,1])/2
    for ii in range(1,lon_n_regrid):
        Lon_bound_regrid[ii,0]=Lon_bound_regrid[ii-1,1]
        Lon_bound_regrid[ii,1]=Lon_bound_regrid[ii,0] +  (360/lon_n_regrid)
        Lon_regrid_1D[ii]=(Lon_bound_regrid[ii,0]+Lon_bound_regrid[ii,1])/2
    
    return Lat_regrid_1D, Lon_regrid_1D, Lat_bound_regrid, Lon_bound_regrid


def func_gridcell_area(Lat_bound_regrid, Lon_bound_regrid): 
    #### Calculate Grid Cell areas in million km2 ####
    earth_R = 6378 # Earth Radius - Unit is kilometer (km)
    GridCell_Area = empty((Lat_bound_regrid.shape[0], Lon_bound_regrid.shape[0] )) *nan
    for ii in range(Lat_bound_regrid.shape[0]):
        for jj in range(Lon_bound_regrid.shape[0]):
            GridCell_Area [ii,jj] = math.fabs( (earth_R**2) * (math.pi/180) * (Lon_bound_regrid[jj,1] - Lon_bound_regrid[jj,0])  * ( math.sin(math.radians(Lat_bound_regrid[ii,1])) - math.sin(math.radians(Lat_bound_regrid[ii,0]))) )
    GridCell_Area = GridCell_Area / 1e6 # to convert the area to million km2
    
    return GridCell_Area



def func_oceanlandmask(Lat_regrid_2D, Lon_regrid_2D):
    lat_n_regrid, lon_n_regrid =Lat_regrid_2D.shape[0], Lat_regrid_2D.shape[1]
    Ocean_Land_mask = empty ((lat_n_regrid, lon_n_regrid)) * nan
    ocean_mask= maskoceans(Lon_regrid_2D-180, Lat_regrid_2D, Ocean_Land_mask)
    for ii in range(lat_n_regrid):
        for jj in range(lon_n_regrid):
            if ma.is_masked(ocean_mask[ii,jj]):
                Ocean_Land_mask[ii,jj]=1 # Land_Ocean_mask=1 means grid cell is ocean (not on land)
            else:
                Ocean_Land_mask[ii,jj]=0 # Land_Ocean_mask=0 means grid cell is land
    land_mask2 = copy.deepcopy ( Ocean_Land_mask ) # The created land_mask's longitude is from -180-180 - following lines transfer it to 0-360
    Ocean_Land_mask=empty((Lat_regrid_2D.shape[0], Lat_regrid_2D.shape[1])) *nan
    Ocean_Land_mask[:,0:int(Ocean_Land_mask.shape[1]/2)]=land_mask2[:,int(Ocean_Land_mask.shape[1]/2):]
    Ocean_Land_mask[:,int(Ocean_Land_mask.shape[1]/2):]=land_mask2[:,0:int(Ocean_Land_mask.shape[1]/2)]
    
    return Ocean_Land_mask # 1= ocean, 0= land

def func_oceanindex (Lat_regrid_2D, Lon_regrid_2D):
    
    Ocean_Land_mask = func_oceanlandmask(Lat_regrid_2D, Lon_regrid_2D) # 1= ocean, 0= land
    
    #directory= '/data1/home/basadieh/behzadcodes/behzadlibrary/'
    directory = os.path.dirname(os.path.realpath(__file__)) # Gets the directory where the code is located - The gx3v5_OceanIndex.nc should be placed in the same directory
    file_name='gx3v5_OceanIndex.nc'
    dset_n = Dataset(directory+'/'+file_name)
    
    REGION_MASK=np.asarray(dset_n.variables['REGION_MASK'][:])
    TLAT=np.asarray(dset_n.variables['TLAT'][:])
    TLONG=np.asarray(dset_n.variables['TLONG'][:])
    
    REGION_MASK_regrid = func_regrid(REGION_MASK, TLAT, TLONG, Lat_regrid_2D, Lon_regrid_2D)
    Ocean_Index = copy.deepcopy(REGION_MASK_regrid)    
    for tt in range(0,6): # Smoothing the coastal gridcells - If a cell in the regrid has fallen on land but in Ocean_Land_mask it's in ocean, a neighboring Ocean_Index value will be assigned to it
        for ii in range(Ocean_Index.shape[0]):
            for jj in range(Ocean_Index.shape[1]):
                
                if Ocean_Index[ii,jj] == 0 and Ocean_Land_mask[ii,jj] == 1:
                    if ii>2 and jj>2:
                        Ocean_Index[ii,jj] = np.max(Ocean_Index[ii-1:ii+2,jj-1:jj+2])

    Ocean_Index[ np.where( np.logical_and(   np.logical_and( 0 <= Lon_regrid_2D , Lon_regrid_2D < 20 ) , np.logical_and( Lat_regrid_2D <= -30 , Ocean_Index != 0 )   ) ) ] = 6 ## Assigning Atlantic South of 30S to Atlantic Ocean Index
    Ocean_Index[ np.where( np.logical_and(   np.logical_and( 290 <= Lon_regrid_2D , Lon_regrid_2D <= 360 ) , np.logical_and( Lat_regrid_2D <= -30 , Ocean_Index != 0 )   ) ) ] = 6   
    Ocean_Index[ np.where( np.logical_and(   np.logical_and( 20 <= Lon_regrid_2D , Lon_regrid_2D < 150 ) , np.logical_and( Lat_regrid_2D <= -30 , Ocean_Index != 0 )   ) ) ] = 3 ## Assigning Pacifi South of 30S to Atlantic Ocean Index   
    Ocean_Index[ np.where( np.logical_and(   np.logical_and( 150 <= Lon_regrid_2D , Lon_regrid_2D < 290 ) , np.logical_and( Lat_regrid_2D <= -30 , Ocean_Index != 0 )   ) ) ] = 2 ## Assigning Pacifi South of 30S to Atlantic Ocean Index
    
    return Ocean_Index # [0=land] [2=Pacific] [3=Indian Ocean] [6=Atlantic] [10=Arctic] [8=Baffin Bay (west of Greenland)] [9=Norwegian Sea (east of Greenland)] [11=Hudson Bay (Canada)] 
                       # [-7=Mediterranean] [-12=Baltic Sea] [-13=Black Sea] [-5=Red Sea] [-4=Persian Gulf] [-14=Caspian Sea]

def func_regrid(Data_orig, Lat_orig, Lon_orig, Lat_regrid_2D, Lon_regrid_2D):    
    
    Lon_orig[Lon_orig < 0] +=360
    
    if np.ndim(Lon_orig)==1: # If the GCM grid is not curvlinear
        Lon_orig,Lat_orig=np.meshgrid(Lon_orig, Lat_orig)
        
    lon_vec = np.asarray(Lon_orig)
    lat_vec = np.asarray(Lat_orig)
    lon_vec = lon_vec.flatten()
    lat_vec = lat_vec.flatten()
    coords=np.squeeze(np.dstack((lon_vec,lat_vec)))

    Data_orig=np.squeeze(Data_orig)
    if Data_orig.ndim==2:#this is for 2d regridding
        data_vec = np.asarray(Data_orig)
        if np.ndim(data_vec)>1:
            data_vec = data_vec.flatten()
        Data_regrid = griddata(coords, data_vec, (Lon_regrid_2D, Lat_regrid_2D), method='nearest')
        return np.asarray(Data_regrid)
    if Data_orig.ndim==3:#this is for 3d regridding
        Data_regrid=[]
        for d in range(len(Data_orig)):
            z = np.asarray(Data_orig[d,:,:])
            if np.ndim(z)>1:
                z = z.flatten()
            zi = griddata(coords, z, (Lon_regrid_2D, Lat_regrid_2D), method='nearest')
            Data_regrid.append(zi)
        return np.asarray(Data_regrid)


def func_EOF (Calc_Var, Calc_Lat): # Empirical Orthogonal Functions maps and indices
#%% Example :
#Calc_Var = data_set_regrid [:,10:61,300:]
#Calc_Lat = Lat_regrid_2D [10:61,300:]
#Calc_Lon = Lon_regrid_2D [10:61,300:]
#
#EOF_spatial_pattern, EOF_time_series, EOF_variance_prcnt = func_EOF (Calc_Var, Calc_Lat)
#%%
    EOF_all=[]
    for i in range(Calc_Var.shape[0]):
        
        print ('EOF calc - Year: ', i)
        data_i=Calc_Var[i,:,:]
        data_i=np.squeeze(data_i)        

        data_EOF=[]
        if i==0:
            [lat_ii,lon_jj] = np.where(~np.isnan(data_i))

        for kk in range(len(lat_ii)):
            EOF_i=data_i[lat_ii[kk],lon_jj[kk]]*np.sqrt(np.cos(np.deg2rad(Calc_Lat[lat_ii[kk],lon_jj[kk]])))
            data_EOF.append(EOF_i)
    
        EOF_all.append(data_EOF)    
    
    EOF_all=np.asarray(EOF_all)
    
    C=np.cov(np.transpose(EOF_all))
    #C= np.array(C, dtype=np.float32)
    eigval,eigvec=np.linalg.eig(C)
    eigval=np.real(eigval)
    eigvec=np.real(eigvec)
    
    EOF_spatial_pattern = empty((10,Calc_Var.shape[1],Calc_Var.shape[2]))*nan # Stores first 10 EOFs for spatial pattern map
    for ss in range(EOF_spatial_pattern.shape[0]):
        for kk in range(len(lat_ii)):
            EOF_spatial_pattern[ss,lat_ii[kk],lon_jj[kk]] = eigvec[kk,ss]

    EOF_time_series = empty((10,Calc_Var.shape[0]))*nan # Stores first 10 EOFs times series
    for ss in range(EOF_time_series.shape[0]):
        EOF_time_series[ss,:] = np.dot(np.transpose(eigvec[:,ss]),np.transpose(EOF_all))
        
    EOF_variance_prcnt = empty((10))*nan # Stores first 10 EOFs variance percentage
    for ss in range(EOF_variance_prcnt.shape[0]):
        EOF_variance_prcnt[ss]=( eigval[ss]/np.nansum(eigval,axis=0) ) * 100        

    return EOF_spatial_pattern, EOF_time_series, EOF_variance_prcnt































