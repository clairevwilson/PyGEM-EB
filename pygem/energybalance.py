# External libraries
import scipy.optimize as opt
import numpy as np
import xarray as xr
import suncalc as solar
# Internal libraries
import pygem.pygem_input as pygem_prms
import pygem.oggm_compat as oggm
import pygem.pygem_modelsetup as modelsetup
import class_climate

# DEFINE A BUNCH OF CONSTANTS
density_ice = 900           # Density of ice [kg m-3] (or Gt / 1000 km3)
density_water = 1000        # Density of water [kg m-3]
area_ocean = 362.5 * 1e12   # Area of ocean [m2] (Cogley, 2012 from Marzeion et al. 2020)
k_ice = 2.33                # Thermal conductivity of ice [J s-1 K-1 m-1] recall (W = J s-1)
k_air = 0.023               # Thermal conductivity of air [J s-1 K-1 m-1] (Mellor, 1997)
k_air = 0.001               # Thermal conductivity of air [J s-1 K-1 m-1]
ch_ice = 1890000            # Volumetric heat capacity of ice [J K-1 m-3] (density=900, heat_capacity=2100 J K-1 kg-1)
ch_air = 1297               # Volumetric Heat capacity of air [J K-1 m-3] (density=1.29, heat_capacity=1005 J K-1 kg-1)
Lh_rf = 333550              # Latent heat of fusion [J kg-1]
tolerance = 1e-12           # Model tolerance (used to remove low values caused by rounding errors)
gravity = 9.81              # Gravity [m s-2]
pressure_std = 101325       # Standard pressure [Pa]
temp_std = 288.15           # Standard temperature [K]
R_gas = 8.3144598           # Universal gas constant [J mol-1 K-1]
molarmass_air = 0.0289644   # Molar mass of Earth's air [kg mol-1]

assert pygem_prms.glac_no not in ['01.00570'], 'EB model can currently only run Gulkana glacier'

# GLACIER SETUP
main_glac_rgi = modelsetup.selectglaciersrgitable(pygem_prms.glac_no,
                rgi_regionsO1=pygem_prms.rgi_regionsO1, rgi_regionsO2=pygem_prms.rgi_regionsO2,
                rgi_glac_number=pygem_prms.rgi_glac_number, include_landterm=pygem_prms.include_landterm,
                include_laketerm=pygem_prms.include_laketerm, include_tidewater=pygem_prms.include_tidewater)
#def getEBpoints(glac_no): (eventually this will be a function to handle EB points)
# get OGGM flowlines to select bins for point balance
gdir = oggm.single_flowline_glacier_directory(pygem_prms.glac_no[0], logging_level='CRITICAL')
fls = oggm.get_glacier_zwh(gdir)
#filter out zero bins to get only initial glacier volume
fls = fls.iloc[np.nonzero(fls['h'].to_numpy())]
#setup three points at minimum, median and maximum elevation band from OGGM
z_stats = np.array([np.min(fls['z']),np.median(fls['z']),np.max(fls['z'])])
median_index = np.where(fls['z']==z_stats[1])[0][0]
w_stats = np.array([fls['w'][-1],fls['w'][median_index],fls['w'][0]])
h_stats = np.array([fls['h'][-1],fls['h'][median_index],fls['h'][0]])

#manually set number of exponentially scaling bins
n_vert_bins = 10
n_points = len(z_stats)
option_bin = 0  #0 for preset depths, 1 for exp scaling

#create variable to store glacier geometry
vert_bins = xr.Dataset(data_vars = dict(
    bin_depth = (['pt','vert_idx'],np.zeros((n_points,n_vert_bins))),
    bin_width = (['pt'],w_stats),
    bin_elev = (['pt'],z_stats)),
    coords=dict(
        point=(['pt'],['bottom','middle','top']),
        vert_idx=range(n_vert_bins)
        )
    )

bin_depths = np.zeros((n_points,n_vert_bins))
#fill vertical bin heights based on ice thickness
for g in range(n_points):
    #get ice thickness of current point
    pt_h = h_stats[g]
    if option_bin==0:
        hs = [0.1,.25,.5,.75,1,2,5,10,20,pt_h-39.6]
        bin_depths[g,:] = hs
    else:
        c = opt.fsolve(lambda c: pt_h-np.sum(np.exp(np.arange(n_vert_bins)*c)),10)
        bin_depths[g,:] = np.exp(c*range(1,n_vert_bins))
vert_bins['bin_depth'] = (['pt','vert_idx'],bin_depths)

#set bin content as a constant snow, firn or ice (s,f,i)
content_arr = np.empty((n_points,n_vert_bins),dtype=str)
content_arr[0,:] = ['i']*n_vert_bins
content_arr[1,:] = ['f']*round(n_vert_bins*.3)+['i']*round(n_vert_bins*.7)
content_arr[2,:] = ['s']*round(n_vert_bins*.2)+['f']*round(n_vert_bins*.2)+['i']*round(n_vert_bins*.6)
vert_bins['bin_content'] = (['pt','vert_idx'],content_arr)

# DATES TABLE
startyr = 1980
endyr = 2021
dates_table = modelsetup.datesmodelrun(startyear=startyr, endyear=endyr, spinupyears=pygem_prms.gcm_spinupyears,
            option_wateryear=pygem_prms.gcm_wateryear)

# CLIMATE DATA
gcm = class_climate.GCM(name='ERA5-hourly')
gcm_temp, gcm_hours = gcm.importGCMvarnearestneighbor_xarray(gcm.temp_fn, gcm.temp_vn, main_glac_rgi, dates_table)
gcm_prec, gcm_hours = gcm.importGCMvarnearestneighbor_xarray(gcm.prec_fn, gcm.prec_vn, main_glac_rgi,dates_table)
gcm_dtemp, gcm_hours = gcm.importGCMvarnearestneighbor_xarray(gcm.dtemp_fn, gcm.dtemp_vn, main_glac_rgi,dates_table)
gcm_tcc, gcm_hours = gcm.importGCMvarnearestneighbor_xarray(gcm.tcc_fn, gcm.tcc_vn, main_glac_rgi, dates_table)
gcm_surfrad, gcm_hours = gcm.importGCMvarnearestneighbor_xarray(gcm.surfrad_fn, gcm.surfrad_vn, main_glac_rgi, dates_table) 
gcm_uwind, gcm_hours = gcm.importGCMvarnearestneighbor_xarray(gcm.uwind_fn, gcm.uwind_vn, main_glac_rgi, dates_table)                                                      
gcm_vwind, gcm_hours = gcm.importGCMvarnearestneighbor_xarray(gcm.vwind_fn, gcm.vwind_vn, main_glac_rgi, dates_table)

gcm_elev = gcm.importGCMfxnearestneighbor_xarray(gcm.elev_fn, gcm.elev_vn, main_glac_rgi)

#create dataset for variables that need to be adjusted
pt_climate_ds = xr.Dataset(data_vars = dict(
    bin_T = (['pt','time'],np.zeros((n_points,len(gcm_hours)))),
    bin_P = (['pt','time'],np.zeros((n_points,len(gcm_hours)))),
    bin_snow = (['pt','time'],np.zeros((n_points,len(gcm_hours)))),
    bin_elev = (vert_bins['bin_elev'])),
    coords=dict(
        point=(['pt'],geo_index),
        vert_idx=range(n_vert_bins),
        time=gcm_hours
        ),
    attrs=dict(description="Climate data adjusted for points in EB."))

# adjust bin temperature and precipitation using linear scaling
temp_adj = [list(gcm_temp[0] + pygem_prms.lapserate*(gcm_elev[0]-z)) for z in pt_climate_ds['bin_elev'].values]
prec_adj = [list(gcm_temp[0]*kp*(1+precgrad*(gcm_elev[0]-z))) for z in pt_climate_ds['bin_elev'].values]
pt_climate_ds['bin_T'] = (['pt','time'],list(temp_adj))
pt_climate_ds['bin_P'] = (['pt','time'],list(prec_adj))
# currently setting solid precipitation using threshold value
pt_climate_ds['bin_snow'] = (['pt','time'],np.where(np.array(temp_adj)<2,1,0))

climate_ds = xr.Dataset(data_vars = dict(
    dtemp = (['glacier','time'],gcm_dtemp),
    surfrad = (['glacier','time'],gcm_surfrad),
    tcc = (['glacier','time'],gcm_tcc),
    uwind = (['glacier','time'],gcm_uwind),
    vwind = (['glacier','time'],gcm_vwind),
),coords = dict(glacier=glac_no,time=gcm_hours))

