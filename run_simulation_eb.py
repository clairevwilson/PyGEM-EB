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

assert pygem_prms.glac_no not in ['01.00570'], 'EB model can currently only run Gulkana glacier'

# ===== GLACIER AND TIME PERIOD SETUP =====
glacier_table = modelsetup.selectglaciersrgitable(pygem_prms.glac_no,
                rgi_regionsO1=pygem_prms.rgi_regionsO1, rgi_regionsO2=pygem_prms.rgi_regionsO2,
                rgi_glac_number=pygem_prms.rgi_glac_number, include_landterm=pygem_prms.include_landterm,
                include_laketerm=pygem_prms.include_laketerm, include_tidewater=pygem_prms.include_tidewater)
dates_table = modelsetup.datesmodelrun(startyear=pygem_prms.gcm_startyear,endyear=pygem_prms.gcm_endyear, 
                                       spinupyears=pygem_prms.gcm_spinupyears,option_wateryear=pygem_prms.gcm_wateryear)

# WHOLE BUNCHA CONSTANTS
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
#---
kp = 1                              # precipitation factor [-] (referred to as k_p in Radic etal 2013; c_prec in HH2015)
tbias = 5                           # temperature bias [deg C]
ddfsnow = 0.0041                    # degree-day factor of snow [m w.e. d-1 degC-1]
ddfsnow_iceratio = 0.7              # Ratio degree-day factor snow snow to ice
ddfice = ddfsnow / ddfsnow_iceratio # degree-day factor of ice [m w.e. d-1 degC-1]
precgrad = 0.0001                   # precipitation gradient on glacier [m-1]
lapserate = -0.0065                 # temperature lapse rate for both gcm to glacier and on glacier between elevation bins [K m-1]
tsnow_threshold = 1                 # temperature threshold for snow [deg C] (HH2015 used 1.5 degC +/- 1 degC)
calving_k = 0.7                     # frontal ablation rate [yr-1]

# READ GULKANA ELEVATIONS FROM OGGM GDIRS
#this step is specific to the EB for three points on Gulkana
#to generalize, need a variable geometry containing zwh for the points for the EB
gdir = oggm.single_flowline_glacier_directory(pygem_prms.glac_no[0], logging_level='CRITICAL')
fls = oggm.get_glacier_zwh(gdir)
fls = fls.iloc[np.nonzero(fls['h'].to_numpy())] #filter out zero bins to get only initial glacier volume
z_stats = np.array([np.min(fls['z']),np.median(fls['z']),np.max(fls['z'])])

#setup three points at minimum, median and maximum elevation band from OGGM
median_index = np.where(fls['z']==z_stats[1])[0][0]
w_stats = np.array([fls['w'][len(fls)-1],fls['w'][median_index],fls['w'][0]])
h_stats = np.array([fls['h'][len(fls)-1],fls['h'][median_index],fls['h'][0]])
geo_index = ['Bottom','Middle','Top']
geometry = pd.DataFrame({'z':z_stats,'w':w_stats,'h':h_stats},index=geo_index)
n_points = len(geo_index)
# END GULKANA-SPECIFIC SECTION

# ===== LOAD CLIMATE DATA =====
gcm = class_climate.GCM(name='ERA5-hourly')
gcm_prec, gcm_hours = gcm.importGCMvarnearestneighbor_xarray(gcm.prec_fn, gcm.prec_vn, glacier_table,dates_table)
gcm_temp, gcm_hours = gcm.importGCMvarnearestneighbor_xarray(gcm.temp_fn, gcm.temp_vn, glacier_table,dates_table)
gcm_dtemp, gcm_hours = gcm.importGCMvarnearestneighbor_xarray(gcm.dtemp_fn, gcm.dtemp_vn, glacier_table,dates_table)
gcm_sp, gcm_hours = gcm.importGCMvarnearestneighbor_xarray(gcm.press_fn, gcm.press_vn, glacier_table,dates_table)
gcm_tcc, gcm_hours = gcm.importGCMvarnearestneighbor_xarray(gcm.tcc_fn, gcm.tcc_vn, glacier_table,dates_table)
gcm_surfrad, gcm_hours = gcm.importGCMvarnearestneighbor_xarray(gcm.surfrad_fn, gcm.surfrad_vn, glacier_table,dates_table) 
gcm_uwind, gcm_hours = gcm.importGCMvarnearestneighbor_xarray(gcm.uwind_fn, gcm.uwind_vn, glacier_table,dates_table)                                                      
gcm_vwind, gcm_hours = gcm.importGCMvarnearestneighbor_xarray(gcm.vwind_fn, gcm.vwind_vn, glacier_table,dates_table)

gcm_elev = gcm.importGCMfxnearestneighbor_xarray(gcm.elev_fn, gcm.elev_vn, glacier_table)

# ===== SET UP CLIMATE DATASETS =====
#create dataset to store variables that need to be downscaled by elevation
#if we want to be able to run multiple glaciers at once, this will need to be updated to add 'glacier' as a coordinate
climateds = xr.Dataset(data_vars = dict(
    pt_elev = ((['pt'],geometry['z'])),
    dtemp = (['time'],gcm_dtemp[0]),
    surfrad = (['time'],gcm_surfrad[0]),
    tcc = (['time'],gcm_tcc[0]),
    uwind = (['time'],gcm_uwind[0]),
    vwind = (['time'],gcm_vwind[0])),
    coords=dict(
        pt=(['pt'],geo_index),
        #glacier=(['glacier'],glac_no),
        time=gcm_hours
        ),
    attrs=dict(description="Climate data adjusted for points in EB."))

#initialize variables to be adjusted
temp_adj = np.zeros((n_points,len(gcm_hours)))
prec_adj = np.zeros((n_points,len(gcm_hours)))
sp_adj = np.zeros((n_points,len(gcm_hours)))
rh_adj = np.zeros((n_points,len(gcm_hours)))

# define function to calculate RH from temp and dewpoint temp
e_func = lambda T: 6.1078*np.exp(17.1*T/(235+T)) #vapor pressure in hPa

#loop through each elevation bin and adjust climate variables by lapse rate/barometric law
for i in range(len(climateds['pt_elev'].values)):
    z = climateds['pt_elev'].values[i]
    temp_adj[i,:] = gcm_temp + pygem_prms.lapserate*(gcm_elev-z)
    prec_adj[i,:] = gcm_prec*kp*(1+precgrad*(gcm_elev-z))
    sp_adj[i,:] = np.power(gcm_sp*(gcm_temp + pygem_prms.lapserate*(gcm_elev-z))/gcm_temp,
                           -gravity*molarmass_air/(R_gas*pygem_prms.lapserate))
    rh_adj[i,:] = e_func(temp_adj[i,:])/e_func(gcm_dtemp)

climateds = climateds.assign(pt_temp = (['pt','time'],temp_adj))
climateds = climateds.assign(pt_prec = (['pt','time'],prec_adj))
climateds = climateds.assign(pt_sp = (['pt','time'],sp_adj))
climateds = climateds.assign(pt_rh = (['pt','time'],rh_adj))
climateds = climateds.assign(pt_snow = (['pt','time'],np.where(temp_adj<(tsnow_threshold+273),1,0)))

# ===== RUN ENERGY BALANCE =====
