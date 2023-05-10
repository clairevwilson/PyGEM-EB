# External libraries
import scipy.optimize as opt
import numpy as np
import xarray as xr
import pandas as pd
#import suncalc as solar
# Internal libraries
import pygem.pygem_input as pygem_prms
import pygem.oggm_compat as oggm
import pygem.pygem_modelsetup as modelsetup
import class_climate
import pygem.energybalance as eb

assert pygem_prms.glac_no not in ['01.00570'], 'EB model can currently only run Gulkana glacier'

# ===== GLACIER AND TIME PERIOD SETUP =====
glacier_table = modelsetup.selectglaciersrgitable(pygem_prms.glac_no,
                rgi_regionsO1=pygem_prms.rgi_regionsO1, rgi_regionsO2=pygem_prms.rgi_regionsO2,
                rgi_glac_number=pygem_prms.rgi_glac_number, include_landterm=pygem_prms.include_landterm,
                include_laketerm=pygem_prms.include_laketerm, include_tidewater=pygem_prms.include_tidewater)
dates_table = modelsetup.datesmodelrun(startyear=pygem_prms.gcm_startyear,endyear=pygem_prms.gcm_endyear, 
                                       spinupyears=pygem_prms.gcm_spinupyears,option_wateryear=pygem_prms.gcm_wateryear)


# ===== BEGIN TRIAL-ERA GULKANA SETUP =====
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
# ===== END TRIAL-ERA GULKANA SETUP =====

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

# ===== SET UP CLIMATE DATASET =====
climateds = xr.Dataset(data_vars = dict(
    bin_elev = ((['bin'],geometry['z'])),
    dtemp = (['time'],gcm_dtemp[0]),
    surfrad = (['time'],gcm_surfrad[0]),
    tcc = (['time'],gcm_tcc[0]),
    uwind = (['time'],gcm_uwind[0]),
    vwind = (['time'],gcm_vwind[0])),
    coords=dict(
        bin=(['bin'],geo_index),
        #glacier=(['glacier'],glac_no),
        time=gcm_hours
        ),
    attrs=dict(description="Climate data adjusted for points in EB."))

#initialize variables to be adjusted
temp_adj = np.zeros((n_points,len(gcm_hours)))
prec_adj = np.zeros((n_points,len(gcm_hours)))
sp_adj = np.zeros((n_points,len(gcm_hours)))
rh_adj = np.zeros((n_points,len(gcm_hours)))

# define function to calculate vapor pressure (needed for RH)
e_func = lambda T: 6.1078*np.exp(17.1*T/(235+T)) #vapor pressure in hPa

#loop through each elevation bin and adjust climate variables by lapse rate/barometric law
for i in range(len(climateds['bin_elev'].values)):
    z = climateds['bin_elev'].values[i]
    temp_adj[i,:] = gcm_temp + pygem_prms.lapserate*(gcm_elev-z)
    prec_adj[i,:] = gcm_prec*pygem_prms.kp*(1+pygem_prms.precgrad*(gcm_elev-z))
    sp_adj[i,:] = gcm_sp*np.power((gcm_temp + pygem_prms.lapserate*(gcm_elev-z)+273.15)/(gcm_temp+273.15),
                           -pygem_prms.gravity*pygem_prms.molarmass_air/(pygem_prms.R_gas*pygem_prms.lapserate))
    rh_adj[i,:] = e_func(temp_adj[i,:])/e_func(gcm_dtemp)

climateds = climateds.assign(bin_temp = (['bin','time'],temp_adj))
climateds = climateds.assign(bin_prec = (['bin','time'],prec_adj))
climateds = climateds.assign(bin_sp = (['bin','time'],sp_adj))
climateds = climateds.assign(bin_rh = (['bin','time'],rh_adj))
climateds = climateds.assign(bin_snow = (['bin','time'],np.where(temp_adj<(pygem_prms.tsnow_threshold+273),1,0)))
print('! GCM is giving weird values because of high elevation. Check!')
print('! Using constant (not calibrated) kp and lapserate')

# ===== RUN ENERGY BALANCE =====
#extremely arbitrary values for temperature and density profile
tempprof_arb = np.array([[0,2],[1,-10],[5,-1],[10,0]])
densprof_arb = np.array([[0,1.005],[1,5],[5,8],[10,10]])
meltModel = eb.meltProfile(climateds,10,tempprof_arb,densprof_arb)
#meltModel.EnergyMassBalance()