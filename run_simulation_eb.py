# External libraries
import numpy as np
import xarray as xr
import pandas as pd
# Internal libraries
import pygem_eb.input as eb_prms
import pygem.oggm_compat as oggm
import pygem.pygem_modelsetup as modelsetup
import class_climate
import pygem_eb.massbalance as mb
import pygem_eb.layers as eb_layers

assert eb_prms.glac_no not in ['01.00570'], 'EB model can currently only run Gulkana glacier'

# ===== GLACIER AND TIME PERIOD SETUP =====
glacier_table = modelsetup.selectglaciersrgitable(eb_prms.glac_no,
                rgi_regionsO1=eb_prms.rgi_regionsO1, rgi_regionsO2=eb_prms.rgi_regionsO2,
                rgi_glac_number=eb_prms.rgi_glac_number, include_landterm=eb_prms.include_landterm,
                include_laketerm=eb_prms.include_laketerm, include_tidewater=eb_prms.include_tidewater)
dates_table = modelsetup.datesmodelrun(startyear=eb_prms.gcm_startyear,endyear=eb_prms.gcm_endyear, 
                                       spinupyears=eb_prms.gcm_spinupyears,option_wateryear=eb_prms.gcm_wateryear)


# ===== BEGIN TRIAL-ERA GULKANA SETUP =====
# READ GULKANA ELEVATIONS FROM OGGM GDIRS
#this step is specific to the EB for three points on Gulkana
#to generalize, need a variable geometry containing zwh for the points for the EB
gdir = oggm.single_flowline_glacier_directory(eb_prms.glac_no[0], logging_level='CRITICAL')
fls = oggm.get_glacier_zwh(gdir)
fls = fls.iloc[np.nonzero(fls['h'].to_numpy())] #filter out zero bins to get only initial glacier volume
z_stats = np.array([np.min(fls['z']),np.median(fls['z']),np.max(fls['z'])])

#setup three points at minimum, median and maximum elevation band from OGGM
median_index = np.where(fls['z']==z_stats[1])[0][0]
w_stats = np.array([fls['w'][len(fls)-1],fls['w'][median_index],fls['w'][0]])
h_stats = np.array([fls['h'][len(fls)-1],fls['h'][median_index],fls['h'][0]])
bin_name = ['Bottom','Middle','Top']
bin_idx = range(len(bin_name))
geometry = pd.DataFrame({'z':z_stats,'w':w_stats,'h':h_stats,'idx':bin_idx},index=bin_name)
n_points = len(bin_name)
# ===== END TRIAL-ERA GULKANA SETUP =====

# ===== LOAD CLIMATE DATA =====
gcm = class_climate.GCM(name=eb_prms.ref_gcm_name)
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
    bin_elev = (['bin'],z_stats,{'units':'m'}),
    bin_idx = (['bin'],bin_idx),
    surfrad = (['time'],gcm_surfrad[0],{'units':'J m-2'}),
    tcc = (['time'],gcm_tcc[0],{'units':'0-1'}),
    uwind = (['time'],gcm_uwind[0],{'units':'m s-1'}),
    vwind = (['time'],gcm_vwind[0],{'units':'m s-1'})),
    coords=dict(
        bin=(['bin'],bin_name),
        time=gcm_hours
        ),
    attrs=dict(description="Climate data adjusted for points in EB."))

#initialize variables to be adjusted
temp_adj = np.zeros((n_points,len(gcm_hours)))
prec_adj = np.zeros((n_points,len(gcm_hours)))
sp_adj = np.zeros((n_points,len(gcm_hours)))
rh_adj = np.zeros((n_points,len(gcm_hours)))
density_adj = np.zeros((n_points,len(gcm_hours)))
dtemp_adj = np.zeros((n_points,len(gcm_hours)))

# define function to calculate vapor pressure (needed for RH)
e_func = lambda T_C: 610.94*np.exp(17.625*T_C/(T_C+243.04))  #vapor pressure in Pa, T in Celsius

#loop through each elevation bin and adjust climate variables by lapse rate/barometric law
for idx,z in enumerate(climateds['bin_elev'].values):
    temp_adj[idx,:] = gcm_temp + eb_prms.lapserate*(z-gcm_elev)
    dtemp_adj[idx,:] = gcm_dtemp + eb_prms.lapserate_dew*(z-gcm_elev) - 273.15
    prec_adj[idx,:] = gcm_prec*eb_prms.kp*(1+eb_prms.precgrad*(z-gcm_elev))
    sp_adj[idx,:] = gcm_sp*np.power((gcm_temp + eb_prms.lapserate*(z-gcm_elev)+273.15)/(gcm_temp+273.15),
                           -eb_prms.gravity*eb_prms.molarmass_air/(eb_prms.R_gas*eb_prms.lapserate))
    rh_adj[idx,:] = e_func(dtemp_adj[idx,:]) / e_func(temp_adj[idx,:]) * 100
    density_adj[idx,:] = sp_adj[idx,:]/eb_prms.R_gas/(temp_adj[idx,:]+273.15)*eb_prms.molarmass_air

climateds = climateds.assign(bin_temp = (['bin','time'],temp_adj,{'units':'C'}))
climateds = climateds.assign(bin_dtemp = (['bin','time'],dtemp_adj,{'units':'C'}))
climateds = climateds.assign(bin_prec = (['bin','time'],prec_adj,{'units':'m'}))
climateds = climateds.assign(bin_sp = (['bin','time'],sp_adj,{'units':'Pa'}))
climateds = climateds.assign(bin_rh = (['bin','time'],rh_adj,{'units':'%'}))
climateds = climateds.assign(bin_density = (['bin','time'],density_adj,{'units':'kg m-3'}))
print('!! Using constant (not calibrated) kp and lapserate')

# ===== RUN ENERGY BALANCE =====
# Read in data for initial temperatures and densities
temp_prof = pd.read_csv(eb_prms.initTemp_fp).to_numpy()[:,1:]
density_prof = pd.read_csv(eb_prms.initDensity_fp).to_numpy()[:,1:]

#loop through bins here so EB script is set up for only one bin (1D data)
for bin in climateds['bin_idx'][0:1]:
    initial_layers = eb_layers.Layers(temp_prof,density_prof,[10,2,18])
    results = mb.main(initial_layers,climateds,bin,eb_prms.dt)