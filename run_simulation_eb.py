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
dates_table = pd.DataFrame({'date' : pd.date_range(eb_prms.startdate,eb_prms.enddate,freq='h')})
# Extract attributes for dates_table
dates_table['year'] = dates_table['date'].dt.year
dates_table['month'] = dates_table['date'].dt.month
dates_table['day'] = dates_table['date'].dt.day
dates_table['hour'] = dates_table['date'].dt.hour
dates_table['daysinmonth'] = dates_table['date'].dt.daysinmonth
dates_table['timestep'] = np.arange(len(dates_table['date']))
# Set date as index
dates_table.set_index('timestep', inplace=True)

# Remove leap year days if user selected this with option_leapyear
if eb_prms.option_leapyear == 0:
    # First, change 'daysinmonth' number
    mask1 = dates_table['daysinmonth'] == 29
    dates_table.loc[mask1,'daysinmonth'] = 28
    # Next, remove the 29th days from the dates
    mask2 = ((dates_table['month'] == 2) & (dates_table['day'] == 29))
    dates_table.drop(dates_table[mask2].index, inplace=True)
    dates_table['timestep'] = np.arange(len(dates_table['date']))
    dates_table.set_index('timestep', inplace=True)

# Add column for water year
# Water year for northern hemisphere using USGS definition (October 1 - September 30th),
# e.g., water year for 2000 is from October 1, 1999 - September 30, 2000
dates_table['wateryear'] = dates_table['year']
for step in range(dates_table.shape[0]):
    if dates_table.loc[step,'month'] >= 10:
        dates_table.loc[step,'wateryear'] = dates_table.loc[step,'year'] + 1

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
        time=(['time'],gcm_hours)
        ))

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

# Read in data for initial temperatures and densities
# temp_prof = pd.read_csv(eb_prms.initTemp_fp).to_numpy()[:,1:]
# density_prof = pd.read_csv(eb_prms.initDensity_fp).to_numpy()[:,1:]
temp_prof = np.array([[0,-30],[1,-10],[5,-8],[10,0]])
density_prof = np.array([[0,100],[1,300],[3,350],[8,600]])
layer_depths = [[1,0,20],[4,1,40],[5,2,40]]
#print(climateds.sel(time=gcm_hours[0])['bin_temp'])

# Set up files for storage
if eb_prms.store_data:
    time_to_store = pd.date_range(eb_prms.startdate,eb_prms.enddate,freq=eb_prms.storage_freq)
    zeros = np.zeros([len(time_to_store),eb_prms.n_bins,eb_prms.max_nlayers])
    all_variables = xr.Dataset(data_vars = dict(
            SWin = (['time','bin'],zeros[:,:,0],{'units':'W m-2'}),
            SWout = (['time','bin'],zeros[:,:,0],{'units':'W m-2'}),
            LWin = (['time','bin'],zeros[:,:,0],{'units':'W m-2'}),
            LWout = (['time','bin'],zeros[:,:,0],{'units':'W m-2'}),
            rain = (['time','bin'],zeros[:,:,0],{'units':'W m-2'}),
            sensible = (['time','bin'],zeros[:,:,0],{'units':'W m-2'}),
            latent = (['time','bin'],zeros[:,:,0],{'units':'W m-2'}),
            meltenergy = (['time','bin'],zeros[:,:,0],{'units':'W m-2'}),
            melt = (['time','bin'],zeros[:,:,0],{'units':'kg m-2'}),
            refreeze = (['time','bin'],zeros[:,:,0],{'units':'kg m-2'}),
            runoff = (['time','bin'],zeros[:,:,0],{'units':'m w.e.'}),
            accum = (['time','bin'],zeros[:,:,0],{'units':'m w.e.'}),
            airtemp = (['time','bin'],zeros[:,:,0],{'units':'C'}),
            surftemp = (['time','bin'],zeros[:,:,0],{'units':'C'}),
            snowtemp = (['time','bin','layer'],zeros,{'units':'C'}),
            snowwetmass = (['time','bin','layer'],zeros,{'units':'kg m-2'}),
            snowdrymass = (['time','bin','layer'],zeros,{'units':'kg m-2'}),
            snowdensity = (['time','bin','layer'],zeros,{'units':'kg m-3'})
            ),
            coords=dict(
                time=(['time'],time_to_store),
                bin = (['bin'],np.arange(eb_prms.n_bins)),
                layer=(['layer'],np.arange(eb_prms.max_nlayers))
                ))
    all_variables.to_netcdf(eb_prms.output_name+'.nc')

# ===== RUN ENERGY BALANCE =====
#loop through bins here so EB script is set up for only one bin (1D data)
for bin in np.arange(1,eb_prms.n_bins):
    # initialize layers
    initial_layers = eb_layers.Layers(temp_prof,density_prof,layer_depths[bin])
    # initialize variables to store from mass balance
    massbal = mb.massBalance(climateds)
    # check runtime of main function
    results = massbal.main(initial_layers,climateds,bin)
    print('Number of melted layers:',eb_prms.melt_counter)
    print('Number of split layers:',eb_prms.split_counter)
    print('Number of merged layers:',eb_prms.merge_counter)
    eb_prms.melt_counter = 0
    eb_prms.split_counter = 0
    eb_prms.merge_counter = 0

    print(massbal.monthly_output)
    if eb_prms.store_data:
        massbal.storeVars(bin)

    print('Success: moving onto bin',bin+1)