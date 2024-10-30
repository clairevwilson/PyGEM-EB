# Built-in libraries
import os
import socket
# External libraries
import numpy as np
import pandas as pd
import xarray as xr
# import pygem.oggm_compat as oggm

debug=False           # Print monthly status?
store_data=False      # Save file?
new_file=True         # Make new file or write to scratch?

# ========== USER OPTIONS ========== 
glac_no = ['01.00570']  # List of RGI glacier IDs
timezone = pd.Timedelta(hours=-8)   # local GMT time zone
use_AWS = False          # Use AWS data? (or just reanalysis)

# ========== GLACIER INFO ========== 
glac_props = {'01.00570':{'name':'Gulkana',
                            'site_elev':1693,
                            'AWS_fn':'Preprocessed/gulkana2024.csv'}, 
            '01.01104':{'name':'Lemon Creek',
                            'site_elev':1285,
                            'AWS_fn':'LemonCreek1285_hourly.csv'},
            '01.00709':{'name':'Mendenhall',
                            'site_elev':1316},
            '01.01390':{'name':'Taku',
                            'site_elev':1166},
            '01.00704':{'name':'Gilkey',
                            'site_elev':1459},
            '01.16195':{'name':'South',
                            'site_elev':2280,
                            'AWS_fn':'Preprocessed/south/south2280_2008.csv'},
            '08.00213':{'name':'Storglaciaren',
                            'AWS_fn':'Storglaciaren/SITES_MET_TRS_SGL_dates_15MIN.csv'},
            '11.03674':{'name':'Saint-Sorlin',
                            'site_elev':2720,
                            'AWS_fn':'Preprocessed/saintsorlin/saintsorlin_hourly.csv'},
            '16.02444':{'name':'Artesonraju',
                            'site_elev':4797,
                            'AWS_fn':'Preprocessed/artesonraju/Artesonraju_hourly.csv'}}

# WAYS OF MAKING BIN_ELEV
# dynamics = False
# if dynamics:
#     gdir = oggm.single_flowline_glacier_directory(glac_no[0], logging_level='CRITICAL') #,has_internet=False
#     all_fls = oggm.get_glacier_zwh(gdir)
#     fls = all_fls.iloc[np.nonzero(all_fls['h'].to_numpy())] # remove empty bins
#     bin_indices = np.linspace(len(fls.index)-1,0,n_bins,dtype=int)
#     bin_elev = fls.iloc[bin_indices]['z'].to_numpy()
#     bin_ice_depth = fls.iloc[bin_indices]['h'].to_numpy()
# bin_elev = np.array([1270,1385,1470,1585,1680,1779]) # From Takeuchi 2009
# bin_elev = np.array([1526,1693,1854])
# bin_ice_depth = np.ones(len(bin_elev)) * 200

if glac_no[0] in list(glac_props.keys()):
    elev = glac_props[glac_no[0]]['site_elev']
    site = 'AWS'
else:
    elev = 2000
    site = str(elev)
initial_snow_depth = 2.18
initial_firn_depth = 0
initial_ice_depth = 200

# ========== DIRECTORIES AND FILEPATHS ========== 
machine = socket.gethostname()
main_directory = os.getcwd()
output_filepath = main_directory + '/../Output/EB/'
output_sim_fp = output_filepath + 'simulations/'
model_run_date = str(pd.Timestamp.today()).replace('-','_')[0:10]
glac_name = glac_props[glac_no[0]]['name']

# Filepaths
glac_no_str = str(glac_no[0]).replace('.','_')
# Grain size evolution lookup table
grainsize_fp = main_directory + '/data/grainsize/drygrainsize(SSAin=60).nc'
# SNICAR inputs
snicar_input_fp = main_directory + '/biosnicar-py/biosnicar/inputs.yaml'
# Initial conditions
initial_temp_fp = main_directory + '/data/sample_initial_temp.csv'
initial_density_fp = main_directory + '/data/sample_initial_density.csv'
initial_grains_fp = main_directory + '/data/sample_initial_grains.csv'
initial_LAP_fp = main_directory + f'/data/sample_initial_laps.csv' # f'/../Data/Nagorski/May_Mend-2_BC.csv'
# Shading
shading_fp = main_directory + f'/shading/out/{glac_name}{site}_shade.csv'
# Output filepaths
albedo_out_fp = main_directory + '/../Output/EB/albedo.csv'
output_name = f'{glac_name}_{model_run_date}_'

# ========== CLIMATE AND TIME INPUTS ========== 
reanalysis = 'MERRA2' # 'MERRA2' (or 'ERA5-hourly' -- BROKEN)
temp_bias_adjust = True   # adjust MERRA-2 temperatures according to bias?
MERRA2_filetag = False    # False or string to follow 'MERRA2_VAR_' in MERRA2 filename
AWS_fp = main_directory + '/../climate_data/AWS/'
AWS_fn = AWS_fp+glac_props[glac_no[0]]['AWS_fn']
glac_name = glac_props[glac_no[0]]['name']
wind_ref_height = 10 if reanalysis in ['ERA5-hourly'] else 2
if use_AWS:
    assert os.path.exists(AWS_fn), 'Check AWS filepath or glac_no in input.py'

dates_from_data = False
if dates_from_data:
    cdf = pd.read_csv(AWS_fn,index_col=0)
    cdf = cdf.set_index(pd.to_datetime(cdf.index))
    if glac_no != ['01.00570']:
        elev = cdf['z'].iloc[0]
    startdate = pd.to_datetime(cdf.index[0])
    enddate = pd.to_datetime(cdf.index.to_numpy()[-1])
    if reanalysis == 'MERRA2' and startdate.minute != 30:
        startdate += pd.Timedelta(minutes=30)
        enddate -= pd.Timedelta(minutes=30)
else:
    startdate = pd.to_datetime('2024-04-20 00:00:00') 
    enddate = pd.to_datetime('2024-08-20 00:00:00')
    # enddate = pd.to_datetime('2019-04-25 23:00')
    # startdate = pd.to_datetime('2023-04-20 00:30')    # Gulkana AWS dates
    # enddate = pd.to_datetime('2023-08-10 00:30')
    # startdate = pd.to_datetime('2008-05-04 18:30')    # South dates
    # enddate = pd.to_datetime('2008-09-14 00:30')
    # startdate = pd.to_datetime('2016-05-11 00:30') # JIF sample dates
    # enddate = pd.to_datetime('2016-07-18 00:30')
    
# ========== MODEL OPTIONS ========== 
# INITIALIATION
initialize_temp = 'interpolate'     # 'interpolate' or 'ripe'
initialize_LAPs = 'clean'           # 'interpolate' or 'clean' 
surftemp_guess =  -10               # guess for surface temperature of first timestep
if 6 < startdate.month < 9:         # initialize without snow
    initial_snowdepth = 0

# OUTPUT
store_vars = ['MB','EB','temp','layers']  # Variables to store of the possible set: ['MB','EB','Temp','Layers']
store_bands = False     # Store spectral albedo .csv
store_climate = False   # Store climate dataset .nc

# TIMESTEP
dt = 3600                   # Model timestep [s]
daily_dt = 3600*24          # Seconds in a day [s]
dt_heateq = 3600/5          # Time resolution of heat eq [s], should be integer multiple of 3600s so data can be stored on the hour
end_summer = '2024-08-20'   # Date to consider the end of summer (year is irrelevant) (snow -> firn)

# METHODS
method_turbulent = 'BulkRichardson'     # 'MO-similarity' or 'BulkRichardson' 
method_heateq = 'Crank-Nicholson'       # 'Crank-Nicholson'
method_densification = 'Boone'          # 'Boone', 'HerronLangway', 'Kojima'
method_cooling = 'iterative'            # 'minimize' (slow) or 'iterative' (fast)
method_ground = 'MolgHardy'             # 'MolgHardy'
method_conductivity = 'Sturm'           # 'Sturm','Douville','Jansson','OstinAndersson','VanDusen'

# CONSTANT SWITCHES
constant_snowfall_density = False       # False or density in kg m-3
constant_freshgrainsize = 54.5          # False or grain size in um (54.5 is standard)
constant_drdry = False                  # False or dry metamorphism grain size growth rate [um s-1] (1e-4 seems reasonable)

# ALBEDO SWITCHES
switch_snow = 1             # 0 to turn off fresh snow feedback; 1 to include it
switch_melt = 2             # 0 to turn off melt feedback; 1 for simple degradation; 2 for grain size evolution
switch_LAPs = 1             # 0 to turn off LAPs; 1 to turn on
if switch_snow + switch_melt + switch_LAPs < 4:
    snow_on = 'ON' if switch_snow == 1 else 'OFF'
    melt_on = 'ON' if switch_melt == 2 else 'OFF'
    LAPs_on = 'ON' if switch_LAPs == 1 else 'OFF'
    if switch_melt == 1:
        LAPs_on = melt_on = 'ON (DECAY)'
    print(f'SWITCH RUN WITH SNOW {snow_on}, MELT {melt_on} and LAPs {LAPs_on}')
    output_name = f'{output_filepath}EB/{glac_name}_{model_run_date}_{switch_snow}{switch_melt}{switch_LAPs}'
include_LWC_SNICAR = False
grainshape_SNICAR = 0  # 0: sphere, 1: spheroid, 2: hexagonal plate, 3: koch snowflake, 4: hexagonal prisms

# ALBEDO BANDS
wvs = np.round(np.arange(0.2,5,0.01),2) # 480 bands used by SNICAR
band_indices = {}
for i in np.arange(0,480):
    band_indices['Band '+str(i)] = np.array([i])
grainsize_ds = xr.open_dataset(grainsize_fp)

# ========== PARAMETERS ==========
# site specific
sky_view = 0.936
kp = 3.2
# play with
dep_factor = 1              # multiplicative factor to adjust MERRA-2 deposition
wind_factor = 1             # multiplicative wind scaling factor
albedo_ice = 0.47           # albedo of ice [-] 
kcond_ice = 2.25            # thermal conductivity of ice
kcond_snow = 'Sturm'        # thermal conductivity of snow
Boone_c5 = 0.05             # m3 kg-1 (0.018)
firn_grainsize = 2000       # firn grain size in um
ice_grainsize = 5000        # ice grain size in um (placeholder -- unused)
dz_toplayer = 0.05          # Thickness of the uppermost layer [m]
layer_growth = 0.4          # Rate of exponential growth of layer size (smaller layer growth = more layers) recommend 0.3-.6
# leave
snow_threshold_low = 0      # lower threshold for linear snow-rain scaling [C]
snow_threshold_high = 1     # upper threshold for linear snow-rain scaling [C]
precgrad = 0.0001           # precipitation gradient on glacier [m-1]
lapserate = -0.0065         # temperature lapse rate for both gcm to glacier and on glacier between elevation bins [C m-1]
roughness_ice = 10          # surface roughness length for ice [mm] (Moelg et al. 2012, TC)
roughness_fresh_snow = 0.24 # surface roughness length for fresh snow [mm] (Moelg et al. 2012, TC)
roughness_aged_snow = 4     # surface roughness length for aged snow [mm]
roughness_firn = 4          # surface roughness length for firn [mm] (Moelg et al. 2012, TC)
roughness_aging_rate = 0.1  # rate in mm/day fresh --> aged snow (60 days from 0.24 to 4.0 => 0.06267)
albedo_TOD = [14]           # List of time(s) of day to calculate albedo [hr] 
initSSA = 80                # initial estimate of Specific Surface Area of fresh snowfall (interpolation tables)
BC_freshsnow = 0            # concentration of BC in fresh snow for initialization [kg m-3]
dust_freshsnow = 0          # concentration of dust in fresh snow for initilization [kg m-3]
ksp_BC = 0.5                # 0.1-0.2 meltwater scavenging efficiency of BC (from CLM5)
ksp_dust = 0.2              # 0.015 meltwater scavenging efficiency of dust (from CLM5)
# 1 kg m-3 = 1e6 ppb = ng g-1 = ug L-1
mb_threshold = 0.1          # Threshold to consider not conserving mass (kg m-2 = mm w.e.)


# ========== CONSTANTS ===========
# Discretization
max_nlayers = 50            # Maximum number of vertical layers allowed
max_dz = 1                  # Max layer height
# Boundary conditions
temp_temp = -2              # temperature of temperate ice [C]
temp_depth = 100            # depth of temperate ice [m]
# Physical properties of snow, ice, water and air
density_ice = 900           # Density of ice [kg m-3] (or Gt / 1000 km3)
density_water = 1000        # Density of water [kg m-3]
density_firn = 700          # Density threshold for firn
k_air = 0.023               # Thermal conductivity of air [W K-1 m-1] (Mellor, 1997)
Cp_water = 4184             # Isobaric heat capacity of water [J kg-1 K-1]
Cp_air = 1005               # Isobaric heat capacity of air [J kg-1 K-1]
Cp_ice = 2050               # Isobaric heat capacity of ice [J kg-1 K-1]
Lv_evap = 2514000           # latent heat of evaporation [J kg-1]
Lv_sub = 2849000            # latent heat of sublimation [J kg-1]
Lh_rf = 333550              # Latent heat of fusion of ice [J kg-1]
viscosity_snow = 3.7e7      # Viscosity of snow [Pa-s]
# Universal constants
gravity = 9.81              # Gravity [m s-2]
karman = 0.4                # von Karman's constant [-]
sigma_SB = 5.67037e-8       # Stefan-Boltzmann constant [W m-2 K-4]
# Ideal gas law
R_gas = 8.3144598           # Universal gas constant [J mol-1 K-1]
molarmass_air = 0.0289644   # Molar mass of Earth's air [kg mol-1]
pressure_std = 101325       # Standard pressure [Pa]
temp_std = 293.15           # Standard temperature [K]
density_std = 1.225         # Air density at sea level [kg m-3]
# Model parameterizations
albedo_deg_rate = 15        # Rate of exponential decay of albedo
wet_snow_C = 4.22e-13       # Constant for wet snow metamorphosis [m3 s-1]
rfz_grainsize = 1500        # Grainsize of refrozen snow [um]
Sr = 0.033                  # Fraction of irreducible water content for percolation [-]
diffuse_cloud_limit = 0.6   # Threshold to consider cloudy vs clear-sky in SNICAR [-]
# Constants for switch runs
average_grainsize = 1000    # Grainsize to treat as constant if switch_melt is 0 [um]
albedo_fresh_snow = 0.85    # Albedo of fresh snow [-] (Moelg et al. 2012, TC)
albedo_firn = 0.5           # Albedo of firn [-]
albedo_ground = 0.2         # Albedo of ground [-]
# MERRA-2: LAP binning
ratio_BC2_BCtot = 2.08      # Ratio to transform BC bin 2 deposition to total BC
ratio_DU3_DUtot = 3         # Ratio to transform dust bin 3 deposition to total dust
ratio_DU_bin1 = 0.0751      # Ratio to transform total dust to SNICAR Bin 1 (0.05-0.5um)
ratio_DU_bin2 = 0.20535     # " SNICAR Bin 2 (0.5-1.25um)
ratio_DU_bin3 = 0.481675    # " SNICAR Bin 3 (1.25-2.5um)
ratio_DU_bin4 = 0.203775    # " SNICAR Bin 4 (2.5-5um)
ratio_DU_bin5 = 0.034       # " SNICAR Bin 5 (5-50um)
# MERRA-2: temperature bias
temp_bias_slope = 0.72801   # Slope of linear regression of MERRA-2 --> AWS
temp_bias_intercept = 2.234 # Intercept of linear regression MERRA-2 --> AWS

# ========== OTHER PYGEM INPUTS ========== 
rgi_regionsO1 = [1]
rgi_regionsO2 = [2]
rgi_glac_number = 'all'

# Types of glaciers to include (True) or exclude (False)
include_landterm = True                # Switch to include land-terminating glaciers
include_laketerm = True                # Switch to include lake-terminating glaciers
include_tidewater = True               # Switch to include tidewater glaciers
ignore_calving = False                 # Switch to ignore calving and treat tidewater glaciers as land-terminating
oggm_base_url = 'https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.4/L1-L2_files/elev_bands/'
logging_level = 'DEBUG' # DEBUG, INFO, WARNING, ERROR, WORKFLOW, CRITICAL (recommended WORKFLOW)
option_leapyear = 1 # 0 to exclude leap years

# Reference period runs (runs up to present)
ref_startyear = 1980                # first year of model run (reference dataset)
ref_endyear = 2019                  # last year of model run (reference dataset)
ref_wateryear = 'calendar'          # options for years: 'calendar', 'hydro', 'custom'
ref_spinupyears = 0                 # spin up years

# This is where the simulation runs climate data will be set up once we're there
# Simulation runs (refers to period of simulation and needed separately from reference year to account for bias adjustments)
gcm_startyear = 1980            # first year of model run (simulation dataset)
gcm_endyear = 2019              # last year of model run (simulation dataset)
gcm_wateryear = 'calendar'      # options for years: 'calendar', 'hydro', 'custom'
gcm_spinupyears = 0             # spin up years for simulation (output not set up for spinup years at present)
# constantarea_years = 0          # number of years to not let the area or volume change
# if gcm_spinupyears > 0:
#     assert 0==1, 'Code needs to be tested to enure spinup years are correctly accounted for in output files'