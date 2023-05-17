# Built-in libraries
import os
# External libraries
import numpy as np
import pandas as pd

#%% ===== MODEL SETUP DIRECTORY =====
main_directory = os.getcwd()
# Output directory
output_filepath = main_directory + '/../Output/'
output_sim_fp = output_filepath + 'simulations/'
model_run_date = pd.Timestamp.today()

#%% ===== GLACIER SELECTION =====
rgi_regionsO1 = [1]                 # 1st order region number (RGI V6.0)
rgi_regionsO2 = [2]                 # 2nd order region number (RGI V6.0)

# RGI glacier number (RGI V6.0)
#  Three options: (1) use glacier numbers for a given region (or 'all'), must have glac_no set to None
#                 (2) glac_no is not None, e.g., ['1.00001', 13.0001'], overrides rgi_glac_number
#                 (3) use one of the functions from  utils._funcs_selectglaciers
rgi_glac_number = 'all'
glac_no = ['01.00570']

# Types of glaciers to include (True) or exclude (False)
include_landterm = True                # Switch to include land-terminating glaciers
include_laketerm = True                # Switch to include lake-terminating glaciers
include_tidewater = True               # Switch to include tidewater glaciers
ignore_calving = False                 # Switch to ignore calving and treat tidewater glaciers as land-terminating

oggm_base_url = 'https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.4/L1-L2_files/elev_bands/'
logging_level = 'DEBUG' # DEBUG, INFO, WARNING, ERROR, WORKFLOW, CRITICAL (recommended WORKFLOW)

#%% ===== CLIMATE DATA ===== 
# Reference period runs (runs up to present)
ref_gcm_name = 'ERA5-hourly'        # reference climate dataset
ref_startyear = 1980                # first year of model run (reference dataset)
ref_endyear = 2019                  # last year of model run (reference dataset)
ref_wateryear = 'calendar'          # options for years: 'calendar', 'hydro', 'custom'
ref_spinupyears = 0                 # spin up years
if ref_spinupyears > 0:
    assert 0==1, 'Code needs to be tested to ensure spinup years are correctly accounted for in output files'

# This is where the simulation runs climate data will be set up once we're there
# Simulation runs (refers to period of simulation and needed separately from reference year to account for bias adjustments)
gcm_startyear = 1980            # first year of model run (simulation dataset)
gcm_endyear = 2019              # last year of model run (simulation dataset)
gcm_wateryear = 'calendar'      # options for years: 'calendar', 'hydro', 'custom'
gcm_spinupyears = 0             # spin up years for simulation (output not set up for spinup years at present)
# constantarea_years = 0          # number of years to not let the area or volume change
# if gcm_spinupyears > 0:
#     assert 0==1, 'Code needs to be tested to enure spinup years are correctly accounted for in output files'

#%% MODEL OPTIONS
# Initialization
option_initWater = 'zero_w0'            # 'zero_w0' or 'initial_w0'
option_initTemp = 'piecewise'           # 'piecewise' or 'interp'
option_initDensity = 'piecewise'        # 'piecewise' or 'interp'
# option_start_season = 'acc_end'         # 'acc_end' (end of accumulation), 'abl_end' (end of ablation) or 'other'

# Simulation options
dt = 3600/3         # Time resolution in [s], should be integer multiple of 3600s
method_turbulent = 'MO-similarity'  # 'MO-similarity' or *****
# option_SW
# option_LW

# Albedo switches
switch_snow = 0             # 0 to turn off fresh snow feedback; 1 to include it
switch_melt = 0
switch_LAPs = 0

#%% MODEL PROPERTIES THAT MAY NEED TO BE ADJUSTED
precgrad = 0.0001           # precipitation gradient on glacier [m-1]
lapserate = -0.0065         # temperature lapse rate for both gcm to glacier and on glacier between elevation bins [K m-1]
tsnow_threshold = 1         # Threshold to consider freezing
kp = 1                      # precipitation factor [-] 
temp_temp = 0               # temperature of temperate ice in Celsius

#%% MODEL PROPERTIES
density_ice = 900           # Density of ice [kg m-3] (or Gt / 1000 km3)
density_water = 1000        # Density of water [kg m-3]
k_ice = 2.33                # Thermal conductivity of ice [J s-1 K-1 m-1] recall (W = J s-1)
k_air = 0.023               # Thermal conductivity of air [J s-1 K-1 m-1] (Mellor, 1997)
Lh_rf = 333550              # Latent heat of fusion [J kg-1]
gravity = 9.81              # Gravity [m s-2]
pressure_std = 101325       # Standard pressure [Pa]
temp_std = 288.15           # Standard temperature [K]
R_gas = 8.3144598           # Universal gas constant [J mol-1 K-1]
molarmass_air = 0.0289644   # Molar mass of Earth's air [kg mol-1]
Cp_water = 4184             # Isobaric heat capacity of water [J kg-1 K-1]
Cp_air = 1005               # Isobaric heat capacity of air [J kg-1 K-1]
Cp_ice = 2050               # Isobaric heat capacity of ice [J kg-1 K-1]
Lv_evap = 2.514e6           # latent heat of evaporation [J kg-1]
Lv_sub = 2.849e6            # latent heat of sublimation [J kg-1]
karman = 0.4                # von Karman's constant
density_std = 1.225         # air density at sea level [kg m^-3]
albedo_fresh_snow = 0.85    # albedo of fresh snow [-] (Moelg et al. 2012, TC)
albedo_firn = 0.55          # albedo of firn [-] (Moelg et al. 2012, TC)
albedo_ice = 0.3            # albedo of ice [-] (Moelg et al. 2012, TC)
dz_toplayer = 0.05          # thickness of the uppermost bin [m]
layer_growth = 0.5          # rate of exponential growth of bin size
sigma_SB = 5.67037e-8       # Stefan-Boltzmann constant [W m-2 K-4]