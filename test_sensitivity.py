# Built-in libraries
import os
import time
import copy
import traceback
# External libraries
import pandas as pd
import xarray as xr
import pickle
from multiprocessing import Pool
# Internal libraries
import pygem_eb.input as eb_prms
import run_simulation_eb as sim
import pygem_eb.massbalance as mb
from objectives import *

# Read command line args
args = sim.get_args()
args.store_data = True
args.use_AWS = False
args.site = 'B'
eb_prms.store_vars = ['MB','EB','layers','climate']
args.startdate = pd.to_datetime('2024-04-18 00:00:00')
args.enddate = pd.to_datetime('2024-08-20 00:00:00')

# Create output directory
eb_prms.output_filepath = '/trace/group/rounce/cvwilson/Output/sensitivity/'

# Get climate
climate = sim.initialize_model(args.glac_no[0],args)

# Place for results
result_dict = {'base':{},'kp':{},'Boone_c5':{},'lapserate':{},
               'roughness_fresh_snow':{},'roughness_aged_snow':{},
               'roughness_ice':{},'albedo_ground':{},
               'ksp_BC':{},'ksp_OC':{},'ksp_dust':{}}

def model_run(name):
    # Name the run
    args.out = name
    print(name.split())
    var = name.split('_')[0].replace('-','_')
    percent = name.split('_')[-2]

    # Start timer
    start = time.time()

    # Initialize the mass balance / output
    massbal = mb.massBalance(args,climate)

    # Run the model
    massbal.main()

    # Store the mass balance
    ds = massbal.output.get_output()
    internal_acc = ds.isel(time=-2).cumrefreeze.values
    summer_mb = ds.accum + ds.refreeze - ds.melt - internal_acc
    # plt.plot(summer_mb.time, ds.accum.cumsum().values,label=name+percent)
    result_dict[var][percent] = summer_mb.sum().values
    # os.remove(eb_prms.output_filepath + args.out + '0.nc')

    # Print time
    timer = time.time() - start
    print(f'Time elapsed: {timer:.0f} seconds')

model_run('base_')

# SENSITIVITY
args.kp = 1
model_run('kp_-20_')
args.kp = 3
model_run('kp_+20_')
args.kp = 2

# plt.legend()
# plt.savefig('/trace/group/rounce/cvwilson/Output/TEST_FIG.png')
# assert 1==0
args.Boone_c5 = 0.018
model_run('Boone-c5_-20_')
args.Boone_c5 = 0.03
model_run('Boone-c5_+20_')
args.Boone_c5 = 0.022

eb_prms.lapserate = -0.0055
climate = sim.initialize_model(args.glac_no[0],args)
model_run('lapserate_-20_')
print(np.mean(climate.cds.temp.values))
eb_prms.lapserate = -0.007
climate = sim.initialize_model(args.glac_no[0],args)
model_run('lapserate_+20_')
print(np.mean(climate.cds.temp.values))
eb_prms.lapserate = -0.0065
climate = sim.initialize_model(args.glac_no[0],args)
print('back to normal lapse rate',np.mean(climate.cds.temp.values))

eb_prms.roughness_fresh_snow = 0.1
model_run('roughness-fresh-snow_-20_')
eb_prms.roughness_fresh_snow = 2
model_run('roughness-fresh-snow_+20_')
eb_prms.roughness_fresh_snow = 0.24

eb_prms.roughness_aged_snow = 5
model_run('roughness-aged-snow_-20_')
eb_prms.roughness_aged_snow = 20
model_run('roughness-aged-snow_+20_')
eb_prms.roughness_aged_snow = 10

eb_prms.roughness_ice = 10
model_run('roughness-ice_-20_')
eb_prms.roughness_ice = 50
model_run('roughness-ice_+20_')
eb_prms.roughness_ice = 20

eb_prms.albedo_ground = 0.1
model_run('albedo-ground_-20_')
eb_prms.albedo_ground = 0.3
model_run('albedo-ground_+20_')
eb_prms.albedo_ground = 0.2

eb_prms.ksp_BC = 0.1
model_run('ksp-BC_-20_')
eb_prms.ksp_BC = 1.2
model_run('ksp-BC_+20_')
eb_prms.ksp_BC = 0.9

eb_prms.ksp_OC = 0.1
model_run('ksp-OC_-20_')
eb_prms.ksp_OC = 1.2
model_run('ksp-OC_+20_')
eb_prms.ksp_OC = 0.9

eb_prms.ksp_dust = 0.001
model_run('ksp-dust_-20_')
eb_prms.ksp_dust = 0.1
model_run('ksp-dust_+20_')

with open(eb_prms.output_filepath+'sensitivity_test.pkl', 'wb') as f:
    pickle.dump(result_dict, f)
    print('Done! Saved to ',eb_prms.output_filepath+'sensitivity_test.pkl')