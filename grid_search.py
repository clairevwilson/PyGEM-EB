# Built-in libraries
import os, sys
import time
# External libraries
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
# Internal libraries
import pygem_eb.input as eb_prms
import run_simulation_eb as sim
import pygem_eb.massbalance as mb

# Start timer
start_time = time.time()

# Read command line args
parser = sim.get_args(parse=False)
args = parser.parse_args()

# Force some args
args.store_data = True
args.parallel = False
args.use_AWS = True
args.debug = False
args.startdate = pd.to_datetime('2000-04-20 00:00:00')
args.enddate = pd.to_datetime('2000-05-21 12:00:00')
print('CHANGE THE DATE BACK FROM 2000')

# Get parameters
k_ice = args.k_ice
k_snow = args.k_snow
a_ice = args.a_ice
site = args.site
assert site != 'AWS', 'add flag for site'

path_out = os.getcwd() + '/../Output'
eb_prms.output_name = f'{path_out}/EB/kice{k_ice}_ksnow{k_snow}_aice{a_ice}_site{site}_'

if not os.path.exists(f'{eb_prms.output_name}_0.nc'):
    # initialize the model
    climate = sim.initialize_model(args.glac_no[0],args)

    # specify attributes for output file
    store_attrs = {'k_ice':str(k_ice),'k_snow':str(k_snow),
                    'a_ice':str(a_ice)}
    
    # run the model
    print(f'Beginning run for site {site} with:')
    print(f'       kice: {k_ice}    ksnow: {k_snow}    aice: {a_ice}')
    massbal = mb.massBalance(args,climate)
    massbal.main()

    # completed model run: end timer
    time_elapsed = time.time() - start_time
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print(f'Total Time Elapsed: {time_elapsed:.1f} s')

    # store output
    massbal.output.add_vars()
    massbal.output.add_basic_attrs(args,time_elapsed,climate)
    massbal.output.add_attrs(store_attrs)
else:
    print('      already exists; skipping')