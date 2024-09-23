# Built-in libraries
import os
import time
# External libraries
import pandas as pd
# Internal libraries
import pygem_eb.input as eb_prms
import run_simulation_eb as sim
import pygem_eb.massbalance as mb

# Start timer
start_time = time.time()

# Read command line args
args = sim.get_args()

# Force some args
args.store_data = True              # Ensures output is stored
args.use_AWS = True                 # Use available AWS data
args.debug = False                  # Don't need debug prints
eb_prms.store_vars = ['MB','EB']    # Only store mass and energy balance results

# Date range
args.startdate = pd.to_datetime('2000-04-20 00:00:00')
args.enddate = pd.to_datetime('2022-05-21 12:00:00')

# Get parameters
k_snow = args.k_snow
a_ice = args.a_ice
kw = args.kw
site = args.site
assert site != 'AWS', 'add flag for site'

# Specify output filename
args.out = f'kw_{kw}_site{site}_'
out = eb_prms.output_filepath + args.out + '0.nc'

if not os.path.exists(f'{eb_prms.output_name}0.nc'):
    # initialize the model
    climate = sim.initialize_model(args.glac_no[0],args)

    # specify attributes for output file
    store_attrs = {'k_snow':str(k_snow),'a_ice':str(a_ice),'kw':str(kw)}
    print('Running model with:')
    print(f'    k_snow:{k_snow}    a_ice:{a_ice}     kw:{kw}')
    
    # run the model
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

print(f'finished job {args.task_id}')