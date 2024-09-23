# Built-in libraries
import os
import time
import copy
# External libraries
import pandas as pd
from multiprocessing import Pool
# Internal libraries
import pygem_eb.input as eb_prms
import run_simulation_eb as sim
import pygem_eb.massbalance as mb

# Read command line args
args = sim.get_args()
n_processes = args.n_simultaneous_processes

# Define sets of parameters
params = {'k_snow':['Sturm','Douville','Jansson'],
          'a_ice':[0.2,0.4,0.6],
          'kw':[0.25,0.5,0.75,1]}

# Determine number of runs for each process
n_runs = 3 # three sites
for param in list(params.keys()):
    n_runs *= len(params[param])
n_runs_per_process = n_runs // n_processes
n_runs_with_extra = n_runs % n_processes

# Force some args
args.store_data = True              # Ensures output is stored
args.use_AWS = True                 # Use available AWS data
args.debug = False                  # Don't need debug prints
eb_prms.store_vars = ['MB','EB']    # Only store mass and energy balance results
args.startdate = pd.to_datetime('2000-04-20 00:00:00')
args.enddate = pd.to_datetime('2000-05-21 12:00:00')

# Initialize the model
climate = sim.initialize_model(args.glac_no[0],args)

# Parse list for inputs to Pool function
packed_vars = [[] for _ in range(n_processes)]
run_no = 0  # Counter for runs added to each set
set_no = 0  # Index for the parallel process
for k_snow in params['k_snow']:
    for a_ice in params['a_ice']:
        for kw in params['kw']:
            for site in ['AB','B','D']:
                args_run = copy.deepcopy(args)

                # Set parameters
                args_run.k_snow = k_snow
                args_run.a_ice = a_ice
                args_run.kw = kw
                args_run.site = site

                # Set identifying output filename
                args_run.out = f'kw{kw}_ksnow{k_snow}_aice{a_ice}_site{site}_'

                # Specify attributes for output file
                store_attrs = {'k_snow':str(k_snow),'a_ice':str(a_ice),'kw':str(kw)}

                # Check if moving to the next set
                n_runs_set = n_runs_per_process + (0 if set_no < n_runs_with_extra else 0)
                if run_no >= n_runs_set:
                    set_no += 1
                    run_no = 0

                # Set task ID for SNICAR input file
                args_run.task_id = set_no
            
                # Store model inputs
                packed_vars[set_no].append((args_run,climate,store_attrs))

                # Advance counter
                run_no += 1

def run_model_parallel(list_inputs):
    # Loop through the variable sets
    for inputs in list_inputs:
        # Unpack inputs
        args,climate,store_attrs = inputs
        
        # Check if model run should be performed
        if not os.path.exists(eb_prms.output_filepath + args.out + '0.nc'):
            # Start timer
            start_time = time.time()

            # Run the model
            massbal = mb.massBalance(args,climate)
            massbal.main()

            # Completed model run: end timer
            time_elapsed = time.time() - start_time

            # Store output
            massbal.output.add_vars()
            massbal.output.add_basic_attrs(args,time_elapsed,climate)
            massbal.output.add_attrs(store_attrs)
    return

with Pool(n_processes) as processes_pool:
    processes_pool.map(run_model_parallel,packed_vars)