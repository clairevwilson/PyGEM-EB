# Built-in libraries
import os
import time
import copy
# External libraries
import pandas as pd
from multiprocessing import Pool
# Internal libraries
import run_simulation_eb as sim
import pygem_eb.massbalance as mb

# User info
sites = ['AB','B','D','T'] # Sites to run in parallel
# False or filename of parameters .csv for run, relative to PyGEM-EB/
params_fn = '../Output/params/10_08.csv'
run_date = str(pd.Timestamp.today()).replace('-','_')[:10]

# Read command line args
args = sim.get_args()
args.startdate = '2024-04-20 00:00'
args.enddate = '2024-08-20 00:00'
args.store_data = True              # Ensures output is stored
args.debug = False                  # Don't need debug prints
args.use_AWS = True

# Determine number of runs for each process
n_processes = len(sites)
args.n_processes = n_processes

# Parse list for inputs to Pool function
packed_vars = [[] for _ in range(n_processes)]
run_no = 0
for site in sites:
    # Get current site args
    args_run = copy.deepcopy(args)
    args_run.site = site

    # Set parameters filename (relative to PyGEM-EB/)
    if params_fn:
        args_run.params_fn = params_fn
        store_attrs = {'params_fn':params_fn,'site':site}
    else:
        store_attrs = {'site':site}

    # Output info
    args_run.out = f'Gulkana_{run_date}_{site}_'

    # Set task ID for SNICAR input file
    args_run.task_id = run_no

    # Store model inputs
    climate = sim.initialize_model(args_run.glac_no[0],args_run)
    packed_vars[run_no].append((args_run,climate,store_attrs))

    # Advance counter
    run_no += 1

def run_model_parallel(list_inputs):
    # Loop through the variable sets
    for inputs in list_inputs:
        # Unpack inputs
        args,climate,store_attrs = inputs
        
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

# Run model in parallel
with Pool(n_processes) as processes_pool:
    processes_pool.map(run_model_parallel,packed_vars)