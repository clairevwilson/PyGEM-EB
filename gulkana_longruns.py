# Built-in libraries
import time
import copy
# External libraries
import pandas as pd
from multiprocessing import Pool
# Internal libraries
import run_simulation_eb as sim
import pygem_eb.massbalance as mb
import pygem_eb.input as eb_prms

# User info
use_AWS = False
sites = ['A','AU','B','D'] # Sites to run in parallel 
# False or filename of parameters .csv for run, relative to PyGEM-EB/
params_fn = False # '../Output/params/11_26_best.csv'
run_date = str(pd.Timestamp.today()).replace('-','_')[:10]
n_runs_ahead = 5    # Step if you're going to run this script more than once

# Read command line args
args = sim.get_args()
args.startdate = '2000-04-20 00:00'
args.enddate = '2024-08-20 12:00'
args.store_data = True              # Ensures output is stored
args.glac_no = ['01.00570']
args.use_AWS = use_AWS
eb_prms.AWS_fn = eb_prms.AWS_fp + 'Preprocessed/gulkana_22yrs.csv'
if 'trace' in eb_prms.machine:
    eb_prms.output_filepath = '/trace/group/rounce/cvwilson/Output/'

# Determine number of runs for each process
n_processes = len(sites)
args.n_processes = n_processes

def pack_vars():
    # Parse list for inputs to Pool function
    packed_vars = [[] for _ in range(n_processes)]
    run_no = 0
    for site in sites:
        # Get current site args
        args_run = copy.deepcopy(args)
        args_run.site = site
        if site == 'A':
            args_run.enddate = pd.to_datetime('2015-05-20 00:00:00')
        elif site == 'AU':
            args_run.startdate = pd.to_datetime('2012-04-20 00:00:00')

        # Set parameters filename (relative to PyGEM-EB/)
        if params_fn:
            args_run.params_fn = params_fn
            params = pd.read_csv(params_fn,index_col=0)
            kp = params.loc['kp',args_run.site].astype(float)
            kw = params.loc['kw',args_run.site].astype(float)
            a_ice = params.loc['a_ice',args_run.site].astype(float)
            c5 = params.loc['Boone_c5',args_run.site].astype(float)
            # Command line args override params input
            if args_run.kp == eb_prms.kp:
                args_run.kp = kp
            if args_run.kw == eb_prms.wind_factor:
                args_run.kw = kw
            if args_run.a_ice == eb_prms.albedo_ice:
                args_run.a_ice = a_ice
            if args_run.Boone_c5 == eb_prms.Boone_c5:
                args_run.Boone_c5 = c5
            store_attrs = {'params_fn':params_fn,'site':site,
                        'kp':str(args_run.kp),'kw':str(args_run.kw),
                            'AWS':eb_prms.AWS_fn,'c5':str(c5)}
        else:
            store_attrs = {'site':site}

        # Output info
        args_run.out = f'Gulkana_{run_date}_long{site}_'

        # Set task ID for SNICAR input file
        args_run.task_id = run_no + n_runs_ahead*n_processes

        # Store model inputs
        climate = sim.initialize_model(args_run.glac_no[0],args_run)
        packed_vars[run_no].append((args_run,climate,store_attrs))

        # Advance counter
        run_no += 1
    return packed_vars

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
if __name__ == '__main__':
    packed_vars = pack_vars()
    with Pool(n_processes) as processes_pool:
        processes_pool.map(run_model_parallel,packed_vars)