# Built-in libraries
import os
import time
import copy
# External libraries
import pandas as pd
import xarray as xr
from multiprocessing import Pool
# Internal libraries
import pygem_eb.input as eb_prms
import run_simulation_eb as sim
import pygem_eb.massbalance as mb
from objectives import *

# OPTIONS
run_type = '2024'   # 'long' or '2024'
date = '12_06'
idx = '0'

# Create output directory
if 'trace' in eb_prms.machine:
    eb_prms.output_filepath = '/trace/group/rounce/cvwilson/Output/'
out_fp = f'{date}_{idx}/'

# Load in failed dataset
missing_fn = eb_prms.output_filepath + out_fp + 'missing.txt'
failed = np.genfromtxt(missing_fn,delimiter=',',dtype=str)
if len(failed) < 1:
    print('Successfully finished all runs!')
    quit()
if '__iter__' not in failed[0]:
    failed = [failed]

# Define sites
if run_type == '2024':
    sites = ['AB','ABB','B','BD','D','T'] # 
else:
    sites = ['AU'] # 'A','AU','B','D'

# Read command line args
args = sim.get_args()
n_processes = args.n_simultaneous_processes

# Determine number of runs for each process
n_runs = len(failed)
print(f'Beginning {n_runs} model runs on {n_processes} CPUs')

# Check if we need multiple (serial) runs per (parallel) set
if n_runs <= n_processes:
    n_runs_per_process = 0
    n_process_with_extra = 0
else:
    n_runs_per_process = n_runs // n_processes  # Base number of runs per CPU
    n_process_with_extra = n_runs % n_processes    # Number of CPUs with one extra run

# Force some args
args.store_data = True              # Ensures output is stored
if run_type == '2024': # Short AWS run
    args.use_AWS = True
    eb_prms.AWS_fn = '../climate_data/AWS/Preprocessed/gulkana2024.csv'
    eb_prms.store_vars = ['MB','EB','layers','temp']
    args.startdate = pd.to_datetime('2024-04-18 00:00:00')
    args.enddate = pd.to_datetime('2024-08-20 00:00:00')
else: # Long MERRA-2 run
    args.use_AWS = False
    eb_prms.store_vars = ['MB','layers']     # Only store mass balance results
    args.startdate = pd.to_datetime('2000-04-20 00:00:00')
    args.enddate = pd.to_datetime('2024-08-20 00:00:00')

# Parse list for inputs to Pool function
packed_vars = [[] for _ in range(n_processes)]
run_no = 0  # Counter for runs added to each set
set_no = 0  # Index for the parallel process

# Loop through sites and initialize the model
site_dict = {}
for site in sites:
    site_dict[site] = {}
    # Initialize the model for the site
    args_site = copy.deepcopy(args)
    args_site.site = site

    # Special dates for low sites
    if site == 'A':
        args_site.enddate = pd.to_datetime('2015-05-20 00:00:00')
    elif site == 'AU':
        args_site.startdate = pd.to_datetime('2012-04-20 00:00:00')

    # Get the climate
    climate = sim.initialize_model(args.glac_no[0],args_site)

    # Store
    site_dict[site]['climate'] = climate
    site_dict[site]['args'] = args_site

# Loop through the failed runs
for param in failed:
    # Unpack the parameters 
    site, kw, c5, kp, out = param

    # Get args for the current run
    args_run = copy.deepcopy(site_dict[site]['args'])
    climate_run = copy.deepcopy(site_dict[site]['climate'])

    # Set parameters
    args_run.k_snow = 'VanDusen'
    args_run.kw = kw
    args_run.site = site
    args_run.Boone_c5 = c5
    args_run.kp = kp

    # Set identifying output filename
    args_run.out = out

    # Specify attributes for output file
    store_attrs = {'kw':kw,'c5':c5,'kp':kp,'site':site}

    # Set task ID for SNICAR input file
    args_run.task_id = set_no
    args_run.run_id = run_no

    # Store model inputs
    packed_vars[set_no].append((args_run,climate_run,store_attrs))

    # Check if moving to the next set of runs
    n_runs_set = n_runs_per_process + (1 if set_no < n_process_with_extra else 0)
    if run_no == n_runs_set - 1:
        set_no += 1
        run_no = -1

    # Advance counter
    run_no += 1

def run_model_parallel(list_inputs):
    global outdict
    # Loop through the variable sets
    for inputs in list_inputs:
        # Unpack inputs
        args,climate,store_attrs = inputs

        try:
            # Start timer
            start_time = time.time()

            # Initialize the mass balance / output
            massbal = mb.massBalance(args,climate)

            # Add attributes to output file in case it crashes
            massbal.output.add_attrs(store_attrs)

            # Run the model
            massbal.main()

            # End timer
            time_elapsed = time.time() - start_time

            # Store output
            massbal.output.add_vars()
            massbal.output.add_basic_attrs(args,time_elapsed,climate)
        except:
            print('Failed site',args.site,'with kw =',args.kw,'c5 =',args.Boone_c5,'kp =',args.kp)
            print('Removing:',args.out + '0.nc')
            os.remove(eb_prms.output_filepath + args.out + '0.nc')
            print()

        # If succeeded, process and save only the stats
        if os.path.exists(eb_prms.output_filepath + args.out + '0.nc'):
            with xr.open_dataset(eb_prms.output_filepath + args.out + '0.nc') as dataset:
                ds = dataset.load()
                if run_type == 'long':
                    # seasonal mass balance
                    winter_MAE,summer_MAE = seasonal_mass_balance(ds,method='MAE')
                    winter_ME,summer_ME = seasonal_mass_balance(ds,method='ME')
                    results = {'winter_MAE':winter_MAE,'summer_MAE':summer_MAE,
                            'winter_ME':winter_ME,'summer_ME':summer_ME}

                    # snowpits
                    for method in ['MAE','ME']:
                        snowpit_dict = snowpits(ds,method=method)
                        for var in snowpit_dict:
                            results[var] = snowpit_dict[var]

                elif run_type == '2024':
                    MAE = cumulative_mass_balance(ds,method='MAE')
                    ME = cumulative_mass_balance(ds,method='MAE')
                    results = {'MAE':MAE,'ME':ME}

                # Store the attributes in the results dict
                for attr in ds.attrs:
                    results[attr] = ds.attrs[attr]

            # Pickle the dict
            stats_fn = eb_prms.output_filepath + args.out + '0.pkl'
            with open(stats_fn, 'wb') as file:
                pickle.dump(results,file)

            # Remove the .nc
            os.remove(eb_prms.output_filepath + args.out + '0.nc')
    return

# Run model in parallel
with Pool(n_processes) as processes_pool:
    processes_pool.map(run_model_parallel,packed_vars)

missing = []
for run in failed:
    fn = run[4]
    if not os.path.exists(eb_prms.output_filepath + fn + '0.pkl'):
        missing.append(run)

# Store missing as .txt
np.savetxt(missing_fn,np.array(missing),fmt='%s',delimiter=',')
print(f'Finished param_set_parallel, saved missing to {missing_fn}')