"""
This script executes a grid search in parallel over
multiple parameters. These parameters can be specified
in the params dict below. It is set up to perform the
search over two parameters for ***Paper 1*** 
(Boone c5 densification parameter and kp precipitation
factor) but with minor edits more parameters can be added.

@author: clairevwilson
"""

# Built-in libraries
import os
import time
import copy
import traceback
import pickle
from multiprocessing import Pool
# External libraries
import pandas as pd
import xarray as xr
# Internal libraries
import run_simulation as sim
import pebsi.input as eb_prms
import pebsi.massbalance as mb
from objectives import *

# OPTIONS
repeat_run = False   # True if restarting an already begun run
# Define sets of parameters
# params = {'Boone_c5':[0.018,0.02,0.022,0.024,0.026,0.028,0.03], # 
#           'kp':[1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,3.25,3.5]} # 
params = {'Boone_c5':[0.018,0.02,0.022,0.023,0.024,0.025,0.026,0.027,0.028,0.03], # 
          'kp':[1,1.25,1.5,1.75,2,2.25,2.375,2.5,2.625,2.75,2.875,3]} # 

# Read command line args
parser = sim.get_args(parse=False)
parser.add_argument('-run_type', default='long')
args = parser.parse_args()
n_processes = args.n_simultaneous_processes

# Determine number of runs for each process
n_runs = 1
for param in list(params.keys()):
    n_runs *= len(params[param])
print(f'Beginning {n_runs} model runs on {n_processes} CPUs')

# Check if we need multiple (serial) runs per (parallel) set
if n_runs <= n_processes:
    n_runs_per_process = 1
    n_process_with_extra = 0
else:
    n_runs_per_process = n_runs // n_processes  # Base number of runs per CPU
    n_process_with_extra = n_runs % n_processes    # Number of CPUs with one extra run

# Create output directory
if 'trace' in eb_prms.machine:
    eb_prms.output_filepath = '/trace/group/rounce/cvwilson/Output/'

if repeat_run:
    date = '03_05' if args.run_type == 'long' else '03_06'
    n_today = '1'
    out_fp = f'{date}_{args.site}_{n_today}/'
    if not os.path.exists(eb_prms.output_filepath + out_fp):
        os.mkdir(eb_prms.output_filepath + out_fp)
else:
    # date = str(pd.Timestamp.today()).replace('-','_')[5:10]
    date = '07_01' if args.run_type == 'long' else '07_02'
    print('Forcing run date to be', date)
    n_today = 0
    out_fp = f'{date}_{args.site}_{n_today}/'
    while os.path.exists(eb_prms.output_filepath + out_fp):
        n_today += 1
        out_fp = f'{date}_{args.site}_{n_today}/'
    os.mkdir(eb_prms.output_filepath + out_fp)

# Force some args
args.store_data = True              # Ensures output is stored
if args.run_type == '2024': # Short AWS run
    args.use_AWS = True
    eb_prms.AWS_fn = '../climate_data/AWS/Preprocessed/gulkana2024.csv'
    eb_prms.store_vars = ['MB','EB','layers','climate']
    args.startdate = pd.to_datetime('2024-04-18 00:00:00')
    args.enddate = pd.to_datetime('2024-08-20 00:00:00')
else: # Long MERRA-2 run
    args.use_AWS = False
    eb_prms.store_vars = ['MB','layers','climate','EB']
    args.startdate = pd.to_datetime('2000-04-15 00:00:00')
    args.enddate = pd.to_datetime('2024-08-20 00:00:00')

# Transform params to strings for comparison
for key in params:
    for v,value in enumerate(params[key]):
        params[key][v] = str(value)

# Parse list for inputs to Pool function
packed_vars = [[] for _ in range(n_processes)]
run_no = 0  # Counter for runs added to each set
set_no = 0  # Index for the parallel process

# Storage for failed runs
all_runs = []
missing_fn = eb_prms.output_filepath + out_fp + 'missing.txt'

# Special dates for low sites
if args.run_type == 'long':
    if args.site == 'A':
        args.enddate = pd.to_datetime('2015-05-20 00:00:00')
    elif args.site == 'AU':
        args.startdate = pd.to_datetime('2012-04-20 00:00:00')

# Get the climate
climate, args = sim.initialize_model(args.glac_no,args)

# Loop through parameters
for kp in params['kp']:
    for c5 in params['Boone_c5']:
        # Get args for the current run
        args_run = copy.deepcopy(args)
        climate_run = copy.deepcopy(climate)

        # Set parameters
        args_run.Boone_c5 = c5
        args_run.kp = kp

        # Set identifying output filename
        args_run.out = out_fp + f'grid_{date}_set{set_no}_run{run_no}_'
        all_runs.append((args.site, c5, kp, args_run.out))

        # Specify attributes for output file
        store_attrs = {'c5':c5,'kp':kp,'site':args.site}

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

        # Check if model run should be performed
        if not os.path.exists(eb_prms.output_filepath + args.out + '0.pkl'):
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

            except Exception as e:
                print('An error occurred at site',args.site,'with c5 =',args.Boone_c5,'kp =',args.kp,' ... removing',args.out)
                traceback.print_exc()
                os.remove(eb_prms.output_filepath + args.out + '0.nc')

        # If succeeded, process and save only the stats
        # out_fn = eb_prms.output_filepath + args.out
        # if os.path.exists(out_fn + '0.nc') and not os.path.exists(out_fn + '0.pkl'):
        #     with xr.open_dataset(out_fn + '0.nc') as dataset:
        #         ds = dataset.load()
        #         if args.run_type == 'long':
        #             # seasonal mass balance
        #             error_dict = seasonal_mass_balance(ds,method=['MAE','ME'])
        #             winter_MAE, winter_ME = error_dict['winter']
        #             summer_MAE, summer_ME = error_dict['summer']
        #             annual_MAE, annual_ME = error_dict['annual']
        #             seasonal_MAE = np.mean([winter_MAE, summer_MAE])
        #             seasonal_ME = np.mean([winter_ME, summer_ME])
        #             results = {'winter_MAE':winter_MAE,'summer_MAE':summer_MAE,
        #                     'winter_ME':winter_ME,'summer_ME':summer_ME,
        #                     'seasonal_MAE':seasonal_MAE,'seasonal_ME':seasonal_ME,
        #                     'annual_MAE':annual_MAE,'annual_ME':annual_ME}

        #             # snowpits
        #             for method in ['MAE','ME']:
        #                 snowpit_dict = snowpits(ds,method=method)
        #                 for var in snowpit_dict:
        #                     results[var] = snowpit_dict[var]

        #         elif args.run_type == '2024':
        #             MAE = cumulative_mass_balance(ds,method='MAE')
        #             ME = cumulative_mass_balance(ds,method='ME')
        #             results = {'MAE':MAE,'ME':ME}

        #         # Store the attributes in the results dict
        #         for attr in ds.attrs:
        #             results[attr] = ds.attrs[attr]

        #     # Pickle the dict
        #     stats_fn = out_fn + '0.pkl'
        #     with open(stats_fn, 'wb') as file:
        #         pickle.dump(results,file)

            # Remove the .nc
            # os.remove(eb_prms.output_filepath + args.out + '0.nc')

    return

# Run model in parallel
with Pool(n_processes) as processes_pool:
    processes_pool.map(run_model_parallel,packed_vars)
    
missing = []
for run in all_runs:
    fn = run[-1]
    if not os.path.exists(eb_prms.output_filepath + fn + '0.pkl'):
        missing.append(run)
n_missing = len(missing)

# Store missing as .txt
np.savetxt(missing_fn,np.array(missing),fmt='%s',delimiter=',')
print(f'Finished param_set_parallel with {n_missing} failed: saved to {missing_fn}')