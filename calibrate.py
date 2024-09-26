# Builtin libraries
import os
import time
import copy
# External libraries
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
# Internal libraries
import pygem_eb.input as eb_prms
import run_simulation_eb as sim
import pygem_eb.massbalance as mb
from pygem_eb.processing.objectives import *

# ===== USER OPTIONS =====
param_name = 'a_ice'    # Parameter to calibrate: a_ice or kw
initial_guess = 0.4     # Initial guess for the parameter
tolerance = 100         # Tolerance for MAE we are looking for
step_size = 1e-2        # Size to step parameter on (initially)
best_ksnow = []         # Storage for each guess's best snow parameterization

# ===== DATA FILEPATHS =====
base_mb = '/home/claire/research/MB_data/'
base_AWS = '/home/claire/research/cliate_data/AWS/Preprocessed/'
fp_dict = {'long': {'data_fp': base_mb + 'Gulkana/Input_Gulkana_Glaciological_Data.csv',
                    'AWS_fp': base_AWS + 'gulkana_22yrs.csv'},
            '2023': {'data_fp': base_mb + 'Stakes/gulkanaAB23_ALL.csv',
                     'AWS_fp': base_AWS + 'gulkana_2023.csv'},
            '2024':{'data_fp': base_mb + 'Stakes/gulkanaSITE24_ALL.csv',
                    'AWS_fp': base_AWS + 'gulkana_2024.csv'}}

# ===== RUN PREPROCESSING =====
# Read command line args
args = sim.get_args()
n_processes = args.n_simultaneous_processes
params = {'k_snow':['Sturm','Douville','Jansson','OstinAndersson','VanDusen'],
          'site':['AB','B']}

# Force some args
args.store_data = True              # Ensures output is stored
args.use_AWS = True                 # Use available AWS data
args.debug = False                  # Don't need debug prints
eb_prms.store_vars = ['MB']         # Only store mass balance results

# Determine number of runs for each parallel process
n_runs = len(fp_dict) # number of different time period runs
for param in list(params.keys()):
    n_runs *= len(params[param]) # number of parameter options
n_runs_per_process = n_runs // n_processes  # Base number of runs per CPU
n_runs_with_extra = n_runs % n_processes    # Number of CPUs with one extra run

# Define the dates for three model runs
runs_dict = {'long':{'start':'2000-04-20 00:00:00','end':'2022-05-21 12:00:00'},
        '2023':{'start':'2023-04-20 00:00:00','end':'2023-08-21 00:00:00'},
        '2024':{'start':'2024-04-20 00:00:00','end':'2024-08-21 00:00:00'}}

# Initialize climate for each of the runs
for run in runs_dict:
    # Get the dates for the model run
    args.startdate = pd.to_datetime(runs_dict[run]['start'])
    args.enddate = pd.to_datetime(runs_dict[run]['end'])

    # Set the filepath for AWS data
    eb_prms.AWS_fn = fp_dict[run]['AWS_fp']

    # Initialize the model
    climate = sim.initialize_model(args.glac_no[0],args)

    # Add climate to runs_dict
    runs_dict[run]['climate'] = climate

# ===== FUNCTIONS =====
def run_model_parallel(list_inputs):
    # Loop through the variable sets
    for inputs in list_inputs:
        # Unpack inputs
        args,climate,store_attrs = inputs
        
        # Run the model if the run doesn't already exist
        run_fp = eb_prms.output_filepath + args.out + '0.nc'
        if not os.path.exists(run_fp):
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

# ===== OBJECTIVE FUNCTION =====
def objective(parameter):    
    # Parse the input parameter
    if param_name == 'a_ice':
        a_ice = parameter
    elif param_name == 'kw':
        kw = parameter

    # Parse list for inputs to Pool function
    packed_vars = [[] for _ in range(n_processes)]
    run_no = 0  # Counter for runs added to each set
    set_no = 0  # Index for the parallel process

    # Loop through the options we want to run in parallel
    for run in runs_dict:
        # Get args for the current run
        args_run = copy.deepcopy(args)

        # Grab the preprocessed climate
        climate = runs_dict[run]['climate']
        args_run.startdate = pd.to_datetime(runs_dict[run]['start'])
        args_run.enddate = pd.to_datetime(runs_dict[run]['end'])

        for k_snow in params['k_snow']:
            for site in params['site']:
                # Set parameters
                args_run.k_snow = k_snow
                args_run.kw = kw
                args_run.a_ice = a_ice
                args_run.site = site

                # Set identifying output filename
                args_run.out = f'ksnow{k_snow}_aice{a_ice}_site{site}_{run}_'

                # Specify attributes for output file
                store_attrs = {'k_snow':str(k_snow),'a_ice':str(a_ice),
                               'kw':str(kw),'site':site}

                # Check if moving to the next set of runs
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

    # Run model in parallel
    with Pool(n_processes) as processes_pool:
        processes_pool.map(run_model_parallel,packed_vars)

    # Assess model outputs
    all_loss = []
    for k_snow in params['k_snow']:
        site_loss = []
        for site in params['site']:
            run_loss = []
            for run in runs_dict:
                # 2023 run doesn't need to be assessed at site B
                if site != 'AB' and run == '2023':
                    continue

                # Get the output dataset
                out = f'ksnow{k_snow}_aice{a_ice}_site{site}_{run}_'
                ds = xr.open_dataset(eb_prms.output_filepath + out + '0.nc')

                # Evaluate loss
                loss_fun = seasonal_mass_balance if run == 'long' else cumulative_mass_balance
                data_fp = fp_dict[run]['data_fp'].replace('SITE',site)
                loss = loss_fun(data_fp,ds,site=site)
                if type(loss) == tuple: # seasonal mass balance stores winter and summer
                    loss = np.mean(loss)
                run_loss.append(loss)
            
            # Weight loss by length of run
            weights = np.array([22,1])
            if len(run_loss) == 3:
                weights = np.append(weights,1)
            weighted_loss = np.sum(np.array(run_loss) * weights / np.sum(weights))
            site_loss.append[weighted_loss]

        # Calculate site mean weighted loss
        site_mean_weighted_loss = np.mean(site_loss)
        all_loss.append(site_mean_weighted_loss)
    
    # Determine best snow parameterization for the current parameter set
    best_loss = np.min(all_loss)
    best_idx = np.argmin(all_loss)
    best_ksnow.append(params['k_snow'][best_idx])

    return best_loss

# ===== OPTIMIZATION =====
# Initialize storage
param_storage = [initial_guess]
result_storage = []
# Initialize guess
parameter = initial_guess

# Begin search
while loss > tolerance:
    # Run the objective function
    loss = objective(parameter)

    # Determine which way the parameter needs to move based on the long run
    # Load the best long run
    k_snow = best_ksnow[-1]
    fn_best = f'ksnow{k_snow}_aice{parameter}_site{site}_long_'
    ds = xr.open_dataset(eb_prms.output_filepath + fn_best + '0.nc')

# low = 3e-6
# high = 3e-4
# n_iters = 3
# outputs = []
# best_loss = np.inf

# for guess in np.linspace(low,high,n_iters):
#     eb_prms.Boone_c1 = guess
#     with HiddenPrints():
#         # run the model
#         out = sim.run_model(climate,args,{'c5':str(guess)})
#     result = out.dh.resample(time='d').sum().cumsum().values
#     loss = objective(result.flatten(),stake_df['CMB'].values)

#     # new best
#     if loss < best_loss:
#         best_loss = loss
#         best_guess = guess
        
#     outputs.append(out)

# print(f'After {n_iters} iterations between {low} and {high} the best result was:')
# print(f'      c5 = {guess:.3f}')
# print(f'      mae = {best_loss:.3e}')

# dh_vs_stake(stake_df,outputs,[args.startdate,args.enddate],labels=[str(i) for i in range(len(outputs))])
# plt.savefig('/home/claire/research/dh_best.png',dpi=200)