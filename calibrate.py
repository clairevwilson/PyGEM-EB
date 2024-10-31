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
from objectives import *

# ===== USER OPTIONS =====
sites = ['A','AU','B','D']
n_spc_runs_ahead = 1    # Step if you're going to run this script more than once

# Summer parameter bounds and guesses by site
summer_info = {'A':{'param':'kw','bounds':[0.2,5],'x0':1,'step':0.5},
               'AU':{'param':'kw','bounds':[0.2,5],'x0':1,'step':0.5},
                'B':{'param':'kw','bounds':[0.2,5],'x0':1,'step':0.5},
                # 'AB':{'param':'a_ice','bounds':[0.2,0.4],'x0':0.2,'step':0.05},
                # 'B':{'param':'a_ice','bounds':[0.4,0.55],'x0':0.4,'step':0.02},
                'D':{'param':'kw','bounds':[0.2,5],'x0':1.5,'step':0.5}}
# Winter parameter is always kp
winter_info = {'param':'kp','bounds':[0.5,4],'x0':3,'step':0.1}

# Optimization choices
tolerance = 1e-1          # Tolerance for MAE we are looking for
max_n_iters = 15          # Max number of iterations to run

# ===== INITIAL PRINTS =====
print(f'Starting calibration on {len(sites)} sites:')
for site in sites:
    param = summer_info[site]['param']
    bounds = summer_info[site]['bounds']
    print(f'   For site {site}, calibrating summer mass balance using {param} with bounds {bounds}')
print(f'                       and winter mass balance using kp with bounds [0.5, 4]')

# ===== FILEPATHS =====
today = str(pd.Timestamp.today()).replace('-','_')[5:10]
base_fn = f'calibration_{today}_run#_'
n_today = 0
eb_prms.output_filepath = os.getcwd() + f'/../Output/EB/{today}_{n_today}/'
while os.path.exists(eb_prms.output_filepath):
    n_today += 1
    eb_prms.output_filepath = os.getcwd() + f'/../Output/EB/{today}_{n_today}/'
os.mkdir(eb_prms.output_filepath)

# ===== RUN PREPROCESSING =====
# Read command line args
args = sim.get_args()
n_processes = args.n_simultaneous_processes
# Set parameters that should be run in parallel
params_parallel = {'k_snow':['Sturm','Douville','Jansson','OstinAndersson','VanDusen']}
# Initialize storage
best_runs = {}          # Storage for each iteration's best run name
for site in sites:
    best_runs[site] = []

# Force some args
args.store_data = True              # Ensures output is stored
args.use_AWS = False                 # Use available AWS data
if not args.use_AWS:
    print('Using only MERRA-2 data')
args.a_ice = 0.4
eb_prms.AWS_fn = eb_prms.AWS_fp + 'Preprocessed/gulkana_22yrs.csv'
print('Forcing ice albedo to 0.4')
eb_prms.store_vars = ['MB']         # Only store basic results

# Initialize model
args.startdate = pd.to_datetime('2000-04-20 00:00:00')
if args.use_AWS:
    args.enddate = pd.to_datetime('2022-04-20 00:00:00')
else:
    args.enddate = pd.to_datetime('2024-08-20 00:00:00')
climate_dict = {}
for site in sites:
    args.site = site
    print()
    print('Initializing climate for',args.site,':')
    climate_dict[site] = sim.initialize_model(args.glac_no[0],args)
all_runs_counter = 0

# Determine number of runs for each parallel process
n_runs = len(sites)                         # Number of sites 
for param in params_parallel:
    n_runs *= len(params_parallel[param])   # Number of parallel parameter options
n_runs_per_process = n_runs // n_processes  # Base number of runs per CPU
n_runs_with_extra = n_runs % n_processes    # Number of CPUs with one extra run

# ===== PARALLEL FUNCTION =====
def run_model_parallel(list_inputs):
    """
    Loops through runs per process and runs the model
    """
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

# ===== OBJECTIVE FUNCTION =====
def objective(parameters):    
    """
    Prepares the inputs for parallel runs, executes the model
    in parallel, and assesses loss.
    """
    global all_runs_counter
    global base_fn

    # Parse list for inputs to Pool function
    packed_vars = [[] for _ in range(n_processes)]
    run_no = 0  # Counter for runs added to each set
    set_no = 0  # Index for the parallel process
    start_counter = all_runs_counter # Index for the first run of this iteration

    # Loop through the options we want to run in parallel
    for site in sites:
        site_climate = climate_dict[site]

        # Parse the input parameter
        kp = parameters[site]['kp']
        kw = parameters[site]['kw']
        
        for k_snow in params_parallel['k_snow']:
            # Get args for the current run
            args_run = copy.deepcopy(args)

            # Set parameters
            args_run.k_snow = k_snow
            args_run.kw = kw
            args_run.kp = kp
            # args_run.a_ice = a_ice
            args_run.site = site

            # Set output filename
            args_run.out = base_fn.replace('#',str(all_runs_counter))

            # Specify attributes for output file
            store_attrs = {'k_snow':str(k_snow), # 'a_ice':str(a_ice),
                            'kw':str(kw),'kp':str(kp),'site':site}

            # Check if moving to the next set of runs
            n_runs_set = n_runs_per_process + (0 if set_no < n_runs_with_extra else 0)
            if run_no >= n_runs_set:
                set_no += 1
                run_no = 0

            # Set task ID for SNICAR input file
            args_run.task_id = set_no + n_spc_runs_ahead*n_processes
        
            # Store model inputs
            packed_vars[set_no].append((args_run,site_climate,store_attrs))

            # Advance counter
            run_no += 1
            all_runs_counter += 1

    # Run model in parallel
    with Pool(n_processes) as processes_pool:
        processes_pool.map(run_model_parallel,packed_vars)

    # Storage for each site results
    all_runs = {}
    best = {}
    for site in sites:
        all_runs[site] = {'loss':[],'fns':[]}
        best[site] = {}
    
    # Assess model outputs
    for run in np.arange(start_counter, all_runs_counter):
        # Get the output dataset
        out = base_fn.replace('#',str(run)) + '0.nc'
        ds = xr.open_dataset(eb_prms.output_filepath + out)

        # Evaluate loss
        site_run = ds.attrs['site']
        loss = seasonal_mass_balance(site_run,ds)
        all_runs[site_run]['loss'].append(np.mean(loss)) # average winter and summer loss
        all_runs[site_run]['fns'].append(out)

    # Determine best runs for each site
    for site in sites:
        # Determine best snow parameterization for the current parameter set
        best_idx = np.argmin(all_runs[site]['loss'])
        best[site]['loss'] = all_runs[site]['loss'][best_idx]
        best[site]['fn'] = all_runs[site]['fns'][best_idx]
        # Add best run filepath to big list
        best_runs[site].append(all_runs[site]['fns'][best_idx])
    
    return best

def update_params(param_storage,iter):
    params_out = {}
    # Loop through sites
    for site in sites:
        params_out[site] = {}

        # Load the best run
        fn_best = best_runs[site][-1]

        # Get best dataset
        ds = xr.open_dataset(eb_prms.output_filepath + fn_best)
        k_snow = ds.attrs['k_snow']

        # Extract summer parameters
        summer_param = summer_info[site]['param']
        summer_value = float(ds.attrs[summer_param])
        summer_bounds = summer_info[site]['bounds']
        summer_step = summer_info[site]['step']
        # Extract winter parameters
        winter_param = winter_info['param']
        winter_value = float(ds.attrs[winter_param])
        winter_bounds = winter_info['bounds']
        winter_step = winter_info['step']

        # Get lists of previously used parameters
        previous_summer = [ps[site][summer_param] for ps in param_storage]
        previous_winter = [ps[site][winter_param] for ps in param_storage]

        # Calculate the bias in the best run
        winter_bias,summer_bias = seasonal_mass_balance(site,ds,method='ME')
        print(f'For site {site} with k_snow = {k_snow}, {summer_param} = {summer_value}, {winter_param} = {winter_value}')
        print(f'      Winter bias: {winter_bias} m w.e.    Summer bias: {summer_bias} m w.e.')

        # Adjust parameters according to bias
        # Switch back and forth between updating winter, summer params
        summer_hit_bounds = False
        if iter % 2 == 0:
            # Summer parameter on even iteration
            if summer_param == 'a_ice':
                if summer_bias < 0:
                    summer_direction = 1
                    print(f'      Overestimated melt: increasing {summer_param}')
                elif summer_bias > 0:
                    summer_direction = -1
                    print(f'      Underestimated melt: decreasing {summer_param}')
                else:
                    print(f'      Got a {summer_bias} result from bias with {fn_best}: quitting')
                    quit()
            elif summer_param == 'kw':
                if summer_bias < 0:
                    summer_direction = -1
                    print(f'      Overestimated melt: decreasing {summer_param}')
                elif summer_bias > 0:
                    summer_direction = 1
                    print(f'      Underestimated melt: increasing {summer_param}')
                else:
                    print(f'      Got a {summer_bias} result from bias with {fn_best}: quitting')
                    quit()
            else:
                print(f'Quitting: Need to code how to adjust {summer_param} based on bias')
                quit()

            # Step parameter
            summer_value += summer_step * summer_direction

            # If we already tried this parameter, step halfway back
            diff = np.abs(np.array(previous_summer) - summer_value)
            while np.any(diff < 1e-6):
                summer_step /= 2
                summer_value += summer_step * summer_direction * -1
                diff = np.abs(np.array(previous_summer) - summer_value)
            
            # Bound parameter
            summer_value = max(summer_value,summer_bounds[0])
            summer_value = min(summer_value,summer_bounds[-1])
            if np.any(np.abs(np.array(previous_summer) - summer_value) < 1e-6):
                summer_hit_bounds = True
        if iter % 2 == 1 or summer_hit_bounds:
            # Winter parameter on odd iteration or if summer hit the bounds
            if winter_param == 'kp':
                if winter_bias < 0:
                    winter_direction = 1
                    print(f'      Underestimated winter MB: increasing {winter_param}')
                elif winter_bias > 0:
                    winter_direction = -1
                    print(f'      Overestimated winter MB: decreasing {winter_param}')
                else:
                    print(f'      Got a {winter_bias} result from bias with {fn_best}: quitting')
                    quit()
            else:
                print(f'Quitting: Need to code how to adjust {winter_param} based on bias')
                quit()

            # Step parameter
            winter_value += winter_step * winter_direction

            # winter
            diff = np.abs(np.array(previous_winter) - winter_value)
            while np.any(diff < 1e-6):
                winter_step /= 2
                winter_value += winter_step * winter_direction * -1
                diff = np.abs(np.array(previous_winter) - winter_value)

            # Bound parameter
            winter_value = max(winter_value,winter_bounds[0])
            winter_value = min(winter_value,winter_bounds[-1])

        # Pack parameters
        params_out[site][summer_param] = summer_value
        params_out[site][winter_param] = winter_value

    return params_out

# ===== OPTIMIZATION =====
# Initialize storage
param_storage = []
result_storage = []
# Initialize guess
x0 = {}
for site in sites:
    x0[site] = {}
    x0[site][summer_info[site]['param']] = summer_info[site]['x0']
    x0[site][winter_info['param']] = winter_info['x0']
parameters = copy.deepcopy(x0)
# Initialize loop
overall_loss = np.inf
n_iters = 0

# Begin search
while overall_loss > tolerance and n_iters < max_n_iters:
    # Run the objective function
    print()
    print(f'Testing {parameters}')
    best = objective(parameters)
    
    # Store results
    overall_loss = np.mean([best[site]['loss'] for site in sites])
    print(f'   Overall loss: {overall_loss:.4f}')
    result_storage.append(overall_loss)
    param_storage.append(copy.deepcopy(parameters))

    # If this is the final loop, don't update the parameter
    if overall_loss < tolerance or n_iters + 1 > max_n_iters:
        n_iters += 1
        break

    # Update the parameters
    parameters = update_params(param_storage, n_iters)

    # Next step
    n_iters += 1

print('=======================================')
print(f'Completed calibration in {n_iters} iterations')
for site in sites:
    params_site = parameters[site]
    print(f'Final parameters for site {site}:')
    print(f'       {params_site}')

# Loss plot
plt.plot(np.arange(n_iters),result_storage)
plt.title('Mean MAE across sites')
plt.ylabel('Loss')
plt.xlabel('Iterations')
plt.savefig(eb_prms.output_filepath+f'overall_loss_iterations.png')

# # Best result plot
# k_snow = ds.attrs['k_snow']
# fig, ax = seasonal_mass_balance(data_fp,ds,site=site,method='MAE',plot=True)
# fig.suptitle(f'Current best run at {site}: {summer_param} = {summer_value}      k_snow = {k_snow}')
# plt.savefig(eb_prms.output_filepath+f'mass_balance_{site}.png',dpi=150)

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