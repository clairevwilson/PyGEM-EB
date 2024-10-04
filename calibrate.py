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
site = 'D'
# Parameter tuning summer balance
summer_param = 'kw' if site == 'D' else 'a_ice'
summer_info = {'AB':{'bounds':[0.2,0.4],'x0':0.3},
                 'B':{'bounds':[0.4,0.55],'x0':0.4},
                 'D':{'bounds':[0.2,0.8],'x0':0.4}}
summer_x0 = summer_info[site]['x0']          # Initial guess for the summer parameter
summer_bounds = summer_info[site]['bounds']  # Bounds for parameter search
summer_step = 0.02
# Parameter tuning winter balance
winter_param = 'kp'
winter_x0 = 2.5             # Initial guess for the winter parameter (kp)
winter_bounds = [0.5,4]   # Bounds for kp search
winter_step = 0.5
# Optimization choices
tolerance = 1e-1          # Tolerance for MAE we are looking for
max_n_iters = 10        # Max number of iterations to run

# ===== FILEPATHS =====
data_fp = os.getcwd() + '/../MB_data/Gulkana/Input_Gulkana_Glaciological_Data.csv'
today = str(pd.Timestamp.today()).replace('-','_')[5:10]
base_fn = f'calibration_{site}_{today}_run#_'
n_today = 0
eb_prms.output_filepath = os.getcwd() + f'/../Output/EB/{today}_{n_today}/'
while os.path.exists(eb_prms.output_filepath + base_fn.replace('#','0') + '0.nc'):
    n_today += 1
    eb_prms.output_filepath = os.getcwd() + f'/../Output/EB/{today}_{n_today}/'
if not os.path.exists(eb_prms.output_filepath):
    os.mkdir(eb_prms.output_filepath)

# ===== RUN PREPROCESSING =====
# Read command line args
args = sim.get_args()
n_processes = args.n_simultaneous_processes
# Set parameters that should be run in parallel
params_parallel = {'k_snow':['Sturm','Douville','Jansson','OstinAndersson','VanDusen']}
# Initialize storage
best_runs = []          # Storage for each iteration's best run name

# Force some args
args.store_data = True              # Ensures output is stored
args.use_AWS = True                 # Use available AWS data
args.debug = False                  # Don't need debug prints
eb_prms.store_vars = ['MB']         # Only store basic results

# Initialize model
args.startdate = pd.to_datetime('2000-04-20 00:00:00')
args.enddate = pd.to_datetime('2022-05-20 00:00:00')
args.site = site
climate = sim.initialize_model(args.glac_no[0],args)
all_runs_counter = 0

# Determine number of runs for each parallel process
n_runs = 1                                  # Number of sites 
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

    # Parse the input parameter
    if site == 'D':
        kw, kp = parameters
        a_ice = 0.4
    else:
        a_ice, kp = parameters
        kw = 1

    # Parse list for inputs to Pool function
    packed_vars = [[] for _ in range(n_processes)]
    run_no = 0  # Counter for runs added to each set
    set_no = 0  # Index for the parallel process
    start_counter = all_runs_counter # Index for the first run of this iteration

    # Loop through the options we want to run in parallel
    for k_snow in params_parallel['k_snow']:
        # Get args for the current run
        args_run = copy.deepcopy(args)

        # Set parameters
        args_run.k_snow = k_snow
        args_run.kw = kw
        args_run.kp = kp
        args_run.a_ice = a_ice
        args_run.site = site

        # Set output filename
        args_run.out = base_fn.replace('#',str(all_runs_counter))

        # Specify attributes for output file
        store_attrs = {'k_snow':str(k_snow),'a_ice':str(a_ice),
                        'kw':str(kw),'kp':str(kp),'site':site}

        # Check if moving to the next set of runs
        n_runs_set = n_runs_per_process + (0 if set_no < n_runs_with_extra else 0)
        if run_no >= n_runs_set:
            set_no += 1
            run_no = 0

        # Set task ID for SNICAR input file
        args_run.task_id = set_no 
        if site == 'B':
            args_run.task_id += 5
        elif site == 'D':
            args_run.task_id += 10
    
        # Store model inputs
        packed_vars[set_no].append((args_run,climate,store_attrs))

        # Advance counter
        run_no += 1
        all_runs_counter += 1

    # Run model in parallel
    with Pool(n_processes) as processes_pool:
        processes_pool.map(run_model_parallel,packed_vars)

    # Assess model outputs
    all_loss = []
    all_names = []
    for run in np.arange(start_counter, all_runs_counter):
        # Get the output dataset
        out = base_fn.replace('#',str(run)) + '0.nc'
        ds = xr.open_dataset(eb_prms.output_filepath + out)

        # Evaluate loss
        loss = seasonal_mass_balance(data_fp,ds,site=site)
        all_loss.append(np.mean(loss)) # average winter and summer loss
        all_names.append(out)
    
    # Determine best snow parameterization for the current parameter set
    best_loss = np.min(all_loss)
    best_idx = np.argmin(all_loss)
    best_runs.append(all_names[best_idx])

    return best_loss

# ===== OPTIMIZATION =====
# Initialize storage
param_storage = {'summer':[summer_x0],'winter':[winter_x0]}
result_storage = []
# Initialize guess
parameters = (summer_x0, winter_x0)
loss = np.inf
n_iters = 0

# Begin search
while loss > tolerance and n_iters < max_n_iters:
    # Unpack parameters
    summer_value, winter_value = parameters

    # Run the objective function
    print(f'Testing {winter_param} = {winter_value}, {summer_param} = {summer_value}')
    loss = objective(parameters)
    result_storage.append(loss)

    # Load the best run
    fn_best = best_runs[-1]
    ds = xr.open_dataset(eb_prms.output_filepath + fn_best)
    best_ksnow = ds.attrs['k_snow']

    # If we achieved loss within tolerance, don't update the parameter
    if loss < tolerance:
        n_iters += 1
        break

    # Calculate the bias in the best run
    winter_bias,summer_bias = seasonal_mass_balance(data_fp,ds,site=site,method='ME')
    print(f'Winter bias: {winter_bias} m w.e.    Summer bias: {summer_bias} m w.e.')

    # Adjust parameters according to bias
    if summer_param == 'a_ice':
        if summer_bias < 0:
            summer_direction = 1
            print(f'Overestimated melt: increasing {summer_param}')
        elif summer_bias > 0:
            summer_direction = -1
            print(f'Underestimated melt: decreasing {summer_param}')
        else:
            print(f'Got a {summer_bias} result from bias with {fn_best}: quitting')
            quit()
    elif summer_param == 'kw':
        if summer_bias < 0:
            summer_direction = -1
            print(f'Overestimated melt: decreasing {summer_param}')
        elif summer_bias > 0:
            summer_direction = 1
            print(f'Underestimated melt: increasing {summer_param}')
        else:
            print(f'Got a {summer_bias} result from bias with {fn_best}: quitting')
            quit()
    else:
        print(f'Need to code how to adjust {summer_param} based on bias')
        quit()
    if winter_bias < 0:
        winter_direction = 1
        print(f'Underestimated winter MB: increasing {winter_param}')
    elif winter_bias > 0:
        winter_direction = -1
        print(f'Overestimated winter MB: decreasing {winter_param}')
    else:
        print(f'Got a {winter_bias} result from bias with {fn_best}: quitting')
        quit()

    # Step parameter
    summer_value += summer_step * summer_direction
    winter_value += winter_step * winter_direction

    # If we already tried this parameter, step halfway back
    diff = np.abs(np.array(param_storage['summer']) - summer_value)
    while np.any(diff < 1e-6):
        summer_step /= 2
        summer_value += summer_step * summer_direction * -1
        diff = np.abs(np.array(param_storage['summer']) - summer_value)

    # winter
    diff = np.abs(np.array(param_storage['winter']) - winter_value)
    while np.any(diff < 1e-6):
        winter_step /= 2
        winter_value += winter_step * winter_direction * -1
        diff = np.abs(np.array(param_storage['winter']) - winter_value)

    # Bound parameters
    summer_value = max(summer_value,summer_bounds[0])
    summer_value = min(summer_value,summer_bounds[-1])
    winter_value = max(winter_value,winter_bounds[0])
    winter_value = min(winter_value,winter_bounds[-1])

    # Pack parameters
    parameters = (summer_value, winter_value)
    param_storage['winter'].append(winter_value)
    param_storage['summer'].append(summer_value)

    # Next step
    n_iters += 1

print(f'Completed calibration in {n_iters} iterations')
print(f'     Best {winter_param} = {winter_value}')
print(f'     Best {summer_param} = {summer_value}')
print(f'     Best k_snow = {best_ksnow}')
print(f'     MAE: {loss:.3f} m w.e.')

# Loss plot
plt.plot(np.arange(n_iters),result_storage)
plt.ylabel('Loss (MAE)')
plt.xlabel('Iterations')
plt.savefig(eb_prms.output_filepath+f'loss_iterations_{site}.png')

# Loss with parameter
fig,axes = plt.subplots(2)
axes[0].plot(param_storage['winter'],result_storage)
axes[0].set_xlabel(f'{winter_param}')
axes[1].plot(param_storage['summer'],result_storage)
axes[1].set_xlabel(f'{summer_param}')
fig.supylabel('Loss (MAE)')
plt.savefig(eb_prms.output_filepath+f'loss_parameter_{site}.png')

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