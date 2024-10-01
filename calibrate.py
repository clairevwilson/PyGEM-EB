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
site = 'B'
param_name = 'a_ice'    # Parameter to calibrate: a_ice or kw
initial_guess = 0.4     # Initial guess for the parameter
bounds = [0.38,0.42]     # Bounds for parameter search
tolerance = 1e-1        # Tolerance for MAE we are looking for
step_size = 2e-2        # Initial step size to adjust parameter by
# site = 'D'
# param_name = 'kw'       # Parameter to calibrate: a_ice or kw
# initial_guess = 0.2     # Initial guess for the parameter
# bounds = [0.2,0.8]      # Bounds for parameter search
# tolerance = 1e-1        # Tolerance for MAE we are looking for
# step_size = 1e-1        # Initial step size to adjust parameter by
max_n_iters = 10        # Max number of iterations to run
best_runs = []          # Storage for each iteration's best run name

# ===== FILEPATHS =====
data_fp = os.getcwd() + '/../MB_data/Gulkana/Input_Gulkana_Glaciological_Data.csv'
today = str(pd.Timestamp.today()).replace('-','_')[5:10] #  + '_AB'
# print('REMOVE SITE FROM TODAY')
eb_prms.output_filepath = os.getcwd() + f'/../Output/EB/{today}/'
if not os.path.exists(eb_prms.output_filepath):
    os.mkdir(eb_prms.output_filepath)
base_fn = f'{param_name}_calibration_{today}_run0_#.nc'
n_today = 0
while os.path.exists(eb_prms.output_filepath + base_fn.replace('#',str(n_today))):
    n_today += 1

# ===== RUN PREPROCESSING =====
# Read command line args
args = sim.get_args()
n_processes = args.n_simultaneous_processes
params_parallel = {'k_snow':['Sturm','Douville','Jansson','OstinAndersson','VanDusen']}

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
def objective(parameter):    
    """
    Prepares the inputs for parallel runs, executes the model
    in parallel, and assesses loss.
    """
    global all_runs_counter

    # Parse the input parameter
    if param_name == 'a_ice':
        a_ice = parameter
        kw = 1
    elif param_name == 'kw':
        kw = parameter
        a_ice = 0.42
        print('using uncalibrated a_ice')

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
        args_run.a_ice = a_ice
        args_run.site = site

        # Set output filename
        args_run.out = f'{param_name}_calibration_{today}_run{all_runs_counter}_'

        # Specify attributes for output file
        store_attrs = {'k_snow':str(k_snow),'a_ice':str(a_ice),
                        'kw':str(kw),'site':site}

        # Check if moving to the next set of runs
        n_runs_set = n_runs_per_process + (0 if set_no < n_runs_with_extra else 0)
        if run_no >= n_runs_set:
            set_no += 1
            run_no = 0

        # Set task ID for SNICAR input file
        args_run.task_id = set_no +5
    
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
        out = f'{param_name}_calibration_{today}_run{run}_{n_today}.nc'
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
param_storage = [initial_guess]
result_storage = []
# Initialize guess
parameter = initial_guess
loss = np.inf
n_iters = 0

# Begin search
while loss > tolerance and n_iters < max_n_iters:
    # Run the objective function
    print(f'Testing {param_name} = {parameter}')
    loss = objective(parameter)
    result_storage.append(loss)

    # Load the best run
    fn_best = best_runs[-1]
    ds = xr.open_dataset(eb_prms.output_filepath + fn_best)
    best_ksnow = ds.attrs['k_snow']

    # If we achieved loss within tolerance, don't update the parameter
    print(loss,tolerance)
    if loss < tolerance:
        break

    # Calculate the bias in the best run
    winter_bias,summer_bias = seasonal_mass_balance(data_fp,ds,site=site,method='ME')
    print('bias',summer_bias)

    # Adjust parameter according to bias
    if param_name == 'a_ice':
        if summer_bias < 0:
            direction = 1
            print(f'Overestimated melt: increasing {param_name}')
        elif summer_bias > 0:
            direction = -1
            print(f'Underestimated melt: decreasing {param_name}')
        else:
            print(f'Got a {summer_bias} result from bias with {fn_best}: quitting')
            quit()
    elif param_name == 'kw':
        if summer_bias < 0:
            direction = -1
            print(f'Overestimated melt: decreasing {param_name}')
        elif summer_bias > 0:
            direction = 1
            print(f'Underestimated melt: icreasing {param_name}')
        else:
            print(f'Got a {summer_bias} result from bias with {fn_best}: quitting')
            quit()
    else:
        print(f'Need to code how to adjust {param_name} based on bias')
        quit()
    
    # Step parameter
    last = parameter
    parameter += step_size * direction

    # If we already tried this parameter, step halfway back
    diff = np.abs(np.array(param_storage) - parameter)
    while np.any(diff < 1e-6):
        step_size /= 2
        parameter += step_size * direction * -1
        diff = np.abs(np.array(param_storage) - parameter)

    # Bound parameter
    parameter = max(parameter,bounds[0])
    parameter = min(parameter,bounds[-1])
    diff = np.abs(np.array(param_storage) - parameter)
    if parameter in bounds and np.any(diff < 1e-6):
        bound_hit = 'lower' if parameter == bounds[0] else 'upper'
        n_iters += 1
        print(f'Warning: parameter hit the {bound_hit} bounds')
        break
    else:
        param_storage.append(parameter)

    # Next step
    n_iters += 1

print(f'Completed calibration in {n_iters} iterations')
print(f'     Best {param_name} = {parameter}       Best k_snow = {best_ksnow}')
print(f'     MAE: {loss:.3f} m w.e.')

# Loss plot
plt.plot(np.arange(n_iters),result_storage)
plt.ylabel('Loss (MAE)')
plt.xlabel('Iterations')
plt.savefig(eb_prms.output_filepath+'loss_iterations.png',dpi=150)

# Loss with parameter
plt.plot(param_storage,result_storage)
plt.ylabel('Loss (MAE)')
plt.xlabel(f'{param_name}')
plt.savefig(eb_prms.output_filepath+'loss_parameter.png',dpi=150)

# Best result plot
k_snow = ds.attrs['k_snow']
fig, ax = seasonal_mass_balance(data_fp,ds,site=site,method='MAE',plot=True)
fig.suptitle(f'Current best run at {site}: {param_name} = {parameter}   k_snow = {k_snow}')
plt.savefig(eb_prms.output_filepath+f'mass_balance_{site}.png',dpi=150)

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