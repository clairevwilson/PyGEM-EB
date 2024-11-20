import socket
import pickle
# Local libraries
from pygem_eb.processing.plotting_fxns import *
from objectives import *

# Define some information on the model runs
sitedict = {'2024':['AB','ABB','B','BD','D','T'],
            'long':['AU','B','D']}
# Number of runs to assess
n_sets = 200    # Doesn't matter as long as it's above the actual number of sets/runs
n_runs = 5

# Filenames: input and output
machine = socket.gethostname()
if 'trace' in machine:
    data_path = '/trace/group/rounce/cvwilson/Output/'
else:
    data_path = '/home/claire/research/Output/EB/'

def process_grid_search(date,run_type,params,date_idx=0):
    """
    Takes in a given output run and processes an output
    pickle file containing the MAE, ME, and run/set number
    for each site and parameter combination.

    Parameters
    ----------
    date : str
        'MM_DD' formatted string of the run date
    run_type : str
        'long' or '2024' 
    params : dict
        Dictionary containing all parameters
    date_idx : str or int
        Index of the run on the date (default = 0)
    
    """
    # Define filepaths
    data_path = '/trace/group/rounce/cvwilson/Output/' + f'{date}_{date_idx}/'
    output_fn = f'{date}_{date_idx}_out2.pkl'

    # Store which runs are missing
    missing = []
    dsdict = {}
    for site in sitedict[run_type]:
        dsdict[site] = {}
        for kw in params['kw']:
            dsdict[site][kw] = {}
            for c5 in params['Boone_c5']:
                dsdict[site][kw][c5] = {}
                for kp in params['kp']:
                    dsdict[site][kw][c5][kp] = {}

    # Loop through sets and runs
    for set in range(n_sets):
        for run in range(n_runs):
            # Open the files that exist
            try:
                fn = f'grid_{date}_set{set}_run{run}_0.nc'
                ds,_,_ = getds(data_path+fn)
            except FileNotFoundError:
                missing.append(f'set{set}_run{run}')
                continue

            if ds.melt.sum().values <= 1e-5:
                # Found an empty file: skip
                print(set,run,'is empty')
                continue
            
            # Parse the parameters
            kp = ds.attrs['kp']
            c5 = ds.attrs['c5']
            kw = ds.attrs['kw']
            site = ds.attrs['site']

            # Calculate error
            if run_type == 'long':
                winter_MAE,summer_MAE = seasonal_mass_balance(site,ds,method='MAE')
                winter_ME,summer_ME = seasonal_mass_balance(site,ds,method='ME')
                results = {'winter_MAE':winter_MAE,'summer_MAE':summer_MAE,
                        'winter_ME':winter_ME,'summer_ME':summer_ME}
            else:
                MAE = cumulative_mass_balance(site,ds,method='MAE')
                ME = cumulative_mass_balance(site,ds,method='MAE')
                results = {'MAE':MAE,'ME':ME}

            # Store the set/run for each parameter combination
            results['set'] = set
            results['run'] = run

            # Add the result to a dictionary
            if kp in params['kp'] and kw in params['kw'] and c5 in params['Boone_c5']:
                for result in results:
                    dsdict[site][kw][c5][kp][result] = results[result]
            print(dsdict[site][kw][c5][kp]['summer_MAE'],run,set)

    # Pickle the dict
    with open(data_path + output_fn, 'wb') as file:
        pickle.dump(dsdict,file)