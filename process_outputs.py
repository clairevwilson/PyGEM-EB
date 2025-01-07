import socket
import pickle
# Local libraries
from pygem_eb.processing.plotting_fxns import *
from objectives import *

# Define some information on the model runs
sitedict = {'2024':['AB','ABB','B','BD','D','T'],
            'long':['A','AU','B','D']}
# Number of runs to assess
n_sets = 200    # Doesn't matter as long as it's above the actual number of sets/runs
n_runs = 7

# Filenames: input and output
machine = socket.gethostname()

def process_grid_search(dates,run_type,params,date_idx=0):
    """
    Takes in a given output run and processes an output
    pickle file containing the MAE, ME, and run/set number
    for each site and parameter combination.

    Parameters
    ----------
    dates : list of str
        list of 'MM_DD' formatted string(s) of the run date(s)
    run_type : str
        'long' or '2024' 
    params : dict
        Dictionary containing all parameters
    date_idx : str or int
        Index of the run on the date (default = 0)
    
    """
    # Define filepaths
    if 'trace' in machine:
        base_path = '/trace/group/rounce/cvwilson/Output/'
    else:
        base_path = '/home/claire/research/Output/EB/'
    output_fn = f'{dates[0]}_{date_idx}_out.pkl'

    if os.path.exists(base_path + output_fn):
        with open(base_path + output_fn, 'rb') as file:
            dsdict = pickle.load(file)
    else:
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
            for date in dates:
                # Open the files that exist
                try:
                    fn = f'{date}_{date_idx}/grid_{date}_set{set}_run{run}_0.nc'
                    ds,_,_ = getds(base_path+fn)
                except FileNotFoundError:
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

                if 'set' not in dsdict[site][kw][c5][kp]:
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
                    
                    print('Done run',run,'set',set)

    # Pickle the dict
    with open(base_path + output_fn, 'wb') as file:
        pickle.dump(dsdict,file)