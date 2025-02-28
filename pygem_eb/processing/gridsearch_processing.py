import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from pygem_eb.processing.plotting_fxns import *
from objectives import *
import process_outputs as po
import pickle
import os
import copy
import socket, time

# Get the base filepath
machine = socket.gethostname()
if 'trace' in machine:
    base_fp = '/trace/group/rounce/cvwilson/Output/'
elif os.path.exists('/mnt/d/'):
    base_fp = '/mnt/d/grid_search/'
else:
    base_fp = '/home/claire/research/Output/EB/'

# Create some dictionaries with useful information
labels = {'kp':'Precipitation factor','kw':'Wind factor','c5':'Densification c$_5$'}      # Labels for the different parameters we varied
methodlabels = {'MAE':'MAE','ME':'Bias','RMSE':'RMSE','MdAE':'MdAE'}                                          # Labels for the different error methods
errorlabels = {'seasonal':'Seasonal mass balance (m w.e.)',                                     # Labels for the different error metrics with units
               'winter':'Winter mass balance (m w.e.)', 
               'summer':'Summer mass balance (m w.e.)', 
               'annual':'Annual mass balance (m w.e.)',
               'snowdepth':'End-of-winter snow depth (m)',
               'snowmass':'End-of-winter snow mass (m w.e.)',
               'snowdensity':'End-of-winter snow density (kg m-3)',
               '2024':'2024 surface height change (m)'} 
shorterrorlabels = {'2024':'2024 surface height','snowdensity':'Snow density','snowdepth':'Snow depth',
                    'seasonal':'Seasonal MB','winter':'Winter MB','summer':'Summer MB','annual':'Annual MB'}
param_labels = {'kp':'Precipitation factor','c5':'Densification parameter'}
medians = {'kp':'2.6','c5':'0.024'}                                              # Median value of each of the parameters
sitedict = {'2024':['AB','ABB','B','BD','D','T'],'long':['A','AU','B','D']}      # Dictionary of sites in the 2024 and long run
all_sites = sitedict['long']+sitedict['2024']+['mean','median']                  # List all sites

# USER OPTIONS
run_info = {'long':{'date':'02_11', 'idx':'0'},                     # Date and index of the grid search (12_04) (01_16)
            '2024':{'date':'02_12', 'idx':'0'}}                     # (12_06)
params = {'c5':[0.018,0.02,0.022,0.024,0.026,0.028,0.03], # 
          'kp':[1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,3.25,3.5]} # 
for key in params:                                                  # Convert params to strings for processing
    for v,value in enumerate(params[key]):
        params[key][v] = str(value)

# Cutoffs for error to normalize on
error_lims = {'2024':0.5,'snowdensity':100,
              'winter':0.4,'summer':0.9,
              'snowdepth':1,'annual':1}

def get_any(c5,kp,site,result_dict,run_type='long'):
    if run_type == '2024':
        date = run_info['2024']['date']
        idx = run_info['2024']['idx']
        setno = result_dict[c5][kp][site]['2024_set_no']
        runno = result_dict[c5][kp][site]['2024_run_no']
    else:
        date = run_info['long']['date']
        idx = run_info['long']['idx']
        setno = result_dict[c5][kp][site]['set_no']
        runno = result_dict[c5][kp][site]['run_no']
    ds,_,_ = getds(f'/trace/group/rounce/cvwilson/Output/{date}_{site}_{idx}/grid_{date}_set{setno}_run{runno}_0.nc')
    return ds

def get_all_error(ds, run_type):
    if run_type == 'long':
        # seasonal mass balance
        error_dict = seasonal_mass_balance(ds,method=['MAE','MdAE','ME'])
        winter_MAE, winter_MdAE, winter_ME = error_dict['winter']
        summer_MAE, summer_MdAE, summer_ME = error_dict['summer']
        annual_MAE, annual_MdAE, annual_ME = error_dict['annual']
        seasonal_MAE = np.mean([winter_MAE, summer_MAE])
        seasonal_MdAE = np.mean([winter_MdAE, summer_MdAE])
        seasonal_ME = np.mean([winter_ME, summer_ME])
        results = {'winter_MAE':winter_MAE,'winter_MdAE':winter_MdAE,'winter_ME':winter_ME,
                   'summer_MAE':summer_MAE,'summer_MdAE':summer_MdAE,'summer_ME':summer_ME,
                   'seasonal_MAE':seasonal_MAE,'seasonal_MdAE':seasonal_MdAE,'seasonal_ME':seasonal_ME,
                   'annual_MAE':annual_MAE,'annual_MdAE':annual_MdAE,'annual_ME':annual_ME}

        # snowpits
        for method in ['MAE','ME','MdAE']:
            snowpit_dict = snowpits(ds,method=method)
            for var in snowpit_dict:
                results[var] = snowpit_dict[var]

    elif run_type == '2024':
        MAE = cumulative_mass_balance(ds,method='MAE')
        MdAE = cumulative_mass_balance(ds,method='MdAE')
        ME = cumulative_mass_balance(ds,method='ME')
        results = {'MAE':MAE,'ME':ME,'MdAE':MdAE}

    return results

def process_runs(run_type, calib_valid = True, all = False, sites = 'all'):
    date = run_info[run_type]['date']
    idx = run_info[run_type]['idx']
    if sites == 'all':
        sites = sitedict[run_type]

    for site in sites:
        start_time = time.time()
        fp = base_fp+f'{date}_{site}_{idx}/'
        for f in os.listdir(fp):
            results = {}
            if '.nc' in f:
                ds = xr.open_dataset(fp + f)

                if all:
                    results = get_all_error(ds, run_type)

                if calib_valid:
                    years = np.unique(pd.to_datetime(ds.time.values).year)
                    years_calib = years[:int(len(years) * 0.7)]

                    # Calibration dataset
                    calib_start = ds.time.values[0]
                    calib_end = f'{years_calib[-1]}-10-31'
                    dates_calibration = pd.date_range(calib_start, calib_end, freq='h')
                    ds_calib = ds.sel(time=dates_calibration)

                    # Validation dataset
                    valid_start = f'{years_calib[-1]}-08-01'
                    valid_end = ds.time.values[-1]
                    dates_validation = pd.date_range(valid_start, valid_end, freq='h')
                    ds_valid = ds.sel(time=dates_validation)
                    ds_dict = {'calib': ds_calib, 'valid': ds_valid}

                    # Get all error metrics
                    for subset in ['calib','valid']:
                        results[subset] = get_all_error(ds_dict[subset], run_type)

                # Store the attributes in the results dict
                for attr in ds.attrs:
                    results[attr] = ds.attrs[attr]

                f_pickle = f.replace('.nc','.pkl')
                with open(fp + f_pickle, 'wb') as file:
                    pickle.dump(results, file)
        print('Completed site',site,'in',time.time() - start_time,'s')
    return

def create_dict(run_type):
    date = run_info[run_type]['date']
    idx = run_info[run_type]['idx']
    
    # Create storage for the grid search results
    grid_dict = {}
    for c5 in params['c5']:
        grid_dict[c5] = {}
        for kp in params['kp']:
            grid_dict[c5][kp] = {}
            for site in sitedict[run_type]:
                grid_dict[c5][kp][site] = {}

    # Loop through all the sites
    for site in sitedict[run_type]:
        for f in os.listdir(base_fp+f'{date}_{site}_{idx}/'):
            if 'pkl' in f:
                # Open individual output pickle 
                with open(base_fp + f'{date}_{site}_{idx}/' + f, 'rb') as file:
                    run_dict = pickle.load(file)
                set_no = f.split('_')[3].split('set')[1]
                run_no = f.split('_')[4].split('run')[1]
                c5 = run_dict['c5']
                kp = run_dict['kp']
                for var in run_dict:
                    if 'AE' in var or 'ME' in var:
                        if type(run_dict[var]) == dict:
                            grid_dict[c5][kp][site][var] = run_dict[var]['mean']
                        else:
                            grid_dict[c5][kp][site][var] = run_dict[var]
                    if 'calib' in var or 'valid' in var:
                        for thing in run_dict[var]:
                            if 'AE' in thing or 'ME' in thing:
                                if type(run_dict[thing]) == dict:
                                    grid_dict[c5][kp][site][thing+'_'+var] = run_dict[var][thing]['mean']
                                else:
                                    grid_dict[c5][kp][site][thing+'_'+var] = run_dict[var][thing]
                grid_dict[c5][kp][site]['set_no'] = set_no
                grid_dict[c5][kp][site]['run_no'] = run_no

    # Store compiled pickle
    with open(base_fp + f'{date}_{idx}_out.pkl', 'wb') as file:
        pickle.dump(grid_dict, file)

    return grid_dict

def get_result_dict(force_redo=False):
    # ===== Open or parse output dictionary for both runs =====
    both_dict = {}
    for run_type in ['long','2024']:  # 
        date = run_info[run_type]['date']
        idx = run_info[run_type]['idx']
        if not os.path.exists(base_fp + f'{date}_{idx}_out.pkl') or force_redo:
            # No compiled pickle exists: create dictionary
            grid_dict = create_dict(run_type)
        else:
            # Compiled pickle exists: load it
            with open(base_fp + f'{date}_{idx}_out.pkl', 'rb') as file:
                grid_dict = pickle.load(file)
        
        # Store the dictionary under the run type (long or 2024)
        both_dict[run_type] = grid_dict

    # Condense long and 2024 runs into to a single result_dict
    result_dict = both_dict['long']
    for c5 in params['c5']:
        for kp in params['kp']:
            # Add the 2024 error stats to the result_dict
            for site in sitedict['2024']:
                # Some sites are different from long run, so add a slot for these runs
                if site not in result_dict[c5][kp]:
                    result_dict[c5][kp][site] = {}
                # Add the 2024 error stats
                for var in both_dict['2024'][c5][kp][site]:
                    result_dict[c5][kp][site]['2024_'+var] = both_dict['2024'][c5][kp][site][var] 

    # ===== Find site means of each error type =====
    # List out all error types
    all_error = list(result_dict['0.026']['2.5']['B'].keys())
    all_error.remove('run_no')
    all_error.remove('set_no')
    all_error.remove('2024_run_no')
    all_error.remove('2024_set_no')
        
    # Create dictionary to store site means
    sites_error_dict = {}
    # Loop through all parameters and sites
    for c5 in params['c5']:
        for kp in params['kp']:
            result_dict[c5][kp]['mean'] = {}
            result_dict[c5][kp]['median'] = {}
            for error_type in all_error:
                sites_error_dict[error_type] = []
                for site in all_sites[:-2]:
                    if '2024' in error_type and site in sitedict['2024']:
                        sites_error_dict[error_type].append(result_dict[c5][kp][site][error_type])
                    if '2024' not in error_type and site in sitedict['long']:
                        sites_error_dict[error_type].append(result_dict[c5][kp][site][error_type])
                if len(sites_error_dict[error_type]) > 0:
                    result_dict[c5][kp]['mean'][error_type] = np.mean(sites_error_dict[error_type])
                    result_dict[c5][kp]['median'][error_type] = np.median(sites_error_dict[error_type])
    return result_dict

def plot_error_cdf(result_dict, error_type, site='mean'):
    """Plots the cumulative distribution function for error metrics"""
    # Choose error type to plot
    fig,ax = plt.subplots(figsize=(4,3))

    # List out all errors in that type
    error_list = []
    for c5 in params['c5']:
        for kp in params['kp']:
            error_list.append(result_dict[c5][kp][site][error_type])

    # Evaluate CDF
    sorted_list = np.sort(error_list)
    cdf = np.arange(1, len(sorted_list)+1) / len(sorted_list)

    # Plot
    ax.plot(sorted_list, cdf)
    ax.set_xlabel('Value',fontsize=12)
    ax.set_ylabel('CDF',fontsize=12)
    ax.set_ylim(0,1)
    ax.set_xlim(sorted_list[0],sorted_list[-1])
    ax.tick_params(length=5)
    label = error_type.replace('_',' ').replace('snow','snow ')
    ax.set_title(f'Cumulative distribution of {label}')
    plt.show()

def add_normalized(result_dict, error_lims=error_lims, subset='calib', pareto=False):
    """
    Normalizes error between 0-1 for the min-max of a given metric
    
    Returns result_dict with normalized error
    """

    # Grab list of all error metrics
    all_error = list(result_dict['0.026']['2.5']['B'].keys())
    all_error.remove('run_no')
    all_error.remove('set_no')
    all_error.remove('2024_run_no')
    all_error.remove('2024_set_no')

    # # For calibration data
    # if subset in ['calib','valid']:
    #     error_lims = {f'{k}_calib': v for k, v in error_lims.items()}

    # Create storage for minimum/maximum error of each error type
    error_extremes_dict = {}
    for site in all_sites:
        error_extremes_dict[site] = {}
        for error_type in all_error:
            error_extremes_dict[site][error_type] = {'min':np.inf,'max':0}

    # Go through every error and store the extremes
    if pareto:
        all_combos = pareto
    else:
        all_combos = itertools.product(params['c5'],params['kp'])
    
    for c5,kp in all_combos:
        for site in all_sites:
            if site in result_dict[c5][kp]:
                for error_type in result_dict[c5][kp][site]:
                    if 'AE' in error_type:
                        current_value = result_dict[c5][kp][site][error_type]

                        # Check if it's a bad run and skip if so
                        error_name = error_type.split('_')[0]
                        if error_name in error_lims:
                            if current_value > error_lims[error_name]:
                                continue
                        else:
                            continue

                        # Acceptable run: compare error metrics to the running extremes
                        if current_value < error_extremes_dict[site][error_type]['min']:
                            error_extremes_dict[site][error_type]['min'] = current_value
                        if current_value > error_extremes_dict[site][error_type]['max']:
                            error_extremes_dict[site][error_type]['max'] = current_value

    # Divide each value by the minimum to get error_norm
    for c5 in params['c5']:
        for kp in params['kp']:
            if pareto:
                include = True if (c5,kp) in pareto else False
            else:
                include = True
            for site in all_sites:
                if site in result_dict[c5][kp]:
                    list_errors = copy.deepcopy(result_dict[c5][kp][site])
                    for error_type in list_errors:
                        if '_no' not in error_type:
                            current_value = result_dict[c5][kp][site][error_type]
                            min_value = error_extremes_dict[site][error_type]['min']
                            max_value = error_extremes_dict[site][error_type]['max']
                            if (max_value - min_value) > 0 and include:
                                scaled_value = (current_value - min_value) / (max_value - min_value)
                                result_dict[c5][kp][site][error_type+'_norm'] = scaled_value
                            else:
                                result_dict[c5][kp][site][error_type+'_norm'] = np.inf
    
    return result_dict

def pareto_sweep(points):
    """
    Returns the indices of the pareto front points in the passed array
    """
    # Sort points by x-coordinate, then by y-coordinate
    sorted_indices = np.argsort(points[:, 0])
    sorted_points = points[sorted_indices]
    
    pareto_indices = []
    current_min_y = np.inf

    for i, point in enumerate(sorted_points):
        if point[1] < current_min_y: # F
            pareto_indices.append(sorted_indices[i])  # Store the original index
            current_min_y = point[1]

    return np.array(pareto_indices)

def get_best(error_list, result_dict, add_weights=None, site='mean', subset='calib', prints=False):
    """
    Uses normalized error 
    """
    # Get name of error
    error_name = '_MAE_norm'
    if subset in ['calib','valid']:
        error_name = error_name.replace('_norm','_'+key+'_norm')

    # Get weights arrays
    weights = []
    for i in range(len(error_list)):
        zeros = [0]*len(error_list)
        zeros[i] = 1
        weights.append(zeros)
    weights.append(np.ones(len(error_list)) * 1/len(error_list))
    if add_weights is not None:
        for item in add_weights:
            assert len(item) == len(error_list), 'Check formatting of add_weights'
            weights.append(item)

    # Loop through weights
    if prints:
        print('Weights:')
        print(' '.join(f'{x:>10}' for x in error_list),'    best')
    best_all = []
    for weight in weights:
        assert np.abs(np.sum(weight) - 1) < 1e-3
        weighted_list = [np.inf]
        for c5 in params['c5']:
            for kp in params['kp']:
                errors = []
                for error in error_list:
                    errors.append(result_dict[c5][kp][site][error+'_MAE_calib_norm'])
                weighted_error = np.sum(np.array(weight) * np.array(errors))
                if len(weighted_list) > 0 and weighted_error < np.nanmin(weighted_list):
                    best = (c5,kp)
                weighted_list.append(weighted_error)
        if prints:
            print(' '.join(f'{x:>10}' for x in weight),'    ',best)
        best_all.append(best)
    return best_all