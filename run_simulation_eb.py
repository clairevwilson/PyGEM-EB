import argparse
import time
import os
# External libraries
import numpy as np
import xarray as xr
import pandas as pd
from multiprocessing import Pool
# Internal libraries
import pygem_eb.input as eb_prms
import pygem_eb.massbalance as mb
import pygem_eb.climate as climutils
from shading.shading import Shading

# Start timer
start_time = time.time()

# ===== INITIALIZE UTILITIES =====
def get_args(parse=True):
    parser = argparse.ArgumentParser(description='pygem-eb model runs')
    # GLACIER INFORMATION
    parser.add_argument('-glac_no', action='store', default=eb_prms.glac_no,
                        help='',nargs='+')
    parser.add_argument('-elev',action='store',default=eb_prms.elev,type=float,
                        help='Elevation in m a.s.l.')
    parser.add_argument('-site',action='store',default='',type=str,
                        help='Site name')
    parser.add_argument('-lat',action='store',default=eb_prms.lat,type=float,
                        help='Site latitude')
    parser.add_argument('-lon',action='store',default=eb_prms.lon,type=float,
                        help='Site longitude')
    parser.add_argument('-slope',action='store',default=eb_prms.slope,type=float,
                        help='Site slope')
    parser.add_argument('-aspect',action='store',default=eb_prms.aspect,type=float,
                        help='Site aspect')
    parser.add_argument('-initial_snow_depth',action='store',type=float,
                        default=eb_prms.initial_snow_depth,
                        help='Snow depth in m')
    parser.add_argument('-initial_firn_depth',action='store',type=float,
                        default=eb_prms.initial_firn_depth,
                        help='Firn depth in m')
    
    # MODEL TIME
    parser.add_argument('-start','--startdate', action='store', type=str, 
                        default=eb_prms.startdate,
                        help='pass str like datetime of model run start')
    parser.add_argument('-end','--enddate', action='store', type=str,
                        default=eb_prms.enddate,
                        help='pass str like datetime of model run end')
    
    # USER OPTIONS
    parser.add_argument('-use_AWS', action='store_true',
                        default=eb_prms.use_AWS,help='use AWS or just reanalysis?')
    parser.add_argument('-use_threads', action='store_true',
                        help='use threading to import climate data?')
    parser.add_argument('-store_data', action='store_true', 
                        help='store the model output?')
    parser.add_argument('-out',action='store',type=str,default=eb_prms.output_name,
                        help='output file name excluding extension (.nc)')
    parser.add_argument('-debug', action='store_true', 
                        default=eb_prms.debug, help='print debug statements?')
    
    # ALBEDO SWITCHES
    parser.add_argument('-switch_LAPs',action='store', type=int,
                        default=eb_prms.switch_LAPs, help='')
    parser.add_argument('-switch_melt',action='store', type=int, 
                        default=eb_prms.switch_melt, help='')
    parser.add_argument('-switch_snow',action='store', type=int,
                        default=eb_prms.switch_snow, help='')
    
    # CALIBRATED PARAMETERS
    parser.add_argument('-k_snow',default=eb_prms.method_conductivity,action='store',
                        help='Thermal conductivity of snow')
    parser.add_argument('-a_ice',default=eb_prms.albedo_ice,action='store',type=float,
                        help='Broadband albedo of ice')
    parser.add_argument('-kw',default=eb_prms.wind_factor,action='store',type=float,
                        help='Multiplicative wind factor')
    parser.add_argument('-kp',default=eb_prms.kp,action='store',type=float,
                        help='Multiplicative precipitation factor')
    parser.add_argument('-Boone_c5',default=eb_prms.Boone_c5,action='store',type=float,
                        help='Parameter for Boone densification scheme')
    parser.add_argument('-roughness_ice',default=eb_prms.roughness_ice,action='store',type=float,
                        help='Surface roughness of ice')
    
    # PARALLELIZATION
    parser.add_argument('-n','--n_simultaneous_processes',default=1,type=int,
                        help='Number of parallel processes to run')
    parser.add_argument('-task_id',default=-1,type=int,
                        help='Task ID if submitted as batch job')
    
    # DUMMY ARG FOR JUPYTER COMPATIBILITY
    parser.add_argument('-f', '--fff', help='Dummy arg to fool ipython', default='1')

    if parse:
        args = parser.parse_args()
        return args
    else:
        return parser
    
def get_site_table(site_df, args):
    """
    Loads the table for sites at this glacier and
    stores information in args
    """
    # Special handling for Gulkana AWS site
    if args.site == 'AWS' and args.glac_name == 'gulkana':
        site = 'B'
    else:
        site = args.site

    # Get site-specific variables
    args.lat = site_df.loc[args.site]['lat']
    args.lon = site_df.loc[args.site]['lon']
    args.elev = site_df.loc[site]['elevation']
    args.slope = site_df.loc[site]['slope']
    args.aspect = site_df.loc[site]['aspect']
    args.sky_view = site_df.loc[site]['sky_view']

    # Snow and firn depth may be specified in the site_constants table
    if 'snowdepth' in site_df.columns:
        args.initial_snow_depth = site_df.loc[site]['snowdepth']
    if 'firndepth' in site_df.columns:
        args.initial_firn_depth = site_df.loc[site]['firndepth']

    # *****Special HARD-CODED handling for Gulkana*****
    if eb_prms.glac_name == 'gulkana':
        # Set scaling albedo
        slope = (0.485 - 0.315)/(site_df.loc['B','elevation'] - site_df.loc['A','elevation'])
        intercept = 0.315
        args.a_ice = intercept + (args.elev - site_df.loc['A','elevation'])*slope
        args.a_ice = min(0.485,args.a_ice)

        # Set initial density profile from measurements
        if args.site not in ['AB','ABB','BD']:
            if pd.to_datetime(args.startdate) > pd.to_datetime('2023-12-31'):
                args.initial_density_fp = f'data/Gulkana/gulkana{args.site}density24.csv'
            else:
                args.initial_density_fp = f'data/Gulkana/gulkana{args.site}meandensity.csv'
        elif args.site in ['ABB','BD']:
            args.initial_density_fp = 'data/Gulkana/gulkanaBdensity24.csv'
        elif args.site in ['AB']:
            args.initial_density_fp = 'data/Gulkana/gulkanaAUdensity24.csv'
    
    return args
    
def check_inputs(glac_no, args):
    """
    Checks that the glacier point has all required inputs.
    First the 'shade.csv' file is checked. If it doesn't exist,
    the shading model is run. This produces the 'site_constants.csv'
    file which contains the following for the specified point:
    - Elevation
    - Slope/aspect
    - Sky-view factor
    """
    # Files are associated with the glacier number or name
    glacier = glac_no
    if glacier in eb_prms.glac_props:
        args.AWS_fn = eb_prms.AWS_fp + eb_prms.glac_props[glacier]['AWS_fn']
        glacier = eb_prms.glac_props[glacier]['name']
    else:
        args.AWS_fn = eb_prms.AWS_fn
        glacier = glacier.replace('.','_')
    args.glac_name = glacier

    # Specify filepaths
    data_fp = os.getcwd() + '/data/'
    args.shading_fp = f'shading/out/{glacier}{args.site}_shade.csv'
    args.dem_fp = f'shading/in/{glacier}/dem.tif'
    args.initial_density_fp = eb_prms.initial_density_fp
    args.initial_temp_fp = eb_prms.initial_temp_fp
    args.initial_grains_fp = eb_prms.initial_grains_fp
    args.initial_LAP_fp = eb_prms.initial_LAP_fp

    # Check if the shading file exists
    if not os.path.exists(args.shading_fp):
        print(args.shading_fp)
        print(f'No shading file for',args.shading_fp.split('/')[-1])
        # check DEM exists
        if os.path.exists(args.dem_fp):
            print('Running shading model...')
            model = Shading(args.dem_fp)
            # ****** Need to specify the point on the glacier to do, lat lon...
            print('Done...')
        else:
            print(f'DEM not found: add to shading/in/{glacier}/')
    
    # Check if the site_constants table exists
    if glacier in os.listdir(data_fp):
        # Load site constants table
        site_fp = os.path.join(data_fp,glacier+'/site_constants.csv')
        site_df = pd.read_csv(site_fp,index_col='site')

        # Update args from the site table
        args = get_site_table(site_df, args)

        # Check if there's a usable density profile*****temp?laps?etc
        for fn in os.listdir(data_fp+args.glac_name):
            if args.site in fn and 'density.csv' in fn:
                args.initial_density_fp = f'{data_fp}{args.glac_name}/{fn}'

    else:
        print('WARNING! Glacier site_constants not found: using defaults from input.py')

    # Add site to output filename
    if args.site not in args.out:
        args.out += f'{args.site}_'
       
    return args

def initialize_model(glac_no,args):
    """
    Loads glacier table and climate dataset for one glacier to initialize
    the model inputs.

    Parameters
    ==========
    glac_no : str
        RGI glacier ID
    args : command-line arguments
    
    Returns
    -------
    climate
        Class object from climate.py
    """
    # ===== CHECK GLACIER INPUTS (LAT,LON,ELEV,...) =====
    args = check_inputs(glac_no, args)

    # ===== GET GLACIER CLIMATE =====
    # Initialize the climate class
    climate = climutils.Climate(args)
    # Load in available AWS data, then reanalysis
    if args.use_AWS:
        need_vars = climate.get_AWS(args.AWS_fn)
        climate.get_reanalysis(need_vars)
    else:
        climate.get_reanalysis(climate.all_vars)
    # Check the dataset is ready to go
    climate.check_ds()

    # ===== PRINT MODEL RUN INFO =====
    start = pd.to_datetime(args.startdate)
    end = pd.to_datetime(args.enddate)
    n_months = np.round((end-start)/pd.Timedelta(days=30))
    start_fmtd = start.month_name()+', '+str(start.year)
    print(f'Running {glac_no} at {args.elev} m a.s.l. for {n_months} months starting in {start_fmtd}')

    return climate, args

def run_model(climate,args,store_attrs=None):
    """
    Executes model functions in parallel or series and
    stores output data.

    Parameters
    ==========
    climate
        Class object with climate data from initialize_model
    args
        Command line arguments from get_args
    store_attrs : dict
        Dictionary of additional metadata to store in the .nc
    """
    # ===== RUN ENERGY BALANCE =====
    massbal = mb.massBalance(args,climate)
    massbal.main()

    # ===== END ENERGY BALANCE =====
    # Get final model run time
    end_time = time.time()
    time_elapsed = end_time-start_time
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print(f'Total Time Elapsed: {time_elapsed:.1f} s')

    # Store metadata in netcdf and save result
    if args.store_data:
        massbal.output.add_vars()
        massbal.output.add_basic_attrs(args,time_elapsed,climate)
        massbal.output.add_attrs(store_attrs)
        out = massbal.output.get_output()
    else:
        print('Success: data was not saved')
        out = None
    
    return out

if __name__ == '__main__':
    args = get_args()
    for gn in args.glac_no:
        climate, args = initialize_model(gn,args)
        time_elapsed = time.time()-start_time
        print(f'Got climate in {time_elapsed:.1f} s')
        out = run_model(climate,args)
        if isinstance(out, xr.Dataset):
            # Get final mass balance
            print(f'Total Mass Loss: {out.melt.sum():.3f} m w.e.')