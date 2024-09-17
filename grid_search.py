import pandas as pd
import numpy as np
import xarray as xr
import os, sys
import matplotlib.pyplot as plt

run_model = True

class HiddenPrints:
    """
    Class to hide prints when running SNICAR
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self,exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        return
    
# import model
import pygem_eb.input as eb_prms
eb_prms.startdate = pd.to_datetime('2000-04-20 00:00')
eb_prms.enddate = eb_prms.startdate + pd.Timedelta(hours=2)
eb_prms.debug = False
import run_simulation_eb as sim

# model parameters
params = {
    'albedo_ice':[0.4,0.5,0.6],
    'k_ice': [2,3,4]
}

# read command line args
args = sim.get_args()
args.enddate = pd.to_datetime('2022-05-21 12:00:00')

# force some args
args.store_data = True
args.parallel = True
args.use_AWS = True

ds_list = []
for albedo_ice in params['albedo_ice']:
    for thermal_cond in params['k_ice']:
        eb_prms.output_name = f'{eb_prms.output_filepath}EB/a_{albedo_ice}_k_{thermal_cond}_'
        eb_prms.constant_conductivity = thermal_cond
        eb_prms.albedo_ice = albedo_ice

        print()
        print('Starting model run with a_ice = ',albedo_ice,'and k_ice = ',thermal_cond)

        if not os.path.exists(eb_prms.output_name+'0_bin0.nc'):
            # initialize the model
            climate = sim.initialize_model(args.glac_no[0],args)

            # run the model
            sim.run_model(climate,args,{'a_ice':str(albedo_ice),
                                            'k_ice':str(thermal_cond)})
        else:
            print('      already exists; skipping')