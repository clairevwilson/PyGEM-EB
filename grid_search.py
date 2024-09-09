import pandas as pd
import numpy as np
import xarray as xr
import os, sys
import matplotlib.pyplot as plt
from pygem_eb.processing.plotting_fxns import plot_multiyear_mb

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
eb_prms.startdate = pd.to_datetime('2000-04-21 00:30')
eb_prms.enddate = eb_prms.startdate + pd.Timedelta(hours=2)
eb_prms.new_file = False
eb_prms.debug = False
with HiddenPrints():
    import run_simulation_eb as sim
eb_prms.store_data = True

# model parameters
params = {
    'albedo_ice':[0.1,0.3],
    'k_ice':[0.5,2]
}

# read command line args
args = sim.get_args()
args.enddate = pd.to_datetime('2002-08-21 00:30')

ds_list = []
for albedo_ice in params['albedo_ice']:
    for thermal_cond in params['k_ice']:
        eb_prms.output_name = f'{eb_prms.output_filepath}EB/a_{albedo_ice}_k_{thermal_cond}'
        eb_prms.constant_conductivity = thermal_cond
        eb_prms.albedo_ice = albedo_ice
        print('Starting a_ice = ',albedo_ice,'and k_ice = ',thermal_cond)

        if run_model and not os.path.exists(eb_prms.output_name+'.nc'):
            # with HiddenPrints():
            if True:
                # initialize the model
                climate = sim.initialize_model(args.glac_no[0],args)

                # run the model
                out = sim.run_model(climate,args,{'a_ice':str(albedo_ice),
                                                'k_ice':str(thermal_cond)})
                ds_list.append(out)
        else:
            ds = xr.open_dataset(f'{eb_prms.output_filepath}EB/a_{albedo_ice}_k_{thermal_cond}.nc')
            ds = ds.assign_coords(time=ds.time.values + pd.Timedelta(days=365*10))
            ds_list.append(ds)
        
        print('Finished')

mb_fp = os.getcwd()+'/../MB_data/Gulkana/Input_Gulkana_Glaciological_Data.csv'
end_year = args.enddate.year if args.enddate.month > 7 else args.enddate.year-1
years_model = np.arange(args.startdate.year+1,end_year+1)
mb_df = pd.read_csv(mb_fp)
for site in ['AB','B','D']:
    years_site = np.unique(mb_df.loc[mb_df['site_name']==site]['Year'])
    years = list(set(years_site)&set(years_model))
    years = np.sort(np.array(years))
    years = [2011]
    print(site,years)
    fig,ax = plot_multiyear_mb(ds_list,mb_df,years,site)
    plt.savefig(f'multiyear_run_{site}.png',dpi=200)