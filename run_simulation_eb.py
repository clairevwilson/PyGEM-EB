# External libraries
import numpy as np
import xarray as xr
import pandas as pd
from multiprocessing import Pool
# Internal libraries
import pygem_eb.input as eb_prms
import pygem_eb.climate_processing as climate_processing
import pygem_eb.massbalance as mb

# assert eb_prms.glac_no not in ['01.00570'], 'EB model can currently only run Gulkana glacier'
climateds = climate_processing.getClimateData()

# Set up files for storage
bin_idx = range(eb_prms.n_bins)
if eb_prms.store_data:
    time_to_store = pd.date_range(eb_prms.startdate,eb_prms.enddate,freq=eb_prms.storage_freq)
    zeros = np.zeros([len(time_to_store),eb_prms.n_bins,eb_prms.max_nlayers])
    all_variables = xr.Dataset(data_vars = dict(
            SWin = (['time','bin'],zeros[:,:,0],{'units':'W m-2'}),
            SWout = (['time','bin'],zeros[:,:,0],{'units':'W m-2'}),
            LWin = (['time','bin'],zeros[:,:,0],{'units':'W m-2'}),
            LWout = (['time','bin'],zeros[:,:,0],{'units':'W m-2'}),
            rain = (['time','bin'],zeros[:,:,0],{'units':'W m-2'}),
            sensible = (['time','bin'],zeros[:,:,0],{'units':'W m-2'}),
            latent = (['time','bin'],zeros[:,:,0],{'units':'W m-2'}),
            meltenergy = (['time','bin'],zeros[:,:,0],{'units':'W m-2'}),
            melt = (['time','bin'],zeros[:,:,0],{'units':'m w.e.'}),
            refreeze = (['time','bin'],zeros[:,:,0],{'units':'m w.e.'}),
            runoff = (['time','bin'],zeros[:,:,0],{'units':'m w.e.'}),
            accum = (['time','bin'],zeros[:,:,0],{'units':'m w.e.'}),
            airtemp = (['time','bin'],zeros[:,:,0],{'units':'C'}),
            surftemp = (['time','bin'],zeros[:,:,0],{'units':'C'}),
            snowtemp = (['time','bin','layer'],zeros,{'units':'C'}),
            watercont = (['time','bin','layer'],zeros,{'units':'kg m-2'}),
            layerheight = (['time','bin','layer'],zeros,{'units':'m'}),
            snowdensity = (['time','bin','layer'],zeros,{'units':'kg m-3'}),
            snowdepth = (['time','bin'],zeros[:,:,0],{'units':'m'})
            ),
            coords=dict(
                time=(['time'],time_to_store),
                bin = (['bin'],bin_idx),
                layer=(['layer'],np.arange(eb_prms.max_nlayers))
                ))
    all_variables.to_netcdf(eb_prms.output_name+'.nc')

# ===== RUN ENERGY BALANCE =====
#loop through bins here so EB script is set up for only one bin (1D data)
if eb_prms.parallel:
    def run_mass_balance(bin):
        massbal = mb.massBalance(bin,climateds)
        massbal.main(climateds)
    processes_pool = Pool(eb_prms.n_bins)
    processes_pool.map(run_mass_balance,range(eb_prms.n_bins))
else:
    for bin in np.arange(eb_prms.n_bins):
        # initialize variables to store from mass balance
        massbal = mb.massBalance(bin,climateds)
        results = massbal.main(climateds)
        
        if bin<eb_prms.n_bins:
            print('Success: moving onto bin',bin+1)