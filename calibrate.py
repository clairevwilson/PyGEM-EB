import pandas as pd
import numpy as np
import xarray as xr
import sys,os

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self,exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        return
    
# import model
import pygem_eb.input as eb_prms
eb_prms.enddate = eb_prms.startdate + pd.Timedelta(hours=2)
with HiddenPrints():
    import run_simulation_eb as sim

# read command line args
args = sim.get_args()
args.startdate = pd.to_datetime('2016-05-11 00:30')
args.enddate = pd.to_datetime('2016-07-18 00:30')
args.store_data = True
args.new_file = False
args.debug = False

# ===== CALIBRATION DATA =====
# GULKANA
# surface height change
stake_df = pd.read_csv('~/research/MB_data/Stakes/gulkanaAB23_ALL.csv')
stake_df.index = pd.to_datetime(stake_df['Date'])
dates_index = pd.date_range(eb_prms.startdate,eb_prms.enddate) - pd.Timedelta(minutes=30)
stake_df = stake_df.loc[dates_index]
# snow temperatures
temp_df = pd.read_csv('~/research/MB_data/Gulkana/field_data/iButton_2023_all.csv')
temp_df.index = pd.to_datetime(temp_df['Datetime'])
temp_df = temp_df.loc[eb_prms.startdate:eb_prms.enddate]
h0 = np.array([10,40,80,120,160,200,240,280,320,350])

# function for adjusting parameters
def choose_param(current,past,sign,varname):
    if varname in ['ksp_BC']:
        dx = d_BC*sign
    elif varname in ['ksp_dust']:
        dx = d_dust*sign
    
    new = current + dx
    diff = np.abs(np.array(past) - new)
    while np.any(diff < 1e-4):
        new = np.mean(np.array([current,new]))
        diff = np.abs(np.array(past) - new)
    if new < 0:
        if 'BC' in varname:
            new = ksp_BC_0 + 0.1
            d_BC /= 2
        elif 'dust' in varname:
            new = ksp_dust_0 + 0.1
            d_dust /= 2
    return new