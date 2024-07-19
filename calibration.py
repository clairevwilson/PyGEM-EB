import pandas as pd
import numpy as np
import xarray as xr
import os
import pygem_eb.input as eb_prms

# import model
eb_prms.startdate = pd.to_datetime('2023-04-18 00:30')
eb_prms.enddate = eb_prms.startdate + pd.Timedelta(hours=2)
eb_prms.new_file = False
import run_simulation_eb as sim
eb_prms.enddate = pd.to_datetime('2023-08-09 00:30')

# ===== CALIBRATION DATA =====
# surface height change
stake_df = pd.read_csv('~/research/MB_data/Stakes/gulkanaAB23_ALL.csv')
stake_df.index = pd.to_datetime(stake_df['Date'])
dates_index = pd.date_range(eb_prms.startdate,eb_prms.enddate) - pd.Timedelta(minutes=30)
stake_df = stake_df.loc[dates_index]
# snow temperatures
temp_df = pd.read_csv('~/research/MB_data/Gulkana/field_data/iButton_2023_all.csv')
temp_df.index = pd.to_datetime(temp_df['Datetime'])
temp_df = temp_df.loc[eb_prms.startdate:eb_prms.enddate]
h = np.array([10,40,80,120,160,200,240,280,320,350])

# model parameters
params = {
    'kp':[0.5,0.8],
    'albedo_ice':[0.1,0.3],
    'k_ice':[1,4]
}

# JIF
args = sim.get_args()
args.startdate = pd.to_datetime('2016-05-11 00:30') # snow sample dates
args.enddate = pd.to_datetime('2016-07-18 00:30')

# initial guess
ksp_BC = 0.5
ksp_dust = 0.05
i = 0
while i < 10:
    eb_prms.output_name = f'{eb_prms.output_filepath}EB/{eb_prms.glac_name}_iter{i+1}'
    eb_prms.ksp_BC = ksp_BC
    eb_prms.ksp_dust = ksp_dust
    i += 1

# i = 0
# parser = sim.getparser()
# args = parser.parse_args()
# for kp in params['kp']:
#     for aice in params['albedo_ice']:
#         for ksp in params['ksp_BC']:
#             for k_ice in params['k_ice']:
#                 eb_prms.output_name = f'{eb_prms.output_filepath}EB/{eb_prms.glac_name}_cal{i}'
#                 eb_prms.ksp_BC = ksp
#                 eb_prms.albedo_ice = aice
#                 eb_prms.kp = kp
#                 eb_prms.k_ice = k_ice

#                 climateds,dates_table,utils = sim.initialize_model('01.00570',args)
#                 ds_run = sim.run_model(climateds,dates_table,utils,args,
#                                     {'ksp_BC':ksp,'albedo_ice':aice,'kp':kp,'k_ice':k_ice})
                
#                 i += 1