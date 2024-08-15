import pandas as pd
import numpy as np
import xarray as xr
import os, sys

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
eb_prms.store_data = True
eb_prms.new_file = False
eb_prms.debug = False

# ===== CALIBRATION DATA =====
# JIF
data_fp = '/home/claire/research/Data/Nagorski/bcdust.csv'
df = pd.read_csv(data_fp,index_col=0).iloc[:-1]

# read command line args
args = sim.get_args()
args.startdate = pd.to_datetime('2016-05-11 00:30') # snow sample dates
args.enddate = pd.to_datetime('2016-07-18 00:30')

# initial guess
ksp_BC_0 = 0.8
ksp_dust_0 = 0.1
d_BC = 0.1
d_dust = 0.05

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

def get_outputs(out):
    final_snowdepth = out.snowdepth.isel(bin=0,time=-1)
    assert final_snowdepth < 1e-3, 'snow is gone: adjust initial guess'
    BC_surf_final = out.layerBC.isel(bin=0,layer=0,time=-1).values
    dust_surf_final = out.layerdust.isel(bin=0,layer=0,time=-1).values
    return BC_surf_final, dust_surf_final

for glacier in df.index[3:]:
    # snowdepth = df.loc[glacier,'snow_depth']
    snowdepth = 5
    print('Calibrating ksp for',glacier,f'with {snowdepth}m of snow')
    # inputs
    eb_prms.glac_name = glacier
    eb_prms.initial_snowdepth = [snowdepth]
    eb_prms.initial_LAP_fp = f'~/research/Data/Nagorski/May_{glacier}_BC.csv'
    
    # measurements
    depth_BC_meas = np.arange(0.125,1,0.25)
    depth_dust_meas = np.arange(0.25,1,0.5)
    BC_surf_meas = df.loc[glacier,'Surface_BC_July']
    dust_surf_meas = df.loc[glacier,'Surface_dust_July']
    if glacier in ['Mend-2','Taku']:
        varnames = {'BC':[],'dust':[]}
        for j in depth_BC_meas*100:
            varnames['BC'].append(f'BC_July_{int(j-12.5)}_{int(j+12.5)}')
        for j in depth_dust_meas*100:
            varnames['dust'].append(f'dust_July_{int(j-25)}_{int(j+25)}')
        BC_meas = df.loc[glacier,varnames['BC']].to_numpy()
        dust_meas = df.loc[glacier,varnames['dust']].to_numpy()

    # initial guess
    ksp_BC = ksp_BC_0
    ksp_dust = ksp_dust_0
    i = 0
    result = {'BC':{'ksp':[],'error':[]},
                 'dust':{'ksp':[],'error':[]}}
    while i < 10:
        result['BC']['ksp'].append(ksp_BC)
        result['dust']['ksp'].append(ksp_dust)
        eb_prms.output_name = f'{eb_prms.output_filepath}EB/{glacier}_iter{i+1}'
        eb_prms.ksp_BC = ksp_BC
        eb_prms.ksp_dust = ksp_dust
        args.bin_elev = [df.loc[glacier,'Elev']]

        with HiddenPrints():
            # initialize the model
            gn = df.loc[glacier]['RGIId'][1:-1]
            args.glac_no[0] = gn
            climate = sim.initialize_model(gn,args)

            # run the model
            out = sim.run_model(climate,args,{'ksp_BC':str(ksp_BC),
                                              'ksp_dust':str(ksp_dust)})
        
        # parse the output
        BC_surf_final,dust_surf_final = get_outputs(out,ksp_BC,ksp_dust)

        # compare
        BC_surf_err = (BC_surf_meas - BC_surf_final) / BC_surf_meas * 100
        dust_surf_err = (dust_surf_meas - dust_surf_final) / dust_surf_meas * 100
        
        # prints
        print(f'Iteration {i+1}:   BC ksp = {ksp_BC:.3f}      Error = {BC_surf_err:.1f} %')
        print(f'               dust ksp = {ksp_dust:.3f}    Error: {dust_surf_err:.1f} %')

        # update parameter
        sign_BC = 1 if BC_surf_err < 0 else -1
        sign_dust = 1 if dust_surf_err < 0 else -1
        ksp_BC = choose_param(ksp_BC,result['BC']['ksp'],sign_BC,'ksp_BC')
        ksp_dust = choose_param(ksp_dust,result['dust']['ksp'],sign_dust,'ksp_dust')

        result['BC']['error'].append(BC_surf_err)
        result['dust']['error'].append(dust_surf_err)

        prev_sign_BC = sign_BC
        prev_sign_dust = sign_dust
        i += 1

    # determine best of the parameter set
    iBC = np.argmin(np.abs(np.array(result['BC']['error'])))
    ksp_BC = result['BC']['ksp'][iBC]
    idust = np.argmin(np.abs(np.array(result['dust']['error'])))
    ksp_dust = result['dust']['ksp'][idust]

    print(f'Best parameters: ksp_BC = {ksp_BC}      ksp_dust = {ksp_dust}')

    # rerun model on best parameters
    eb_prms.output_name = f'{eb_prms.output_filepath}EB/{glacier}_best'
    eb_prms.ksp_BC = ksp_BC
    eb_prms.ksp_dust = ksp_dust
    with HiddenPrints():
        out = sim.run_model(climate,args,{'ksp_BC':str(ksp_BC),
                                          'ksp_dust':str(ksp_dust)})

    BC_surf_err,dust_surf_err = get_outputs(out,ksp_BC,ksp_dust)
    print(f'                 BC error: {BC_surf_err:.1f} %    Dust error: {dust_surf_err:.1f} %')

    if glacier in ['Mend-2','Taku']:
        BC_final = out.layerBC.isel(bin=0,time=-1).values
        dust_final = out.layerdust.isel(bin=0,time=-1).values
        lh = out.layerheight.isel(bin=0,time=-1).values

        # get depth of final model layers
        n = len(lh)
        depth_final = np.array([np.sum(lh[:i+1])-(lh[i]/2) for i in range(n)])

        # interpolate to new depths
        BC_final = np.interp(depth_BC_meas,depth_final,BC_final)
        dust_final = np.interp(depth_dust_meas,depth_final,dust_final)

        # calculate error
        BC_err = np.mean((BC_meas - BC_final)/BC_meas*100)
        dust_err = np.mean((dust_meas - dust_final)/dust_meas*100)
        print(f'Final iteration for {glacier}: ')
        print(f'    MSE BC = {BC_err:.1f}     MSE dust = {dust_err:.1f}')
    print('======================================================')

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