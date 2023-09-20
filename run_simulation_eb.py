import argparse
# External libraries
import numpy as np
import xarray as xr
import pandas as pd
from multiprocessing import Pool
# Internal libraries
import pygem_eb.input as eb_prms
import pygem.class_climate as class_climate
import pygem_eb.massbalance as mb
import pygem.pygem_modelsetup as modelsetup

def getparser():
    parser = argparse.ArgumentParser(description='pygem-eb model runs')
    # add arguments
    parser.add_argument('-glac_no', action='store', default=eb_prms.glac_no,
                        help='',nargs='+')
    parser.add_argument('-start','--startdate', action='store', type=str, default=eb_prms.startdate,
                        help='pass str like datetime of model run start')
    parser.add_argument('-end','--enddate', action='store', type=str, default=eb_prms.enddate,
                        help='pass str like datetime of model run end')
    parser.add_argument('-climate_input', action='store', type=str, default=eb_prms.climate_input,
                        help='pass str-like datetime of model run start')
    parser.add_argument('-store_data', action='store', type=str, default=eb_prms.store_data,
                        help='')
    parser.add_argument('-n_bins',action='store',type=int,default=eb_prms.n_bins,
                        help='number of elevation bins')
    return parser

# Initialize arg parser
parser = getparser()
args = parser.parse_args()

n_bins = args.n_bins
# ===== GLACIER AND TIME PERIOD SETUP =====
glacier_table = modelsetup.selectglaciersrgitable(args.glac_no,
                rgi_regionsO1=eb_prms.rgi_regionsO1, rgi_regionsO2=eb_prms.rgi_regionsO2,
                rgi_glac_number=eb_prms.rgi_glac_number, include_landterm=eb_prms.include_landterm,
                include_laketerm=eb_prms.include_laketerm, include_tidewater=eb_prms.include_tidewater)

dates_table = modelsetup.datesmodelrun(startyear=args.startdate, endyear=args.enddate)

gcm = class_climate.GCM(name=eb_prms.ref_gcm_name)
if args.climate_input in ['GCM']:
    # ===== LOAD CLIMATE DATA =====
    tp_data, data_hours = gcm.importGCMvarnearestneighbor_xarray(gcm.prec_fn, gcm.prec_vn, glacier_table,dates_table)
    temp_data, data_hours = gcm.importGCMvarnearestneighbor_xarray(gcm.temp_fn, gcm.temp_vn, glacier_table,dates_table)
    dtemp_data, data_hours = gcm.importGCMvarnearestneighbor_xarray(gcm.dtemp_fn, gcm.dtemp_vn, glacier_table,dates_table)
    sp_data, data_hours = gcm.importGCMvarnearestneighbor_xarray(gcm.sp_fn, gcm.sp_vn, glacier_table,dates_table)
    tcc, data_hours = gcm.importGCMvarnearestneighbor_xarray(gcm.tcc_fn, gcm.tcc_vn, glacier_table,dates_table)
    SWin, data_hours = gcm.importGCMvarnearestneighbor_xarray(gcm.surfrad_fn, gcm.surfrad_vn, glacier_table,dates_table) 
    uwind, data_hours = gcm.importGCMvarnearestneighbor_xarray(gcm.uwind_fn, gcm.uwind_vn, glacier_table,dates_table)                                                      
    vwind, data_hours = gcm.importGCMvarnearestneighbor_xarray(gcm.vwind_fn, gcm.vwind_vn, glacier_table,dates_table)
    elev_data = gcm.importGCMfxnearestneighbor_xarray(gcm.elev_fn, gcm.elev_vn, glacier_table)
    wind = np.sqrt(np.power(uwind[0],2)+np.power(vwind[0],2))
    winddir = np.arctan2(-uwind[0],-vwind[0]) * 180 / np.pi
    LWin = np.empty(len(data_hours))
    LWin[:] = np.nan
    LWout = LWin.copy()
    SWout = LWin.copy()
    rh_data = LWin.copy()
    SWin = SWin[0]
    tcc = tcc[0]
    ntimesteps = len(data_hours)
elif args.climate_input in ['AWS']:
    aws = class_climate.AWS(eb_prms.AWS_fn,dates_table)
    temp_data = aws.temp
    tp_data = aws.tp
    dtemp_data = aws.dtemp
    rh_data = aws.rh
    SWin = aws.SWin
    SWout = aws.SWout
    LWin = aws.LWin
    LWout = aws.LWout
    NR = aws.NR
    wind = aws.wind
    winddir = aws.winddir
    tcc = aws.tcc
    sp_data = aws.sp
    elev_data = aws.elev
    ntimesteps = len(temp_data)

#initialize variables to be adjusted
temp = np.zeros((n_bins,ntimesteps))
tp = np.zeros((n_bins,ntimesteps))
sp = np.zeros((n_bins,ntimesteps))
dtemp = np.zeros((n_bins,ntimesteps))

# define function to calculate vapor pressure (needed for RH)
e_func = lambda T_C: 610.94*np.exp(17.625*T_C/(T_C+243.04))  #vapor pressure in Pa, T in Celsius
#loop through each elevation bin and adjust climate variables by lapse rate/barometric law
for idx,z in enumerate(eb_prms.bin_elev):
    temp[idx,:] = temp_data + eb_prms.lapserate*(z-elev_data)
    tp[idx,:] = tp_data*(1+eb_prms.precgrad*(z-elev_data))*2 # *eb_prms.kp for GCM******
    sp[idx,:] = sp_data*np.power((temp_data + eb_prms.lapserate*(z-elev_data)+273.15)/(temp_data+273.15),
                        -eb_prms.gravity*eb_prms.molarmass_air/(eb_prms.R_gas*eb_prms.lapserate))
    if not np.all(np.isnan(rh_data)): # if RH is not empty, get dtemp data from it
        dtemp_data = rh_data / 100 * e_func(temp[idx,:])
        dtemp[idx,:] = dtemp_data + eb_prms.lapserate_dew*(z-elev_data)-273.15
        rh = np.array([rh_data]*n_bins)
    else:
        rh = e_func(dtemp) / e_func(temp) * 100

dates = pd.date_range(args.startdate,args.enddate,freq='h')

# ===== SET UP CLIMATE DATASET =====
bin_idx = np.arange(0,n_bins)
climateds = xr.Dataset(data_vars = dict(
    bin_elev = (['bin'],eb_prms.bin_elev,{'units':'m'}),
    SWin = (['time'],SWin,{'units':'J m-2'}),
    SWout = (['time'],SWout,{'units':'J m-2'}),
    LWin = (['time'],LWin,{'units':'J m-2'}),
    LWout = (['time'],LWout,{'units':'J m-2'}),
    NR = (['time'],NR,{'units':'J m-2'}),
    tcc = (['time'],tcc,{'units':'0-1'}),
    wind = (['time'],wind,{'units':'m s-1'}),
    winddir = (['time'],winddir,{'units':'deg'})),
    coords = dict(
        bin=(['bin'],bin_idx),
        time=(['time'],dates)
        ))

climateds = climateds.assign(bin_temp = (['bin','time'],temp,{'units':'C'}))
climateds = climateds.assign(bin_tp = (['bin','time'],tp,{'units':'m'}))
climateds = climateds.assign(bin_sp = (['bin','time'],sp,{'units':'Pa'}))
climateds = climateds.assign(bin_rh = (['bin','time'],rh,{'units':'%'}))

# ===== RUN ENERGY BALANCE =====
#loop through bins here so EB script is set up for only one bin (1D data)
if eb_prms.parallel:
    def run_mass_balance(bin):
        massbal = mb.massBalance(bin,dates_table)
        massbal.main(args,climateds)
    processes_pool = Pool(args.n_bins)
    processes_pool.map(run_mass_balance,range(args.n_bins))
else:
    for bin in np.arange(args.n_bins):
        # initialize variables to store from mass balance
        massbal = mb.massBalance(bin,dates_table)
        results = massbal.main(args,climateds)
        
        if bin<args.n_bins-1:
            print('Success: moving onto bin',bin+1)
        else:
            massbal.output.addAttrs(args)
            if str(args.store_data)=='True':
                print('Success: saving to',eb_prms.output_name+'.nc')