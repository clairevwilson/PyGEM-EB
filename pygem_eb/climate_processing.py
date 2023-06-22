import numpy as np
import os
import xarray as xr
import pandas as pd
import pygem_eb.input as eb_prms
import pygem.pygem_modelsetup as modelsetup
import class_climate

def getClimateData():
    # ===== GLACIER AND TIME PERIOD SETUP =====
    glacier_table = modelsetup.selectglaciersrgitable(eb_prms.glac_no,
                    rgi_regionsO1=eb_prms.rgi_regionsO1, rgi_regionsO2=eb_prms.rgi_regionsO2,
                    rgi_glac_number=eb_prms.rgi_glac_number, include_landterm=eb_prms.include_landterm,
                    include_laketerm=eb_prms.include_laketerm, include_tidewater=eb_prms.include_tidewater)

    dates_table = pd.DataFrame({'date' : pd.date_range(eb_prms.startdate,eb_prms.enddate,freq='h')})
    # Extract attributes for dates_table
    dates_table['year'] = dates_table['date'].dt.year
    dates_table['month'] = dates_table['date'].dt.month
    dates_table['day'] = dates_table['date'].dt.day
    dates_table['hour'] = dates_table['date'].dt.hour
    dates_table['daysinmonth'] = dates_table['date'].dt.daysinmonth
    dates_table['timestep'] = np.arange(len(dates_table['date']))
    # Set index
    dates_table.set_index('timestep', inplace=True)

    # Remove leap year days if user selected this with option_leapyear
    if eb_prms.option_leapyear == 0:
        # First, change 'daysinmonth' number
        mask1 = dates_table['daysinmonth'] == 29
        dates_table.loc[mask1,'daysinmonth'] = 28
        # Next, remove the 29th days from the dates
        mask2 = ((dates_table['month'] == 2) & (dates_table['day'] == 29))
        dates_table.drop(dates_table[mask2].index, inplace=True)
        dates_table['timestep'] = np.arange(len(dates_table['date']))
        dates_table.set_index('timestep', inplace=True)

    # Add column for water year
    # Water year for northern hemisphere using USGS definition (October 1 - September 30th),
    # e.g., water year for 2000 is from October 1, 1999 - September 30, 2000
    dates_table['wateryear'] = dates_table['year']
    for step in range(dates_table.shape[0]):
        if dates_table.loc[step,'month'] >= 10:
            dates_table.loc[step,'wateryear'] = dates_table.loc[step,'year'] + 1

    # GET EACH DATA VARIABLE FROM GCM or AWS
    gcm = class_climate.GCM(name=eb_prms.ref_gcm_name)
    if eb_prms.climate_input in ['GCM']:
        # ===== LOAD CLIMATE DATA =====
        gcm_tp, data_hours = gcm.importGCMvarnearestneighbor_xarray(gcm.prec_fn, gcm.prec_vn, glacier_table,dates_table)
        temp, data_hours = gcm.importGCMvarnearestneighbor_xarray(gcm.temp_fn, gcm.temp_vn, glacier_table,dates_table)
        dtemp, data_hours = gcm.importGCMvarnearestneighbor_xarray(gcm.dtemp_fn, gcm.dtemp_vn, glacier_table,dates_table)
        sp, data_hours = gcm.importGCMvarnearestneighbor_xarray(gcm.press_fn, gcm.press_vn, glacier_table,dates_table)
        tcc, data_hours = gcm.importGCMvarnearestneighbor_xarray(gcm.tcc_fn, gcm.tcc_vn, glacier_table,dates_table)
        SWin, data_hours = gcm.importGCMvarnearestneighbor_xarray(gcm.surfrad_fn, gcm.surfrad_vn, glacier_table,dates_table) 
        uwind, data_hours = gcm.importGCMvarnearestneighbor_xarray(gcm.uwind_fn, gcm.uwind_vn, glacier_table,dates_table)                                                      
        vwind, data_hours = gcm.importGCMvarnearestneighbor_xarray(gcm.vwind_fn, gcm.vwind_vn, glacier_table,dates_table)
        gcm_elev = gcm.importGCMfxnearestneighbor_xarray(gcm.elev_fn, gcm.elev_vn, glacier_table)
        wind = np.sqrt(np.power(uwind[0],2)+np.power(vwind[0],2))
        LWin = np.empty(len(data_hours))
        LWout = np.empty(len(data_hours))
        SWout = np.empty(len(data_hours))
        SWin = SWin[0]
        tcc = tcc[0]

        #initialize variables to be adjusted
        temp = np.zeros((n_bins,len(data_hours)))
        tp = np.zeros((n_bins,len(data_hours)))
        sp = np.zeros((n_bins,len(data_hours)))
        rh = np.zeros((n_bins,len(data_hours)))
        dtemp = np.zeros((n_bins,len(data_hours)))

        # define function to calculate vapor pressure (needed for RH)
        e_func = lambda T_C: 610.94*np.exp(17.625*T_C/(T_C+243.04))  #vapor pressure in Pa, T in Celsius
        #loop through each elevation bin and adjust climate variables by lapse rate/barometric law
        for idx,z in enumerate(climateds['bin_elev'].values):
            temp[idx,:] = temp + eb_prms.lapserate*(z-gcm_elev)
            dtemp[idx,:] = dtemp + eb_prms.lapserate_dew*(z-gcm_elev) - 273.15
            tp[idx,:] = gcm_tp*eb_prms.kp*(1+eb_prms.precgrad*(z-gcm_elev))
            sp[idx,:] = sp*np.power((temp + eb_prms.lapserate*(z-gcm_elev)+273.15)/(temp+273.15),
                                -eb_prms.gravity*eb_prms.molarmass_air/(eb_prms.R_gas*eb_prms.lapserate))
            rh[idx,:] = e_func(dtemp[idx,:]) / e_func(temp[idx,:]) * 100
    elif eb_prms.climate_input in ['AWS']:
        dates = str(eb_prms.startdate)[0:10].replace('-','')+'-'+str(eb_prms.enddate)[0:10].replace('-','')
        df = pd.read_csv(eb_prms.AWS_fn.replace('dates',dates),skiprows = 34)
        df = df.set_index(pd.date_range(eb_prms.startdate,eb_prms.enddate,freq='15min'))
        df = df.interpolate('time')
        vars = df.columns[1:]
        for var in vars:
            if var in ['T','t2m','T2','TA_2.0m']:
                temp = df[var].resample('H').mean().to_numpy()
            elif var in ['P','tp']:
                tp = df[var].resample('H').sum().to_numpy()
            elif var in ['RH','RH_2.0m']:
                rh = df[var].resample('H').mean().to_numpy()
            elif var in ['SW_IN']:
                SWin = df[var].resample('H').mean().to_numpy()
            elif var in ['SW_OUT']:
                SWout = df[var].resample('H').mean().to_numpy()
            elif var in ['LW_IN']:
                LWin = df[var].resample('H').mean().to_numpy()
            elif var in ['LW_OUT']:
                LWout = df[var].resample('H').mean().to_numpy()
            elif var in ['WS']:
                wind = df[var].resample('H').mean().to_numpy()
            elif var in ['tcc']:
                tcc = df[var].resample('H').mean().to_numpy()
            elif var in ['sp']:
                sp = df[var].resample('H').mean().to_numpy()
        try:
            clouds = tcc
        except:
            tcc, data_hours = gcm.importGCMvarnearestneighbor_xarray(gcm.tcc_fn, gcm.tcc_vn, glacier_table,dates_table)
            tcc = tcc[0]

        try:
            pressure = sp
        except:
            sp, data_hours = gcm.importGCMvarnearestneighbor_xarray(gcm.press_fn, gcm.press_vn, glacier_table,dates_table)
            sp = sp[0]

    # ===== SET UP CLIMATE DATASET =====
    n_bins = eb_prms.n_bins
    bin_idx = np.arange(0,n_bins)
    climateds = xr.Dataset(data_vars = dict(
        bin_elev = (['bin'],eb_prms.bin_elev,{'units':'m'}),
        SWin = (['time'],SWin,{'units':'J m-2'}),
        SWout = (['time'],SWout,{'units':'J m-2'}),
        LWin = (['time'],LWin,{'units':'J m-2'}),
        LWout = (['time'],LWout,{'units':'J m-2'}),
        tcc = (['time'],tcc,{'units':'0-1'}),
        wind = (['time'],wind,{'units':'m s-1'})),
        coords = dict(
            bin=(['bin'],bin_idx),
            time=(['time'],data_hours)
            ))
    climateds = climateds.assign(bin_temp = (['bin','time'],[temp],{'units':'C'}))
    climateds = climateds.assign(bin_tp = (['bin','time'],[tp],{'units':'m'}))
    climateds = climateds.assign(bin_sp = (['bin','time'],[sp],{'units':'Pa'}))
    climateds = climateds.assign(bin_rh = (['bin','time'],[rh],{'units':'%'}))
    return climateds