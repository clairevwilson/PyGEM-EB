"""
Climate class for PEBSI

Handles generation of climate dataset for
the model run duration and 

@author: clairevwilson
"""
# Built-in libraries
import threading
import os,sys
import time
# External libraries
import pandas as pd
import numpy as np
import xarray as xr
# Local libraries
import pebsi.input as prms

class Climate():
    """
    Climate-related functions which build the climate dataset
    for a single simulation.

    If use_AWS = True in the input, the climate dataset will be 
    filled with as many AWS variables as exist before turning 
    to reanalysis data to fill the necessary variables.

    If use_AWS = False, only reanalysis data will be used.
    """
    def __init__(self,args):
        """
        Initializes glacier information and creates the dataset where 
        climate data will be stored.
        """
        # start timer
        self.start_time = time.time()

        # load args and run information
        self.args = args
        self.dates = pd.date_range(args.startdate,args.enddate,freq='h')
        self.dates_UTC = self.dates - args.timezone
        n_time = len(self.dates)

        # specify glacier and time information
        self.lat = args.lat
        self.lon = args.lon
        self.n_time = n_time
        self.elev = args.elev

        # find median elevation of the glacier from RGI
        RGI_region = args.glac_no.split('.')[0]
        if float(RGI_region) > 0:
            for fn in os.listdir(prms.RGI_fp):
                # open the attributes .csv for the correct region
                if fn[:2] == RGI_region and fn[-3:] == 'csv':
                    RGI_df = pd.read_csv(prms.RGI_fp + fn)
                    RGI_df.index = [f.split('-')[-1] for f in RGI_df['RGIId']]
            self.median_elev = RGI_df.loc[args.glac_no,'Zmed']
        else:
            self.median_elev = self.elev

        # define reanalysis variables
        self.get_vardict()
        self.all_vars = ['temp','tp','rh','uwind','vwind','sp','SWin','LWin',
                            'bcwet','bcdry','ocwet','ocdry','dustwet','dustdry']
        if not self.args.use_AWS:
            self.measured_vars = []
        
        # default to wind direction being calculated
        self.wind_direction = True

        # create empty dataset
        nans = np.ones(n_time)*np.nan
        self.cds = xr.Dataset(data_vars = dict(
                SWin = (['time'],nans,{'units':'J m-2'}),
                SWout = (['time'],nans,{'units':'J m-2'}),
                albedo = (['time'],nans,{'units':'-'}),
                LWin = (['time'],nans,{'units':'J m-2'}),
                LWout = (['time'],nans,{'units':'J m-2'}),
                NR = (['time'],nans,{'units':'J m-2'}),
                tcc = (['time'],nans,{'units':'-'}),
                rh = (['time'],nans,{'units':'%'}),
                uwind = (['time'],nans,{'units':'m s-1'}),
                vwind = (['time'],nans,{'units':'m s-1'}),
                wind = (['time'],nans,{'units':'m s-1'}),
                winddir = (['time'],nans,{'units':'o'}),
                bcdry = (['time'],nans,{'units':'kg m-2 s-1'}),
                bcwet = (['time'],nans,{'units':'kg m-2 s-1'}),
                ocdry = (['time'],nans,{'units':'kg m-2 s-1'}),
                ocwet = (['time'],nans,{'units':'kg m-2 s-1'}),
                dustdry = (['time'],nans,{'units':'kg m-2 s-1'}),
                dustwet = (['time'],nans,{'units':'kg m-2 s-1'}),
                temp = (['time'],nans,{'units':'C'}),
                tp = (['time'],nans,{'units':'m'}),
                sp = (['time'],nans,{'units':'Pa'})
                ),
                coords = dict(time=(['time'],self.dates)))
        return
    
    def get_AWS(self,fp):
        """
        Loads available AWS data and determines which variables
        are missing and need come from reanalysis.
        """
        # load data
        df = pd.read_csv(fp,index_col=0)
        df = df.set_index(pd.to_datetime(df.index))

        # check dates of data match input dates
        data_start = pd.to_datetime(df.index.to_numpy()[0])
        data_end = pd.to_datetime(df.index.to_numpy()[-1])
        assert self.dates[0] >= data_start, f'Check input dates: start date before range of AWS data ({data_start})'
        assert self.dates[len(self.dates)-1] <= data_end, f'Check input dates: end date after range of AWS data ({data_end})'
        
        # reindex in case of MERRA-2 half-hour timesteps
        new_index = pd.DatetimeIndex(self.dates)
        index_joined = df.index.join(new_index, how='outer')
        df = df.reindex(index=index_joined).interpolate().reindex(new_index)

        # get AWS elevation
        self.AWS_elev = df.iloc[0]['z']

        # get the available variables
        all_AWS_vars = ['temp','tp','rh','uwind','vwind','sp','SWin','SWout','albedo',
                        'NR','LWin','LWout','bcwet','bcdry','ocwet','ocdry','dustwet','dustdry']
        AWS_vars = df.columns
        self.measured_vars = list(set(all_AWS_vars) & set(AWS_vars))

        # check if wind direction can be calculated
        uwind_measured = 'uwind' in AWS_vars
        vwind_measured = 'vwind' in AWS_vars
        if uwind_measured ^ vwind_measured:
            self.wind_direction = False
            # print('! Wind speed was input as a scalar. Wind shading is not handled')
        
        # extract and store data
        for var in self.measured_vars:
            self.cds[var].values = df[var].astype(float)

        # determine which data variables are still needed from reanalysis
        need_vars = [e for e in self.all_vars if e not in AWS_vars]

        # if net radiation was measured, don't need LWin
        if 'NR' in self.measured_vars:
            need_vars.remove('LWin')

        # if wind was input as a scalar, don't need the other direction of wind
        if not self.wind_direction:
            if uwind_measured:
                self.cds['vwind'].values = np.zeros(self.n_time)
                need_vars.remove('vwind')
            elif vwind_measured:
                self.cds['uwind'].values = np.zeros(self.n_time)
                need_vars.remove('uwind')
        self.need_vars = need_vars
        return need_vars
    
    def get_reanalysis(self,vars):
        """
        Fetches reanalysis climate data variables.
        """
        # load time and point data
        dates = self.dates_UTC
        lat = self.lat
        lon = self.lon
        use_threads = self.args.use_threads
        
        # interpolate data if time was input on the hour instead of half-hour
        interpolate = dates[0].minute != 30 and prms.reanalysis == 'MERRA2'
        
        # get reanalysis data geopotential
        z_fp = self.reanalysis_fp + self.var_dict['elev']['fn']
        zds = xr.open_dataarray(z_fp)
        zds = zds.sel({self.lat_vn:lat,self.lon_vn:lon},method='nearest')
        zds = self.check_units('elev',zds)
        self.reanalysis_elev = zds.isel(time=0).values.ravel()[0]

        # define worker function for threading
        def access_cell(fn, var, result_dict):
            # open and check units of climate data
            ds = xr.open_dataset(fn)

            # index by lat and lon
            vn = self.var_dict[var]['vn'] 
            lat_vn,lon_vn = [self.lat_vn,self.lon_vn]
            if 'bc' in var or 'oc' in var or 'dust' in var:
                if prms.reanalysis == 'ERA5-hourly':
                    lat_vn,lon_vn = ['lat','lon']
            ds = ds.sel({lat_vn:lat,lon_vn:lon}, method='nearest')[vn]

            # check the units
            ds = self.check_units(var,ds)

            # for time-varying variables, select/interpolate to the model time
            if var != 'elev':
                dep_var = 'bc' in var or 'dust' in var or 'oc' in var
                if not dep_var and prms.reanalysis == 'ERA5-hourly':
                    assert dates[0] >= pd.to_datetime(ds.time.values[0])
                    assert dates[-1] <= pd.to_datetime(ds.time.values[-1])
                    ds = ds.interp(time=dates)
                elif interpolate:
                    ds = ds.interp(time=dates)
                else:
                    ds = ds.sel(time=dates)
            
            # make sure the gridcell corrected is close enough to the glacier
            assert np.abs(ds.coords[lat_vn].values - float(lat)) <= 0.5, 'Wrong grid cell was accessed'
            assert np.abs(ds.coords[lon_vn].values - float(lon)) <= 0.5, 'Wrong grid cell was accessed'

            # store result
            result_dict[var] = ds.values.ravel()
            ds.close()

            # if not threading, return the result
            if not use_threads:
                return result_dict
        
        # initiate variables
        all_data = {}
        threads = []
        # loop through vars
        for var in vars:
            fn = self.reanalysis_fp + self.var_dict[var]['fn']
            if use_threads:
                # initiate threads
                thread = threading.Thread(target=access_cell,
                                        args=(fn, var, all_data))
                thread.start()
                threads.append(thread)
            else:
                # gather data in series
                all_data = access_cell(fn,var,all_data)
        if use_threads:
            # join threads
            for thread in threads:
                thread.join()

        # store data
        for var in vars:
            self.cds[var].values = all_data[var].ravel()
        return

    def adjust_to_elevation(self):
        """
        Adjusts elevation-dependent climate variables (temperature, precip,
        surface pressure).

        Parameters
        =========
        """
        # CONSTANTS
        LAPSE_RATE = prms.lapserate
        PREC_GRAD = prms.precgrad
        PREC_FACTOR = float(self.args.kp)
        GRAVITY = prms.gravity
        R_GAS = prms.R_gas
        MM_AIR = prms.molarmass_air

        # TEMPERATURE: correct according to lapserate
        temp_elev = self.AWS_elev if 'temp' in self.measured_vars else self.reanalysis_elev
        new_temp = self.cds.temp.values + LAPSE_RATE*(self.elev - temp_elev)
        if 'gulkana_22yrs' in self.args.AWS_fn and 'temp' in self.measured_vars:
            new_temp -= 1
            print('Reducing temperature for on-ice weather station')
            
        # PRECIP: correct according to precipitation gradient
        tp_elev = self.median_elev
        new_tp = self.cds.tp.values*PREC_FACTOR*(1+PREC_GRAD*(self.elev-tp_elev))

        # SURFACE PRESSURE: correct according to barometric law
        sp_elev = self.AWS_elev if 'sp' in self.measured_vars else self.reanalysis_elev
        temp_sp_elev = new_temp + LAPSE_RATE*(sp_elev - self.elev) + 273.15
        ratio = ((new_temp + 273.15) / temp_sp_elev) ** (-GRAVITY*MM_AIR/(R_GAS*LAPSE_RATE))
        new_sp = self.cds.sp.values * ratio

        # Store adjusted values
        self.cds.temp.values = new_temp.ravel()
        self.cds.tp.values = new_tp.ravel()
        self.cds.sp.values = new_sp.ravel()
        return
    
    def check_ds(self):
        # calculate wind speed and direction from u and v components
        # ***WINDMAPPER GOES HERE***
        uwind = self.cds['uwind'].values
        vwind = self.cds['vwind'].values
        wind = np.sqrt(np.power(uwind,2)+np.power(vwind,2))
        winddir = np.arctan2(-uwind,-vwind) * 180 / np.pi
        self.cds['wind'].values = wind
        self.cds['winddir'].values = winddir

        if prms.reanalysis == 'MERRA2':
            # correct other MERRA-2 variables
            for var in prms.bias_vars:
                from_MERRA = True if not self.args.use_AWS else var in self.need_vars
                if from_MERRA:
                    self.bias_adjust_quantile(var)
                    #  and prms.bias_vars[var] == 'binned':
                    #     self.bias_adjust_binned(var)
                    # elif from_MERRA and prms.bias_vars[var] == 'quantile':

        # adjust elevation dependent variables
        self.adjust_to_elevation()
        
        # adjust MERRA-2 deposition by reduction coefficient
        if prms.reanalysis == 'MERRA2' and prms.adjust_deposition:
            self.adjust_dep()

        # check all variables are there
        failed = []
        for var in self.all_vars:
            data = self.cds[var].values
            if np.any(np.isnan(data)):
                failed.append(var)

        # can input net radiation instead of incoming LW radiation
        if 'LWin' in failed and 'NR' in self.measured_vars:
            failed.remove('LWin')

        # print any missing data
        if len(failed) > 0:
            print('Missing data from',failed)
            self.exit()

        # store the dataset as a netCDF
        if prms.store_climate:
            out_fp = prms.output_filepath + self.args.out + self.args.site + '_climate'
            self.cds.to_netcdf(out_fp+'.nc')
            print('Climate dataset saved to',out_fp+'.nc')
        
        time_elapsed = time.time()-self.start_time
        if self.args.debug:
            print(f'~ Loaded climate dataset in {time_elapsed:.1f} seconds ~')
        return
    
    def check_units(self,var,ds):
        # define the units the model needs
        model_units = {'temp':'C','uwind':'m s-1','vwind':'m s-1',
                       'rh':'%','sp':'Pa','tp':'m s-1','elev':'m',
                       'SWin':'J m-2', 'LWin':'J m-2', 'NR':'J m-2', 'tcc':'-',
                       'bcdry':'kg m-2 s-1', 'bcwet':'kg m-2 s-1',
                       'ocdry':'kg m-2 s-1', 'ocwet':'kg m-2 s-1',
                       'dustdry':'kg m-2 s-1', 'dustwet':'kg m-2 s-1'}
        
        # get the current variable's units
        units_in = ds.attrs['units'].replace('*','')
        units_out = model_units[var]

        # check and make replacements
        if units_in != units_out:
            if var == 'temp' and units_in == 'K':
                ds = ds - 273.15
            elif var == 'rh' and units_in in ['-','0-1']:
                ds  = ds * 100
            elif var == 'tp':
                if units_in == 'kg m-2 s-1':
                    ds = ds / 1000 * 3600
                elif units_in == 'm':
                    ds = ds / 3600
            elif var in ['SWin','LWin','NR'] and units_in == 'W m-2':
                ds = ds * 3600
            elif var == 'elev' and units_in in ['m+2 s-2','m2 s-2']:
                ds = ds / prms.gravity
            else:
                print(f'WARNING! Units did not match for {var} but were not updated')
                print(f'Previously {units_in}; should be {units_out}')
                print('Make a manual change in check_units (climate.py)')
                self.exit()
        return ds
    
    def adjust_dep(self):
        """
        Updates deposition based on preprocessed reduction coefficients
        """
        print('***Hard-coded MERRA-2 to UK-ESM filepath for deposition adjustment***')
        fn = self.reanalysis_fp + 'merra2_to_ukesm_conversion_map_MERRAgrid.nc'
        ds_f = xr.open_dataarray(fn)
        ds_f = ds_f.sel({self.lat_vn:self.lat,self.lon_vn:self.lon},method='nearest')
        f = ds_f.mean('time').values.ravel()[0]
        # To do time-moving monthly factors:
        # for date in ds_f.time.values:
        #     # select the reduction coefficient of the current month
        #     f = ds_f.sel(time=date).values[0]
        #     # index the climate dataset by the month and year
        #     month = pd.to_datetime(date).month
        #     year = pd.to_datetime(date).year
        #     idx_month = np.where(self.cds.coords['time'].dt.month.values == month)[0]
        #     idx_year = np.where(self.cds.coords['time'].dt.year.values == year)[0]
        #     idx = list(set(idx_month)&set(idx_year))
        #     # update dry and wet BC deposition
        #     self.cds['bcdry'][{'time':idx}] = self.cds['bcdry'][{'time':idx}] * f
        #     self.cds['bcwet'][{'time':idx}] = self.cds['bcwet'][{'time':idx}] * f
        self.cds['bcdry'].values *= f
        self.cds['bcwet'].values *= f
        return
    
    def adjust_temp_bias(self):
        """
        Updates air temperature according to preprocessed bias adjustments
        """
        old_T = self.cds['temp'].values
        new_T = prms.temp_bias_slope * old_T + prms.temp_bias_intercept
        self.cds['temp'].values = new_T 
        return
    
    def bias_adjust_binned(self,var):
        """
        Updated variables according to preprocessed bias adjustments
        """
        # open .csv with binned correction factors
        bias_fp = prms.bias_fp.replace('METHOD','binned').replace('VAR',var)
        assert os.path.exists(bias_fp), f'Binned data does not exist for {var}'
        bias_df = pd.read_csv(bias_fp)

        # define bins and values
        bins = np.append(np.zeros(1), bias_df['bins'])
        values = self.cds[var].values
        scale = np.ones_like(self.cds[var].values)

        # loop through bins and assign scale factor
        for i in range(1,len(bins)):
            index = np.where((values >= bins[i-1]) & (values < bins[i]))[0]
            scale[index] = bias_df['factor'][i-1]

        # update values with scale factor
        self.cds[var].values = values * scale
        return
    
    def bias_adjust_quantile(self,var):
        """
        Updates variables according to preprocessed quantile mapping
        """
        # open .csv with quantile mapping
        bias_fp = prms.bias_fp.replace('METHOD','quantile_mapping').replace('VAR',var)
        assert os.path.exists(bias_fp), f'Quantile mapping file does not exist for {var}'
        bias_df = pd.read_csv(bias_fp)
        
        # interpolate values according to quantile mapping
        values = self.cds[var].values
        adjusted = np.interp(values, bias_df['sorted'], bias_df['mapping'])

        # update values
        self.cds[var].values = adjusted
        return

    def getVaporPressure(self,tempC):
        """
        Returns vapor pressure in Pa from air temperature in Celsius
        """
        return 610.94*np.exp(17.625*tempC/(tempC+243.04))
    
    def getDewTemp(self,vap):
        """
        Returns dewpoint air temperature in C from vapor pressure in Pa
        """
        return 243.04*np.log(vap/610.94)/(17.625-np.log(vap/610.94))

    def RMSE(self,y,pred):
        N = len(y)
        RMSE = np.sqrt(np.sum(np.square(pred-y))/N)
        return RMSE

    def get_vardict(self):
        # determine filetag for MERRA2 lat/lon gridded files
        flat = str(int(np.floor(self.lat/10)*10))
        flon = str(int(np.floor(self.lon/10)*10))
        tag = prms.MERRA2_filetag if prms.MERRA2_filetag else f'{flat}_{flon}'

        # update filenames for MERRA-2 (need grid lat/lon)
        self.reanalysis_fp = prms.climate_fp
        self.var_dict = {'temp':{'fn':[],'vn':[]},
            'rh':{'fn':[],'vn':[]},'sp':{'fn':[],'vn':[]},
            'tp':{'fn':[],'vn':[]},'tcc':{'fn':[],'vn':[]},
            'SWin':{'fn':[],'vn':[]},'LWin':{'fn':[],'vn':[]},
            'uwind':{'fn':[],'vn':[]},'vwind':{'fn':[],'vn':[]},
            'bcdry':{'fn':[],'vn':[]},'bcwet':{'fn':[],'vn':[]},
            'ocdry':{'fn':[],'vn':[]},'ocwet':{'fn':[],'vn':[]},
            'dustdry':{'fn':[],'vn':[]},'dustwet':{'fn':[],'vn':[]},
            'elev':{'fn':[],'vn':[]},'time':{'fn':'','vn':''},
            'lat':{'fn':'','vn':''}, 'lon':{'fn':'','vn':''}}
        if prms.reanalysis == 'MERRA2':
            self.reanalysis_fp += 'MERRA2/'
            self.var_dict['temp']['vn'] = 'T2M'
            self.var_dict['rh']['vn'] = 'RH2M'
            self.var_dict['sp']['vn'] = 'PS'
            self.var_dict['tp']['vn'] = 'PRECTOTCORR'
            self.var_dict['elev']['vn'] = 'PHIS'
            self.var_dict['tcc']['vn'] = 'CLDTOT'
            self.var_dict['SWin']['vn'] = 'SWGDN'
            self.var_dict['LWin']['vn'] = 'LWGAB'
            self.var_dict['uwind']['vn'] = 'U2M'
            self.var_dict['vwind']['vn'] = 'V2M'
            self.var_dict['bcwet']['vn'] = 'BCWT002'
            self.var_dict['bcdry']['vn'] = 'BCDP002'
            self.var_dict['ocwet']['vn'] = 'OCWT002'
            self.var_dict['ocdry']['vn'] = 'OCDP002'
            self.var_dict['dustwet']['vn'] = 'DUWT003'
            self.var_dict['dustdry']['vn'] = 'DUDP003'
            self.time_vn = 'time'
            self.lat_vn = 'lat'
            self.lon_vn = 'lon'
            self.elev_vn = self.var_dict['elev']['vn']

            # Variable filenames
            self.var_dict['temp']['fn'] = f'T2M/MERRA2_T2M_{tag}.nc'
            self.var_dict['rh']['fn'] = f'RH2M/MERRA2_RH2M_{tag}.nc'
            self.var_dict['sp']['fn'] = f'PS/MERRA2_PS_{tag}.nc'
            self.var_dict['tcc']['fn'] = f'CLDTOT/MERRA2_CLDTOT_{tag}.nc'
            self.var_dict['LWin']['fn'] = f'LWGAB/MERRA2_LWGAB_{tag}.nc'
            self.var_dict['SWin']['fn'] = f'SWGDN/MERRA2_SWGDN_{tag}.nc'
            self.var_dict['vwind']['fn'] = f'V2M/MERRA2_V2M_{tag}.nc'
            self.var_dict['uwind']['fn'] = f'U2M/MERRA2_U2M_{tag}.nc'
            self.var_dict['tp']['fn'] = f'PRECTOTCORR/MERRA2_PRECTOTCORR_{tag}.nc'
            self.var_dict['elev']['fn'] = f'MERRA2constants.nc4'
            self.var_dict['bcwet']['fn'] = f'BCWT002/MERRA2_BCWT002_{tag}.nc'
            self.var_dict['bcdry']['fn'] = f'BCDP002/MERRA2_BCDP002_{tag}.nc'
            self.var_dict['ocwet']['fn'] = f'OCWT002/MERRA2_OCWT002_{tag}.nc'
            self.var_dict['ocdry']['fn'] = f'OCDP002/MERRA2_OCDP002_{tag}.nc'
            self.var_dict['dustwet']['fn'] = f'DUWT003/MERRA2_DUWT003_{tag}.nc'
            self.var_dict['dustdry']['fn'] = f'DUDP003/MERRA2_DUDP003_{tag}.nc'
        elif prms.reanalysis == 'ERA5-hourly':
            self.reanalysis_fp += 'ERA5/ERA5_hourly/'

            # Variable names for energy balance
            self.var_dict['temp']['vn'] = 't2m'
            self.var_dict['rh']['vn'] = 'rh'
            self.var_dict['sp']['vn'] = 'sp'
            self.var_dict['tp']['vn'] = 'tp'
            self.var_dict['elev']['vn'] = 'z'
            self.var_dict['tcc']['vn'] = 'tcc'
            self.var_dict['SWin']['vn'] = 'ssrd'
            self.var_dict['LWin']['vn'] = 'strd'
            self.var_dict['uwind']['vn'] = 'u10'
            self.var_dict['vwind']['vn'] = 'v10'
            self.var_dict['bcwet']['vn'] = 'BCWT002'
            self.var_dict['bcdry']['vn'] = 'BCDP002'
            self.var_dict['ocwet']['vn'] = 'OCWT002'
            self.var_dict['ocdry']['vn'] = 'OCDP002'
            self.var_dict['dustwet']['vn'] = 'DUWT003'
            self.var_dict['dustdry']['vn'] = 'DUDP003'
            self.time_vn = 'time'
            self.lat_vn = 'latitude'
            self.lon_vn = 'longitude'
            self.elev_vn = self.var_dict['elev']['vn']

            # Variable filenames
            self.var_dict['temp']['fn'] = 'ERA5_temp_hourly.nc'
            self.var_dict['rh']['fn'] = 'ERA5_rh_hourly.nc'
            self.var_dict['sp']['fn'] = 'ERA5_sp_hourly.nc'
            self.var_dict['tcc']['fn'] = 'ERA5_tcc_hourly.nc'
            self.var_dict['LWin']['fn'] = 'ERA5_LWin_hourly.nc'
            self.var_dict['SWin']['fn'] = 'ERA5_SWin_hourly.nc'
            self.var_dict['vwind']['fn'] = 'ERA5_vwind_hourly.nc'
            self.var_dict['uwind']['fn'] = 'ERA5_uwind_hourly.nc'
            self.var_dict['tp']['fn'] = 'ERA5_tp_hourly.nc'
            self.var_dict['elev']['fn'] = 'ERA5_geopotential_2000.nc'
            self.var_dict['bcwet']['fn'] = f'./../../MERRA2/BCWT002/MERRA2_BCWT002_{tag}.nc'
            self.var_dict['bcdry']['fn'] = f'./../../MERRA2/BCDP002/MERRA2_BCDP002_{tag}.nc'
            self.var_dict['ocwet']['fn'] = f'./../../MERRA2/OCWT002/MERRA2_OCWT002_{tag}.nc'
            self.var_dict['ocdry']['fn'] = f'./../../MERRA2/OCDP002/MERRA2_OCDP002_{tag}.nc'
            self.var_dict['dustwet']['fn'] = f'./../../MERRA2/DUWT003/MERRA2_DUWT003_{tag}.nc'
            self.var_dict['dustdry']['fn'] = f'./../../MERRA2/DUDP003/MERRA2_DUDP003_{tag}.nc'

    def exit(self):
        """Exits model if there is a problem"""
        sys.exit()