"""
Energy balance class for PyGEM Energy Balance

@author: clairevwilson
"""
import pandas as pd
import numpy as np
import pygem_eb.input as eb_prms
import suncalc

class energyBalance():
    """
    Energy balance scheme that calculates the surface energy balance
    and penetrating shortwave radiation. This class is updated within
    MassBalance every timestep, so it stores the current climate data and 
    surface fluxes.
    """ 
    def __init__(self,climate,time,dt,args):
        """
        Loads in the climate data at a given timestep to use in the surface energy balance.

        Parameters
        ----------
        climateds : xr.Dataset
            Climate dataset containing temperature, precipitation, pressure, wind speed,
            shortwave radiation, and total cloud cover.
        local_time : datetime
            Time to index the climate dataset.
        dt : float
            Resolution for the time loop [s]
        """
        # Unpack climate variables
        climateds_now = climate.cds.sel(time=time)
        self.tempC = climateds_now['temp'].values
        self.tp = climateds_now['tp'].values
        self.sp = climateds_now['sp'].values
        self.rh = climateds_now['rh'].values
        self.wind = climateds_now['wind'].values
        self.tcc = climateds_now['tcc'].values
        self.SWin_ds = climateds_now['SWin'].values
        self.SWout_ds = climateds_now['SWout'].values
        self.albedo_ds = climateds_now['albedo'].values
        self.LWin_ds = climateds_now['LWin'].values
        self.LWout_ds = climateds_now['LWout'].values
        self.NR_ds = climateds_now['NR'].values
        self.bcdry = climateds_now['bcdry'].values
        self.bcwet = climateds_now['bcwet'].values
        self.ocdry = climateds_now['ocdry'].values
        self.ocwet = climateds_now['ocwet'].values
        self.dustdry = climateds_now['dustdry'].values
        self.dustwet = climateds_now['dustwet'].values

        # Main variables
        self.climateds = climate.cds
        self.args = args

        # Define additional useful values
        self.tempK = self.tempC + 273.15
        self.prec =  self.tp / 3600     # tp is hourly total precip, prec is the rate in m/s
        self.rh = 100 if self.rh > 100 else self.rh
        
        # Time
        self.time = time
        self.dt = dt

        # Adjust calibrated values
        self.wind *= float(args.kw)

        # Radiation terms
        self.measured_SWin = 'SWin' in climate.measured_vars
        self.nanLWin = True if np.isnan(self.LWin_ds) else False
        self.nanSWout = True if np.isnan(self.SWout_ds) else False
        self.nanLWout = True if np.isnan(self.LWout_ds) else False
        self.nanNR = True if np.isnan(self.NR_ds) else False
        self.nanalbedo = True if np.isnan(self.albedo_ds) else False
        return

    def surface_EB(self,surftemp,layers,surface,days_since_snowfall,mode='sum'):
        """
        Calculates the surface heat fluxes for the current timestep.

        Parameters
        ----------
        surftemp : float
            Temperature of the surface snow in Celsius
        layers
            class object from pygem_eb.layers
        surface : float
            class object from pygem_eb.surface
        days_since_snowfall : int
            Number of days since fresh snowfall
        method_turbulent : str
            'MO-similarity', 'bulk-aerodynamic', or 'Richardson': 
            determines method for calculating turbulent fluxes
        mode : str, default: 'sum'
            'sum' to return sum of heat fluxes, 'list' to return separate fluxes,
            'optim' to return absolute value of heat fluxes (needed for BFGS optimization)

        Returns
        -------
        Qm : float OR np.ndarray
            If mode is 'sum' or 'optim', returns the sum of heat fluxes
            If mode is 'list', returns list in the order of 
                    [SWin, SWout, LWin, LWout, sensible, latent, rain, ground]
        """
        # SHORTWAVE RADIATION  (Snet)
        SWin,SWout = self.get_SW(surface)
        Snet_surf = SWin + SWout
        self.SWin = SWin
        self.SWout = SWout[0] if '__iter__' in dir(SWout) else SWout
                    
        # LONGWAVE RADIATION (Lnet)
        LWin,LWout = self.get_LW(surftemp)
        Lnet = LWin + LWout
        self.LWin = LWin
        self.LWout = LWout[0] if '__iter__' in dir(LWout) else LWout

        # RAIN FLUX (Qp)
        Qp = self.get_rain(surftemp)
        self.rain = Qp[0] if '__iter__' in dir(Qp) else Qp

        # GROUND FLUX (Qg)
        Qg = self.get_ground(surftemp)
        self.ground = Qg[0] if '__iter__' in dir(Qg) else Qg

        # TURBULENT FLUXES (Qs and Ql)
        roughness = self.get_roughness(days_since_snowfall,layers.ltype)
        Qs, Ql = self.get_turbulent(surftemp,roughness)
        self.sens = Qs[0] if '__iter__' in dir(Qs) else Qs
        self.lat = Ql[0] if '__iter__' in dir(Qs) else Ql

        # OUTPUTS
        Qm = Snet_surf + Lnet + Qp + Qs + Ql + Qg

        if mode in ['sum']:
            return Qm
        elif mode in ['optim']:
            return np.abs(Qm)
        elif mode in ['list']:
            return np.array([SWin,SWout,LWin,LWout,Qs,Ql,Qp,Qg])
        else:
            assert 1==0, 'argument \'mode\' in function surfaceEB should be sum, list or optim'
    
    def get_SW(self,surface):
        """
        Simplest parameterization for shortwave radiation which
        adjusts it by modeled spectral albedo. Returns the shortwave
        surface flux and the penetrating shortwave for each layer.
        
        Parameters
        ----------
        surface
            class object from surface.py
        """
        # Constants
        SKY_VIEW = self.args.sky_view
        LAT = self.args.lat
        LON = self.args.lon
        SLOPE = self.args.slope * np.pi/180
        ASPECT = self.args.aspect * np.pi/180

        # Albedo inputs
        albedo = surface.albedo
        spectral_weights = surface.spectral_weights
        assert np.abs(1-np.sum(spectral_weights)) < 1e-5, 'Solar weights dont sum to 1'

        # Get solar position
        time_UTC = self.time - eb_prms.timezone
        sunpos = suncalc.get_position(time_UTC,LON,LAT)
        # suncalc gives azimuth with 0 = South, we want 0 = North
        SUN_AZ = sunpos['azimuth'] + np.pi     # solar azimuth angle
        SUN_ZEN = np.pi/2 - sunpos['altitude'] # solar zenith angle

        # Calculate slope correction
        cos_theta = (np.cos(SUN_ZEN)*np.cos(SLOPE) + 
                    np.sin(SUN_ZEN)*np.sin(SLOPE)*np.cos(SUN_AZ - ASPECT))
        slope_correction = min(cos_theta / np.cos(SUN_ZEN), 5)
        slope_correction = max(slope_correction,0)
        
        # SWin needs to be corrected for shade
        if self.measured_SWin:
            # if point elev != AWS elev
            # is AWS in the sun?
            # if so: is the point in the sun?
                # if so: just calcualte diffuse
                # if not: neglect SWin, just diffuse
            # if not: is the point in the sun?
                # if so: COMPLICATED
                # if not: SWin AWS = SWin point
            SWin = self.SWin_ds/self.dt * slope_correction
            self.SWin_sky = np.nan
            self.SWin_terr = np.nan
        else:
            # get sky (diffuse+direct) and terrain (diffuse) SWin
            SWin_sky = self.SWin_ds/self.dt
            SWin_terrain = SWin_sky*(1-SKY_VIEW)*surface.albedo_surr

            # correct for slope
            SWin_sky *= slope_correction

            # correct for shade
            time_str = str(self.time).replace(str(self.time.year),'2024')
            time_2024 = pd.to_datetime(time_str)
            self.shade = bool(surface.shading_df.loc[time_2024,'shaded'])
            SWin = SWin_terrain if self.shade else SWin_terrain + SWin_sky

            # store sky and terrain portions
            self.SWin_sky = 0 if self.shade else SWin_sky
            self.SWin_terr = SWin_terrain

        # get reflected radiation
        if self.nanSWout and self.nanalbedo:
            albedo = albedo[0] if len(spectral_weights) < 2 else albedo
            SWout = -np.sum(SWin*spectral_weights*albedo)
        elif not self.nanalbedo:
            albedo = self.albedo_ds
            surface.bba = albedo
            SWout = -SWin*albedo
        else:
            SWout = -self.SWout_ds/self.dt
            # store albedo
            if -SWout < SWin and SWin > 0:
                surface.bba = -SWout / SWin
        return SWin,SWout

    def get_LW(self,surftemp):
        """
        If not input in climate data, scheme follows Klok and 
        Oerlemans (2002) for calculating net longwave radiation 
        from the air temperature and cloud cover.
        
        Parameters
        ----------
        surftemp : float
            Surface temperature of snowpack/ice [C]
        """
        if self.nanLWout:
            surftempK = surftemp+273.15
            LWout = -eb_prms.sigma_SB*surftempK**4
        else:
            LWout = -self.LWout_ds/self.dt
        
        if self.nanLWin and self.nanNR:
            ezt = self.vapor_pressure(self.tempC)    # vapor pressure in hPa
            Ecs = .23+ .433*(ezt/self.tempK)**(1/8)  # clear-sky emissivity
            Ecl = 0.984               # cloud emissivity, Klok and Oerlemans, 2002
            Esky = Ecs*(1-self.tcc**2)+Ecl*self.tcc**2    # sky emissivity
            LWin = eb_prms.sigma_SB*(Esky*self.tempK**4)
        elif not self.nanLWin:
            LWin = self.LWin_ds/self.dt
        elif not self.nanNR:
            LWin = self.NR_ds/self.dt - LWout - self.SWin - self.SWout
            
        return LWin,LWout
    
    def get_rain(self,surftemp):
        """
        Calculates amount of energy supplied by precipitation that falls as rain.
        
        Parameters
        ----------
        surftemp : float
            Surface temperature of snowpack/ice [C]
        """
        # CONSTANTS
        SNOW_THRESHOLD_LOW = eb_prms.snow_threshold_low
        SNOW_THRESHOLD_HIGH = eb_prms.snow_threshold_high
        DENSITY_WATER = eb_prms.density_water
        CP_WATER = eb_prms.Cp_water

        # Define rain vs snow scaling
        rain_scale = np.linspace(0,1,20)
        temp_scale = np.linspace(SNOW_THRESHOLD_LOW,SNOW_THRESHOLD_HIGH,20)
        
        # Get fraction of precip that is rain
        if self.tempC < SNOW_THRESHOLD_LOW:
            frac_rain = 0
        elif SNOW_THRESHOLD_LOW < self.tempC < SNOW_THRESHOLD_HIGH:
            frac_rain = np.interp(self.tempC,temp_scale,rain_scale)
        else:
            frac_rain = 1

        Qp = (self.tempC-surftemp)*self.prec*frac_rain*DENSITY_WATER*CP_WATER
        return Qp
    
    def get_ground(self,surftemp):
        """
        Calculates amount of energy supplied by heat conduction from the temperate ice.
        
        Parameters
        ----------
        surftemp : float
            Surface temperature of snowpack/ice [C]
        """
        K_ICE = eb_prms.k_ice
        if eb_prms.method_ground in ['MolgHardy']:
            Qg = -K_ICE * (surftemp - eb_prms.temp_temp) / eb_prms.temp_depth
        else:
            assert 1==0, 'Ground flux method not accepted; choose from [\'MolgHardy\']'
        return Qg
    
    def get_turbulent(self,surftemp,roughness):
        """
        Calculates turbulent fluxes (sensible and latent heat) based on 
        Monin-Obukhov Similarity Theory or Bulk Richardson number, 
        both requiring iteration.
        Follows COSIPY.

        Parameters
        ----------
        surftemp : float
            Surface temperature of snowpack/ice [C]
        roughness : float
            Surface roughness [m]
        """
        # CONSTANTS
        KARMAN = eb_prms.karman
        GRAVITY = eb_prms.gravity
        R_GAS = eb_prms.R_gas
        MM_AIR = eb_prms.molarmass_air
        CP_AIR = eb_prms.Cp_air
        WIND_REF_Z = eb_prms.wind_ref_height

        # ROUGHNESS LENGTHS
        z0 = roughness  # Roughness length for momentum
        z0t = z0/100    # Roughness length for heat
        z0q = z0/10     # Roughness length for moisture

        # # SLOPE (not currently using)
        # if eb_prms.glac_no == ['01.00570']:
        #     slope = eb_prms.slope
        #     cos_slope = np.cos(np.radians(slope))

        # ADJUST WIND SPEED
        z = 2 # reference height in m
        if eb_prms.wind_ref_height != 2:
            wind_2m *= np.log(2/roughness) / np.log(WIND_REF_Z/roughness)
        else:
            wind_2m = self.wind

        # Transform humidity into mixing ratio (q), get air density from PV=nRT
        Ewz = self.vapor_pressure(self.tempC)  # vapor pressure at 2m
        Ew0 = self.vapor_pressure(surftemp)    # vapor pressure at the surface
        qz = (self.rh/100)*0.622*(Ewz/(self.sp-Ewz))
        q0 = 1.0*0.622*(Ew0/(self.sp-Ew0))
        density_air = self.sp/R_GAS/self.tempK*MM_AIR

        # Latent heat term depends on direction of heat exchange
        if surftemp == 0. and (qz-q0) > 0:
            Lv = eb_prms.Lv_evap
        else:
            Lv = eb_prms.Lv_sub 

        # Initiate loop
        loop = True
        counter = 0
        L = 0
        Qs_last = np.inf
        if eb_prms.method_turbulent in ['MO-similarity']:
            while loop:
                # calculate stability terms
                fric_vel = KARMAN*wind_2m / (np.log(z/z0)-self.PhiM(z,L))
                cD = KARMAN**2/np.square(np.log(z/z0) - self.PhiM(z,L) - self.PhiM(z0,L))
                csT = KARMAN*np.sqrt(cD) / (np.log(z/z0t) - self.PhiT(z,L) - self.PhiT(z0,L))
                csQ = KARMAN*np.sqrt(cD) / (np.log(z/z0q) - self.PhiT(z,L) - self.PhiT(z0,L))
                
                # calculate fluxes
                Qs = density_air*CP_AIR*csT*wind_2m*(self.tempC - surftemp)
                Ql = density_air*Lv*csQ*wind_2m*(qz-q0)

                # recalculate L
                if np.abs(Qs) < 1e-5:
                    Qs = 1e-5 # prevent overflow errors
                L = fric_vel**3*(self.tempK)*density_air*CP_AIR/(KARMAN*GRAVITY*Qs)
                L = max(L,0.3) # DEBAM uses this limit to prevent over-stabilization

                # check convergence
                counter += 1
                diff = np.abs(Qs_last-Qs)
                if counter > 10 or diff < 1e-1:
                    loop = False
                    if diff > 1:
                        print('Turbulent fluxes didnt converge; Qs still changing by',diff)

                Qs_last = Qs
        elif eb_prms.method_turbulent in ['BulkRichardson']:   
            if wind_2m != 0:
                RICHARDSON = GRAVITY/self.tempK*(self.tempC-surftemp)*(z-z0)/wind_2m**2
            else:
                RICHARDSON = 0
            csT = KARMAN**2/(np.log(z/z0) * np.log(z/z0t))
            csQ = KARMAN**2/(np.log(z/z0) * np.log(z/z0q))
            if RICHARDSON <= 0.01:
                psi = 1
            elif 0.01 < RICHARDSON <= 0.2:
                psi = np.square(1-5*RICHARDSON)
            else:
                psi = 0
            
            # calculate fluxes
            Qs = density_air*CP_AIR*csT*psi*wind_2m*(self.tempC - surftemp)
            Ql = density_air*Lv*csQ*psi*wind_2m*(qz-q0)
        else:
            assert 1==0, 'Choose turbulent method from MO-similarity or BulkRichardson'
        
        return Qs, Ql
    
    def get_dry_deposition(self, layers):
        BC_RATIO = eb_prms.ratio_BC2_BCtot
        OC_RATIO = eb_prms.ratio_OC2_OCtot
        DUST_RATIO = eb_prms.ratio_DU3_DUtot
        
        # Switch runs have no LAPs
        if eb_prms.switch_LAPs == 0:
            self.bcdry = 0
            self.ocdry = 0
            self.dustdry = 0

        if layers.ltype[0] != 'ice':
            layers.lBC[0] += self.bcdry * self.dt * BC_RATIO
            layers.lOC[0] += self.ocdry * self.dt * OC_RATIO
            layers.ldust[0] += self.dustdry * self.dt * DUST_RATIO
        return 
    
    def get_roughness(self,days_since_snowfall,layertype):
        """
        Function to determine the roughness length of the surface. This assumes the roughness of snow
        linearly degrades with time in 60 days from that of fresh snow to firn.

        Parameters
        ----------
        days_since_snowfall : int
            Number of days since fresh snow occurred
        layertype : np.ndarray
            List of layer types to determine surface roughness
        """
        # CONSTANTS
        ROUGHNESS_FRESH_SNOW = eb_prms.roughness_fresh_snow
        ROUGHNESS_AGED_SNOW = eb_prms.roughness_aged_snow
        ROUGHNESS_FIRN = eb_prms.roughness_firn
        ROUGHNESS_ICE = eb_prms.roughness_ice
        AGING_RATE = eb_prms.roughness_aging_rate

        if layertype[0] in ['snow']:
            sigma = min(ROUGHNESS_FRESH_SNOW + AGING_RATE * days_since_snowfall, ROUGHNESS_AGED_SNOW)
        elif layertype[0] in ['firn']:
            sigma = ROUGHNESS_FIRN
        elif layertype[0] in ['ice']:
            sigma = ROUGHNESS_ICE
        return sigma/1000
    
    def vapor_pressure(self,T,method = 'ARM'):
        """
        Returns vapor pressure in Pa from temperature in Celsius

        Parameters
        ----------
        T : float
            Temperature in C
        """
        if method in ['ARM']:
            P = 0.61094*np.exp(17.625*T/(T+243.04)) # kPa
        elif method in ['Sonntag']:
            # follows COSIPY
            T += 273.15
            if T > 273.15: # over water
                P = 0.6112*np.exp(17.67*(T-273.15)/(T-29.66))
            else: # over ice
                P = 0.6112*np.exp(22.46*(T-273.15)/(T-0.55))
        return P*1000
 
    def stable_PhiM(self,z,L):
        zeta = z/L
        if zeta > 1:
            phim = -4*(1+np.log(zeta)) - zeta
        elif zeta > 0:
            phim = -5*zeta
        else:
            phim = 0
        return phim

    def PhiM(self,z,L):
        if L < 0:
            X = np.power((1-16*z/L),0.25)
            phim = 2*np.log((1+X)/2) + np.log((1+X**2)/2) - 2*np.arctan(X) + np.pi/2
        elif L > 0: # stable
            phim = self.stable_PhiM(z, L)
        else:
            phim = 0.0
        return phim

    def PhiT(self,z,L):
        if L < 0:
            X = np.power((1-19.3*z/L),0.25)
            phit = 2*np.log((1+X**2)/2)
        elif L > 0: # stable
            phit = self.stable_PhiM(z, L)
        else:
            phit = 0.0
        return phit
    