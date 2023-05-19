#import xarray as xr
import numpy as np
import pygem_eb.input as eb_prms

class energyBalance():
    """
    Energy balance scheme that calculates the surface energy balance and penetrating shortwave radiation.
    """ 
    def __init__(self,climateds,time,bin_idx,dt):
        """
        Loads in the climate data at a given timestep to use in the surface energy balance.

        Parameters
        ----------
        climateds : xr.Dataset
            Climate dataset containing temperature, precipitation, pressure, air density, wind speed,
            shortwave radiation, and total cloud cover.
        time : datetime
            Time to index the climate dataset.
        bin_idx : int
            Index number of the bin being run
        dt : float
            Resolution for the time loop [s]
        """
        # Unpack climate variables
        if time.minute == 0:
            # Timestamp is on the hour, no processing needed, just extract the values
            climateds_now = climateds.sel(time=time)
            # Bin-dependent variables indexed by bin_idx
            self.tempC = climateds_now['bin_temp'].to_numpy()[bin_idx]
            self.prec = climateds_now['bin_prec'].to_numpy()[bin_idx]
            self.sp = climateds_now['bin_sp'].to_numpy()[bin_idx]
            self.density = climateds_now['bin_density'].to_numpy()[bin_idx]
            self.rH = climateds_now['bin_rh'].to_numpy()[bin_idx]
            # Elevation-invariant variables
            self.uwind = climateds_now['uwind'].to_numpy()
            self.vwind = climateds_now['vwind'].to_numpy()
            self.tcc = climateds_now['tcc'].to_numpy()
            self.surfrad = climateds_now['surfrad'].to_numpy()
        else:
            # Timestep is between hours, so interpolate using interpClimate function
            # Bin-dependent variables indexed by bin_idx
            self.tempC = self.interpClimate(climateds,time,'bin_temp',bin_idx)
            self.prec = self.interpClimate(climateds,time,'bin_prec',bin_idx)
            self.sp = self.interpClimate(climateds,time,'bin_sp',bin_idx)
            self.density = self.interpClimate(climateds,time,'bin_density',bin_idx)
            self.rH = self.interpClimate(climateds,time,'bin_rh',bin_idx)
            # Elevation-invariant variables
            self.uwind = self.interpClimate(climateds,time,'uwind')
            self.vwind = self.interpClimate(climateds,time,'vwind')
            self.tcc = self.interpClimate(climateds,time,'tcc')
            self.surfrad = self.interpClimate(climateds,time,'surfrad')
        # Define additional useful values
        self.wind = (self.uwind**2 + self.vwind**2)**(1/2)
        self.tempK = self.tempC + 273.15
        self.dt = dt
        return

    def surfaceEB(self,surftemp,layerz,layertype,days_since_snowfall,albedo):
        """
        Calculates the surface heat fluxes at each point on the glacier and applies mass-balance
        scheme to calculate melt and refreeze at each time point.

        Parameters
        ----------
        surftemp : float
            Temperature of the surface snow in Celsius
        layerz : list-like
            List of mid-layer depths
        layertype : list-like
            List of strings that define the type ('snow','firn', or 'ice') of each layer
        days_since_snowfall : int
            Number of days since fresh snowfall
        albedo : float
            Broadband albedo
        method_turbulent : str, default: 'MO-similarity'
            'MO-similarity', 'bulk-aerodynamic', or 'Richardson': determines method for calculating 
            turbulent fluxes
        """
        # SHORTWAVE RADIATION
        Snet_surf, SW_penetrating = self.getSW(self.surfrad,albedo,layerz,layertype)
                    
        # LONGWAVE RADIATION (Lnet)
        Lnet = self.getLW(surftemp,self.tempC,self.tcc)

        # RAIN FLUX (Qp)
        Qp = self.getRain(surftemp,self.tempC,self.prec)

        # TURBULENT FLUXES (Qs and Ql)
        roughness = self.roughness_length(days_since_snowfall,layertype)
        if eb_prms.method_turbulent in ['MO-similarity']:
            Qs, Ql = self.getTurbulentMO(self.tempK,surftemp,self.density,self.wind,self.sp,self.rH,roughness)
        else:
            print('Only MO similarity method is set up for turbulent fluxes')
            Qs, Ql = self.getTurbulentMO(self.tempK,surftemp,self.density,self.wind,self.sp,self.rH,roughness)
                
        # SUM ENERGY FOR MELT
        Qm = Snet_surf + Lnet + Qp + Qs + Ql
        return Qm, SW_penetrating
    
    def getSW(self,surfrad,albedo,layerz,layertype):
        """
        Simplest parameterization for shortwave radiation which just adjusts it by modeled albedo.
        Returns the shortwave surface flux and the penetrating shortwave with each layer.
        """
        # sun_pos = solar.get_position(time,glacier_table['CenLon'],glacier_table['CenLat'])
        Snet_surf = surfrad*(1-albedo)/self.dt #* (cos(theta))

        nlayers = len(layertype)
        frac_absrad = np.zeros(nlayers)
        extinct_coef = np.zeros(nlayers)
        for idx,type in enumerate(layertype):
            if type in ['snow']:
                frac_absrad[idx] = 0.9
                extinct_coef[idx] = 17.1
            elif type in ['firn']:
                frac_absrad[idx] = 0.85
                extinct_coef[idx] = 9.8
            elif type in ['ice']:
                frac_absrad[idx] = 0.8
                extinct_coef[idx] = 2.5
        Snet_penetrating = Snet_surf*frac_absrad*np.exp(-extinct_coef*layerz)/self.dt
        return Snet_surf,Snet_penetrating

    def getLW(self,surftemp,airtemp,tcc):
        """
        Scheme following Klok and Oerlemans (2002) for calculating net longwave radiation
        from the air temperature and cloud cover.
        """
        airtempK = airtemp+273.15
        surftempK = surftemp+273.15
        ezt = self.vapor_pressure(airtemp)    # vapor pressure in hPa
        Ecs = .23+ .433*(ezt/airtempK)**(1/8)  # clear-sky emissivity
        Ecl = 0.984                         # cloud emissivity, Klok and Oerlemans, 2002
        Esky = Ecs*(1-tcc**2)+Ecl*tcc**2    # sky emissivity
        Lnet = eb_prms.sigma_SB*(Esky*airtempK**4 - surftempK**4)
        return Lnet
    
    def getRain(self,surftemp,airtemp,prec):
        """
        Calculates amount of energy supplied by precipitation that falls as rain.
        """
        is_rain = airtemp > eb_prms.tsnow_threshold
        Qp = is_rain*eb_prms.Cp_water*(airtemp-surftemp)*prec/self.dt
        return Qp
    
    def getAlbedo(self,BC_conc,days_since_snowfall,switch_snow=True,switch_melt=True,switch_LAP=True):
        """
        Updates the surface albedo based on the concentration of LAPs and the degradation with time. Switches allow
        for controlled simulations to see the changes associated with snow-albedo feedback, melt-albedo feedback and
        LAP-albedo feedback.

        Parameters
        ----------
        BC_conc : np.ndarray
            Concentration of BC at the surface and in the bulk snowpack [ppb]
        days_since_snowfall : int
            Number of days since the last snowfall
        switch_snow : Bool
            Switch to turn on/off snow-albedo feedback
        switch_melt : Bool
            Switch to turn on/off melt-albedo feedback
        switch_LAP : Bool
            Switch to turn on/off LAP-albedo feedback
        """
        return 0.85

    def getTurbulentMO(self,air_temp,surf_temp,air_density,wind_speed,pressure,rH,roughness):
        """
        Calculates turbulent fluxes (sensible and latent heat) based on Monin-Obukhov Similarity 
        Theory, requiring iteration.

        Parameters
        ----------
        air_temp : float
            Air temperature at reference height [C]
        surf_temp : float
            Surface temperature of snowpack/ice [C]
        air_density : float
            Air density at reference height [kg m-3]
        wind_speed : float
            Wind speed at reference height [m s-1]
        pressure : float
            Air pressure at reference height [Pa]
        rH : float
            Relative humidity at reference height [%]
        roughness : float
            Surface roughness [m]
        """
        chi = lambda zeta: abs(1-16*zeta)**(1/4)
        PsiM = lambda zeta: np.piecewise(zeta,[zeta<0,(zeta>=0)&(zeta<=1),zeta>1],
                            [2*np.log((1+chi(zeta))/2)+np.log((1+chi(zeta)**2)/2)-2*np.arctan(chi(zeta))+np.pi/2,
                            -5*zeta, -4*(1+np.log(zeta))-zeta])
        PsiT = lambda zeta: np.piecewise(zeta,[zeta<0,(zeta>=0)&(zeta<=1),zeta>1],
                            [np.log((1+chi(zeta)**2)/2), -5*zeta, -4*(1+np.log(zeta))-zeta])
        
        Qs = 1000 #initial guess
        Ql = 1000
        converged = False
        zeta = 0.1
        count_iters = 0
        karman = eb_prms.karman
        while not converged:
            previous_zeta = zeta
            z = 2 #reference height, 2m

            Lv = np.piecewise(Ql,[(Ql>0)&(surf_temp<=0),(Ql>0)&(surf_temp>0),Ql<0],
                            [eb_prms.Lv_evap,eb_prms.Lv_sub,eb_prms.Lv_sub])

            z0 = roughness
            z0t = z0/100    # Roughness length for sensible heat
            z0q = z0/10     # Roughness length for moisture

            # calculate friction velocity using previous heat flux to get Obukhov length (L)
            fric_vel = karman*wind_speed/(np.log(z/z0)-PsiM(zeta))
            L = fric_vel**3*(air_temp+273.15)*air_density*eb_prms.Cp_air/(karman*eb_prms.gravity*Qs)
            if L<0.3: # DEBAM uses this correction to ensure it isn't over stablizied
                L = 0.3
            zeta = z/L
                
            cD = karman**2/(np.log(z/z0)-PsiM(zeta)-PsiM(z0/L))**2
            cH = karman*cD**(1/2)/((np.log(z/z0t)-PsiT(zeta)-PsiT(z0t/L)))
            cE = karman*cD**(1/2)/((np.log(z/z0q)-PsiT(zeta)-PsiT(z0q/L)))
            Qs = air_density*eb_prms.Cp_air*cH*wind_speed*(air_temp-surf_temp)

            Ewz = self.vapor_pressure(air_temp)*100 # vapor pressure at 2m
            Ew0 = self.vapor_pressure(surf_temp)*100 # vapor pressure at the surface
            qz = (rH/100)*0.622*(Ewz/(pressure-Ewz))
            q0 = 1.0*0.622*(Ew0/(pressure-Ew0))
            # qz = (rH2 * 0.622 * (Ew / (p - Ew))) / 100.0
            # q0 = (100.0 * 0.622 * (Ew0 / (p - Ew0))) / 100.0
            Ql = air_density*Lv*cE*wind_speed*(qz-q0)

            count_iters += 1
            if count_iters > 10 or abs(previous_zeta - zeta) < .1:
                converged = True
        return Qs, Ql
    
    def roughness_length(self,days_since_snowfall,layertype):
        """
        Function to determine the roughness length of the surface. This assumes the roughness of snow
        linearly degrades with time in 60 days from that of fresh snow to firn.

        Parameters
        ----------
        days_since_snowfall : int
            Number of days since fresh snow occurred
        """
        roughness_fresh_snow = 0.24                     # surface roughness length for fresh snow [mm] (Moelg et al. 2012, TC)
        roughness_ice = 1.7                             # surface roughness length for ice [mm] (Moelg et al. 2012, TC)
        roughness_firn = 4.0                            # surface roughness length for aged snow [mm] (Moelg et al. 2012, TC)
        aging_factor_roughness = 0.06267                # effect of ageing on roughness length: 60 days from 0.24 to 4.0 => 0.06267

        if layertype[0] in ['snow']:
            sigma = min(roughness_fresh_snow + aging_factor_roughness * days_since_snowfall, roughness_firn)
        elif layertype[0] in ['firn']:
            sigma = roughness_firn
        elif layertype[0] in ['ice']:
            sigma = roughness_ice
        return sigma/1000
    
    def vapor_pressure(self,T):
        return 6.1078*np.exp(17.1*T/(235+T))
    
    def interpClimate(self,climateds,time,varname,bin_idx=-1):
        """
        Interpolates climate variables from the hourly dataset to get sub-hourly data.

        Parameters
        ----------
        climateds : xr.Dataset
            Climate dataset containing temperature, precipitation, pressure, air density, wind speed,
            shortwave radiation, and total cloud cover.
        time : datetime
            Timestamp to interpolate the climate variable.
        bin_idx : int, default = -1
            Index number of the bin being run. Unspecified for running a variable that is elevation-independent.
        """
        climate_before = climateds.sel(time=time.floor('H'))
        climate_after = climateds.sel(time=time.ceil('H'))
        if bin_idx == -1:
            before = climate_before[varname].to_numpy()
            after = climate_after[varname].to_numpy()
        else:
            before = climate_before[varname].to_numpy()[bin_idx]
            after = climate_after[varname].to_numpy()[bin_idx]
        return before+(after-before)*(time.minute/60)
    