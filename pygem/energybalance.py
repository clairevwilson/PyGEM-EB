#import xarray as xr
import numpy as np
import pandas as pd
import pygem.pygem_input as pygem_prms

class meltProfile():
    """
    Temperature and density profile function to distribute melt through vertical layers and define
    the content (snow or firn) of each layer.
    
    Attributes
    ----------
    climateds : 

    """ 
    def __init__(self,snow_temp,snow_density,sfi_dz0,
                 option_initTemp='piecewise',option_initDensity='piecewise',option_initwater='zero_w0'):
        """
        Initialize the temperature, density and water content profile of the vertical layers and
        instanciate the model.

        Attributes
        ----------
        snow_temp : np.ndarray
            Array containing the initial snow temperatures in Celsius at associated depths.
        snow_density : np.ndarray
            Array containing the initial snow density in kg m**-3 at associated depths.
        sfi_dz0 : np.ndarray
            Array containing the initial snow, firn, and ice thicknesses [m]
        option_initwater : str, default: 'zero_w0'
            'zero_w0' or 'initial_w0': defines the water content in the snowpack at the first time step.
        option_initTemp : str, default: 'piecewise'
            'piecewise': bases scheme on turning points following DEBAM
            'interp': uses iButton data to interpolate temperatures
        option_initDensity : str, default: 'piecewise'
            'piecewise': bases scheme on turning points following DEBAM
            'interp': uses snowpit data to interpolate density
        """
        # Calculate the layer depths based on initial snow, firn and ice depths
        layer_dz,layer_z = self.getLayerdz(sfi_dz0)

        # Initialize SNOW layer temperatures based on chosen method and data (snow_temp)  
        if option_initTemp in ['piecewise']:
            Tprofile = self.initProfilesPiecewise(layer_z,snow_temp,'temp')
        elif option_initTemp in ['interp']:
            Tprofile = np.interp(layer_z,snow_temp[0,:],snow_temp[1,:])
        else:
            assert 1==0, "Choose between 'piecewise' and 'interp' methods for temp initialization"

        # Initialize SNOW layer density based on chosen method and data (snow_density)
        if option_initDensity in ['piecewise']:
            pprofile = self.initProfilesPiecewise(layer_z,snow_density,'density')
        elif option_initDensity in ['interp']:
            pprofile = np.interp(layer_z,snow_density[0,:],snow_density[1,:])
        else:
            assert 1==0, "Choose between 'piecewise' and 'interp' methods for density initialization"

        # Add firn and ice layers, assuming constant temperature and density?
        # layer_dz.append(sfi_dz0[1:])
        # Tprofile.append(pygem_prms.density_ice)
        self.nlayers = len(layer_dz)

        # Initialize water content
        if option_initwater in ['zero_w0']:
            wprofile = np.zeros(self.nlayers)
        elif option_initwater in ['initial_w0']:
            assert 1==0, "Only default water content method is set up"

        # Initialize layer type based on density
        firn_density_cutoff = 400
        ice_density = 900
        self.layertype = np.where(pprofile<firn_density_cutoff,'snow','firn')
        self.layertype[np.where(pprofile>=ice_density)] = 'ice'

        frac_absrad = np.zeros(self.nlayers)
        extinct_coef = np.zeros(self.nlayers)
        for idx,type in enumerate(self.layertype):
            if type in ['snow']:
                frac_absrad[idx] = 0.9
                extinct_coef[idx] = 17.1
            elif type in ['firn']:
                frac_absrad[idx] = 0.85
                extinct_coef[idx] = 9.8
            elif type in ['ice']:
                frac_absrad[idx] = 0.8
                extinct_coef[idx] = 2.5
        
        self.Tprofile = Tprofile
        self.pprofile = pprofile
        self.wprofile = wprofile
        self.layerdz = layer_dz
        self.layerz = layer_z
        self.frac_absrad = frac_absrad
        self.extinct_coef = extinct_coef
        return 
    
    def EnergyMassBalance(self,climateds,bin_idx,method_turbulent='MO-similarity'):
        """
        Calculates the surface heat fluxes at each point on the glacier and applies mass-balance
        scheme to calculate melt and refreeze at each time point.

        Attributes
        ----------
        climateds : xr.Dataset
            Climate dataset containing temperature, precipitation, pressure, air density, wind speed,
            shortwave radiation, and total cloud cover.
        time : list-like
            List of datetime objects containing timestamps for each data value in the climateds.
        bin_idx : int
            Index number of the bin being run
        method_turbulent : str, default: 'MO-similarity'
            'MO-similarity', 'bulk-aerodynamic', or 'Richardson': determines method for calculating 
            turbulent fluxes
        """
        # Initiate variables to be filled
        melt_monthly = []
        refreeze_monthly = []
        time_idx = 0
        Ts = 0 #initial guess is needed to start the model
        albedo = pygem_prms.albedo_fresh_snow #initial timestep albedo
        BC = [pygem_prms.freshsnow_BC,pygem_prms.freshsnow_BC]
        dt = 3600/3  # in s, must be a multiple of 3600

        start_time = climateds.coords['time'][0].values
        end_time = climateds.coords['time'][climateds.dims['time']-1].values
        time_dt = pd.date_range(start_time,end_time,freq=str(dt)+'S')

        # ===== ENTER TIME LOOP =====
        for hour in time_dt[0:10]:
            # Unpack climate variables
            vars_to_unpack = ['bin_temp','bin_prec','bin_sp','bin_density','bin_snow','uwind','vwind','tcc','surfrad']
            # Process climate variables to get values for timesteps between hours
            if hour.minute == 0:
                climatearray = climateds.sel(time=hour)
                temperature_C,precipitation,pressure,density,is_snow = [climatearray[var].to_numpy()[bin_idx] for var in vars_to_unpack[0:5]]
                uwind,vwind,tcc,surfrad = [climatearray[var].to_numpy() for var in vars_to_unpack[5:]]
            else:
                climatearray_before = climateds.sel(time=hour.floor('H'))
                climatearray_after = climateds.sel(time=hour.ceil('H'))
                temperature_C,precipitation,pressure,density,is_snow = [self.interpClimate(climatearray_before,climatearray_after,
                                                                                           var,hour.minute,bin_idx) for var in vars_to_unpack[0:5]]
                uwind,vwind,tcc,surfrad = [self.interpClimate(climatearray_before,climatearray_after,
                                                              var,hour.minute) for var in vars_to_unpack[5:]]
            wind_speed = (uwind**2 + vwind**2)**(1/2)
            temperature_K = temperature_C + 273.15

            # Check if snowfall occurred and set variable for last_snow_day
            if precipitation > 0.01 and is_snow:
                last_snow_day = hour.day

            # Check for special time cases: first timestep, daily and monthly occurrences
            if time_idx<1:
                hourly_Q = [] #initialize variable to store hourly heat fluxes
            elif hour.is_month_start and hour.hour < 1:
                # any monthly happenings go here!!!
                # convert previous month Q to M, summing hourly Q_melts
                monthly_M = np.sum(hourly_Q)*3600/(pygem_prms.density_water*pygem_prms.Lh_rf)
                melt_monthly.append(monthly_M)

                #re-initialize storage for hourly heat fluxes
                hourly_Q = []
            elif hour.hour < 1:
                # any daily happenings go here!!!
                days_since_snowfall = hour.day - last_snow_day
                #update albedo
                albedo = self.getAlbedo(BC,days_since_snowfall)
                roughness = self.roughness_length(days_since_snowfall)

            # For stability, temporal resolution needs to be upped. to run hourly, simply set dt to 3600s
            # SHORTWAVE RADIATION (Snet)
            # sun_pos = solar.get_position(hour,glacier_table['CenLon'],glacier_table['CenLat'])
            Snet_surf = surfrad*(1-albedo)/dt #* (cos(theta))
            Snet_penetrating = Snet_surf*self.frac_absrad*np.exp(-self.extinct_coef*self.layerz)
            # if I decide to include shading effects, move this to a separate function
                
            # LONGWAVE RADIATION (Lnet)
            # calculate vapor pressure from air temp
            ezt = self.vapor_pressure(temperature_K)
            # clear=sky emissivity
            Ecs = .23+ .433*(ezt/(temperature_K))**(1/8)
            # cloud emissivity
            Ecl = 0.984 # Klok and Oerlemans, 2002
            Lnet = Ecs*(1-tcc**2)+Ecl*tcc**2

            # RAIN FLUX (Qp)
            rain_mask = -(is_snow-1)
            Qp = rain_mask*pygem_prms.Cp_water*(temperature_C-Ts)*precipitation/dt

            # TURBULENT FLUXES (Qs and Ql)
            # ******
            # if method_turbulent in ['MO-similarity']:
            #     Ql, Qs = self.getTurbulentMO(temperature_K,Ts,wind_speed,roughness)
            # else:
            #     print('Only MO similarity method is set up for turbulent fluxes')
            #     Ql, Qs = self.getTurbulentMO(temperature_K,Ts,wind_speed,roughness)
            # ******
            
            # SUM ENERGY FOR MELT AND STORE
            Qm = Snet_surf + Lnet + Qp
            hourly_Q.append(Qm)

            # RECALCULATE TEMPERATURE PROFILE
            # ******
            K = 0.0138e-1 - 1.01e-3*self.pprofile + 3.233e-6*np.square(self.pprofile)
            dTdz = [(self.Tprofile[i+1]-self.Tprofile[i])/(self.layerdz[i+1]) for i in range(self.nlayers-1)]
            dTdz = np.insert(dTdz,0,(self.Tprofile[0]-Ts)/self.layerdz[0])
            ddzKdtdz = [(dTdz[i+1]*K[i+1]-dTdz[i]*K[i])/(self.layerdz[i+1]) for i in range(self.nlayers-1)]
            ddzKdtdz = np.insert(ddzKdtdz,0,(dTdz[0]/self.layerdz[0])) #assumes dTdz at the surface is 0
            #print(ddzKdtdz,'ddzKdtdz')
            dQdz = Snet_penetrating
            dQdz[0] = Qm/self.layerdz[0]
            dT = (dQdz+ddzKdtdz)*dt/(self.pprofile*pygem_prms.Cp_ice)
            new_T = self.Tprofile + dT
            # ******

            leftovers = 0
            # ENTER LAYER LOOP
            for layer in range(self.nlayers):
                m_layer = self.layerdz[layer]*self.pprofile[layer]
                if new_T[layer] > 0:
                    dm_melt = new_T[layer]*m_layer*pygem_prms.Cp_ice/pygem_prms.Lh_rf
                    self.wprofile[layer] += dm_melt + leftovers
                    irreducible_water = 0.0143*np.exp(3.3*(1-self.pprofile[layer]/pygem_prms.density_ice))

                    # check if meltwater exceeds the irreducible water content of the snow
                    if self.wprofile[layer] > irreducible_water:
                        # set water content to irreducible water content and add the difference to leftovers
                        leftovers = irreducible_water - self.wprofile[layer]
                        self.wprofile[layer] = irreducible_water
                    else: #if not overflowing, leftovers should be set back to 0
                        leftovers = 0
                    self.Tprofile[layer] = 0
                if new_T[layer] < 0 and self.wprofile[layer] > 0:
                    E_temperature = np.abs(new_T[layer])*m_layer*pygem_prms.Cp_ice
                    E_water = self.wprofile[layer]*pygem_prms.Lh_rf
                    E_pore = (self.pprofile[layer]-pygem_prms.density_ice)*self.layerdz[layer]*pygem_prms.Lh_rf
                    dm = np.min([E_temperature,E_water,E_pore])/pygem_prms.Lh_rf
                    m_layer += dm
                    self.pprofile[layer] = m_layer/self.layerdz[layer]
                    self.Tprofile[layer] = -(E_temperature-dm*pygem_prms.Lh_rf/pygem_prms.Cp_ice/m_layer)
            
            Ts_new = np.interp(0,self.layerz[0:2],self.Tprofile[0:2])
            time_idx +=1
            
    def getLayerdz(self,sfi_dz0):
        """
        Calculates layer depths based on an exponential growth function with prescribed rate of growth and 
        initial layer depth (from pygem_input). 

        Attributes
        ----------
        sfi_dz0 : np.ndarray
            Initial thicknesses of the snow, firn and ice layers [m]
        """
        dz_toplayer = pygem_prms.dz_toplayer
        layer_growth = pygem_prms.layer_growth

        #Initialize variables to get looped
        total_depth = 0
        layer_no = 1
        layer_dz = [dz_toplayer]
        while total_depth < sfi_dz0[0]:
            layer_dz.append(dz_toplayer * np.exp(layer_no*layer_growth))
            layer_no += 1
            total_depth = np.sum(layer_dz)
        layer_dz[-1] = layer_dz[-1] - (total_depth-sfi_dz0[0])
        layer_z = [np.sum(layer_dz[:i]) for i in range(len(layer_dz))]
        
        # Add firn and ice bins
        return layer_dz, layer_z
    
    def initProfilesPiecewise(self,layer_z,snow_var,varname):
        """
        Based on the DEBAM scheme for temperature and density that assumes linear changes with depth 
        in three piecewise sections.

        Attributes
        ----------
        layer_z : np.ndarray
            Bottom depth of the layers to be filled.
        snow_var : np.ndarray
            Turning point snow temperatures or densities and the associated depths in pairs 
            by (depth,temp/density value). If a surface value (z=0) is not prescribed, temperature 
            is assumed to be 0C, or density to be 100 kg m-3.
        varname : str
            'temp' or 'density': which variable is being calculated
        """
        #check if inputs are the correct dimensions
        assert np.shape(snow_var) in [(4,2),(3,2)], "! Snow inputs data is improperly formatted"

        #check if a surface value is given; if not, add a row at z=0, T=0C or p=100kg/m3
        if np.shape(snow_var) == (3,2):
            assert snow_var[0,0] == 1.0, "! Snow inputs data is improperly formatted"
            if varname in ['temp']:
                np.insert(snow_var,0,[0,0],axis=0)
            elif varname in ['density']:
                np.insert(snow_var,0,[0,100],axis=0)

        #calculate slopes and intercepts for the piecewise function
        slopes = [(snow_var[i,1]-snow_var[i+1,1])/(snow_var[i,0]-snow_var[i+1,0]) for i in range(3)]
        intercepts = [snow_var[i+1,1] - slopes[i]*snow_var[i+1,0] for i in range(3)]

        #solve piecewise functions at each layer bottom depth
        layer_var = np.piecewise(layer_z,
                     [layer_z <=snow_var[1,0], (layer_z <= snow_var[2,0]) & (layer_z > snow_var[1,0]),
                      (layer_z <= snow_var[3,0]) & (layer_z > snow_var[2,0])],
                      [lambda x: slopes[0]*x+intercepts[0],lambda x:slopes[1]*x+intercepts[1],
                       lambda x: slopes[2]*x+intercepts[2]])
        return layer_var
    
    def getAlbedo(self,BC_conc,days_since_snowfall,switch_snow,switch_melt,switch_LAP):
        """
        Updates the surface albedo based on the concentration of LAPs and the degradation with time. Switches allow
        for controlled simulations to see the changes associated with snow-albedo feedback, melt-albedo feedback and
        LAP-albedo feedback.

        Attributes
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

    def getTurbulentMO(self,air_temp,surf_temp,wind_speed,roughness):
        """
        Calculates turbulent fluxes (sensible and latent heat) based on Monin-Obukhov Similarity 
        Theory, requiring iteration.

        Attributes
        ----------
        air_temp : float
            Air temperature [C]
        surf_temp : float
            Surface temperature of snowpack/ice [C]
        """
    def roughness_length(self,days_since_snowfall):
        """
        Function taken pretty much exactly from COSIPY to determine the roughness length based
        on the time since the last snowfall. 
        *****THIS DOES NOT WORK WHEN THE SURFACE IS BARE ICE!

        Attributes
        ----------
        days_since_snowfall : int
            Number of days since fresh snow occurred
        """
        roughness_fresh_snow = 0.24                     # surface roughness length for fresh snow [mm] (Moelg et al. 2012, TC)
        roughness_ice = 1.7                             # surface roughness length for ice [mm] (Moelg et al. 2012, TC)
        roughness_firn = 4.0                            # surface roughness length for aged snow [mm] (Moelg et al. 2012, TC)
        aging_factor_roughness = 0.06267                # effect of ageing on roughness lenght (days) 60 days from 0.24 to 4.0 => 0.06267

        sigma = min(roughness_fresh_snow + aging_factor_roughness * days_since_snowfall, roughness_firn)
        return sigma/1000
    def vapor_pressure(self,T):
        return 6.1078*np.exp(17.1*T/(235+T))
    
    def interpClimate(self,climatearray_before,climatearray_after,varname,min,bin_idx=-1):
        """
        Interpolates two climate variables based on the climatearray (climateds selected at a
        given hour) to get sub-hourly data.
        """
        if bin_idx == -1:
            before = climatearray_before[varname].to_numpy()
            after = climatearray_after[varname].to_numpy()
        else:
            before = climatearray_before[varname].to_numpy()[bin_idx]
            after = climatearray_after[varname].to_numpy()[bin_idx]
        return before+(after-before)*(min/60)
    