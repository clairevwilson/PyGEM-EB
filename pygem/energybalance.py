#import xarray as xr
import numpy as np
import pygem.pygem_input as pygem_prms

class meltProfile():
    """
    Temperature and density profile function to distribute melt through vertical layers and define
    the content (snow or firn) of each layer.
    
    Attributes
    ----------
    climateds : 

    """ 
    def __init__(self,snow_depth,snow_temp,snow_density,option_initTemp='piecewise',option_initDensity='piecewise',option_initwater='zero_w0'):
        """
        Initialize the temperature, density and water content profile of the vertical layers and
        instanciate the model.

        Attributes
        ----------
        climateds : xr.Dataset
            Dataset containing hourly data for T, P, p, density, wind speed, 
            cloud cover, and SW radiation.
        snow_depth : int
            Depth where snow rests on ice or firn.
        snow_temp : np.ndarray
            Array containing the initial snow temperatures at associated depths.
        snow_density : np.ndarray
            Array containing the initial snow density at associated depths.
        option_initwater : str, default: 'zero_w0'
            'zero_w0' or 'initial_w0': defines the water content in the snowpack at the first time step.
        option_initTemp : str, default: 'piecewise'
            'piecewise': bases scheme on turning points following DEBAM
            'interp': uses iButton data to interpolate temperatures
        option_initDensity : str, default: 'piecewise'
            'piecewise': bases scheme on turning points following DEBAM
            'interp': uses snowpit data to interpolate density
        """
        # Calculate the layer depths based on exponential function and total initial snow depth.
        dz_top = 0.01           # depth of the top bin in m
        layer_growth = 0.5      # factor for exponential growth of the bin depth
        total_depth = 0
        layer_dz = [dz_top]
        layer = 1
        while total_depth < snow_depth:
            layer_dz.append(dz_top * np.exp(layer*layer_growth))
            layer += 1
            total_depth = np.sum(layer_dz)
        layer_dz[-1] = layer_dz[-1] - (total_depth-snow_depth)
        layer_z = [np.sum(layer_dz[:i]) for i in range(len(layer_dz))]
        n_layers = len(layer_z)
        
        # Initialize layer temperatures based on chosen method and data (snow_temp)  
        if option_initTemp in ['piecewise']:
            Tprofile = self.initProfilesPiecewise(layer_z,snow_temp,'temp')
        elif option_initTemp in ['interp']:
            Tprofile = np.interp(layer_z,snow_density[0,:],snow_density[1,:])
        else:
            assert 1==0, "Choose between 'piecewise' and 'interp' methods for temp initialization"

        # Initialize layer density based on chosen method and data (snow_density)
        if option_initDensity in ['piecewise']:
            pprofile = self.initProfilesPiecewise(layer_z,snow_density,'density')
        elif option_initDensity in ['interp']:
            pprofile = np.interp(layer_z,snow_density[0,:],snow_density[1,:])
        else:
            assert 1==0, "Choose between 'piecewise' and 'interp' methods for density initialization"

        # Initialize water content
        if option_initwater in ['zero_w0']:
            wprofile = np.zeros(n_layers)
        elif option_initwater in ['initial_w0']:
            assert 1==0, "Only default water content method is set up"

        self.Tprofile = Tprofile
        self.pprofile = pprofile
        self.wprofile = wprofile
        self.layerdz = layer_dz
        self.layerz = layer_z
        return 
    
    def EnergyMassBalance(self,climateds,bin_no):
        """
        Calculates the surface heat fluxes at each point on the glacier and applies mass-balance
        scheme to calculate melt and refreeze at each time point.

        Attributes
        ----------
        climateds : xr.Dataset
            Climate dataset containing temperature, precipitation, pressure, air density, wind speed,
            shortwave radiation, and total cloud cover.
        
        """
        # Initiate variables to be filled
        melt_monthly = []
        time_idx = 0
        Ts = 0 #initial guess is needed to start the model
        albedo = pygem_prms.albedo_clean_snow #initial timestep albedo

        # ===== ENTER HOURLY LOOP!! =====
        for hour in climateds:
            if time_idx<1:
                hourly_Q = [] #initialize variable to store hourly heat fluxes
            elif hour.is_month_start and hour.hour < 1:
                # convert previous month Q to M, summing hourly Q_melts
                monthly_M = np.sum(hourly_Q)*3600/(pygem_prms.density_water*pygem_prms.Lh_rf)
                melt_monthly.append(monthly_M)

                #re-initialize storage for hourly heat fluxes
                hourly_Q = []

                #update albedo
                #albedo = self.getAlbedo()

            # Unpack climate variables
            vars_to_unpack = ['bin_temp','bin_prec','bin_sp','bin_density','bin_snow','wind','tcc','surfrad']
            climatearray = climateds.sel(time=hour)
            temperature_C,precipitation,pressure,density,is_snow = [climatearray[var].to_numpy()[bin_no] for var in vars_to_unpack[0:5]]
            wind_speed,tcc,surfrad = [climatearray[var].to_numpy() for var in vars_to_unpack[5:]]
            temperature_K = temperature_C + 273.15


            # SHORTWAVE RADIATION (Snet)
            # sun_pos = solar.get_position(hour,glacier_table['CenLon'],glacier_table['CenLat'])
            Snet_surf = surfrad*(np.array([1,1,1])-albedo) #* (cos(theta))
            #Snet = Snet_surf*vert_bins['lambdas']*np.exp(-vert_bins['bin_z']*vert_bins['rs'])
                
            # LONGWAVE RADIATION (Lnet)
            #if option_LW in ['COSIPY-LIKE']:
            #vapor pressure based on air temp
            ezt = self.vapor_pressure(temperature_K)
            #clear=sky emissivity
            Ecs = .23+ .433*(ezt/(temperature_K))**(1/8)
            #cloud emissivity
            Ecl = 0.984 # Klok and Oerlemans, 2002
            Lnet = Ecs*(1-temperature_K**2)+Ecl*temperature_K**2
            #elif option_LW in ['other options']:

            # RAIN FLUX (Qp)
            rain_mask = -(is_snow-1)
            Qp = rain_mask*pygem_prms.Cp_water*(temperature_C-Ts)*precipitation

    
    def initProfilesPiecewise(self,layer_z,snow_var,varname):
        """
        Based on the DEBAM scheme for temperature and density that assumes linear changes with depth 
        in three piecewise sections.

        Attributes
        ----------
        layer_z : np.ndarray
            Bottom depth of the layers to be filled.
        snow_var : np.ndarray
            Turning point snow temperatures or densities and the associated depths.
            If a surface value (z=0) is not prescribed, temperature is assumed to be 0C,
            or density to be 100 kg/m3.
        varname : str
            'temp' or 'density': which variable is being calculated
        """
        #check if inputs are the correct dimensions
        assert np.shape(snow_var) in [(4,2),(3,2)], "! Snow inputs data is improperly formatted"

        #check if a surface temperature is given; if not, add a row at z=0, T=0C
        if snow_var[1,1] != 0:
            if var in ['temp']:
                np.insert(snow_var,0,[0,0],axis=0)
            elif var in ['density']:
                np.insert(snow_var,0,[0,100],axis=0)

        #calculate slopes and intercepts for the piecewise function
        slopes = [(snow_var[i,1]-snow_var[i+1,1])/(snow_var[i,0]-snow_var[i+1,0]) for i in range(3)]
        intercepts = [snow_var[i+1,1] - slopes[i]*snow_var[i+1,0] for i in range(3)]

        #solve piecewise functions at each layer bottom depth
        layer_var = np.piecewise(layer_z,
                     [layer_z <=snow_var[1,0], (layer_z <= snow_var[2,0]) & (layer_z > snow_var[1,0]),
                      (layer_z <= snow_var[3,0]) & (layer_z > snow_var[2,0]), layer_z > snow_var[3,0]],
                      [lambda x: slopes[0]*x+intercepts[0],lambda x:slopes[1]*x+intercepts[1],
                       lambda x: slopes[2]*x+intercepts[2],lambda x:slopes[3]*x+intercepts[3]])
        return layer_var
    
    def getAlbedo(self,BC_conc,grain_size,time_since_snowfall,switch_snow,switch_melt,switch_LAP):
        """
        Updates the surface albedo based on the concentration of LAPs and the degradation with time. Switches allow
        for controlled simulations to see the changes associated with snow-albedo feedback, melt-albedo feedback and
        LAP-albedo feedback.

        Attributes
        ----------
        BC_conc : np.ndarray
            Concentration of BC at the surface and in the bulk snowpack [ppb]
        time_since_snowfall : int
            Number of days since the last snowfall
        switch_snow : Bool
            Switch to turn on/off snow-albedo feedback
        switch_melt : Bool
            Switch to turn on/off melt-albedo feedback
        switch_LAP : Bool
            Switch to turn on/off LAP-albedo feedback
        """

    def getTurbulentFluxes(self,temperature,method='MO-similarity'):
        """
        Calculates turbulent fluxes (sensible and latent heat) based on the chosen method. The default method is based
        on Monin-Obukhov Similarity Theory and requires iteration. 

        """

    def vapor_pressure(self,T):
        return 6.1078*np.exp(17.1*T/(235+T))
    