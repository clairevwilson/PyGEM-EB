import numpy as np
import pandas as pd
import pygem_eb.energybalance as eb
import pygem.pygem_input as pygem_prms

class massBalance():
    """
    Scheme for the multi-layer snowpack model which takes in the energy balance and calculates the change
    in layer properties and amount of melt and refreeze.
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
        layerh,layerz = self.getLayerdz(sfi_dz0)
        print('layerh',layerh)
        print('layerz',layerz)

        # Initialize SNOW layer temperatures based on chosen method and data (snow_temp)  
        if option_initTemp in ['piecewise']:
            Tprofile = self.initProfilesPiecewise(layerz,snow_temp,'temp')
        elif option_initTemp in ['interp']:
            Tprofile = np.interp(layerz,snow_temp[0,:],snow_temp[1,:])
        else:
            assert 1==0, "Choose between 'piecewise' and 'interp' methods for temp initialization"

        # Initialize SNOW layer density based on chosen method and data (snow_density)
        if option_initDensity in ['piecewise']:
            pprofile = self.initProfilesPiecewise(layerz,snow_density,'density')
        elif option_initDensity in ['interp']:
            pprofile = np.interp(layerz,snow_density[0,:],snow_density[1,:])
        else:
            assert 1==0, "Choose between 'piecewise' and 'interp' methods for density initialization"

        # Add firn and ice layers, assuming constant temperature and density?
        # layer_dz.append(sfi_dz0[1:])
        # Tprofile.append(pygem_prms.density_ice)
        self.nlayers = len(layerh)

        # Initialize water content
        if option_initwater in ['zero_w0']:
            wprofile = np.zeros(self.nlayers)
        elif option_initwater in ['initial_w0']:
            assert 1==0, "Only zero water content method is set up"

        # Initialize layer type based on density
        firn_density_cutoff = 400
        ice_density = 900
        self.layertype = np.where(pprofile<firn_density_cutoff,'snow','firn')
        self.layertype[np.where(pprofile>=ice_density)] = 'ice'
        
        self.Tprofile = Tprofile
        self.pprofile = pprofile
        self.wprofile = wprofile
        self.layerh = layerh
        self.layerz = layerz
        return 
    
    def massBalance(self,climateds,bin_idx):
        # Initiate variables to be filled
        time_idx = 0
        surftemp = 0 #initial guess is needed to start the model
        albedo = pygem_prms.albedo_fresh_snow #initial timestep albedo
        BC = [pygem_prms.freshsnow_BC,pygem_prms.freshsnow_BC]

        dt = 3600/3  # in s, must be a multiple of 3600
        start_time = climateds.coords['time'][0].values
        end_time = climateds.coords['time'][climateds.dims['time']-1].values
        time_dt = pd.date_range(start_time,end_time,freq=str(dt)+'S')
        snow_timestamp = time_dt[0]

        # ===== ENTER TIME LOOP =====
        for time in time_dt[0:2]:
            # initiate the energy balance to get climate data unpacked
            enbal = eb.energyBalance(climateds,time,bin_idx,dt)

            # check if snowfall occurred
            if enbal.prec > 0.01 and enbal.tempC < pygem_prms.tsnow_threshold:
                # set timestamp
                snow_timestamp = time
                # add snowfall to uppermost bin
                # *******

            if time.hour < 1:
                # any daily happenings go here!!!
                days_since_snowfall = time.day - snow_timestamp.day
                #update albedo
                #albedo = self.getAlbedo(BC,days_since_snowfall)

            # Run the surface energy balance
            Qm = enbal.surfaceEB(surftemp,self.layerz,self.layertype,days_since_snowfall,albedo)
            trash, Snet_penetrating = enbal.getSW(enbal.surfrad,albedo,self.layerz,self.layertype)

            # Run Crank-Nicholson scheme to 
            surftemp_loop = True
            iteration_count = 0
            while surftemp_loop:
                # RECALCULATE TEMPERATURE PROFILE
                # ******
                K = 0.0138e-1 - 1.01e-3*self.pprofile + 3.233e-6*np.square(self.pprofile)
                #print('K',K)
                dTdz = [(self.Tprofile[i+1]-self.Tprofile[i])/(self.layerh[i+1]) for i in range(self.nlayers-1)]
                dTdz = np.insert(dTdz,0,(self.Tprofile[0]-surftemp)/self.layerh[0])
                #print('T: ',self.Tprofile[0:5])
                #print('dTdz: ',dTdz[0:5])
                ddzKdtdz = [(dTdz[i+1]*K[i+1]-dTdz[i]*K[i])/(self.layerh[i+1]) for i in range(self.nlayers-1)]
                ddzKdtdz = np.insert(ddzKdtdz,0,(dTdz[0]/self.layerh[0])) #assumes dTdz at the surface is 0
                #print('ddzKdtdz: ',ddzKdtdz[0:5])
                dQdz = Snet_penetrating
                dQdz[0] = Qm/self.layerh[0]
                dT = (dQdz+ddzKdtdz)*dt/(self.pprofile*pygem_prms.Cp_ice)
                #print('dT: ',dT[0:5])
                new_T = self.Tprofile + dT
                #print('new T', new_T)
                # ******

                # ENTER LAYER LOOP
                leftovers = 0
                for layer in range(self.nlayers):
                    m_layer = self.layerh[layer]*self.pprofile[layer]
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
                        E_pore = (self.pprofile[layer]-pygem_prms.density_ice)*self.layerh[layer]*pygem_prms.Lh_rf
                        dm = np.min([E_temperature,E_water,E_pore])/pygem_prms.Lh_rf
                        m_layer += dm
                        self.pprofile[layer] = m_layer/self.layerh[layer]
                        self.Tprofile[layer] = -(E_temperature-dm*pygem_prms.Lh_rf/pygem_prms.Cp_ice/m_layer)
                
                surftemp_new = np.interp(0,self.layerz[0:2],self.Tprofile[0:2])
                iteration_count += 1
                if abs(surftemp_new - surftemp) < 0.25:
                    surftemp_loop = True
                else:
                    surftemp = surftemp_new
                if iteration_count > 3:
                    print('surftemp loop failing to converge!')
                    break

            time_idx +=1

                        # if time.is_month_start and time.hour < 1:
            #     # any monthly happenings go here!!!
            #     # convert previous month Q to M, summing hourly Q_melts
            #     monthly_M = np.sum(hourly_Q)*3600/(pygem_prms.density_water*pygem_prms.Lh_rf)
            #     melt_monthly.append(monthly_M)
            #     #re-initialize storage for hourly heat fluxes
            #     hourly_Q = []

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
        layeridx = 0
        layerh = []
        while total_depth < sfi_dz0[0]:
            layerh.append(dz_toplayer * np.exp(layeridx*layer_growth))
            layeridx += 1
            total_depth = np.sum(layerh)
        layerh[-1] = layerh[-1] - (total_depth-sfi_dz0[0])
        layerz = [np.sum(layerh[:i+1])-(layerh[i]/2) for i in range(len(layerh))]
    
        # Add firn and ice bins


        return layerh, layerz
    
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