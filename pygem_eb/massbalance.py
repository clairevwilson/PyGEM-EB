import numpy as np
import pandas as pd
import pygem_eb.energybalance as eb
import pygem_eb.input as eb_prms
import pygem_eb.albedo as eb_albedo

class massBalance():
    """
    Scheme for the multi-layer snowpack model which takes in the energy balance and calculates the change
    in layer properties and amount of melt and refreeze.
    """
    def __init__(self,snow_temp,snow_density,sfi_h0):
        """
        Initialize the temperature, density and water content profile of the vertical layers.

        Parameters
        ----------
        snow_temp : np.ndarray
            Array containing the initial snow temperatures in Celsius at associated depths.
        snow_density : np.ndarray
            Array containing the initial snow density in kg m**-3 at associated depths.
        sfi_h0 : np.ndarray
            Array containing the initial snow, firn, and ice thicknesses [m]
        """
        # Calculate the layer depths based on initial snow, firn and ice depths
        layerh,layerz,layertype = self.getLayers(sfi_h0)
        self.nlayers = len(layerh)

        # Initialize SNOW layer temperatures based on chosen method and data (snow_temp)
        snow_idx =  np.where(layertype=='snow')[0]
        snow_layerz = layerz[snow_idx] 
        if eb_prms.option_initTemp in ['piecewise']:
            Tprofile = self.initProfilesPiecewise(snow_layerz,snow_temp,'temp')
        elif eb_prms.option_initTemp in ['interp']:
            Tprofile = np.interp(snow_layerz,snow_temp[0,:],snow_temp[1,:])
        else:
            assert 1==0, "Choose between 'piecewise' and 'interp' methods for temp initialization"

        # Initialize SNOW layer density based on chosen method and data (snow_density)
        if eb_prms.option_initDensity in ['piecewise']:
            pprofile = self.initProfilesPiecewise(snow_layerz,snow_density,'density')
        elif eb_prms.option_initDensity in ['interp']:
            pprofile = np.interp(snow_layerz,snow_density[0,:],snow_density[1,:])
        else:
            assert 1==0, "Choose between 'piecewise' and 'interp' methods for density initialization"

        # Initialize FIRN AND ICE temperature and density
        # Calculate slope that linearly increases density from the bottom snow bin to the top of the ice layer
        pslope = (eb_prms.density_ice - pprofile[-1])/(np.sum(sfi_h0[0:2])-layerz[snow_idx[-1]])
        for idx,type in enumerate(layertype):
            if type not in ['snow']:
                Tprofile = np.append(Tprofile,eb_prms.temp_temp)
            if type in['firn']:
                pprofile = np.append(pprofile,pprofile[snow_idx[-1]] + pslope*(layerz[idx]-layerz[snow_idx[-1]]))
            elif type in ['ice']:
                pprofile = np.append(pprofile,eb_prms.density_ice)

        # Initialize water content
        if eb_prms.option_initWater in ['zero_w0']:
            wprofile = np.zeros(self.nlayers)
        elif eb_prms.option_initWater in ['initial_w0']:
            assert 1==0, "Only zero water content method is set up"

        # Initialize BC and dust content
        if eb_prms.switch_LAPs == 0:
            BC = [0,0]
            dust = [0,0]
        elif eb_prms.switch_LAPs == 2 and eb_prms.initLAPs is not None:
            BC = eb_prms.initLAPs[0,:]
            dust = eb_prms.initLAPs[1,:]
        else:
            BC = [eb_prms.BC_freshsnow,eb_prms.BC_freshsnow]
            dust = [eb_prms.dust_freshsnow,eb_prms.dust_freshsnow]
        
        self.Tprofile = Tprofile
        self.pprofile = pprofile
        self.wprofile = wprofile
        self.layerh = layerh
        self.layerz = layerz
        self.layertype = layertype
        return 
    
    def main(self,climateds,bin_idx,dt):
        """
        Main function running the time loop and mass balance scheme to solve layer temperature
        and density profiles. 

        Parameters
        ----------
        climateds : xr.Dataset
            Climate dataset containing temperature, precipitation, pressure, air density, wind speed,
            shortwave radiation, and total cloud cover.
        bin_idx : int
            Index number of the bin being run.
        dt : float
            Resolution for the time loop [s]
        """
        start_time = climateds.coords['time'][0].values
        end_time = climateds.coords['time'][climateds.dims['time']-1].values
        time_dt = pd.date_range(start_time,end_time,freq=str(dt)+'S')

        # Set initial albedo based on surface type
        if self.layertype[0] in ['snow']:
            albedo = eb_prms.albedo_fresh_snow
            snow_timestamp = time_dt[0]
        elif self.layertype[0] in ['firn']:
            albedo = eb_prms.albedo_firn
        elif self.layertype[0] in ['ice']:
            albedo = eb_prms.albedo_ice

        # Initiate time loop
        time_idx = 0
        surftemp = 0 # initial guess, will be solved iteratively

        # Place to store melt for each month and for each timestep
        monthly_melt = []
        monthly_refreeze = []
        running_melt = 0
        running_refreeze = 0

        # ===== ENTER TIME LOOP =====
        for time in time_dt[0:2]:
            # Initiate the energy balance to unpack climate data
            enbal = eb.energyBalance(climateds,time,bin_idx,dt)

            # Check if snowfall occurred
            if enbal.prec > 0.01 and enbal.tempC < eb_prms.tsnow_threshold:
                # set timestamp
                snow_timestamp = time
                snowfall = enbal.prec
            else:
                snowfall = 0

            if time.hour < 1:
                # any daily happenings go here!!!
                days_since_snowfall = time.day - snow_timestamp.day
                #update albedo
                #albedo = eb_albedo.getAlbedo(BC,days_since_snowfall)

            # ENTER ITERATIVE LOOP FOR SURFACE TEMPERATURE
            surftemp_loop = True
            iteration_count = 0
            while surftemp_loop:
                # Run the surface energy balance
                Qm, SW_penetrating = enbal.surfaceEB(surftemp,self.layerz,self.layertype,days_since_snowfall,albedo)

                # Recalculate temperature profile
                K = 0.0138e-1 - 1.01e-3*self.pprofile + 3.233e-6*np.square(self.pprofile)
                if eb_prms.method_heateq in ['Crank-Nicholson']:
                    # ******* This does not work.
                    C_Crank = K*dt/(2*self.pprofile[0]*eb_prms.Cp_ice*self.layerh[0]**2)
                    if time_idx < 1:
                        T_past, Ts_past = [0,0]
                    new_T = self.CrankNicholson(time_idx,C_Crank,self.Tprofile,surftemp,T_past,Ts_past)
                elif eb_prms.method_heateq in ['ugly']:
                    # Neither does this oop *****
                    new_T = self.solveHeat(K,surftemp,dt,Qm)
                    print(new_T)
                else:
                    assert 0==1, 'Only Crank-Nicholson method is currently supported for T profiles'

                # ENTER LAYER LOOP
                leftovers = 0
                for layer in range(self.nlayers):
                    m_layer = self.layerh[layer]*self.pprofile[layer]
                    if new_T[layer] > 0:
                        # calculate amount of melt
                        dm_melt = new_T[layer]*m_layer*eb_prms.Cp_ice/eb_prms.Lh_rf

                        # add melt to running sum
                        running_melt += dm_melt

                        # add melt to layer water content
                        self.wprofile[layer] += dm_melt + leftovers

                        # check if meltwater exceeds the irreducible water content of the snow
                        irreducible_water = 0.0143*np.exp(3.3*(1-self.pprofile[layer]/eb_prms.density_ice))
                        if self.wprofile[layer] > irreducible_water:
                            # set water content to irreducible water content and add the difference to leftovers
                            leftovers = irreducible_water - self.wprofile[layer]
                            self.wprofile[layer] = irreducible_water
                        else: #if not overflowing, leftovers should be set back to 0
                            leftovers = 0
                        self.Tprofile[layer] = 0
                        # update self.pprofile from self.layerh, or vice versa?

                        # update surface BC concentration
                        # if layer == 0:
                        #     BC[0] = BC[0] + 
                    if new_T[layer] < 0 and self.wprofile[layer] > 0:
                        # calculate potential for refreeze 
                        E_temperature = np.abs(new_T[layer])*m_layer*eb_prms.Cp_ice
                        E_water = self.wprofile[layer]*eb_prms.Lh_rf
                        E_pore = (self.pprofile[layer]-eb_prms.density_ice)*self.layerh[layer]*eb_prms.Lh_rf

                        # calculate amount of refreeze 
                        dm_ref = np.min([E_temperature,E_water,E_pore])/eb_prms.Lh_rf

                        # add refreeze to running sum
                        running_refreeze += dm_ref

                        # add refreeze to layer ice mass
                        m_layer += dm_ref

                        # update the density and temperature of the layer
                        self.pprofile[layer] = m_layer/self.layerh[layer]
                        self.Tprofile[layer] = -(E_temperature-dm_ref*eb_prms.Lh_rf/eb_prms.Cp_ice/m_layer)
                
                surftemp_new = np.interp(0,self.layerz[0:2],self.Tprofile[0:2])
                iteration_count += 1
                if abs(surftemp_new - surftemp) < 0.25:
                    surftemp_loop = True
                else:
                    surftemp = surftemp_new
                if iteration_count > 3:
                    print('surftemp loop failing to converge!')
                    break

            if time.is_month_start and time.hour < 1:
                # any monthly happenings go here!!
                monthly_melt.append(running_melt)
                monthly_refreeze.append(running_refreeze)
                running_melt = 0
                running_refreeze = 0

            if time.is_month_start and time.month == 10:
                print('Update glacier geometry!')

            time_idx +=1
            T_past = self.Tprofile
            Ts_past = surftemp_new

    def getLayers(self,sfi_h0):
        """
        Calculates layer depths based on an exponential growth function with prescribed rate of growth and 
        initial layer depth (from pygem_input). 

        Parameters
        ----------
        sfi_h0 : np.ndarray
            Initial thicknesses of the snow, firn and ice layers [m]
        """
        dz_toplayer = eb_prms.dz_toplayer
        layer_growth = eb_prms.layer_growth

        #Initialize variables to get looped
        layerh = []
        layertype = []

        # Case where there is snow
        if sfi_h0[0] > 0:
            total_depth = 0
            layeridx = 0

            # Loop and make exponentially growing layers
            while total_depth < sfi_h0[0]:
                layerh.append(dz_toplayer * np.exp(layeridx*layer_growth))
                layertype.append('snow')
                layeridx += 1
                total_depth = np.sum(layerh)
            layerh[-1] = layerh[-1] - (total_depth-sfi_h0[0])
        
            # Add firn layers
            n_firn_layers = round(sfi_h0[1],0)
            layerh.extend([sfi_h0[1]/n_firn_layers]*n_firn_layers)
            layertype.extend(['firn']*n_firn_layers)
        # Case where there is no snow, but there is firn*****
        elif sfi_h0[1] > 0:
            # Add firn layers that are approximately 0.2m deep
            n_firn_layers = sfi_h0 // 0.2
            layerh.extend([sfi_h0[1]/n_firn_layers]*n_firn_layers)
            layertype.extend(['firn']*n_firn_layers)

        # Add ice bin
        layerh.append(sfi_h0[2])
        layertype.append('ice')

        # Calculate layer depths (mid-points)
        layerz = [np.sum(layerh[:i+1])-(layerh[i]/2) for i in range(len(layerh))]
 
        return np.array(layerh), np.array(layerz), np.array(layertype)
    
    def initProfilesPiecewise(self,layerz,snow_var,varname):
        """
        Based on the DEBAM scheme for temperature and density that assumes linear changes with depth 
        in three piecewise sections.

        Parameters
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

        #solve piecewise functions at each layer depth
        layer_var = np.piecewise(layerz,
                     [layerz <= snow_var[1,0], (layerz <= snow_var[2,0]) & (layerz > snow_var[1,0]),
                      (layerz <= snow_var[3,0]) & (layerz > snow_var[2,0])],
                      [lambda x: slopes[0]*x+intercepts[0],lambda x:slopes[1]*x+intercepts[1],
                       lambda x: slopes[2]*x+intercepts[2]])
        return layer_var
    
    def CrankNicholson(self,i,C,T,Ts,T_past,Ts_past):
        """
        Solves the heat equation using the Crank-Nicholson scheme to recalculate snowpack temperatures.

        Parameters
        ----------
        i : int
            Index for the timestep
        C : np.ndarray
            Crank-Nicholson constant
        T : np.ndarray
            Current version of temperature profile
        Ts : float
            Current version of surface temperature
        T_past : np.ndarray
            Temperature profile of the previous timestep
        Ts_past : float
            Surface temperature of the previous timestep
        """
        a_Crank = np.zeros((self.nlayers))
        b_Crank = np.zeros((self.nlayers))
        c_Crank = np.zeros((self.nlayers))
        d_Crank = np.zeros((self.nlayers))
        A_Crank = np.zeros((self.nlayers))
        S_Crank = np.zeros((self.nlayers))
        T_new = np.zeros((self.nlayers))
        if i < 1:
            # First timestep requires no adjustment, just use the initial conditions
            T_new = self.Tprofile
        else:
            for j in range(0,self.nlayers):
                a_Crank[j] = C
                b_Crank[j] = 2*C+1
                c_Crank[j] = C

                if j == 0:
                    d_Crank[j] = C*Ts + C*Ts_past + (1-2*C)*T_past[j] + C*T_past[j+1]
                elif j < self.nlayers-1:
                    d_Crank[j] = C*T_past[j-1] + (1-2*C)*T_past[j] + C*T_past[j+1]
                else:
                    d_Crank[j] = 2*C*eb_prms.temp_temp + C*T_past[j-1] + (1-2*C)*T_past[j]

                if j == 0:
                    A_Crank[j] = b_Crank[j]
                    S_Crank[j] = d_Crank[j]
                else:
                    A_Crank[j] = b_Crank[j] - a_Crank[j]/A_Crank[j-1] * c_Crank[j-1]
                    S_Crank[j] = d_Crank[j] + a_Crank[j]/A_Crank[j-1] * S_Crank[j-1]
            for j in range(self.nlayers - 1,0,-1):
                if j == self.nlayers-1:
                    T_new[j] = S_Crank[j]/A_Crank[j]
                else:
                    T_new[j] = 1/A_Crank[j] * (S_Crank[j]+c_Crank[j]*T_new[j+1])
        return T_new
    
    def solveHeat(self,K,surftemp,dt,Qm):
        """
        Recalculate temperature profile by brute force method like DEBAM
        """
        conduction = []
        Tprofile_new = []


        for layer in range(self.nlayers):
            if layer == 0:
                conduction.append(K[0]*self.Tprofile[0] - surftemp/self.layerh[0])
            else:
                dzl = 0.5*(self.layerh[layer]+self.layerh[layer-1])
                layerconduct = 0.5/dzl*(K[layer]*self.layerh[layer]+K[layer-1]*self.layerh[layer-1])*(self.Tprofile[layer]-self.Tprofile[layer-1])/dzl
                conduction.append(layerconduct)
        for layer in range(self.nlayers-1):
            if layer == 0:
                dT = dt*2/eb_prms.Cp_ice/(self.pprofile[0]+self.pprofile[1])*(conduction[1]-Qm)/self.layerh[0]
            else:
                dT = dt*2/eb_prms.Cp_ice/(self.pprofile[layer]+self.pprofile[layer+1])*(conduction[layer+1]-conduction[layer])/self.layerh[layer]
            Tprofile_new.append(self.Tprofile[layer] + dT)
        return Tprofile_new
