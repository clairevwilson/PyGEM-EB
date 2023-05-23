import numpy as np
import pandas as pd
import pygem_eb.energybalance as eb
import pygem_eb.input as eb_prms
import pygem_eb.albedo as eb_albedo


class Layers():
    """
    Scheme for the multi-layer snowpack model.
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
        heights,depths,types = self.getLayers(sfi_h0)
        self.nlayers = len(heights)

        # Initialize SNOW layer temperatures based on chosen method and data (snow_temp)
        snow_idx =  np.where(types=='snow')[0]
        snow_layerz = depths[snow_idx] 
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
        pslope = (eb_prms.density_ice - pprofile[-1])/(np.sum(sfi_h0[0:2])-depths[snow_idx[-1]])
        for idx,type in enumerate(types):
            if type not in ['snow']:
                Tprofile = np.append(Tprofile,eb_prms.temp_temp)
            if type in['firn']:
                pprofile = np.append(pprofile,pprofile[snow_idx[-1]] + pslope*(depths[idx]-depths[snow_idx[-1]]))
            elif type in ['ice']:
                pprofile = np.append(pprofile,eb_prms.density_ice)

        # Initialize water content
        if eb_prms.option_initWater in ['zero_w0']:
            wprofile = np.zeros(self.nlayers)
        elif eb_prms.option_initWater in ['initial_w0']:
            assert 1==0, "Only zero water content method is set up"

        # Define dry (solid) mass of each layer
        dry_masses = pprofile*heights
        # Define irreducible water content of each layer and set saturated value
        irrwatercont = 0.0143*np.exp(3.3*(1-pprofile/eb_prms.density_ice))
        saturated = np.where(wprofile == irrwatercont,1,0)

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
        self.heights = heights
        self.depths = depths
        self.types = types
        self.dry_masses = dry_masses
        self.irrwatercont = irrwatercont
        self.saturated = saturated
        self.BC = BC
        self.dust = dust
        return 
    
    def getLayers(self,sfi_h0):
        """
        Initializes layer depths based on an exponential growth function with prescribed rate of growth and 
        initial layer depth (from pygem_input). 

        Parameters
        ----------
        sfi_h0 : np.ndarray
            Initial thicknesses of the snow, firn and ice layers [m]

        Returns
        -------
        layerh : np.ndarray
            Height of the layer [m]
        layerz : np.ndarray
            Depth of the middle of the layer [m]
        layertype : np.ndarray
            Type of layer, 'snow' 'firn' or 'ice'
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
        layerz : np.ndarray
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
    
    def subsurfaceMelt(self,Snet_surf,dt):
        # Where to put this function?? 
        # Fraction of radiation absorbed at the surface depends on surface type
        if self.types[0] in ['snow']:
            frac_absrad = 0.9
        else:
            frac_absrad = 0.8

        # Extinction coefficient depends on layer type
        extinct_coef = np.ones(self.nlayers)*1e8 # ensures unfilled layers have 0 heat
        for idx,type in enumerate(self.types):
            if type in ['snow']:
                extinct_coef[idx] = 17.1
            else:
                extinct_coef[idx] = 2.5
            # Cut off if the flux reaches zero threshold (1e-6)
            if np.exp(-extinct_coef[idx]*self.depths[idx]) < 1e-6:
                extinct_coef[idx] = 1e8
                break
        Snet_pen = Snet_surf*frac_absrad*np.exp(-extinct_coef*self.depths)/dt

        # recalculate layer temperatures, leaving out the surface since it's also forced by other fluxes
        new_Tprofile = self.Tprofile
        new_Tprofile[1:] += Snet_pen[1:]/(self.dry_masses[1:]*eb_prms.Cp_ice)*dt

        # calculate melt from meltLayers function
        layermelt = np.zeros(self.nlayers)
        for layer,new_T in enumerate(new_Tprofile):
            # check if temperature is above 0
            if new_T > 0:
                # calculate melt from the energy that raised the temperature above 0
                melt = (new_T-0)*self.dry_masses[layer]*eb_prms.Cp_ice/eb_prms.Lh_rf
                self.Tprofile[layer] = 0
            else:
                melt = 0
                self.Tprofile[layer] = new_T
            layermelt[layer] = melt
        
        return layermelt
        

    def percolate(self,layermelt,extra_water=0):
        """
        Calculates the liquid water content in each layer by downward percolation and adjusts 
        layer heights.

        Parameters
        ----------
        layermelt: np.ndarray
            Array containing melt amount for each layer
        extra_water : float
            Additional liquid water input (eg. rainfall) [kg/m2 = kg]

        Returns
        -------
        runoff : float
            Runoff that was not absorbed into void space [m3]
        """
        melted_layers = []
        for layer,melt in enumerate(layermelt):
            # check if the layer fully melted
            if melt >= self.dry_masses[layer]:
                melted_layers.append(layer)
                # pass the meltwater to the next layer
                extra_water += melt
            else:
                # remove melt from the dry mass
                self.dry_masses[layer] += -1*melt

                # add melt and extra_water (melt from above) to layer water content
                added_water = melt + extra_water
                self.wprofile[layer] += added_water

                # check if meltwater exceeds the irreducible water content of the layer
                if self.wprofile[layer] >= self.irrwatercont[layer]:
                    # set water content to irr. water content and add the difference to extra_water
                    extra_water = self.wprofile[layer] - self.irrwatercont[layer]
                    self.wprofile[layer] = self.irrwatercont[layer]
                    self.saturated[layer] = 1 # set the layer to saturated
                else: #if not overflowing, extra_water should be set back to 0
                    extra_water = 0
                
                # get the change in layer height due to loss of solid mass
                dh = -melt/self.pprofile[layer]
                self.heights[layer] += dh

        runoff = extra_water
        self.removeLayers(melted_layers)
        return runoff

    def refreeze(self,Tprofile):
        refreeze = 0
        for layer, T in enumerate(Tprofile):
            if T < 0 and self.wprofile[layer] > 0:
                # calculate potential for refreeze 
                E_temperature = np.abs(T)*self.dry_masses[layer]*eb_prms.Cp_ice
                E_water = self.wprofile[layer]*eb_prms.Lh_rf
                E_pore = (self.pprofile[layer]-eb_prms.density_ice)*self.heights[layer]*eb_prms.Lh_rf

                # calculate amount of refreeze 
                dm_ref = np.min([E_temperature,E_water,E_pore])/eb_prms.Lh_rf

                # add refreeze to running sum
                refreeze += dm_ref

                # add refreeze to layer ice mass
                self.dry_masses[layer] += dm_ref
                self.heights[layer] = self.dry_masses[layer]/self.pprofile[layer]
                self.Tprofile[layer] = -(E_temperature-dm_ref*eb_prms.Lh_rf/eb_prms.Cp_ice/self.dry_masses[layer])
        return refreeze
    
    def removeLayers(self,layers_to_remove):
        self.nlayers += -len(layers_to_remove)
        self.pprofile = np.delete(self.pprofile,layers_to_remove)
        self.Tprofile = np.delete(self.Tprofile,layers_to_remove)
        self.wprofile = np.delete(self.wprofile,layers_to_remove)
        self.heights = np.delete(self.heights,layers_to_remove)
        self.types = np.delete(self.types,layers_to_remove)
        self.dry_masses = np.delete(self.dry_masses,layers_to_remove)
        self.irrwatercont = np.delete(self.irrwatercont,layers_to_remove)
        self.saturated = np.delete(self.saturated,layers_to_remove)

        # recalculate layer depths
        self.depths = np.array([np.sum(self.heights[:i+1])-(self.heights[i]/2) for i in range(self.nlayers)])
        return
    
    def solveHeatEq(self,dt):
        # COPIED DIRECTLY FROM COSIPY
        # number of layers
        nl = self.nlayers

        # Define index arrays 
        k   = np.arange(1,nl-1) # center points
        kl  = np.arange(2,nl)   # lower points
        ku  = np.arange(0,nl-2) # upper points
        
        # Get thermal diffusivity [m2 s-1]
        K = 2.2*np.power(self.pprofile/eb_prms.density_ice,1.88)
        
        # Get snow layer heights    
        hlayers = self.heights

        # Get grid spacing
        diff = ((hlayers[0:nl-1]/2.0)+(hlayers[1:nl]/2.0))
        hk = diff[0:nl-2]  # between z-1 and z
        hk1 = diff[1:nl-1] # between z and z+1
        
        # Get temperature array from grid|
        T = self.Tprofile
        Tnew = T.copy()
        
        Kl = (K[1:nl-1]+K[2:nl])/2.0
        Ku = (K[0:nl-2]+K[1:nl-1])/2.0
        
        stab_t = 0.0
        c_stab = 0.8
        dt_stab  = c_stab * (min([min(diff[0:nl-2]**2/(2*Ku)),min(diff[1:nl-1]**2/(2*Kl))]))
        
        n_iters = 0
        dt = 100 # SHOULD BE 3600s
        while stab_t < dt:
            dt_use = np.minimum(dt_stab, dt-stab_t)
            stab_t = stab_t + dt_use

            # Update the temperatures
            Tnew[k] += ((Kl*dt_use*(T[kl]-T[k])/(hk1)) - (Ku*dt_use*(T[k]-T[ku])/(hk))) / (0.5*(hk+hk1))
            T = Tnew.copy()
            n_iters += 1
        # print(n_iters)
        return T
