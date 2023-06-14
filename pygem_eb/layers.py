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
        self.initial_heights = heights.copy()
        self.types = types
        self.heights = heights
        self.depths = depths

        # Initialize SNOW layer temperatures based on chosen method and data (snow_temp)
        snow_idx =  np.where(types=='snow')[0]
        self.snow_idx0 = snow_idx.copy()
        snow_layerz = depths[snow_idx] 
        if eb_prms.option_initTemp in ['piecewise']:
            Tprofile = self.initProfilesPiecewise(snow_layerz,snow_temp,'temp')
        elif eb_prms.option_initTemp in ['interp']:
            Tprofile = np.interp(snow_layerz,snow_temp[0,:],snow_temp[1,:])
        else:
            assert 1==0, "Choose between 'piecewise' and 'interp' methods for temp initialization"

        # Initialize SNOW layer density based on chosen method and data (snow_density)
        if eb_prms.option_initDensity in ['piecewise']:
            density = self.initProfilesPiecewise(snow_layerz,snow_density,'density')
        elif eb_prms.option_initDensity in ['interp']:
            density = np.interp(snow_layerz,snow_density[0,:],snow_density[1,:])
        else:
            assert 1==0, "Choose between 'piecewise' and 'interp' methods for density initialization"

        # Initialize FIRN AND ICE temperature and density
        # Calculate slope that linearly increases density from the bottom snow bin to the top of the ice layer
        pslope = (eb_prms.density_ice - density[-1])/(np.sum(sfi_h0[0:2])-depths[snow_idx[-1]])
        for idx,type in enumerate(types):
            if type not in ['snow']:
                Tprofile = np.append(Tprofile,eb_prms.temp_temp)
            if type in ['firn']:
                density = np.append(density,density[snow_idx[-1]] + pslope*(depths[idx]-depths[snow_idx[-1]]))
            elif type in ['ice']:
                density = np.append(density,eb_prms.density_ice)

        # Initialize water content [kg m-2]
        if eb_prms.option_initWater in ['zero_w0']:
            watercont = np.zeros(self.nlayers)
        elif eb_prms.option_initWater in ['initial_w0']:
            assert 1==0, "Only zero water content method is set up"

        # Define dry (solid) and wet (total) mass of each layer [kg m-2]
        dry_mass = density*heights
        wet_mass = dry_mass + watercont
        # Define irreducible water content of each layer and set saturated value
        irrwatercont = self.getIrrWaterCont(density)
        # saturated = np.where(watercont == irrwatercont,1,0)

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
        self.density = density
        self.watercont = watercont
        self.heights = heights
        self.depths = depths
        self.types = types
        self.dry_mass = dry_mass
        self.wet_mass = wet_mass
        self.irrwatercont = irrwatercont
        self.BC = BC
        self.dust = dust

        print('Layers initialized')
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
            if sfi_h0[1] > 0:
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
    
    def addLayers(self,layers_to_add):
        """
        Adds layers to layers class.

        Parameters
        ----------
        layers_to_add : pd.Dataframe
            Contains temperature 'T', water content 'w', height 'h', type 't', dry mass 'drym'
        """
        self.nlayers += len(layers_to_add.loc['T'].values)
        self.Tprofile = np.append(layers_to_add.loc['T'].values,self.Tprofile)
        self.watercont = np.append(layers_to_add.loc['w'].values,self.watercont)
        self.heights = np.append(layers_to_add.loc['h'].values,self.heights)
        self.types = np.append(layers_to_add.loc['t'].values,self.types)
        self.dry_mass = np.append(layers_to_add.loc['drym'].values,self.dry_mass)
        self.updateLayerProperties()
        return
    
    def removeLayer(self,layer_to_remove):
        """
        Removes layer from layers class.

        Parameters
        ----------
        layer_to_remove : int
            index of layer to remove
        """
        eb_prms.melt_counter += 1
        self.nlayers -= 1
        self.Tprofile = np.delete(self.Tprofile,layer_to_remove)
        self.watercont = np.delete(self.watercont,layer_to_remove)
        self.heights = np.delete(self.heights,layer_to_remove)
        self.types = np.delete(self.types,layer_to_remove)
        self.dry_mass = np.delete(self.dry_mass,layer_to_remove)
        self.updateLayerProperties()
        return
    
    def splitLayer(self,layer_to_split):
        eb_prms.split_counter += 1
        l = layer_to_split
        self.nlayers += 1
        self.Tprofile = np.insert(self.Tprofile,l,self.Tprofile[l])
        self.types = np.insert(self.types,l,self.types[l])
        self.watercont[l] = self.watercont[l].copy()/2
        self.watercont = np.insert(self.watercont,l,self.watercont[l])
        self.heights[l] = self.heights[l].copy()/2
        self.heights = np.insert(self.heights,l,self.heights[l]/2)
        self.dry_mass[l] = self.dry_mass[l].copy()/2
        self.dry_mass = np.insert(self.dry_mass,l,self.dry_mass[l]/2)
        self.updateLayerProperties()
        return

    def mergeLayers(self,layer_to_merge):
        l = layer_to_merge
        if self.types[l+1] != 'ice':
            eb_prms.merge_counter += 1
            self.density[l+1] = np.sum(self.density[l:l+2]*self.dry_mass[l:l+2]/np.sum(self.dry_mass[l:l+2]))
            self.watercont[l+1] = np.sum(self.watercont[l:l+2]) # *****can cause water to overflow irrwatercont
            self.Tprofile[l+1] = np.mean(self.Tprofile[l:l+2])
            self.heights[l+1] = np.sum(self.heights[l:l+2])
            self.dry_mass[l+1] = np.sum(self.dry_mass[l:l+2])
            self.removeLayer(l)
        return
    
    def updateLayers(self):
        for layer,dz in enumerate(self.heights):
            try:
                if self.types[layer] == 'snow':
                    if dz < self.initial_heights[0]*0.5:
                        self.mergeLayers(layer) # merges layer with next layer down
                    elif dz > eb_prms.max_dz:
                        self.splitLayer(layer)
            except: # override error when layers are taken out during the loop and index gets too high
                pass
        return
    
    def updateLayerProperties(self,do=['depth','wetmass','density','irrwater']):
        """
        Recalculates nlayers, depths, wet mass, density, and irreducible water
        content from DRY density. Can specify to only update certain properties.
        """
        self.nlayers = len(self.heights)
        if 'depth' in do:
            self.depths = np.array([np.sum(self.heights[:i+1])-(self.heights[i]/2) for i in range(self.nlayers)])
        if 'wetmass' in do:
            self.wet_mass = self.watercont + self.dry_mass
        if 'density' in do:
            self.density = self.dry_mass / self.heights
        if 'irrwater' in do:
            self.irrwatercont = self.getIrrWaterCont(self.density)
        return
    
    def updateLayerTypes(self):
        for layer in range(self.addLayers):
            if self.density[layer] < eb_prms.density_firn:
                self.types[layer] = 'snow'
            elif self.density[layer] < eb_prms.density_ice:
                self.types[layer] = 'firn'
            else:
                self.types[layer] = 'ice'
        return
    
    def addSnow(self,snowfall,airtemp,new_density=eb_prms.density_fresh_snow):
        """
        snowfall = fresh snow MASS in kg / m2
        """
        if self.types[0] in 'ice':
            new_layer = pd.DataFrame([airtemp,0,snowfall/new_density,'snow',snowfall],index=['T','w','h','t','drym'])
            self.addLayers(new_layer)
        else:
            new_layermass = self.dry_mass[0] + snowfall
            self.density[0] = (self.density[0]*self.dry_mass[0] + new_density*snowfall)/(new_layermass)
            self.dry_mass[0] = new_layermass
            self.heights[0] += snowfall/new_density
            if self.heights[0] > (eb_prms.dz_toplayer * 1.5) and (self.nlayers+1)<eb_prms.max_nlayers:
                self.splitLayer(0)
        self.updateLayerProperties()
        return
    
    def getIrrWaterCont(self,density=[0]):
        if sum(density) == 0: # default condition for function
            density = self.dry_mass / self.heights
        density = density.astype(float)
        density_ice = eb_prms.density_ice
        ice_idx = np.where(self.types=='ice')[0][0]
        porosity = (density_ice - density[:ice_idx])/density_ice
        irrwatercont = 0.0143*np.exp(3.3*porosity)
        irrwatersat = irrwatercont*density[:ice_idx]/porosity # kg m-3, mass of liquid over pore volume
        irrwatercont = irrwatersat*self.heights[:ice_idx] # kg m-2, mass of liquid in a layer
        
        irrwatercont = np.append(irrwatercont,0) # ice layer cannot hold water
        return irrwatercont