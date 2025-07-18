"""
Layers class for PEBSI

Tracks layer properties and contains utility
functions to maintain layer arrays.

@author: clairevwilson
"""
# Built-in libraries
import warnings, sys
warnings.simplefilter('error', RuntimeWarning)
# External libraries
import numpy as np
import pandas as pd
import xarray as xr
# Local libraries
import pebsi.input as prms

class Layers():
    """
    Layer scheme for the 1D snowpack model.

    All mass terms are stored in kg m-2.
    """
    def __init__(self,climate,args):
        """
        Initialize the layer properties (temperature, 
        density, water content, LAPs, etc.)

        Parameters
        ==========
        climate
            Class object from pebsi.climate
        args : command line arguments
        """
        # INPUTS
        self.climate = climate 
        self.args = args

        # load in initial depths of snow, firn and ice in m
        dz_snow = args.initial_snow_depth
        dz_firn = args.initial_firn_depth
        dz_ice = prms.initial_ice_depth

        # calculate the layer depths based on initial snow, firn and ice depths
        lheight,ldepth,ltype,nlayers = self.make_layers(dz_snow,dz_firn,dz_ice)
        self.nlayers = nlayers              # NUMBER OF LAYERS
        self.ltype = ltype                  # LAYER TYPE (snow, firn, or ice)
        self.lheight = lheight              # LAYER HEIGHT (dz) [m]
        self.ldepth = ldepth                # LAYER DEPTH (midlayer) [m]

        # initialize layer temperature, density, water content
        ltemp,ldensity,lwater,lgrainsize = self.initialize_layers(dz_snow,dz_firn)
        self.ltemp = ltemp                  # LAYER TEMPERATURE [C]
        self.ldensity = ldensity            # LAYER DENSITY [kg m-3]
        self.lice = ldensity*lheight        # LAYER ICE (SOLID) MASS [kg m-2]
        self.lwater = lwater                # LAYER WATER (LIQUID) MASS [kg m-2]
        self.lgrainsize = lgrainsize        # LAYER GRAIN SIZE [um]

        # initialize LAPs (black carbon, organic carbon, and dust)
        if prms.switch_LAPs == 1:
            lBC,lOC,ldust = self.initialize_LAPs()
        else:
            lBC = np.zeros(self.nlayers)
            lOC = np.zeros(self.nlayers)
            ldust = np.zeros(self.nlayers)
        self.lBC = lBC                      # LAYER BLACK CARBON MASS [kg m-2]
        self.lOC = lOC                      # LAYER ORGANIC CARBON MASS [kg m-2]
        self.ldust = ldust                  # LAYER DUST MASS [kg m-2]
       
        # additional layer properties
        self.update_layer_props()
        self.drefreeze = np.zeros_like(self.ltemp)   # LAYER MASS OF REFREEZE [kg m-2]
        self.lnewsnow = np.zeros_like(self.ltemp)    # LAYER MASS OF NEW SNOW [kg m-2]
        self.cumrefreeze = np.zeros_like(self.ltemp) # TRACK CUM. REFREEZE [kg m-2]

        # initialize bucket for 'delayed snow' and running max snow mass
        self.delayed_snow = 0
        self.max_snow = np.sum(self.lice[self.snow_idx])
        
        if args.debug:
            print(f'~ {self.nlayers} layers initialized ~')
        return 
    
    def make_layers(self,snow_height,firn_height,ice_height):
        """
        Initializes layer depths based on an exponential
        growth function with prescribed rate of growth 
        and initial layer height. 

        Parameters
        ==========
        snow_height : float
        firn_height : float
        ice_height : float
            Initial depth of snow, firn, and ice [m]

        Returns
        -------
        lheight : np.ndarray
            Height of the layer [m]
        ldepth : np.ndarray
            Depth of the middle of the layer [m]
        ltype : np.ndarray
            Layers types (snow, firn, ice)
        """
        # CONSTANTS
        DZ_TOP = prms.dz_toplayer
        LAYER_GROWTH = prms.layer_growth

        # initialize variables to get looped
        lheight = []
        ltype = []
        current_depth = 0
        layer = 0

        # make exponentially growing snow layers
        while current_depth < snow_height:
            lheight.append(DZ_TOP * np.exp(layer*LAYER_GROWTH))
            ltype.append('snow')
            layer += 1
            current_depth = np.sum(lheight)
        if snow_height > 0:
            lheight[-1] = lheight[-1] - (current_depth-snow_height)
            # can end up with a small bottom layer: merge it
            if lheight[-1] < lheight[-2] / 2:
                lheight[-2] += lheight[-1]
                lheight = lheight[:-1]
                ltype = ltype[:-1]
                layer -= 1
    
        # add firn layers
        if firn_height > 0.75:
            n_firn_layers = int(round(firn_height,0))
            lheight.extend([firn_height/n_firn_layers]*n_firn_layers)
            ltype.extend(['firn']*n_firn_layers)
        elif firn_height > 0:
            lheight.extend([firn_height])
            ltype.extend(['firn'])

        # add ice layers
        current_depth = 0
        while current_depth < ice_height:
            lheight.append(DZ_TOP * np.exp(layer*LAYER_GROWTH))
            ltype.append('ice')
            layer += 1
            ice_idx = np.where(np.array(ltype)=='ice')[0]
            current_depth = np.sum(np.array(lheight)[ice_idx])
        lheight[-1] = lheight[-1] - (current_depth-ice_height)
        
        # get depth of layers (distance from surface to midpoint of layer) [m]
        nlayers = len(lheight)
        if 'layers' in prms.store_vars:
            assert nlayers <= prms.max_nlayers, f'Need >= {nlayers} in prms.max_nlayers'
        ldepth = [np.sum(lheight[:i+1])-(lheight[i]/2) for i in range(nlayers)]

        # make into arrays
        lheight = np.array(lheight)
        ldepth = np.array(ldepth)
        ltype = np.array(ltype)

        # assign indices
        self.snow_idx = np.where(ltype=='snow')[0]
        self.firn_idx = np.where(ltype=='firn')[0]
        self.ice_idx = np.where(ltype=='ice')[0]
        return np.array(lheight), np.array(ldepth), np.array(ltype), nlayers

    def initialize_layers(self,snow_height,firn_height):
        """
        Initializes the layer temperature, density, 
        water content and grain size.

        Parameters:
        ==========
        snow_height : float
        firn_height : float
            Initial depth of snow and firn [m]
        
        Returns:
        --------
        ltemp, ldensity, lwater, lgrainsize : np.ndarray
            Arrays containing layer temperature [C], 
            density [kg m-3], water content [kg m-2],
            and grain size [um]
        """
        snow_idx = self.snow_idx
        firn_idx = self.firn_idx
        ice_idx = self.ice_idx

        # read in depth profiles
        temp_data = pd.read_csv(self.args.initial_temp_fp)
        density_data = pd.read_csv(self.args.initial_density_fp)
        grainsize_data = pd.read_csv(self.args.initial_grains_fp)

        # TEMPERATURE [C]
        if prms.initialize_temp == 'interpolate':
            ltemp = np.interp(self.ldepth,temp_data['depth'],temp_data['temp'])
        elif prms.initialize_temp == 'ripe':
            ltemp = np.ones(self.nlayers)*0
        else:
            print('Choose between ripe and interpolate in initialize_temp')
            self.exit()
        
        # GRAIN SIZE [um]
        lgrainsize = np.interp(self.ldepth,grainsize_data['depth'],
                               grainsize_data['grainsize'])
        lgrainsize[self.ltype == 'firn'] = prms.firn_grainsize
        lgrainsize[self.ltype == 'ice'] = prms.ice_grainsize

        # DENSITY [kg m-3]
        if prms.initialize_density == 'interpolate':
            # SNOW layers initialized by interpolation
            ldensity = np.interp(self.ldepth[snow_idx],
                                density_data['depth'],density_data['density'])
            if len(firn_idx) > 0:
                # calculate firn density slope from snow --> ice
                if snow_height > 0 and firn_height > 0:
                    pslope = (prms.density_ice - ldensity[-1]) / (
                        self.ldepth[ice_idx[0]]-self.ldepth[snow_idx[-1]])
                # no snow: set boundary tp constant density_firn
                elif firn_height > 0:
                    pslope = (prms.density_ice - prms.density_firn)/(firn_height)
        elif prms.initialize_density == 'constant':
            ldensity = np.ones_like(snow_idx) * prms.density_snow
        else:
            print('Choose between constant and interpolate in initialize_density')
            self.exit()
            
        # append firn and ice layer densities
        for (type,depth) in zip(self.ltype,self.ldepth):
            if type in ['firn']: 
                ldensity = np.append(ldensity,
                    ldensity[snow_idx[-1]] + pslope*(depth-self.ldepth[snow_idx[-1]]))
            elif type in ['ice']:
                ldensity = np.append(ldensity,prms.density_ice)

        # WATER CONTENT [kg m-2]
        if prms.initialize_water == 'dry':
            lwater = np.zeros(self.nlayers)
        elif prms.initialize_water == 'saturated':
            porosity = 1 - ldensity / prms.density_ice
            lwater = porosity * prms.Sr * self.lheight * prms.density_water
        else:
            print('Choose between dry and saturated in initialize_water')
            self.exit()

        return ltemp,ldensity,lwater,lgrainsize
    
    def initialize_LAPs(self):
        """
        Initializes light-absorbing particle content
        of the snow and firn layers.
        """
        # CONSTANTS
        BC_FRESH = prms.BC_freshsnow
        OC_FRESH = prms.OC_freshsnow
        DUST_FRESH = prms.dust_freshsnow

        # INPUTS
        n = self.nlayers
        lheight = self.lheight
        ldepth = self.ldepth

        if prms.initialize_LAPs in ['clean']:
            # snowpack is clean; initialize as constant values
            lBC = np.ones(n)*BC_FRESH*lheight
            lOC = np.ones(n)*OC_FRESH*lheight
            ldust = np.ones(n)*DUST_FRESH*lheight 
        elif prms.initialize_LAPs in ['interpolate']:
            # read in LAP data
            lap_data = pd.read_csv(self.args.initial_LAP_fp,index_col=0)

            # add boundaries for interpolation
            lap_data.loc[0,'BC'] = BC_FRESH
            lap_data.loc[0,'OC'] = OC_FRESH
            lap_data.loc[0,'dust'] = DUST_FRESH
            lap_data.loc[100,'BC'] = BC_FRESH
            lap_data.loc[100,'OC'] = OC_FRESH
            lap_data.loc[100,'dust'] = DUST_FRESH
            lap_data = lap_data.sort_index()

            # interpolate concentration by depth
            data_depth = lap_data.index.to_numpy()
            cBC = np.interp(ldepth,data_depth,lap_data['BC'].values.flatten())
            cOC = np.interp(ldepth,data_depth,lap_data['OC'].values.flatten())
            cdust = np.interp(ldepth,data_depth,lap_data['dust'].values.flatten())

            # calculate mass from concentration
            lBC = cBC * lheight
            lOC = cOC * lheight
            ldust = cdust * lheight
        else:
            print('Choose between clean and interpolate in initialize_LAPs')
            self.exit()
        lBC[self.ice_idx] = 0
        lOC[self.ice_idx] = 0
        ldust[self.ice_idx] = 0
        return lBC, lOC, ldust
    
    # ========= UTILITY FUNCTIONS ==========
    def add_layers(self,layers_to_add):
        """
        Adds layers to layers class.

        Parameters
        ==========
        layers_to_add : pd.Dataframe
            Contains temperature 'T', water mass 'w', 
            solid mass 'm', height 'h', type 't', 
            mass of new snow 'new', grain size 'g', 
            and impurities 'BC','OC' and 'dust'
        """
        self.nlayers += len(layers_to_add.loc['T'].values)
        self.ltemp = np.append(layers_to_add.loc['T'].values,self.ltemp).astype(float)
        self.lwater = np.append(layers_to_add.loc['w'].values,self.lwater).astype(float)
        self.lheight = np.append(layers_to_add.loc['h'].values,self.lheight).astype(float)
        self.ltype = np.append(layers_to_add.loc['t'].values,self.ltype)
        self.lice = np.append(layers_to_add.loc['m'].values,self.lice).astype(float)
        self.lnewsnow = np.append(layers_to_add.loc['new'].values,self.lnewsnow).astype(float)
        self.lgrainsize = np.append(layers_to_add.loc['g'].values,self.lgrainsize).astype(float)
        new_layer_BC = layers_to_add.loc['BC'].values.astype(float)*self.lheight[0]
        self.lBC = np.append(new_layer_BC,self.lBC)
        new_layer_OC = layers_to_add.loc['OC'].values.astype(float)*self.lheight[0]
        self.lOC = np.append(new_layer_OC,self.lOC)
        new_layer_dust = layers_to_add.loc['dust'].values.astype(float)*self.lheight[0]
        self.ldust = np.append(new_layer_dust,self.ldust)
        # new layers start with 0 refreeze
        self.drefreeze = np.append(0,self.drefreeze) 
        self.cumrefreeze = np.append(0,self.cumrefreeze)
        self.update_layer_props()
        return
    
    def remove_layer(self,layer_to_remove):
        """
        Removes a single layer from layers class.

        Parameters
        ==========
        layer_to_remove : int
            index of layer to remove
        """
        self.nlayers -= 1
        self.ltemp = np.delete(self.ltemp,layer_to_remove)
        self.lwater = np.delete(self.lwater,layer_to_remove)
        self.lheight = np.delete(self.lheight,layer_to_remove)
        self.ltype = np.delete(self.ltype,layer_to_remove)
        self.lice = np.delete(self.lice,layer_to_remove)
        self.drefreeze = np.delete(self.drefreeze,layer_to_remove)
        self.cumrefreeze = np.delete(self.cumrefreeze,layer_to_remove)
        self.lnewsnow = np.delete(self.lnewsnow,layer_to_remove)
        self.lgrainsize = np.delete(self.lgrainsize,layer_to_remove)
        self.lBC = np.delete(self.lBC,layer_to_remove)
        self.lOC = np.delete(self.lOC,layer_to_remove)
        self.ldust = np.delete(self.ldust,layer_to_remove)
        self.update_layer_props()
        return
    
    def split_layer(self,layer_to_split):
        """
        Splits a single layer into two layers. Extensive
        properties are halved and intensive properties 
        are maintained.

        Parameters
        ==========
        layer_to_split : int
            Index of the layer to split
        """
        if (self.nlayers+1) > prms.max_nlayers and 'layers' in prms.store_vars:
            print(f'Need bigger max_nlayers: currently have {self.nlayers+1} layers')
            self.exit()
        l = layer_to_split
        self.nlayers += 1
        self.ltemp = np.insert(self.ltemp,l,self.ltemp[l])
        self.ltype = np.insert(self.ltype,l,self.ltype[l])
        self.lgrainsize = np.insert(self.lgrainsize,l,self.lgrainsize[l])
        self.lwater[l] = self.lwater[l]/2
        self.lwater = np.insert(self.lwater,l,self.lwater[l])
        self.lheight[l] = self.lheight[l]/2
        self.lheight = np.insert(self.lheight,l,self.lheight[l])
        self.lice[l] = self.lice[l]/2
        self.lice = np.insert(self.lice,l,self.lice[l])
        self.drefreeze[l] = self.drefreeze[l]/2
        self.drefreeze = np.insert(self.drefreeze,l,self.drefreeze[l])
        self.cumrefreeze[l] = self.cumrefreeze[l]/2
        self.cumrefreeze = np.insert(self.cumrefreeze,l,self.cumrefreeze[l])
        self.lnewsnow[l] = self.lnewsnow[l]/2
        self.lnewsnow = np.insert(self.lnewsnow,l,self.lnewsnow[l])
        self.lBC[l] = self.lBC[l]/2
        self.lBC = np.insert(self.lBC,l,self.lBC[l])
        self.lOC[l] = self.lOC[l]/2
        self.lOC = np.insert(self.lOC,l,self.lOC[l])
        self.ldust[l] = self.ldust[l]/2
        self.ldust = np.insert(self.ldust,l,self.ldust[l])
        self.update_layer_props()
        return

    def merge_layers(self,layer_to_merge):
        """
        Merges two layers into one. Extensive properties
        are added and intensive properties are averaged.

        Parameters
        ==========
        layer_to_merge : int
            Index of the layer to merge with the layer below
        """
        l = layer_to_merge
        self.ldensity[l+1] = np.sum(self.ldensity[l:l+2]*self.lice[l:l+2])/np.sum(self.lice[l:l+2])
        self.lwater[l+1] = np.sum(self.lwater[l:l+2])
        self.ltemp[l+1] = np.mean(self.ltemp[l:l+2])
        self.lheight[l+1] = np.sum(self.lheight[l:l+2])
        self.lice[l+1] = np.sum(self.lice[l:l+2])
        self.drefreeze[l+1] = np.sum(self.drefreeze[l:l+2])
        self.cumrefreeze[l+1] = np.sum(self.cumrefreeze[l:l+2])
        self.lnewsnow[l+1] = np.sum(self.lnewsnow[l:l+2])
        self.lgrainsize[l+1] = np.mean(self.lgrainsize[l:l+2])
        self.lBC[l+1] = np.sum(self.lBC[l:l+2])
        self.lOC[l+1] = np.sum(self.lOC[l:l+2])
        self.ldust[l+1] = np.sum(self.ldust[l:l+2])
        self.remove_layer(l)
        return
    
    def check_layers(self,out):
        """
        Checks the layer heights against the initial sizes.
        
        If layers have become too small (less than half their
        original size), they are merged with the layer below.
        
        If layers have become too large (more than double their
        original size), they are split into two layers.

        Parameters
        ==========
        out : str
            Output filename to raise in case of an error.
        """
        # define initial mass for conservation check
        initial_mass = np.sum(self.lice + self.lwater)
 
        # layer heights
        if self.ltype[0] in ['snow','firn']:
            DZ0 = prms.dz_toplayer
        else:
            DZ0 = 0.3
        layer = 0
        min_heights = lambda i: DZ0 * np.exp((i-1)*prms.layer_growth)/2
        max_heights = lambda i: DZ0 * np.exp((i-1)*prms.layer_growth)*2

        # loop through layers 
        while layer < self.nlayers:
            layer_split = False
            dz = self.lheight[layer]
            if self.ltype[layer] in ['snow']:
                if dz < min_heights(layer) and self.ltype[layer]==self.ltype[layer+1]:
                    # layer too small. Merge if it is the same type as the layer underneath
                    self.merge_layers(layer)
                elif dz > max_heights(layer):
                    # layer too big. Split into two equal size layers
                    self.split_layer(layer)
                    layer_split = True
            if self.ltype[layer] in ['ice']:
                if dz < min_heights(layer) and layer < self.nlayers - 1:
                    # layer too small. Merge if it is not the bottom layer
                    self.merge_layers(layer)
                elif dz > max_heights(layer):
                    # layer too big. Split into two equal size layers
                    self.split_layer(layer)
                    layer_split = True
            # firn layers are not checked
            if not layer_split:
                layer += 1

        # CHECK MASS CONSERVATION
        change = np.sum(self.lice + self.lwater) - initial_mass
        assert np.abs(change) < prms.mb_threshold, f'check_layers failed mass conservation in {out}'
        return
    
    def update_layer_props(self,do=['depth','density']):
        """
        Recalculates nlayers, depths, and density. 
        Can specify to only update certain properties.

        Parameters
        ==========
        do : list-like
            List of any combination of depth, density and irrwater to be updated
        """
        self.nlayers = len(self.lheight)
        self.snow_idx =  np.where(self.ltype=='snow')[0]
        self.firn_idx =  np.where(self.ltype=='firn')[0]
        self.ice_idx =  np.where(self.ltype=='ice')[0]
        
        lh = self.lheight.copy()
        if 'depth' in do:
            self.ldepth = np.array([np.sum(lh[:i+1])-(lh[i]/2) for i in range(self.nlayers)])
        if 'density' in do:
            self.ldensity = self.lice / self.lheight
        return
    
    def update_layer_types(self):
        """
        Checks if new ice layers have been created by 
        densification of firn.
        """
        # CONSTANTS
        DENSITY_ICE = prms.density_ice

        layer = 0
        while layer < self.nlayers:
            # new ice
            if self.ldensity[layer] >= DENSITY_ICE and self.ltype[layer] == 'firn':
                self.ltype[layer] = 'ice'
                self.ldensity[layer] = DENSITY_ICE
                # merge into ice below
                if self.ltype[layer+1] in ['ice']:
                    self.merge_layers(layer)

            else:
                layer += 1

        # bound density of superimposed ice (fake snow)
        self.ldensity[self.snow_idx][self.ldensity[self.snow_idx] > DENSITY_ICE] = DENSITY_ICE
        return
    
    def add_snow(self,snowfall,enbal,surface,timestamp):
        """
        Adds snowfall to the layers. If the existing top 
        layer has a large enough difference in density 
        (eg. firn or ice), the fresh snow is a new layer,
        otherwise it is merged with the top snow layer.
        
        Parameters
        ==========
        snowfall : float
            Fresh snow mass in kg / m2
        enbal
            Class object from pebsi.energybalance
        surface
            Class object from pebsi.surface
        timestamp : pd.Datetime
            Current timestep
        """
        snowfall += self.delayed_snow
        if snowfall == 0.:
            return 0
        
        # define initial mass for conservation check
        initial_mass = np.sum(self.lice + self.lwater)

        if self.args.switch_snow == 0:
            # snow falls with the same properties as the current top layer
            new_density = self.ldensity[0]
            new_height = snowfall/new_density
            new_grainsize = self.lgrainsize[0]
            new_BC = self.lBC[0]/self.lheight[0]*new_height
            new_OC = self.lOC[0]/self.lheight[0]*new_height
            new_dust = self.ldust[0]/self.lheight[0]*new_height
            new_snow = 0
        elif self.args.switch_snow == 1:
            if prms.constant_snowfall_density:
                new_density = prms.constant_snowfall_density
            else:
                # CROCUS formulation of density (Vionnet et al. 2012)
                new_density = max(109+6*(enbal.tempC-0.)+26*enbal.wind**0.5,50)
            
            if prms.constant_freshgrainsize:
                new_grainsize = prms.constant_freshgrainsize
            else:
                # CLM formulation of grain size (CLM5.0 Documentation)
                airtemp = enbal.tempC
                new_grainsize = np.piecewise(airtemp,
                                    [airtemp<=-30,-30<airtemp<0,airtemp>=0],
                                    [54.5,54.5+5*(airtemp+30),204.5])

            new_height = snowfall/new_density
            new_BC = enbal.bcwet * enbal.dt
            new_OC = enbal.ocwet * enbal.dt
            new_dust = enbal.dustwet * enbal.dt
            new_snow = snowfall
            surface.snow_timestamp = timestamp

        if prms.switch_LAPs != 1:
            new_BC = 0
            new_OC = 0
            new_dust = 0
            
        # conditions: if any are TRUE, create a new layer
        new_layer_conds = np.array([self.ltype[0] in 'ice',
                            self.ltype[0] in 'firn',
                            self.ldensity[0] > new_density*3])
        if np.any(new_layer_conds):
            if snowfall/new_density < 1e-4:
                # delay small amounts of snowfall: avoids computational issues
                self.delayed_snow = snowfall
                return 0
            else:
                new_layer = pd.DataFrame([enbal.tempC,0,snowfall/new_density,'snow',snowfall,
                                      new_grainsize,new_BC,new_OC,new_dust,new_snow],
                                     index=['T','w','h','t','m','g','BC','OC','dust','new'])
                self.add_layers(new_layer)
                self.delayed_snow = 0
        else:
            # take weighted average of density and temperature of surface layer and new snow
            self.delayed_snow = 0
            new_layermass = self.lice[0] + snowfall
            self.lnewsnow[0] = snowfall if prms.switch_snow == 1 else 0
            self.ldensity[0] = (self.ldensity[0]*self.lice[0] + new_density*snowfall)/(new_layermass)
            self.ltemp[0] = (self.ltemp[0]*self.lice[0] + enbal.tempC*snowfall)/(new_layermass)
            self.lgrainsize[0] = (self.lgrainsize[0]*self.lice[0] + new_grainsize*snowfall)/(new_layermass)
            self.lice[0] = new_layermass
            self.lheight[0] += snowfall/new_density
            self.lBC[0] = self.lBC[0] + new_BC
            self.lOC[0] = self.lOC[0] + new_OC
            self.ldust[0] = self.ldust[0] + new_dust
            if self.lheight[0] > (prms.dz_toplayer * 2):
                self.split_layer(0)
    
        # CHECK MASS CONSERVATION
        change = np.sum(self.lice + self.lwater) - initial_mass
        assert np.abs(change - snowfall) < prms.mb_threshold

        self.update_layer_props()
        return snowfall

    def get_grain_size(self,airtemp,surftemp):
        """
        Updates grain size according to wet and dry
        metamorphism, refreeze, and addition of fresh
        snow.

        Parameters
        ==========
        airtemp : float
            Air temperature [C]
        surftemp : float
            Surface temperature [C]
        """
        # CONSTANTS
        WET_C = prms.wet_snow_C
        PI = np.pi
        RFZ_GRAINSIZE = prms.rfz_grainsize
        FIRN_GRAINSIZE = prms.firn_grainsize
        ICE_GRAINSIZE = prms.ice_grainsize
        dt = prms.daily_dt

        if prms.constant_freshgrainsize:
            FRESH_GRAINSIZE = prms.constant_freshgrainsize
        else:
            FRESH_GRAINSIZE = np.piecewise(airtemp,[airtemp<=-30,-30<airtemp<0,airtemp>=0],
                                       [54.5,54.5+5*(airtemp+30),204.5])
            
        # only run if we have snow layers
        if len(self.snow_idx) > 0:
            idx = self.snow_idx
            n = len(idx)
            
            # get fractions of refreeze, new snow and old snow
            refreeze = self.drefreeze[idx]
            new_snow = self.lnewsnow[idx]
            old_snow = self.lice[idx] - refreeze - new_snow
            f_old = old_snow / self.lice[idx]
            f_new = new_snow / self.lice[idx]
            f_rfz = refreeze / self.lice[idx]
            f_liq = self.lwater[idx] / (self.lwater[idx] + self.lice[idx])

            # define values for lookup table
            dz = self.lheight.copy()[idx]
            T = self.ltemp.copy()[idx] + 273.15
            surftempK = surftemp + 273.15
            p = self.ldensity.copy()[idx]
            grainsize = self.lgrainsize.copy()[idx]

            # dry metamorphism
            if prms.constant_drdry:
                drdry = np.ones(len(idx))*prms.constant_drdry * dt # um
                drdry[np.where(grainsize>RFZ_GRAINSIZE)[0]] = 0
            else:
                # calculate temperature gradient
                dTdz = np.zeros_like(T)
                if len(idx) > 2:
                    dTdz[0] = (surftempK - (T[0]*dz[0]+T[1]*dz[1]) / (dz[0]+dz[1]))/dz[0]
                    dTdz[1:-1] = ((T[:-2]*dz[:-2] + T[1:-1]*dz[1:-1]) / (dz[:-2] + dz[1:-1]) -
                            (T[1:-1]*dz[1:-1] + T[2:]*dz[2:]) / (dz[1:-1] + dz[2:])) / dz[1:-1]
                    dTdz[-1] = dTdz[-2] # bottom temp gradient -- not used
                elif len(idx) == 2: # use top ice layer for temp gradient
                    T_2layer = np.array([surftempK,T[0],T[1],self.ltemp[2]+273.15])
                    depth_2layer = np.array([0,self.ldepth[0],self.ldepth[1],self.ldepth[2]])
                    dTdz = (T_2layer[0:2] - T_2layer[2:]) / (depth_2layer[0:2] - depth_2layer[2:])
                else: # single layer
                    dTdz = (self.ltemp[2]+273.15-surftempK) / self.ldepth[2]
                    dTdz = np.array([dTdz])
                dTdz = np.abs(dTdz)

                # force values to be within lookup table ranges
                p[np.where(p < 50)[0]] = 50
                p[np.where(p > 400)[0]] = 400
                dTdz[np.where(dTdz > 300)[0]] = 300
                T[np.where(T < 223.15)[0]] = 223.15
                T[np.where(T > 273.15)[0]] = 273.15

                # Interpolate lookup table at the values of T,dTdz,p
                ds = prms.grainsize_ds.copy(deep=True)
                ds = ds.interp(TVals=T.astype(float),
                            DTDZVals=dTdz.astype(float),
                            DENSVals=p.astype(float))
                # Extract values
                diag = np.zeros((n,n,n),dtype=bool)
                for i in range(n):
                    diag[i,i,i] = True
                tau = ds.taumat.to_numpy()[diag].astype(float)
                kap = ds.kapmat.to_numpy()[diag].astype(float)
                dr0 = ds.dr0mat.to_numpy()[diag].astype(float)

                # dry metamorphism
                drdrydt = []
                for r,t,k,g in zip(dr0,tau,kap,grainsize):
                    if t + g <= FRESH_GRAINSIZE:
                        drdrydt.append(r*np.power(t/(t + 1e-6),1/k)/dt)
                    else:
                        drdrydt.append(r*np.power(t/(t + g - FRESH_GRAINSIZE),1/k)/dt)
                drdry = np.array(drdrydt) * dt

            # wet metamorphism
            drwetdt = WET_C*f_liq**3/(4*PI*(grainsize/1e6)**2)
            drwet = drwetdt * dt * 1e6

            # get change in grain size due to aging
            aged_grainsize = grainsize + drdry + drwet
                      
            # sum contributions of old snow, new snow and refreeze
            grainsize = aged_grainsize*f_old + FRESH_GRAINSIZE*f_new + RFZ_GRAINSIZE*f_rfz

            # enforce maximum grainsize
            grainsize[np.where(grainsize > FIRN_GRAINSIZE)[0]] = FIRN_GRAINSIZE
            self.lgrainsize[idx] = grainsize
            self.lgrainsize[self.firn_idx] = FIRN_GRAINSIZE 
            self.lgrainsize[self.ice_idx] = ICE_GRAINSIZE

        elif len(self.firn_idx) > 0: # no snow, but there is firn
            self.lgrainsize[self.firn_idx] = FIRN_GRAINSIZE
            self.lgrainsize[self.ice_idx] = ICE_GRAINSIZE
        else: # no snow or firn, just ice
            self.lgrainsize[self.ice_idx] = ICE_GRAINSIZE
        
        return 
    
    def exit(self):
        if self.args.debug:
            print('Failed in layers')
            print('Layer temperature:',self.ltemp)
            print('Layer density:',self.ldensity)
        sys.exit()