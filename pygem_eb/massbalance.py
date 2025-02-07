"""
Mass balance class and main functions for PyGEM Energy Balance

@author: clairevwilson
"""
import os, sys
import numpy as np
import pandas as pd
import xarray as xr
import pygem_eb.input as eb_prms
import pygem_eb.energybalance as eb
import pygem_eb.layers as eb_layers
import pygem_eb.surface as eb_surface

class massBalance():
    """
    Mass balance scheme which calculates layer and mass balance 
    from melt, refreeze and accumulation. main() executes the core 
    of the model.
    """
    def __init__(self,args,climate):
        """
        Initializes the layers and surface classes and model 
        time for the mass balance scheme.

        Parameters
        ----------
        dates : array-like (pd.datetime)
            List of local time dates
        """
        # Set up model time
        self.dt = eb_prms.dt
        self.days_since_snowfall = 0
        self.time_list = climate.dates
        self.firn_converted = False

        # Initialize climate, layers and surface classes
        self.args = args
        self.climate = climate
        self.layers = eb_layers.Layers(climate,args)
        self.surface = eb_surface.Surface(self.layers,self.time_list,args,climate)

        # Initialize output class
        self.output = Output(self.time_list,args)

        # Initialize mass balance check variable
        self.previous_mass = np.sum(self.layers.lice + self.layers.lwater)
        self.lice_before = np.sum(self.layers.lice)
        self.lwater_before = np.sum(self.layers.lwater)
        return
    
    def main(self):
        """
        Runs the time loop and mass balance scheme to solve for melt, refreeze, 
        accumulation and runoff.
        """
        # Get classes and time
        layers = self.layers
        surface = self.surface
        dt = self.dt

        # CONSTANTS
        DENSITY_WATER = eb_prms.density_water

        # ===== ENTER TIME LOOP =====
        for time in self.time_list:
            # BEGIN MASS BALANCE
            self.time = time

            # Initiate the energy balance to unpack climate data
            enbal = eb.energyBalance(self.climate,time,dt,self.args)
            self.enbal = enbal 

            # Get rain and snowfall amounts [kg m-2]
            rainfall,snowfall = self.get_precip(enbal)

            # Add fresh snow to layers
            snowfall = layers.add_snow(snowfall,enbal,surface,time)

            # Add dry deposited BC, OC and dust to layers
            enbal.get_dry_deposition(layers)

            # Update daily properties
            if time.hour == 0:
                surface.daily_updates(layers,enbal.tempC,surface.stemp,time)
                self.days_since_snowfall = surface.days_since_snowfall
                layers.lnewsnow = np.zeros(layers.nlayers)

            if time.hour in eb_prms.albedo_TOD:
                surface.get_albedo(layers,time)

            # Calculate surface energy balance by updating surface temperature
            surface.get_surftemp(enbal,layers)

            # Calculate subsurface heating from penetrating SW
            SWin,SWout = enbal.get_SW(surface)
            subsurf_melt = self.subsurface_heating(layers,SWin+SWout)
            
            # Calculate column melt including the surface
            layermelt = self.melting(layers,subsurf_melt)
            # Sum melt for output
            melt = np.sum(layermelt)
            if self.melted_layers != 0:
                melt += np.sum(self.melted_layers.mass)
            
            # Percolate the meltwater, rain and LAPs
            runoff = self.percolation(enbal,layers,layermelt,rainfall)
            
            # Recalculate the temperature profile considering conduction
            self.thermal_conduction(layers,surface.stemp)

            # Calculate refreeze
            refreeze = self.refreezing(layers)

            # Run densification daily
            if time.hour == 0:
                self.densification(layers)

            # Calculate mass from phase changes
            self.phase_changes(enbal,surface,layers)

            # Check and update layer sizes
            layers.check_layers(time,self.output.out_fn)
            
            # If towards the end of summer, check if old snow should become firn
            if time.day_of_year >= eb_prms.end_summer_doy and time.hour == 0:
                if not self.firn_converted:
                    self.end_of_summer()

            # Check mass conserves
            self.check_mass_conservation(snowfall+rainfall, runoff)

            # END MASS BALANCE
            # Convert units of mass balance terms
            self.runoff = runoff / DENSITY_WATER
            self.melt = melt / DENSITY_WATER
            self.refreeze = refreeze / DENSITY_WATER
            self.accum = snowfall / DENSITY_WATER

            # Store timestep data
            self.output.store_timestep(self,enbal,surface,layers,time)   

            # Debugging: print current state and monthly melt at the end of each month
            if time.is_month_start and time.hour == 0 and self.args.debug:
                self.current_state(time,enbal.tempC)

            # Updated yearly properties
            if time.day_of_year == 1 and time.hour == 0:
                self.firn_converted = False

            # Check if we still have a glacier
            self.check_glacier_exists()

            # Advance timestep
            pass

        # Completed run: store data
        if self.args.store_data:
            self.output.store_data()

        if eb_prms.store_bands:
            surface.albedo_df.to_csv(eb_prms.albedo_out_fp.replace('.csv',f'_{self.args.elev}.csv'))
        return
    
    def get_precip(self,enbal):
        """
        Determines whether rain or snowfall occurred and outputs amounts.

        Parameters:
        -----------
        enbal
            class object from pygem_eb.energybalance
        surface
            class object from pygem_eb.surface
        time : pd.Datetime
            Current timestep
            
        Returns:
        --------
        rain, snowfall : float
            Specific mass of liquid and solid precipitation [kg m-2]
        """
        # CONSTANTS
        SNOW_THRESHOLD_LOW = eb_prms.snow_threshold_low
        SNOW_THRESHOLD_HIGH = eb_prms.snow_threshold_high
        DENSITY_WATER = eb_prms.density_water

        # Define rain vs snow scaling 
        rain_scale = np.linspace(0,1,20)
        temp_scale = np.linspace(SNOW_THRESHOLD_LOW,SNOW_THRESHOLD_HIGH,20)

        if enbal.tempC <= SNOW_THRESHOLD_LOW: 
            # precip falls as snow
            rain = 0
            snow = enbal.tp*DENSITY_WATER
        elif SNOW_THRESHOLD_LOW < enbal.tempC < SNOW_THRESHOLD_HIGH:
            # mix of rain and snow
            fraction_rain = np.interp(enbal.tempC,temp_scale,rain_scale)
            rain = enbal.tp*fraction_rain*DENSITY_WATER
            snow = enbal.tp*(1-fraction_rain)*DENSITY_WATER
        else:
            # precip falls as rain
            rain = enbal.tp*DENSITY_WATER
            snow = 0
        
        # Adjust snow by precipitation factor
        snow *= float(self.args.kp)
        
        return rain,snow  # kg m-2

    def subsurface_heating(self,layers,surface_SW):
        """
        Calculates melt in subsurface layers (excluding layer 0) 
        due to penetrating shortwave radiation.

        Parameters
        ----------
        surface_SW : float
            Incoming SW radiation [W m-2]

        Returns
        -------
        layermelt : np.ndarray
            Array containing subsurface melt amounts [kg m-2]
        """
        if layers.nlayers == 1: 
            return [0.]
        
        # CONSTANTS
        HEAT_CAPACITY_ICE = eb_prms.Cp_ice
        LH_RF = eb_prms.Lh_rf

        # LAYERS IN
        lt = layers.ltype.copy()
        ld = layers.ldepth.copy()
        lT = layers.ltemp.copy()
        lm = layers.lice.copy()

        # Fraction of radiation absorbed at the surface depends on surface type
        FRAC_ABSRAD = 0.9 if lt[0] in ['snow'] else 0.8

        # Extinction coefficient depends on layer type
        EXTINCT_COEF = np.ones(layers.nlayers)*1e8 # ensures unfilled layers have 0 heat
        for layer,type in enumerate(lt):
            EXTINCT_COEF[layer] = 17.1 if type in ['snow'] else 2.5
            if np.exp(-EXTINCT_COEF[layer]*ld[layer]) < 1e-6:
                break # Cut off when flux reaches ~0
        pen_SW = surface_SW*(1-FRAC_ABSRAD)*np.exp(-EXTINCT_COEF*ld)/self.dt

        # recalculate layer temperatures, leaving out the surface (calculated separately)
        lT[1:] = lT[1:] + pen_SW[1:]/(lm[1:]*HEAT_CAPACITY_ICE)*self.dt

        # calculate melt from temperatures above 0
        layermelt = np.zeros(layers.nlayers)
        for layer,temp in enumerate(lT):
            # check if temperature is above 0
            if temp > 0.:
                # calculate melt from the energy that raised the temperature above 0
                melt = (temp-0.)*lm[layer]*HEAT_CAPACITY_ICE/LH_RF
                lT[layer] = 0.
            else:
                melt = 0
            layermelt[layer] = melt

        # LAYERS OUT
        layers.ltemp = lT
        return layermelt

    def melting(self,layers,subsurf_melt):
        """
        For cases when layers are melting. Can melt multiple surface layers
        at once if Qm is sufficiently high. Otherwise, adds the surface
        layer melt to the array containing subsurface melt to return the
        total layer melt. (This function does NOT REMOVE MELTED MASS from 
        layers. That is done in self.percolation.)

        Parameters
        ----------
        layers
            class object from pygem_eb.layers
        surface
            class object from pygem_eb.surface
        subsurf_melt : np.ndarray
            Array containing melt of subsurface layers [kg m-2]
        Returns
        -------
        layermelt : np.ndarray
            Array containing layer melt amounts [kg m-2]
        
        """
        if self.surface.Qm <= 0:
            layermelt = subsurf_melt.copy()
            layermelt[0] = 0
            self.melted_layers = 0
            return layermelt
        
        # CONSTANTS
        LH_RF = eb_prms.Lh_rf

        # LAYERS IN
        lm = layers.lice.copy()
        lw = layers.lwater.copy()
        layermelt = subsurf_melt.copy()       # mass of melt due to penetrating SW [kg m-2]
        surface_melt = self.surface.Qm*self.dt/LH_RF     # mass of melt due to SEB [kg m-2]
        initial_mass = np.sum(layers.lice + layers.lwater)

        if surface_melt > lm[0]: # melt by surface energy balance completely melts surface layer
            # check if it melts further layers
            fully_melted = np.where(np.cumsum(lm) <= surface_melt)[0]

            # calculate how much more melt occurs in the first layer not fully melted
            newsurface_melt = surface_melt - np.sum(lm[fully_melted])
            newsurface_idx = fully_melted[-1] + 1

            # possible to fully melt that layer too when combined with penetrating SW melt:
            if newsurface_melt + layermelt[newsurface_idx] > lm[newsurface_idx]:
                fully_melted = np.append(fully_melted,newsurface_idx)
                # push new surface to the next layer down
                newsurface_melt -= lm[newsurface_idx]
                newsurface_idx += 1

            # set melt amounts from surface melt into melt array
            layermelt[fully_melted] = lm[fully_melted] + lw[fully_melted]
            layermelt[newsurface_idx] += newsurface_melt 
        else:
            # only surface layer is melting
            layermelt[0] = surface_melt
            fully_melted = []
        
        class MeltedLayers():
            def __init__(self):
                try:
                    self.mass = np.array(layermelt)[fully_melted]
                except:
                    print(fully_melted)
                    self.mass = 0
                self.BC = layers.lBC[fully_melted]
                self.OC = layers.lOC[fully_melted]
                self.dust = layers.ldust[fully_melted]

        self.melted_layers = MeltedLayers()

        # Remove layers that were completely melted 
        removed = 0 # Accounts for indexes of layers changing with loop
        for layer in fully_melted:
            layers.remove_layer(layer-removed)
            layermelt = np.delete(layermelt,layer-removed)
            removed += 1 

        # CHECK MASS CONSERVATION
        change = np.sum(layers.lice + layers.lwater) - initial_mass
        if len(fully_melted) > 0:
            change += np.sum(self.melted_layers.mass)
        assert np.abs(change) < eb_prms.mb_threshold, f'melting failed mass conservation in {self.output.out_fn}'
        return layermelt
        
    def percolation(self,enbal,layers,layermelt,rainfall=0):
        """
        Calculates the liquid water content in each layer by downward
        percolation and applies melt.

        Parameters
        ----------
        enbal
            class object from pygem_eb.energybalance
        layers
            class object from pygem_eb.layers
        layermelt: np.ndarray
            Array containing melt amount for each layer
        rainfall : float
            Additional liquid water input from rainfall [kg m-2]

        Returns
        -------
        runoff : float
            Runoff of liquid water lost to system [kg m-2]
        melted_layers : list
            List of layer indices that were fully melted
        """
        # CONSTANTS
        DENSITY_WATER = eb_prms.density_water
        DENSITY_ICE = eb_prms.density_ice
        FRAC_IRREDUC = eb_prms.Sr
        dt = self.dt

        # Get index of percolating layers (snow and firn)
        snow_firn_idx = np.concatenate([layers.snow_idx,layers.firn_idx])
        if len(snow_firn_idx) > 0 and layers.ice_idx[0] < snow_firn_idx[-1]:
            if layers.ice_idx[0] != 0: # impermeable ice layer
                snow_firn_idx = snow_firn_idx[:layers.ice_idx[0]]
            else: # surface ice layer: all water runs off
                snow_firn_idx = []

        # Initialize variables
        initial_mass = np.sum(layers.lice + layers.lwater)
        lmi,lwi = (layers.lice,layers.lwater)
        rain_bool = rainfall > 0
        runoff = 0  # Any flow that leaves the point laterally

        # Add water from completely melted layers
        if self.melted_layers != 0:
            water_in = rainfall + np.sum(self.melted_layers.mass)
        else:
            water_in = rainfall

        if len(snow_firn_idx) > 0:
            # LAYERS IN
            lm = layers.lice.copy()[snow_firn_idx]
            lw = layers.lwater.copy()[snow_firn_idx]
            lh = layers.lheight.copy()[snow_firn_idx]
            layermelt_sf = layermelt[snow_firn_idx]

            # Calculate volumetric fractions (theta)
            vol_f_liq = lw / (lh*DENSITY_WATER)
            vol_f_ice = lm / (lh*DENSITY_ICE)
            porosity = 1 - vol_f_ice
            vol_f_liq[vol_f_liq > porosity] = porosity[vol_f_liq > porosity]

            # Remove / move snow melt to layer water
            lm -= layermelt_sf
            lh -= layermelt_sf / layers.ldensity[snow_firn_idx]
            lw += layermelt_sf

            # Reduce layer refreeze (refreeze melts first)
            layers.cumrefreeze[snow_firn_idx] -= layermelt_sf
            layers.cumrefreeze[layers.cumrefreeze < 0] = 0

            # Initialize flow into the top layer
            q_out = water_in / dt
            q_in_store = []
            q_out_store = []
            for layer in snow_firn_idx:
                # Set flow in equal to flow out of the previous layer
                q_in = q_out

                # Calculate flow out of layer i
                q_out = DENSITY_WATER*lh[layer]/dt * (
                        vol_f_liq[layer]-FRAC_IRREDUC*porosity[layer])
                
                # Check limits on flow out (q_out)
                # check limit of qi based on underlying layer holding capacity
                if layer < len(porosity) - 1 and vol_f_liq[layer] <= 0.3:
                    next = layer+1
                    lim = DENSITY_WATER*lh[next]/dt * (1-vol_f_ice[next]-vol_f_liq[next])
                else: # no limit on bottom layer (1e6 sufficiently high)
                    lim = 1e6
                # cannot have more flow out than flow in + existing water
                lim = min(lim,q_in + lw[layer])
                q_out = min(q_out,lim)
                # cannot be negative
                q_out = max(0,q_out)

                # layer mass balance
                lw[layer] += (q_in - q_out)*dt
                q_in_store.append(q_in)
                q_out_store.append(q_out)

                # layer cannot contain more water than there is pore space
                layer_porosity = max(1 - lm[layer] / (lh[layer]*DENSITY_ICE),0)
                water_lim = lh[layer]*layer_porosity*DENSITY_WATER
                if lw[layer] > water_lim: # excess runs off
                    runoff += lw[layer] - water_lim
                    lw[layer] = water_lim

            # LAYERS OUT
            layers.lheight[snow_firn_idx] = lh
            layers.lwater[snow_firn_idx] = lw
            layers.lice[snow_firn_idx] = lm
            runoff += q_out*dt + np.sum(layermelt[layers.ice_idx])

            # Separately remove melted ice mass
            for layer in layers.ice_idx:
                layers.lice[layer] -= layermelt[layer]

            # Advect LAPs 
            if self.args.switch_LAPs == 1:
                self.advect_LAPs(np.array(q_out_store),
                                  enbal,layers,rain_bool,
                                  snow_firn_idx)
        else:
            # No percolation, but need to move melt to runoff
            layers.lice -= layermelt
            layers.lheight -= layermelt / layers.ldensity
            runoff += water_in + np.sum(layermelt)
        
        # CHECK MASS CONSERVATION
        ins = water_in
        outs = runoff
        change = np.sum(layers.lice + layers.lwater) - initial_mass
        if np.abs(change - (ins-outs)) >= eb_prms.mb_threshold:
            print(self.output.out_fn)
            print('percolation ins',water_in,'outs',runoff,'now',np.sum(layers.lice + layers.lwater),'initial',initial_mass)
            print('now',layers.lice,layers.lwater)
            print('initial',lmi,lwi)
        assert np.abs(change - (ins-outs)) < eb_prms.mb_threshold, f'percolation failed mass conservation in {self.output.out_fn}'
        return runoff
        
    def advect_LAPs(self,q_out,enbal,layers,rain_bool,snow_firn_idx):
        """
        Advects LAPs vertically through the snow and firn layers based on
        inter-layer water fluxes from percolation.

        Parameters
        ----------
        layers
            class object from pygem_eb.layers
        q_out : np.ndarray
            Array containing water flow out of a layer [kg m-2 s-1]
        enbal
            class object from pygem_eb.energybalance
        rain_bool : Bool
            Raining or not?
        """
        # CONSTANTS
        PARTITION_COEF_BC = eb_prms.ksp_BC
        PARTITION_COEF_OC = eb_prms.ksp_OC
        PARTITION_COEF_DUST = eb_prms.ksp_dust
        dt = eb_prms.dt

        # LAYERS IN
        lw = layers.lwater[snow_firn_idx]
        lm = layers.lice[snow_firn_idx]

        # Layer mass of each species in kg m-2
        mBC = layers.lBC[snow_firn_idx]
        mOC = layers.lOC[snow_firn_idx]
        mdust = layers.ldust[snow_firn_idx]

        # Get wet deposition into top layer if it's raining
        if rain_bool and eb_prms.switch_LAPs == 1: # Switch runs have no BC
            mBC[0] += enbal.bcwet * dt
            mOC[0] += enbal.ocwet * dt
            mdust[0] += enbal.dustwet * eb_prms.ratio_DU3_DUtot * dt

        # Layer mass mixing ratio in kg kg-1
        cBC = mBC / (lw + lm)
        cOC = mOC / (lw + lm)
        cdust = mdust / (lw + lm)

        # Add LAPs from fully melted layers
        if self.melted_layers != 0:
            m_BC_in_top = np.array(np.sum(self.melted_layers.BC) / dt)
            m_OC_in_top = np.array(np.sum(self.melted_layers.OC) / dt)
            m_dust_in_top = np.array(np.sum(self.melted_layers.dust) / dt)
        else:
            m_BC_in_top = np.array([0],dtype=float) 
            m_OC_in_top = np.array([0],dtype=float) 
            m_dust_in_top = np.array([0],dtype=float)
        # Partition in aqueous phase
        m_BC_in_top *= PARTITION_COEF_BC
        m_OC_in_top *= PARTITION_COEF_OC
        m_dust_in_top *= PARTITION_COEF_DUST

        # inward fluxes = outward fluxes from previous layer
        m_BC_in = PARTITION_COEF_BC*q_out[:-1]*cBC[:-1]
        m_OC_in = PARTITION_COEF_OC*q_out[:-1]*cOC[:-1]
        m_dust_in = PARTITION_COEF_DUST*q_out[:-1]*cdust[:-1]
        m_BC_in = np.append(m_BC_in_top,m_BC_in)
        m_OC_in = np.append(m_OC_in_top,m_OC_in)
        m_dust_in = np.append(m_dust_in_top,m_dust_in)

        # outward fluxes are simply (flow out)*(concentration of the layer)
        m_BC_out = PARTITION_COEF_BC*q_out*cBC
        m_OC_out = PARTITION_COEF_OC*q_out*cOC
        m_dust_out = PARTITION_COEF_DUST*q_out*cdust

        # mass balance on each constituent
        dmBC = (m_BC_in - m_BC_out)*dt
        dmOC = (m_OC_in - m_OC_out)*dt
        dmdust = (m_dust_in - m_dust_out)*dt
        mBC += dmBC.astype(float)
        mOC += dmOC.astype(float)
        mdust += dmdust.astype(float)

        # LAYERS OUT
        layers.lBC[snow_firn_idx] = mBC
        layers.lOC[snow_firn_idx] = mOC
        layers.ldust[snow_firn_idx] = mdust
        return
    
    def refreezing(self,layers):
        """
        Calculates refreeze in layers due to temperatures below freezing
        with liquid water content.

        Parameters:
        -----------
        layers
            class object from pygem_eb.layers
        Returns:
        --------
        refreeze : float
            Total amount of refreeze [kg m-2]
        """
        # CONSTANTS
        HEAT_CAPACITY_ICE = eb_prms.Cp_ice
        DENSITY_ICE = eb_prms.density_ice
        LH_RF = eb_prms.Lh_rf

        # LAYERS IN
        snow_firn_idx = np.concatenate([layers.snow_idx,layers.firn_idx])
        lT = layers.ltemp.copy()[snow_firn_idx]
        lw = layers.lwater.copy()[snow_firn_idx]
        lm = layers.lice.copy()[snow_firn_idx]
        lh = layers.lheight.copy()[snow_firn_idx]

        initial_mass = np.sum(layers.lice + layers.lwater)
        refreeze = np.zeros(len(snow_firn_idx))
        for layer, T in enumerate(lT):
            if T < 0. and lw[layer] > 0:
                # calculate potential for refreeze [J m-2]
                E_cold = np.abs(T)*lm[layer]*HEAT_CAPACITY_ICE  # cold content available 
                E_water = lw[layer]*LH_RF  # amount of water to freeze
                E_pore = (DENSITY_ICE*lh[layer]-lm[layer])*LH_RF # pore space available
                
                # calculate amount of refreeze in kg m-2
                dm_ref = np.min([abs(E_cold),abs(E_water),abs(E_pore)])/LH_RF

                # add refreeze to array in kg m-2
                refreeze[layer] = dm_ref

                # add refreeze to layer ice mass
                lm[layer] += dm_ref
                # update layer temperature from latent heat
                lT[layer] = min(0,-(E_cold-dm_ref*LH_RF)/(HEAT_CAPACITY_ICE*lm[layer]))

                # update water content
                lw[layer] = max(0,lw[layer]-dm_ref)
        
        # Update refreeze with new refreeze content
        layers.drefreeze[snow_firn_idx] = refreeze
        layers.cumrefreeze[snow_firn_idx] += refreeze

        # LAYERS OUT
        layers.ltemp[snow_firn_idx] = lT
        layers.lwater[snow_firn_idx] = lw
        layers.lice[snow_firn_idx] = lm
        layers.lheight[snow_firn_idx] = lh
        layers.update_layer_props()

        # CHECK MASS CONSERVATION
        change = np.sum(layers.lice + layers.lwater) - initial_mass
        if np.abs(change) >= eb_prms.mb_threshold:
            print('rfz change',change, 'initial',initial_mass,'dry, water',layers.lice, layers.lwater)
        assert np.abs(change) < eb_prms.mb_threshold, f'refreezing failed mass conservation in {self.output.out_fn}'
        return np.sum(refreeze)
    
    def densification(self,layers):
        """
        Calculates densification of layers due to compression from overlying mass.

        Parameters:
        -----------
        layers
            class object from pygem_eb.layers
        """
        # CONSTANTS
        GRAVITY = eb_prms.gravity
        R = eb_prms.R_gas
        VISCOSITY_SNOW = eb_prms.viscosity_snow
        rho = eb_prms.constant_snowfall_density
        DENSITY_FRESH_SNOW = rho if rho else 50
        DENSITY_ICE = eb_prms.density_ice
        dt = eb_prms.daily_dt

        # LAYERS IN
        snowfirn_idx = np.append(layers.snow_idx,layers.firn_idx)
        lp = layers.ldensity.copy()
        lT = layers.ltemp.copy()
        lm = layers.lice.copy()
        lw = layers.lwater.copy()

        initial_mass = np.sum(layers.lice + layers.lwater)

        if eb_prms.method_densification in ['Boone']:
            # EMPIRICAL PARAMETERS
            c1 = 2.7e-6     # s-1 (2.7e-6)
            c2 = 0.042      # K-1 (0.042)
            c3 = 0.046      # m3 kg-1 (0.046)
            c4 = 0.081      # K-1 (0.081)
            c5 = float(self.args.Boone_c5)

            for layer in snowfirn_idx:
                weight_above = GRAVITY*np.sum(lm[:layer]+lw[:layer])
                viscosity = VISCOSITY_SNOW*np.exp(c4*(0.-lT[layer])+c5*lp[layer])

                # get change in density
                mass_term = (weight_above*GRAVITY)/viscosity
                temp_term = -c2*(0.-lT[layer])
                dens_term = -c3*max(0,lp[layer]-DENSITY_FRESH_SNOW)
                dRho = (mass_term+c1*np.exp(temp_term+dens_term))*lp[layer]*dt
                lp[layer] += dRho

            # LAYERS OUT
            layers.ldensity = lp
            layers.lheight = lm / lp
            layers.update_layer_props('depth')

        # Herron Langway (1980) method
        elif eb_prms.method_densification in ['HerronLangway']:
            # yearly accumulation is the maximum layer snow mass in mm w.e. yr-1
            a = layers.max_snow / (dt*365) # kg m-2 = mm w.e.
            k = np.zeros_like(lp)
            b = np.zeros_like(lp)
            for layer,density in enumerate(lp[snowfirn_idx]):
                lTK = lT[layer] + 273.15
                if density < 550:
                    b[layer] = 1
                    k[layer] = 11*np.exp(-10160/(R*lTK))
                else:
                    b[layer] = 0.5
                    k[layer] = 575*np.exp(-21400/(R*lTK))
            dRho = k*a**b*(DENSITY_ICE - lp)/DENSITY_ICE*dt
            lp += dRho

            # LAYERS OUT
            layers.ldensity = lp
            layers.lheight = lm / lp
            layers.update_layer_props('depth')

        # Kojima (1967) method (JULES)
        elif eb_prms.method_densification in ['Kojima']:
            NU_0 = 1e7      # Pa s
            RHO_0 = 50      # kg m-3
            k_S = 4000      # K
            T_m = 0. + 273.15
            for layer in snowfirn_idx:
                weight_above = GRAVITY*np.sum(lm[:layer]+lw[:layer])

                # get change in density
                T_K = lT[layer] + 273.15
                exp_term = np.exp(k_S/T_m - k_S/T_K - lp[layer]/RHO_0)
                dRho = lp[layer]*weight_above/NU_0*exp_term
                lp[layer] += dRho

            # LAYERS OUT
            layers.ldensity = lp
            layers.lheight = lm / lp
            layers.update_layer_props('depth')

        # Check if new firn or ice layers were created
        layers.update_layer_types()

        # CHECK MASS CONSERVATION
        change = np.sum(layers.lice + layers.lwater) - initial_mass
        assert np.abs(change) < eb_prms.mb_threshold, f'densification failed mass conservation in {self.output.out_fn}'
        return
    
    def phase_changes(self,enbal,surface,layers):
        # CONSTANTS
        DENSITY_WATER = eb_prms.density_water
        LV_SUB = eb_prms.Lv_sub
        LV_VAP = eb_prms.Lv_evap

        initial_mass = np.sum(layers.lice + layers.lwater)

        # Get mass fluxes from latent heat
        if surface.stemp < 0.:
            # Sublimation / deposition
            dm = enbal.lat*self.dt/(DENSITY_WATER*LV_SUB)
            # Solid-vapor fluxes
            sublimation = -1*min(dm,0)
            deposition = max(dm,0)
            # Liquid
            evaporation = 0
            condensation = 0
            if layers.lice[0] + dm < 0: # Check if first layer goes negative
                # Remove mass from next layer
                layers.lice[1] += dm + layers.lice[0] 
                layers.lice[0] = 0
            else:
                # Add mass to layer if it doesn't cause negativity
                layers.lice[0] += dm
        else:
            # Evaporation / condensation
            dm = enbal.lat*self.dt/(DENSITY_WATER*LV_VAP)
            # No solid-vapor fluxes
            sublimation = 0
            deposition = 0
            # Liquid-vapor fluxes
            evaporation = -1*min(dm,0)
            condensation = max(dm,0)
            if layers.lwater[0] + dm < 0:  # Check if the first layer goes negative
                layer = 0
                while np.abs(dm) > 0 and layer < layers.nlayers:
                    # Calculate the maximum water loss possible for the current layer
                    change = min(np.abs(dm), layers.lwater[layer])
                    layers.lwater[layer] -= change
                    
                    # Reduce the absolute magnitude of dm
                    if dm < 0:
                        dm += change  # Increase dm towards 0 when negative
                    else:
                        dm -= change  # Decrease dm towards 0 when positive
                    layer += 1
                
                if layer == layers.nlayers:
                    # No water is left to handle the remaining dm
                    evaporation += dm  # Add the leftover dm to evaporation
            else:
                # Add water to the first layer if it doesn't cause negativity
                layers.lwater[0] += dm
        
        # CHECK MASS CONSERVATION
        ins = deposition + condensation
        outs = sublimation + evaporation
        change = np.sum(layers.lice + layers.lwater) - initial_mass
        assert np.abs(change - (ins-outs)) < eb_prms.mb_threshold, f'phase change failed mass conservation in {self.output.out_fn}'
        return
      
    def thermal_conduction(self,layers,surftemp,dt_heat=eb_prms.dt_heateq):
        """
        Resolves the temperature profile from conduction of heat using 
        Forward-in-Time-Central-in-Space (FTCS) scheme

        Parameters:
        -----------
        layers
            class object from pygem_eb.layers
        surftemp : float
            Surface temperature [C]
        dt_heat : int
            Timestep to loop the heat equation solver [s]
        Returns:
        --------
        new_T : np.ndarray
            Array of new layer temperatures
        """
        if np.sum(layers.ltemp) == 0.:
            return
        
        # CONSTANTS
        CP_ICE = eb_prms.Cp_ice
        DENSITY_ICE = eb_prms.density_ice
        DENSITY_WATER = eb_prms.density_water
        TEMP_TEMP = eb_prms.temp_temp
        TEMP_DEPTH = eb_prms.temp_depth
        K_ICE = eb_prms.k_ice
        K_WATER = eb_prms.k_water
        K_AIR = eb_prms.k_air

        # determine layers that are below temperate ice depth
        if layers.ice_idx[0] > 0:
            # if there is snow/firn adjust to be relative to the ice surface
            TEMP_DEPTH += layers.ldepth[layers.ice_idx[0] - 1]
        temperate_idx = np.where(layers.ldepth > TEMP_DEPTH)[0]
        if len(temperate_idx) < 1:
            temperate_idx = [layers.nlayers - 1]
        diffusing_idx = np.arange(temperate_idx[0])
        layers.ltemp[temperate_idx] = TEMP_TEMP

        # LAYERS IN
        nl = len(diffusing_idx)
        lh = layers.lheight[diffusing_idx]
        lp = layers.ldensity[diffusing_idx]
        lT_old = layers.ltemp[diffusing_idx]
        lm = layers.lice[diffusing_idx]
        lw = layers.lwater[diffusing_idx]
        lT = layers.ltemp[diffusing_idx]

        # get conductivity 
        ice_idx = layers.ice_idx
        if len(self.args.k_snow) < 5:
            lcond = float(self.args.k_snow)*np.ones_like(lp)
        elif self.args.k_snow in ['Sauter']:
            f_ice = (lm/DENSITY_ICE) / lh
            f_liq = (lw/DENSITY_WATER) / lh
            f_air = 1 - f_ice - f_liq
            f_air[f_air < 0] = 0
            lcond = f_ice*K_ICE + f_liq*K_WATER + f_air*K_AIR
        elif self.args.k_snow in ['VanDusen']:
            lcond = 0.21e-01 + 0.42e-03*lp + 0.22e-08*lp**3
        elif self.args.k_snow in ['Douville']:
            lcond = 2.2*np.power(lp/DENSITY_ICE,1.88)
        elif self.args.k_snow in ['Jansson']:
            lcond = 0.02093 + 0.7953e-3*lp + 1.512e-12*lp**4
        elif self.args.k_snow in ['OstinAndersson']:
            lcond = -8.71e-3 + 0.439e-3*lp + 1.05e-6*lp**2
        # ice has constant conductivity
        diffusing_ice_idx = list(set(ice_idx)&set(diffusing_idx))
        if len(diffusing_ice_idx) > 0:
            lcond[diffusing_ice_idx] = K_ICE

        if nl > 2:
            # heights of imaginary average bins between layers
            up_lh = np.array([np.mean(lh[i:i+2]) for i in range(nl-2)])  # upper layer 
            dn_lh = np.array([np.mean(lh[i+1:i+3]) for i in range(nl-2)])  # lower layer

            # conductivity
            up_kcond = np.array([np.mean(lcond[i:i+2]*lh[i:i+2]) for i in range(nl-2)]) / up_lh
            dn_kcond = np.array([np.mean(lcond[i+1:i+3]*lh[i+1:i+3]) for i in range(nl-2)]) / dn_lh

            # density
            up_dens = np.array([np.mean(lp[i:i+2]) for i in range(nl-2)]) / up_lh
            dn_dens = np.array([np.mean(lp[i+1:i+3]) for i in range(nl-2)]) / dn_lh

            # Top layer uses surftemp boundary condition
            surf_heat_0 = up_kcond[0]*2/(up_dens[0]*lh[0])*(surftemp-lT_old[0])
            subsurf_heat_0 = dn_kcond[0]/(up_dens[0]*up_lh[0])*(lT_old[0]-lT_old[1])
            lT[0] = lT_old[0] + (surf_heat_0 - subsurf_heat_0)*dt_heat/(CP_ICE*lh[0])
            # print('dT0',dt_heat*(surf_heat_0 - subsurf_heat_0)/(CP_ICE*lh[0]),lT[0])
            # If top layer of snow is very thin on top of ice, it can break this calculation
            if lT[0] > 0 or lT[0] < -50: 
                lT[0] = np.mean([surftemp,lT_old[1]])

            # Middle layers solve heat equation
            surf_heat = up_kcond/(up_dens*up_lh)*(lT_old[:-2]-lT_old[1:-1])
            subsurf_heat = dn_kcond/(dn_dens*dn_lh)*(lT_old[1:-1]-lT_old[2:])
            lT[1:-1] = lT_old[1:-1] + (surf_heat - subsurf_heat)*dt_heat/(CP_ICE*lh[1:-1])

        elif nl > 1:
            lT = np.array([surftemp/2,0])
        else:
            lT = np.array([0])

        # LAYERS OUT
        layers.ltemp[diffusing_idx] = lT
        return 
    
    def end_of_summer(self):
        # CONSTANTS
        NDAYS = eb_prms.new_snow_days
        SNOW_THRESHOLD = eb_prms.new_snow_threshold
        T_LOW = eb_prms.snow_threshold_low
        T_HIGH = eb_prms.snow_threshold_high

        # Only merge firn if there is snow and firn below that snow
        if len(self.layers.snow_idx) > 0 and len(self.layers.firn_idx) > 0:
            # Define rain vs snow scaling 
            rain_scale = np.linspace(1,0,20)
            temp_scale = np.linspace(T_LOW,T_HIGH,20)

            # Index the temperature and precipitation of the upcoming period
            end_time = min(self.time_list[-1],self.time+pd.Timedelta(days=NDAYS))
            check_dates = pd.date_range(self.time,end_time,freq='h')
            check_temp = self.climate.cds.sel(time=check_dates)['temp'].values
            check_tp = self.climate.cds.sel(time=check_dates)['tp'].values

            # Create array to mask tp to snow amounts
            mask = np.interp(check_temp,temp_scale,rain_scale)
            upcoming_snow = np.sum(check_tp*mask)
            if upcoming_snow < SNOW_THRESHOLD:
                # Not getting new snow, so we are not at the end of summer yet
                return
        
            # We're getting new snow, so today is the end of summer: merge snow to firn
            merge_count = max(0,len(self.layers.snow_idx) - 1)
            for _ in range(merge_count):
                self.layers.merge_layers(0)
                self.layers.ltype[0] = 'firn'
            if self.args.debug:
                print('Converted firn on ',self.time)
            self.firn_converted = True

            # Reset cumulative refreeze
            self.layers.cumrefreeze *= 0
            return

    def current_state(self,time,airtemp):
        """Prints some useful information to keep track of a model run"""
        # gather variables to print out
        layers = self.layers
        surftemp = self.surface.stemp
        albedo = self.surface.bba
        melte = np.mean(self.output.meltenergy_output[-720:])
        melt = np.sum(self.output.melt_output[-720:])
        accum = np.sum(self.output.accum_output[-720:])
        ended_month = (time - pd.Timedelta(days=1)).month_name()
        year = time.year if ended_month != 'December' else time.year - 1

        layers.update_layer_props()
        snowdepth = np.sum(layers.lheight[layers.snow_idx])
        firndepth = np.sum(layers.lheight[layers.firn_idx])
        icedepth = np.sum(layers.lheight[layers.ice_idx])

        # Begin prints
        print(f'MONTH COMPLETED: {ended_month} {year} with +{accum:.2f} and -{melt:.2f} m w.e.')
        print(f'Currently {airtemp:.2f} C with {melte:.0f} W m-2 melt energy')
        print(f'----------surface albedo: {albedo:.3f} -----------')
        print(f'-----------surface temp: {surftemp:.2f} C-----------')
        if len(layers.snow_idx) > 0:
            print(f'|       snow depth: {snowdepth:.2f} m      {len(layers.snow_idx)} layers      |')
        if len(layers.firn_idx) > 0:
            print(f'|       firn depth: {firndepth:.2f} m      {len(layers.firn_idx)} layers      |')
        print(f'|       ice depth: {icedepth:.2f} m      {len(layers.ice_idx)} layers      |')
        for l in range(min(2,layers.nlayers)):
            print(f'--------------------layer {l}---------------------')
            print(f'     T = {layers.ltemp[l]:.1f} C                 h = {layers.lheight[l]:.3f} m ')
            print(f'                 p = {layers.ldensity[l]:.0f} kg/m3')
            print(f'Water Mass : {layers.lwater[l]:.2f} kg/m2   Dry Mass : {layers.lice[l]:.2f} kg/m2')
        print('================================================')
        return
    
    def check_mass_conservation(self,mass_in,mass_out):
        """
        Checks mass was conserved within the last timestep
        mass_in:    sum of precipitation (kg m-2)
        mass_out:   sum of runoff (kg m-2)
        """
        # Difference in mass since the last timestep
        current_mass = np.sum(self.layers.lice + self.layers.lwater)
        diff = current_mass - self.previous_mass
        in_out = mass_in - mass_out
        if np.abs(diff - in_out) >= eb_prms.mb_threshold:
            print(self.time,'discrepancy of',np.abs(diff - in_out) - eb_prms.mb_threshold,self.output.out_fn)
            print('in',mass_in,'out',mass_out,'currently',current_mass,'was',self.previous_mass)
            print('ice before',self.lice_before,'ice after',np.sum(self.layers.lice))
            print('w before',self.lwater_before,'w after',np.sum(self.layers.lwater))
            print('melt',self.melt,'rfz',self.refreeze,'accum',self.accum)
        assert np.abs(diff - in_out) < eb_prms.mb_threshold, f'Timestep {self.time} failed mass conservation in {self.output.out_fn}'
        
        # New initial mass
        self.previous_mass = current_mass
        self.lice_before = np.sum(self.layers.lice)
        self.lwater_before = np.sum(self.layers.lwater)
        return
    
    def check_glacier_exists(self):
        """
        Checks there is still a glacier. If not, ends the run and
        saves the output.
        """
        # Load layer height
        layerheight = np.sum(self.layers.lheight)
        if layerheight < 2: # Cuts off at 2 meters
            # New end date
            end = self.time
            start = self.time_list[0]
            new_time = pd.date_range(start,end,freq='h')
            self.output.n_timesteps = len(new_time)

            # Load the output
            with xr.open_dataset(self.output.out_fn) as dataset:
                ds = dataset.load()
                # Chop it to the new end date
                ds = ds.sel(time=new_time)
            # Store output
            ds.to_netcdf(self.output.out_fn)

            # Save the data
            if self.args.store_data:
                self.output.store_data()
            print('Glacier disappeared for',self.args.out)
            
            # End the run
            self.exit(failed=False)
        return

    def exit(self,failed=True):
        """
        Exit function. Default usage sends an error message if
        debug is on. Otherwise, exits the run.
        """
        if self.args.debug and failed:
            print('Failed in mass balance')
            print('Current layers',self.ltype)
            print('Layer temp:',self.ltemp)
            print('Layer density:',self.ldensity)
        sys.exit()    

class Output():
    def __init__(self,time,args):
        """
        Creates netcdf file where the model output will be saved.

        Parameters
        ----------
        """
        # Get unique filename (or scratch filename)
        self.out_fn = eb_prms.output_filepath + args.out
        if eb_prms.new_file:
            i = 0
            while os.path.exists(self.out_fn+f'{i}.nc'):
                i += 1
            self.out_fn += str(i)
        else:
            self.out_fn = self.out_fn+'scratch'
        self.out_fn += '.nc'

        # Info needed to create the output file
        self.n_timesteps = len(time)
        zeros = np.zeros([self.n_timesteps,eb_prms.max_nlayers])

        # Create variable name dict
        vn_dict = {'EB':['SWin','SWout','LWin','LWout','rain','ground',
                         'sensible','latent','meltenergy','albedo','vis_albedo',
                         'SWin_sky','SWin_terr','surftemp'],
                   'MB':['melt','refreeze','runoff','accum','snowdepth','cumrefreeze','dh'],
                   'climate':['airtemp','wind'],
                   'layers':['layertemp','layerdensity','layerwater','layerheight',
                             'layerBC','layerOC','layerdust','layergrainsize','layerrefreeze']}

        # Create file to store outputs
        all_variables = xr.Dataset(data_vars = dict(
                SWin = (['time'],zeros[:,0],{'units':'W m-2'}),
                SWout = (['time'],zeros[:,0],{'units':'W m-2'}),
                SWin_sky = (['time'],zeros[:,0],{'units':'W m-2'}),
                SWin_terr = (['time'],zeros[:,0],{'units':'W m-2'}),
                LWin = (['time'],zeros[:,0],{'units':'W m-2'}),
                LWout = (['time'],zeros[:,0],{'units':'W m-2'}),
                rain = (['time'],zeros[:,0],{'units':'W m-2'}),
                ground = (['time'],zeros[:,0],{'units':'W m-2'}),
                sensible = (['time'],zeros[:,0],{'units':'W m-2'}),
                latent = (['time'],zeros[:,0],{'units':'W m-2'}),
                meltenergy = (['time'],zeros[:,0],{'units':'W m-2'}),
                albedo = (['time'],zeros[:,0],{'units':'0-1'}),
                vis_albedo = (['time'],zeros[:,0],{'units':'0-1'}),
                melt = (['time'],zeros[:,0],{'units':'m w.e.'}),
                refreeze = (['time'],zeros[:,0],{'units':'m w.e.'}),
                cumrefreeze = (['time'],zeros[:,0],{'units':'m w.e.'}),
                runoff = (['time'],zeros[:,0],{'units':'m w.e.'}),
                accum = (['time'],zeros[:,0],{'units':'m w.e.'}),
                airtemp = (['time'],zeros[:,0],{'units':'C'}),
                wind = (['time'],zeros[:,0],{'units':'m s-1'}),
                surftemp = (['time'],zeros[:,0],{'units':'C'}),
                layertemp = (['time','layer'],zeros,{'units':'C'}),
                layerwater = (['time','layer'],zeros,{'units':'kg m-2'}),
                layerrefreeze = (['time','layer'],zeros,{'units':'kg m-2'}),
                layerheight = (['time','layer'],zeros,{'units':'m'}),
                layerdensity = (['time','layer'],zeros,{'units':'kg m-3'}),
                layerBC = (['time','layer'],zeros,{'units':'ppb'}),
                layerOC = (['time','layer'],zeros,{'units':'ppb'}),
                layerdust = (['time','layer'],zeros,{'units':'ppm'}),
                layergrainsize = (['time','layer'],zeros,{'units':'um'}),
                snowdepth = (['time'],zeros[:,0],{'units':'m'}),
                dh = (['time'],zeros[:,0],{'units':'m'})
                ),
                coords=dict(
                    time=(['time'],time),
                    layer=(['layer'],np.arange(eb_prms.max_nlayers))
                    ))
        # Select variables from the specified input
        vars_list = vn_dict[eb_prms.store_vars[0]]
        for var in eb_prms.store_vars[1:]:
            assert var in vn_dict, 'Choose store_vars from [MB,EB,climate,layers]'
            vars_list.extend(vn_dict[var])
        self.vars_list = vars_list
        
        # Create the netcdf file to store output
        if args.store_data:
            all_variables[self.vars_list].to_netcdf(self.out_fn)

        # Initialize energy balance outputs
        self.SWin_output = []       # incoming shortwave [W m-2]
        self.SWout_output = []      # outgoing shortwave [W m-2]
        self.LWin_output = []       # incoming longwave [W m-2]
        self.LWout_output = []      # outgoing longwave [W m-2]
        self.SWin_sky_output = []   # incoming sky shortwave [W m-2]
        self.SWin_terr_output = []  # incoming terrain shortwave [W m-2]
        self.rain_output = []       # rain energy [W m-2]
        self.ground_output = []     # ground energy [W m-2]
        self.sensible_output = []   # sensible energy [W m-2]
        self.latent_output = []     # latent energy [W m-2]
        self.meltenergy_output = [] # melt energy [W m-2]
        self.albedo_output = []     # surface broadband albedo [-]
        self.vis_albedo_output = [] # surface visible albedo [-]

        # Initialize mass balance outputs
        self.snowdepth_output = []  # depth of snow [m]
        self.melt_output = []       # melt by timestep [m w.e.]
        self.refreeze_output = []   # refreeze by timestep [m w.e.]
        self.cumrefreeze_output = []   # cumulative refreeze by timestep [m w.e.]
        self.accum_output = []      # accumulation by timestep [m w.e.]
        self.runoff_output = []     # runoff by timestep [m w.e.]
        self.dh_output = []         # surface height change by timestep [m]
        self.airtemp_output = []    # downscaled air temperature [C]
        self.surftemp_output = []   # surface temperature [C]
        self.wind_output = []       # wind speed [m s-1]

        # Initialize layer outputs
        self.layertemp_output = dict()      # layer temperature [C]
        self.layerwater_output = dict()     # layer water content [kg m-2]
        self.layerdensity_output = dict()   # layer density [kg m-3]
        self.layerheight_output = dict()    # layer height [m]
        self.layerBC_output = dict()        # layer black carbon content [ppb]
        self.layerOC_output = dict()        # layer organic carbon content [ppb]
        self.layerdust_output = dict()      # layer dust content [ppm]
        self.layergrainsize_output = dict() # layer grain size [um]
        self.layerrefreeze_output = dict()  # layer refreeze [kg m-2]
        self.last_height = eb_prms.initial_ice_depth+eb_prms.initial_firn_depth+eb_prms.initial_snow_depth
        return
    
    def store_timestep(self,massbal,enbal,surface,layers,step):
        step = str(step)
        self.SWin_output.append(float(enbal.SWin))
        self.SWout_output.append(float(enbal.SWout))
        self.LWin_output.append(float(enbal.LWin))
        self.LWout_output.append(float(enbal.LWout))
        self.SWin_sky_output.append(float(enbal.SWin_sky))
        self.SWin_terr_output.append(float(enbal.SWin_terr))
        self.rain_output.append(float(enbal.rain))
        self.ground_output.append(float(enbal.ground))
        self.sensible_output.append(float(enbal.sens))
        self.latent_output.append(float(enbal.lat))
        self.meltenergy_output.append(float(surface.Qm))
        self.albedo_output.append(float(surface.bba))
        self.vis_albedo_output.append(float(surface.vis_a))
        self.surftemp_output.append(float(surface.stemp))

        self.melt_output.append(float(massbal.melt))
        self.refreeze_output.append(float(massbal.refreeze))
        self.cumrefreeze_output.append(float(np.sum(layers.cumrefreeze))/eb_prms.density_water)
        self.runoff_output.append(float(massbal.runoff))
        self.accum_output.append(float(massbal.accum))
        self.snowdepth_output.append(np.sum(layers.lheight[layers.snow_idx]))
        self.dh_output.append(np.sum(layers.lheight)-self.last_height)
        self.last_height = np.sum(layers.lheight)

        self.airtemp_output.append(float(enbal.tempC))
        self.wind_output.append(float(enbal.wind))

        self.layertemp_output[step] = layers.ltemp.copy()
        self.layerwater_output[step] = layers.lwater.copy()
        self.layerheight_output[step] = layers.lheight.copy()
        self.layerdensity_output[step] = layers.ldensity.copy()
        self.layerBC_output[step] = layers.lBC / layers.lheight * 1e6
        self.layerOC_output[step] = layers.lOC / layers.lheight * 1e6
        self.layerdust_output[step] = layers.ldust / layers.lheight * 1e3
        self.layergrainsize_output[step] = layers.lgrainsize.copy()
        self.layerrefreeze_output[step] = layers.cumrefreeze.copy()

    def store_data(self):
        # Load output dataset
        with xr.open_dataset(self.out_fn) as dataset:
            ds = dataset.load()
            # Store variables
            if 'EB' in eb_prms.store_vars:
                ds['SWin'].values = self.SWin_output
                ds['SWout'].values = self.SWout_output
                ds['LWin'].values = self.LWin_output
                ds['LWout'].values = self.LWout_output
                ds['SWin_sky'].values = self.SWin_sky_output
                ds['SWin_terr'].values = self.SWin_terr_output
                ds['rain'].values = self.rain_output
                ds['ground'].values = self.ground_output
                ds['sensible'].values = self.sensible_output
                ds['latent'].values = self.latent_output
                ds['meltenergy'].values = self.meltenergy_output
                ds['albedo'].values = self.albedo_output
                ds['vis_albedo'].values = self.vis_albedo_output
                ds['surftemp'].values = self.surftemp_output
            if 'MB' in eb_prms.store_vars:
                ds['melt'].values = self.melt_output
                ds['refreeze'].values = self.refreeze_output
                ds['runoff'].values = self.runoff_output
                ds['accum'].values = self.accum_output
                ds['snowdepth'].values = self.snowdepth_output
                ds['dh'].values = self.dh_output
                ds['cumrefreeze'].values = self.cumrefreeze_output
            if 'climate' in eb_prms.store_vars:
                ds['airtemp'].values = self.airtemp_output
                ds['wind'].values = self.wind_output
            if 'layers' in eb_prms.store_vars:
                layertemp_output = pd.DataFrame.from_dict(self.layertemp_output,orient='index')
                layerdensity_output = pd.DataFrame.from_dict(self.layerdensity_output,orient='index')
                layerheight_output = pd.DataFrame.from_dict(self.layerheight_output,orient='index')
                layerwater_output = pd.DataFrame.from_dict(self.layerwater_output,orient='index')
                layerBC_output = pd.DataFrame.from_dict(self.layerBC_output,orient='index')
                layerOC_output = pd.DataFrame.from_dict(self.layerOC_output,orient='index')
                layerdust_output = pd.DataFrame.from_dict(self.layerdust_output,orient='index')
                layergrainsize_output = pd.DataFrame.from_dict(self.layergrainsize_output,orient='index')
                layerrefreeze_output = pd.DataFrame.from_dict(self.layerrefreeze_output,orient='index')
                
                if len(layertemp_output.columns) < eb_prms.max_nlayers:
                    n_columns = len(layertemp_output.columns)
                    for i in range(n_columns,eb_prms.max_nlayers):
                        nans = np.zeros(self.n_timesteps)*np.nan
                        layertemp_output[str(i)] = nans
                        layerdensity_output[str(i)] = nans
                        layerheight_output[str(i)] = nans
                        layerwater_output[str(i)] = nans
                        layerBC_output[str(i)] = nans
                        layerOC_output[str(i)] = nans
                        layerdust_output[str(i)] = nans
                        layergrainsize_output[str(i)] = nans
                        layerrefreeze_output[str(i)] = nans
                else:
                    n = len(layertemp_output.columns)
                    print(f'Need to increase max_nlayers: currently have {n} layers')
                    self.exit()

                ds['layertemp'].values = layertemp_output
                ds['layerheight'].values = layerheight_output
                ds['layerdensity'].values = layerdensity_output
                ds['layerwater'].values = layerwater_output
                ds['layerBC'].values = layerBC_output
                ds['layerOC'].values = layerOC_output
                ds['layerdust'].values = layerdust_output
                ds['layergrainsize'].values = layergrainsize_output
                ds['layerrefreeze'].values = layerrefreeze_output

        # Save NetCDF
        ds.to_netcdf(self.out_fn)
        return ds
    
    def add_vars(self):
        """
        Adds additional variables to the output dataset.
        
        SWnet: net shortwave radiation flux [W m-2]
        LWnet: net longwave radiation flux [W m-2]
        NetRad: net radiation flux (SW and LW) [W m-2]
        """
        if 'SWin' in self.vars_list:
            with xr.open_dataset(self.out_fn) as dataset:
                ds = dataset.load()

                # Add summed radiation terms
                SWnet = ds['SWin'] + ds['SWout']
                LWnet = ds['LWin'] + ds['LWout']
                NetRad = SWnet + LWnet
                ds['SWnet'] = (['time'],SWnet.values,{'units':'W m-2'})
                ds['LWnet'] = (['time'],LWnet.values,{'units':'W m-2'})
                ds['NetRad'] = (['time'],NetRad.values,{'units':'W m-2'})

                # Add summed mass balance term
                MB = ds['accum'] + ds['refreeze'] - ds['melt']
                ds['MB'] = (['time'],MB.values,{'units':'m w.e.'})

            # Save NetCDF 
            ds.to_netcdf(self.out_fn)
        return
    
    def add_basic_attrs(self,args,time_elapsed,climate):
        """
        Adds informational attributes to the output dataset.

        time_elapsed
        run_start and run_end
        from_AWS and from_reanalysis list of variables
        elevation
        model_run_date
        switch_melt, switch_LAPs, switch_snow
        """
        time_elapsed = str(time_elapsed) + ' s'
        elev = str(args.elev)+' m a.s.l.'

        # get information on variable sources
        re_str = eb_prms.reanalysis+': '
        props = eb_prms.glac_props[eb_prms.glac_no[0]]
        if args.use_AWS:
            measured = climate.measured_vars
            AWS_name = props['name']
            AWS_elev = climate.AWS_elev
            AWS_str = f'{AWS_name} {AWS_elev}: '
            AWS_str += ', '.join(measured)
            re_vars = [e for e in climate.all_vars if e not in measured]
            re_str += ', '.join(re_vars)
        else:
            re_str += 'all'
            AWS_str = 'none'
        
        with xr.open_dataset(self.out_fn) as dataset:
            ds = dataset.load()
            ds = ds.assign_attrs(from_AWS=AWS_str,
                                 from_reanalysis=re_str,
                                 run_start=str(args.startdate),
                                 run_end=str(args.enddate),
                                 elevation=elev,
                                 site=str(args.site),
                                 model_run_date=str(pd.Timestamp.today()),
                                 switch_melt=str(args.switch_melt),
                                 switch_snow=str(args.switch_snow),
                                 switch_LAPs=str(args.switch_LAPs),
                                 time_elapsed=time_elapsed,
                                 run_by=eb_prms.machine)
            if len(args.glac_no) > 1:
                reg = args.glac_no[0][0:2]
                ds = ds.assign_attrs(glacier=f'{len(args.glac_no)} glaciers in region {reg}')
            else:
                ds = ds.assign_attrs(glacier=eb_prms.glac_name)

            if args.task_id != -1:
                ds.assign_attrs(task_id=str(args.task_id))

        # Save NetCDF
        ds.to_netcdf(self.out_fn)
        print('Success: saved to '+self.out_fn)
        return
    
    def add_attrs(self,new_attrs):
        """
        Adds new attributes as a dict to the output dataset
        """
        with xr.open_dataset(self.out_fn) as dataset:
            ds = dataset.load()
            if not new_attrs:
                return ds
            ds = ds.assign_attrs(new_attrs)
        ds.to_netcdf(self.out_fn)
        return
    
    def get_output(self):
        return xr.open_dataset(self.out_fn)