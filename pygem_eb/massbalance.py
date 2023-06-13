import numpy as np
import pandas as pd
import xarray as xr
import os
from scipy.optimize import minimize
import pygem_eb.input as eb_prms
import pygem_eb.energybalance as eb
import pygem_eb.layers as eb_layers
import pygem_eb.albedo as eb_albedo

class massBalance():
    def __init__(self,climateds):
        # Set up model time (dt is the time loop used for the mass + surface energy balances)
        self.dt = eb_prms.dt
        self.time_list = pd.date_range(eb_prms.startdate,eb_prms.enddate,freq=str(self.dt)+'S')

        # Time for storing data
        self.time_to_store = pd.date_range(eb_prms.startdate,eb_prms.enddate,freq=eb_prms.storage_freq)

        self.days_since_snowfall = 0
    
        # Initialize variables to store fluxes
        self.SWin_output = []
        self.SWout_output = []
        self.LWin_output = []
        self.LWout_output = []
        self.rain_output = []
        self.sensible_output = []
        self.latent_output = []
        self.meltenergy_output = []
        self.melt_output = []
        self.refreeze_output = []
        self.runoff_output = []
        self.accum_output = []
        self.airtemp_output = []
        self.surftemp_output = []
        self.snowtemp_output = []
        self.snowwetmass_output = []
        self.snowdrymass_output = []
        self.snowdensity_output = []
        return
    
    def main(self,layers,climateds,bin_idx):
        """
        Runs the time loop and mass balance scheme to solve layer temperature
        and density profiles. 

        Parameters
        ----------
        climateds : xr.Dataset
            Climate dataset containing temperature, precipitation, pressure, air density, wind speed,
            shortwave radiation, and total cloud cover.
        bin_idx : int
            Index number of the bin being run.
        """
        # Initialize surface class for albedo / LAPs
        surface = eb_albedo.Surface(layers,self.time_list)

        # Initiate variables for loop
        dt = self.dt
        month = 0

        # Store mass balance terms for each month and for each timestep
        startdate = eb_prms.startdate
        enddate = eb_prms.enddate
        n_months = (enddate.year - startdate.year) * 12 + startdate.month - enddate.month
        self.monthly_output = pd.DataFrame(data=np.zeros((n_months,5)),
                                           columns=['melt','runoff','refreeze','accum','MB'],
                                           index=np.arange(n_months))
        # initialize space for "running" MB terms which sum each timestep -- reset to 0 monthly
        running_output = pd.DataFrame([[0],[0],[0],[0]],index=['melt','runoff','refreeze','accum'])    

        # ===== ENTER TIME LOOP =====
        # index [12960:12964]] will start on a summer day (June 29)
        # index 26281 ends the first year (1980)
        for time in self.time_list:
            # Initiate the energy balance to unpack climate data
            enbal = eb.energyBalance(climateds,time,bin_idx,dt)

            # Check if snowfall or rain occurred and update snow timestamp
            rain,snowfall = self.getPrecip(enbal,surface,time)

            if time.hour < 1 and time.minute < 1:
                # any daily happenings go here!!!
                self.days_since_snowfall = (time - surface.snow_timestamp)/pd.Timedelta(days=1)
                #update albedo
                surface.getAlbedo(self.days_since_snowfall)

            # Add fresh snow to layers
            if snowfall > 0:
                layers.addSnow(snowfall,enbal.tempC)
            # Calculate surface energy balance 
            surface.Qm = enbal.surfaceEB(0,layers,surface,self.days_since_snowfall)

            # Calculate subsurface heating/melt from penetrating SW
            if layers.nlayers > 1: # unnecessary to run these fxns if there is bare ice
                Sin,Sout = enbal.getSW(surface.albedo)
                subsurf_melt = self.getSubsurfMelt(layers,Sin+Sout)
                
                # Recalculate surface temperature if freezing (Qm<0) or melting and the surface is subzero (Qm>0,Ts<0)
                self.getSubzero(enbal,layers,surface)
            else: # if just one layer
                subsurf_melt = [0]

            # Calculate melt when there is energy toward the surface and the surface is at 0C
            if surface.Qm > 0 and surface.temp >= 0: # make this ==? should never exceed 0 via above code
                layermelt = self.getMelt(layers,surface,subsurf_melt)
            else:
                layermelt = subsurf_melt.copy()
                layermelt[0] = 0

            # Percolate the meltwater and any liquid precipitation
            runoff,layers_to_remove = self.percolate(layers,layermelt,rain)

            # Remove layers that were completely melted and update layer heights
            for layer in layers_to_remove:
                layers.removeLayer(layer)
            layers.updateLayers()

            # Recalculate the temperature profile considering conduction, needs at least 3 layers
            Tprofile_new = self.solveHeatEq(layers,surface.temp,eb_prms.dt_heateq)
            layers.Tprofile = Tprofile_new

            # Calculate refreeze
            refreeze = self.refreeze(layers)

            # Store running (monthly) values ***** these are the output but aren't currently being used
            running_output.loc['runoff'] += runoff
            running_output.loc['melt'] += np.sum(layermelt)
            running_output.loc['refreeze'] += refreeze
            running_output.loc['accum'] += snowfall

            # Store data
            if time in self.time_to_store:
                self.SWin_output.append(enbal.SWin)
                self.SWout_output.append(enbal.SWout)
                self.LWin_output.append(enbal.LWin)
                self.LWout_output.append(enbal.LWout)
                self.rain_output.append(enbal.rain)
                self.sensible_output.append(enbal.sens)
                self.latent_output.append(enbal.lat)
                self.meltenergy_output.append(surface.Qm)
                self.melt_output.append(np.sum(layermelt))
                self.refreeze_output.append(refreeze)
                self.runoff_output.append(runoff)
                self.accum_output.append(snowfall)
                self.airtemp_output.append(enbal.tempC)
                self.surftemp_output.append(surface.temp)

                snowtemp = [None]*eb_prms.max_nlayers
                snowwetmass = snowtemp.copy()
                snowdrymass = snowtemp.copy()
                snowdensity = snowtemp.copy()
                snowtemp[eb_prms.max_nlayers-layers.nlayers:] = layers.Tprofile
                snowwetmass[eb_prms.max_nlayers-layers.nlayers:] = layers.wet_mass
                snowdrymass[eb_prms.max_nlayers-layers.nlayers:] = layers.dry_mass
                snowdensity[eb_prms.max_nlayers-layers.nlayers:] = layers.density
                self.snowtemp_output.append(snowtemp)
                self.snowwetmass_output.append(snowwetmass)
                self.snowdrymass_output.append(snowdrymass)
                self.snowdensity_output.append(snowdensity)

            # Monthly time check
            if (time+pd.Timedelta(hours=1)).is_month_start and time.hour == 23 and time.minute == 0:
                # monthly prints
                if eb_prms.debug:
                    melt = running_output.loc['melt'][0]
                    accum = running_output.loc['accum'][0]
                    melte = np.mean(self.meltenergy_output[-720:])
                    print(time.month_name(),time.year,'for bin no',bin_idx)
                    print(f'|    Qm: {melte:.0f} W/m2              Melt: {melt:.2f} kg/m2  |')
                    print(f'| Air temp: {enbal.tempC:.3f} C       Accum: {accum:.2f} kg/m2   |')
                    if layers.nlayers > 3:
                        print(f'|    {layers.nlayers} layers total, only displaying top few     |')
                    print(f'-------------surface temp: {surface.temp:.2f} C-------------')
                    for l in range(min(3,layers.nlayers)):
                        print(f'--------------------layer {l}---------------------')
                        print(f'     T = {layers.Tprofile[l]:.1f}                  h = {layers.heights[l]:.3f} m     ')
                        print(f'                 p = {layers.density[l]:.0f} kg/m3')
                        print(f'Water Mass : {layers.watercont[l]:.2f} kg/m2   Dry Mass : {layers.dry_mass[l]:.2f} kg/m2')
                    print('================================================')
                    #print('Melt:',running_output.loc['melt'][0],' New snow:',running_output.loc['accum'][0])

                # any monthly happenings go here!!
                self.getMassBal(running_output,surface.temp,enbal,month)
                month += 1
                running_output[0] = [0,0,0,0] 

                if time.month == 10:
                    print('Update glacier geometry!')

        return

    def getSubsurfMelt(self,layers,Snet_surf):
        """
        Calculates melt in subsurface layers (excluding layer 0) due to penetrating shortwave radiation.

        Parameters
        ----------
        layers
            class object from pygem_eb.layers.py
        Snet_surf : float
            Incoming SW radiation [W m-2]
        Returns
        -------
        layermelt : np.ndarray
            Array containing subsurface melt amounts [kg m-2]
        """
        # Fraction of radiation absorbed at the surface depends on surface type
        if layers.types[0] in ['snow']:
            frac_absrad = 0.9
        else:
            frac_absrad = 0.8

        # Extinction coefficient depends on layer type
        extinct_coef = np.ones(layers.nlayers)*1e8 # ensures unfilled layers have 0 heat
        for idx,type in enumerate(layers.types):
            if type in ['snow']:
                extinct_coef[idx] = 17.1
            else:
                extinct_coef[idx] = 2.5
            # Cut off if the flux reaches zero threshold (1e-6)
            if np.exp(-extinct_coef[idx]*layers.depths[idx]) < 1e-6:
                break
        Snet_pen = Snet_surf*(1-frac_absrad)*np.exp(-extinct_coef*layers.depths)/self.dt

        # recalculate layer temperatures, leaving out the surface since surface temp is calculated separately
        new_Tprofile = layers.Tprofile.copy()
        new_Tprofile[1:] = layers.Tprofile[1:] + Snet_pen[1:]/(layers.dry_mass[1:]*eb_prms.Cp_ice)*self.dt

        # calculate melt from temperatures above 0
        layermelt = np.zeros(layers.nlayers)
        for layer,new_T in enumerate(new_Tprofile):
            # check if temperature is above 0
            if new_T > 0:
                # calculate melt from the energy that raised the temperature above 0
                melt = (new_T-0)*layers.dry_mass[layer]*eb_prms.Cp_ice/eb_prms.Lh_rf
                layers.Tprofile[layer] = 0
            else:
                melt = 0
                layers.Tprofile[layer] = new_T
            layermelt[layer] = melt

        return layermelt
    
    def getSubzero(self,enbal,layers,surface):
        """
        For cases when the ice temperature is below freezing, recalculates melt energy and surface 
        temperature.

        Parameters
        ----------
        enbal
            class object from pygem_eb.energybalance
        layers
            class object from pygem_eb.layers
        surface
            class object from pygem_eb.albedo
        Returns
        -------
        Qm : float
            Updated energy available for melt [W m-2]
        surftemp : float
            Updated skin layer temperature [C]
        """
        surftemp = surface.temp
        Qm = surface.Qm
        if Qm < 0:
            # Energy away from surface: need to change surface temperature to get 0 surface energy flux 
            result = minimize(enbal.surfaceEB,surftemp,method='L-BFGS-B',bounds=((-60,0),),tol=1e-3,
                            args=(layers,surface,self.days_since_snowfall,'optim'))
            surftemp = result.x[0]
            if result.fun > 1:
                # failing to converge, so put extra cold content in upper layer by lowering T
                layers.Tprofile[0] -= result.fun*self.dt/layers.dry_mass[0]/eb_prms.Cp_ice
            Qm = 0
        elif Qm > 0 and layers.Tprofile[0] < 0:
            # Energy toward surface: need to heat top layer up
            layers.Tprofile[0] += Qm*self.dt/(eb_prms.Cp_ice*layers.dry_mass[0])
            if layers.Tprofile[0] > 0:
                # if temperature rises above zero, leave excess energy in Qm
                Qm = layers.Tprofile[0]*eb_prms.Cp_ice*layers.dry_mass[0]/self.dt
                layers.Tprofile[0] = 0
            else:
                Qm = 0

            # Interpolate for new surface temperature, which cannot exceed 0C
            xp = layers.depths[:2].astype(float)
            fp = layers.Tprofile[:2].astype(float)
            surftemp = np.interp(0,xp,fp) if layers.nlayers > 1 else 0
            if surftemp > 0:
                surftemp = 0
        elif Qm > 0: # Energy toward surface, first layer at melting point
            surftemp = 0

        surface.Qm = Qm
        surface.temp = surftemp
        return

    def getMelt(self,layers,surface,subsurf_melt):
        """
        For cases when bins are melting. Can melt multiple surface bins at once if Qm is
        sufficiently high. Otherwise, adds the surface layer melt to the array containing
        subsurface melt to return the total layer melt.

        Parameters
        ----------
        layers
            class object from pygem_eb.layers
        Qm : float
            Input energy available for melt (to be adjusted) [W m-2]
        surftemp : float
            Input skin layer temperature (to be adjusted) [C]
        subsurf_melt : np.ndarray
            Array containing melt of subsurface layers [kg m-2]
        Returns
        -------
        layermelt : np.ndarray
            Array containing melt amount [kg m-2]
        
        """
        layermelt = subsurf_melt.copy()
        # Energy toward surface: need to calculate surface melt
        surface_melt = surface.Qm*self.dt/eb_prms.Lh_rf         # melt in kg/m2
        if surface_melt > layers.dry_mass[0]:
            # melt by surface energy balance completely melts surface layer, so check if it melts further layers
            fully_melted = np.where(np.array([np.sum(layers.dry_mass[:i+1]) for i in range(layers.nlayers)]) <= surface_melt)[0]

            # calculate how much additional melt will occur in the first layer that's not fully melted
            newsurface_melt = surface_melt - np.sum(layers.dry_mass[fully_melted])
            newsurface_idx = fully_melted[-1] + 1
            # it's possible to fully melt that layer too when combined with penetrating SW melt:
            if newsurface_melt + layermelt[newsurface_idx] > layers.dry_mass[newsurface_idx]:
                fully_melted = np.append(fully_melted,newsurface_idx)
                # push new surface to the next layer down
                newsurface_melt -= layers.dry_mass[newsurface_idx]
                newsurface_idx += 1

            # set melt amounts from surface melt into melt array
            layermelt[fully_melted] = layers.dry_mass[fully_melted] 
            layermelt[newsurface_idx] += newsurface_melt 
        else:
            # only surface layer is melting
            layermelt[0] = surface_melt
        return layermelt
        
    def percolate(self,layers,layermelt,extra_water=0):
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
        melted_layers : list
            List of layer indices that were fully melted
        """
        melted_layers = []
        for layer,melt in enumerate(layermelt):
            # check if the layer fully melted
            if melt >= layers.dry_mass[layer]:
                melted_layers.append(layer)
                # pass the meltwater to the next layer
                extra_water += melt+layers.watercont[layer]
            else:
                # remove melt from the dry mass
                layers.dry_mass[layer] -= melt

                # add melt and extra_water (melt from above) to layer water content
                added_water = melt + extra_water
                layers.watercont[layer] += added_water

                # check if meltwater exceeds the irreducible water content of the layer
                layers.updateLayerProperties('irrwater')
                if layers.watercont[layer] >= layers.irrwatercont[layer]:
                    # set water content to irr. water content and add the difference to extra_water
                    extra_water = layers.watercont[layer] - layers.irrwatercont[layer]
                    layers.watercont[layer] = layers.irrwatercont[layer]
                else: #if not overflowing, extra_water should be set back to 0
                    extra_water = 0
                
                # get the change in layer height due to loss of solid mass (volume only considers solid)
                layers.heights[layer] -= melt/layers.density[layer]
                # need to update layer depths
                layers.updateLayerProperties() 

        # extra water goes to runoff
        runoff = extra_water

        return runoff,melted_layers

    def refreeze(self,layers):
        """
        Calculates refreeze in layers due to temperatures below freezing with liquid water content.

        Parameters:
        -----------
        layers
            class object from pygem_eb.layers
        Returns:
        --------
        refreeze : float
            Total amount of refreeze [kg m-2]
        """
        Cp_ice = eb_prms.Cp_ice
        density_ice = eb_prms.density_ice
        Lh_rf = eb_prms.Lh_rf
        refreeze = 0
        for layer, T in enumerate(layers.Tprofile):
            if T < 0 and layers.watercont[layer] > 0:
                # calculate potential for refreeze [J m-2]
                E_temperature = np.abs(T)*layers.dry_mass[layer]*Cp_ice  # cold content available 
                E_water = layers.watercont[layer]*Lh_rf  # amount of water to freeze
                E_pore = (density_ice*layers.heights[layer]-layers.dry_mass[layer])*Lh_rf # pore space available

                # calculate amount of refreeze 
                dm_ref = max(0,np.min([E_temperature,E_water,E_pore])/Lh_rf) # cannot be negative

                # add refreeze to running sum
                refreeze += dm_ref

                # add refreeze to layer ice mass
                layers.dry_mass[layer] += dm_ref
                # update layer temperature from latent heat
                layers.Tprofile[layer] = -(E_temperature-dm_ref*Lh_rf)/Cp_ice/layers.dry_mass[layer]
                # update water content
                layers.watercont[layer] = max(0,layers.watercont[layer]-dm_ref) # cannot be negative
                # recalculate layer heights from new mass and update layers
                layers.heights[layer] = layers.dry_mass[layer]/layers.density[layer]
                layers.updateLayerProperties()
        return refreeze
    
    def densification(self,layers):
        # update density of lower layers based on age?
        # use new density to recalculate layer thicknesses
        for layer,height in enumerate(layers.heights):
            weight_above = eb_prms.gravity*np.sum(layers.density[:layer]*layers.heights[layer])
            assert abs(weight_above - np.sum(layers.dry_mass)*eb_prms.gravity) > 1e-2, 'Mass and densities dont match!!!'
            dD = -height*weight_above/eb_prms.viscosity_snow*self.dt
            layers.heights[layer] += dD
            layers.density[layer] = layers.dry_mass[layer] / layers.heights[layer]

            if layers.dry_mass[layer] / layers.heights[layer] > eb_prms.density_firn: # if dry density exceeds that of ice
                layers.density[layer] = eb_prms.density_firn
                layers.types[layer] = 'firn'
                print('densified firn!')
            if layers.dry_mass[layer] / layers.heights[layer] > eb_prms.density_ice: # if dry density exceeds that of ice
                layers.density[layer] = eb_prms.density_ice 
                layers.types[layer] = 'ice'
                print('densified ice!')

        return
    
    def getPrecip(self,enbal,surface,time):
        if enbal.prec > 1e-8 and enbal.tempC <= eb_prms.tsnow_threshold: 
            # there is precipitation and it falls as snow--set fresh snow timestamp
            surface.snow_timestamp = time
            rain = 0
            density_fresh_snow = max(109*6*(enbal.tempC-0)+26*enbal.wind**0.5,50) # from CROCUS ***** CITE
            snow = enbal.prec*density_fresh_snow*self.dt # kg m-2
            precip_type = 'snow'
        elif enbal.tempC > eb_prms.tsnow_threshold:
            # precipitation falls as rain
            rain = enbal.prec*eb_prms.density_water*self.dt  # kg m-2
            snow = 0
            precip_type = 'rain'
        else:
            precip_type = 'none'
            rain = 0
            snow = 0
        surface.updatePrecip(precip_type,rain+snow)
        return rain,snow
    
    def getMassBal(self,running_values,surftemp,enbal,month):
        if surftemp < 0:
            sublimation = min(enbal.lat/(eb_prms.density_water * eb_prms.Lv_sub), 0)*self.dt
            deposition = max(enbal.lat/(eb_prms.density_water * eb_prms.Lv_sub), 0)*self.dt
            evaporation = 0
            condensation = 0
        else:
            sublimation = 0
            deposition = 0
            evaporation = min(enbal.lat/(eb_prms.density_water * eb_prms.Lv_evap), 0)*self.dt
            condensation = max(enbal.lat/(eb_prms.density_water * eb_prms.Lv_evap), 0)*self.dt
        melt,runoff,refreeze,accum = running_values.loc[['melt','runoff','refreeze','accum']][0]
        self.monthly_output.loc[month] = [melt,runoff,refreeze,accum,0]

        # calculate total mass balance
        MB = accum + refreeze - melt + deposition - evaporation - sublimation
        self.monthly_output.loc[month]['MB'] = MB
        return
    
    def storeVars(self,bin):
        with xr.open_dataset(eb_prms.output_name+'.nc') as dataset:
            ds = dataset.load()
            ds['SWin'].loc[:,bin] = self.SWin_output
            ds['SWout'].loc[:,bin] = self.SWout_output
            ds['LWin'].loc[:,bin] = self.LWin_output
            ds['LWout'].loc[:,bin] = self.LWout_output
            ds['rain'].loc[:,bin] = self.rain_output
            ds['sensible'].loc[:,bin] = self.sensible_output
            ds['latent'].loc[:,bin] = self.latent_output
            ds['meltenergy'].loc[:,bin] = self.meltenergy_output
            ds['melt'].loc[:,bin] = self.melt_output
            ds['refreeze'].loc[:,bin] = self.refreeze_output
            ds['runoff'].loc[:,bin] = self.runoff_output
            ds['airtemp'].loc[:,bin] = self.airtemp_output
            ds['surftemp'].loc[:,bin] = self.surftemp_output
            ds['snowtemp'].loc[:,bin,:] = self.snowtemp_output
            ds['snowwetmass'].loc[:,bin,:] = self.snowwetmass_output
            ds['snowdrymass'].loc[:,bin,:] = self.snowdrymass_output
            ds['snowdensity'].loc[:,bin,:] = self.snowdensity_output
        ds.to_netcdf(eb_prms.output_name+'.nc')
        return
    
    def solveHeatEq(self,layers,surftemp,dt_heat):
        nl = layers.nlayers
        height = layers.heights
        density = layers.density
        old_T = layers.Tprofile
        new_T = old_T.copy()
        if nl > 1:
            if np.max(density[:-1]) >= 900 or np.min(density) < 0:
                print(density)
                print('Height',height)
                print('Dry mass',layers.dry_mass)
                print('Water',layers.watercont)
                print('Wet mass',layers.wet_mass)
        conductivity = 2.2*np.power(density/eb_prms.density_ice,1.88)
        Cp_ice = eb_prms.Cp_ice

        # set boundary conditions
        new_T[-1] = old_T[-1] # ice is ALWAYS isothermal*****

        if nl > 2:
            # heights of imaginary average bins between layers
            up_height = np.array([np.mean(height[i:i+2]) for i in range(nl-2)])  # upper layer 
            dn_height = np.array([np.mean(height[i:i+2]) for i in range(nl-2)])  # lower layer

            # conductivity
            up_cond = np.array([np.mean(conductivity[i:i+2]*height[i:i+2]) for i in range(nl-2)]) / up_height
            dn_cond = np.array([np.mean(conductivity[i+1:i+3]*height[i+1:i+3]) for i in range(nl-2)]) / dn_height

            # density
            up_dens = np.array([np.mean(density[i:i+2]) for i in range(nl-2)]) / up_height
            dn_dens = np.array([np.mean(density[i+1:i+3]) for i in range(nl-2)]) / dn_height

            # find temperature of top layer from surftemp boundary condition
            surf_cond = up_cond[0]*2/(up_dens[0]*up_height[0])*(surftemp-old_T[0])
            subsurf_cond = dn_cond[0]/(up_dens[0]*up_height[0])*(old_T[0]-old_T[1])
            new_T[0] = old_T[0] + dt_heat/(Cp_ice*height[0])*(surf_cond - subsurf_cond)

            surf_cond = up_cond[0]/(up_dens[0]*up_height[0])*(old_T[:-2]-old_T[1:-1])
            subsurf_cond = dn_cond[0]/(dn_dens[0]*dn_height[0])*(old_T[1:-1]-old_T[2:])
            new_T[1:-1] = old_T[1:-1] + dt_heat/(Cp_ice*height[1:-1])*(surf_cond - subsurf_cond)

        elif nl > 1:
            if surftemp < 0:
                new_T = np.array([surftemp/2,0])
            else:
                new_T = np.array([0,0])
        else:
            new_T[0] = 0
         
        return new_T
        # COPIED FROM 1D SEB
        # # boundary condition bottom: zero flux
        # Tnew[-1] = Told[-1]
        # # boundary condition surface: Tsurf used to calculate first level temperature
        # Tnew[0] = Told[0] + dt_heat / (cpice * dz[0]) * (conduct_up[0] / (dens_up[0] * dz[0] / 2) * (Tsurf - Told[0]) - conduct_dn[0] 
        #                                             / (dens_up[0] * dz_up[0]) * (Told[0] - Told[1]))
            # Tnew[1:-1] = Told[1:-1] + dt_heat / (cpice * dz[1:-1]) * (conduct_up / (dens_up * dz_up) * (Told[:-2] - Told[1:-1]) - conduct_dn 
            #                                                  / (dens_dn * dz_dn) * (Told[1:-1] - Told[2:]))
        # heat equation
        

        # # COPIED DIRECTLY FROM COSIPY
        # # number of layers
        # nl = layers.nlayers

        # # Define index arrays 
        # k   = np.arange(1,nl-1) # center points
        # kl  = np.arange(2,nl)   # lower points
        # ku  = np.arange(0,nl-2) # upper points
        
        # # Get thermal diffusivity [m2 s-1]
        # K = 2.2*np.power(layers.density/eb_prms.density_ice,1.88)
        
        # # Get snow layer heights    
        # hlayers = layers.heights

        # # Get grid spacing
        # diff = ((hlayers[0:nl-1]/2.0)+(hlayers[1:nl]/2.0))
        # hk = diff[0:nl-2]  # between z-1 and z
        # hk1 = diff[1:nl-1] # between z and z+1
        
        # # Get temperature array from grid
        # T = layers.Tprofile
        # Tnew = T.copy()
        
        # Kl = (K[1:nl-1]+K[2:nl])/2.0
        # Ku = (K[0:nl-2]+K[1:nl-1])/2.0
        
        # stab_t = 0.0
        # c_stab = 0.8
        # dt_stab  = c_stab * (min([min(diff[0:nl-2]**2/(2*Ku)),min(diff[1:nl-1]**2/(2*Kl))]))
        
        # n_iters = 0
        # dt = 100 # ******
        # while stab_t < dt:
        #     dt_use = np.minimum(dt_stab, dt-stab_t)
        #     stab_t = stab_t + dt_use

        #     # Update the temperatures
        #     Tnew[k] += ((Kl*dt_use*(T[kl]-T[k])/(hk1)) - (Ku*dt_use*(T[k]-T[ku])/(hk))) / (0.5*(hk+hk1))
        #     T = Tnew.copy()
        #     n_iters += 1
        # #print(n_iters)
        # return T



# def CrankNicholson(self,i,C,T,Ts,T_past,Ts_past):
#     """
#     Solves the heat equation using the Crank-Nicholson scheme to recalculate snowpack temperatures.

#     Parameters
#     ----------
#     i : int
#         Index for the timestep
#     C : np.ndarray
#         Crank-Nicholson constant
#     T : np.ndarray
#         Current version of temperature profile
#     Ts : float
#         Current version of surface temperature
#     T_past : np.ndarray
#         Temperature profile of the previous timestep
#     Ts_past : float
#         Surface temperature of the previous timestep
#     """
#     a_Crank = np.zeros((self.nlayers))
#     b_Crank = np.zeros((self.nlayers))
#     c_Crank = np.zeros((self.nlayers))
#     d_Crank = np.zeros((self.nlayers))
#     A_Crank = np.zeros((self.nlayers))
#     S_Crank = np.zeros((self.nlayers))
#     T_new = np.zeros((self.nlayers))
#     if i < 1:
#         # First timestep requires no adjustment, just use the initial conditions
#         T_new = self.Tprofile
#     else:
#         for j in range(0,self.nlayers):
#             a_Crank[j] = C
#             b_Crank[j] = 2*C+1
#             c_Crank[j] = C

#             if j == 0:
#                 d_Crank[j] = C*Ts + C*Ts_past + (1-2*C)*T_past[j] + C*T_past[j+1]
#             elif j < self.nlayers-1:
#                 d_Crank[j] = C*T_past[j-1] + (1-2*C)*T_past[j] + C*T_past[j+1]
#             else:
#                 d_Crank[j] = 2*C*eb_prms.temp_temp + C*T_past[j-1] + (1-2*C)*T_past[j]

#             if j == 0:
#                 A_Crank[j] = b_Crank[j]
#                 S_Crank[j] = d_Crank[j]
#             else:
#                 A_Crank[j] = b_Crank[j] - a_Crank[j]/A_Crank[j-1] * c_Crank[j-1]
#                 S_Crank[j] = d_Crank[j] + a_Crank[j]/A_Crank[j-1] * S_Crank[j-1]
#         for j in range(self.nlayers - 1,0,-1):
#             if j == self.nlayers-1:
#                 T_new[j] = S_Crank[j]/A_Crank[j]
#             else:
#                 T_new[j] = 1/A_Crank[j] * (S_Crank[j]+c_Crank[j]*T_new[j+1])
#     return T_new

# def solveHeat(self,K,surftemp,dt,Qm):
#     """
#     Recalculate temperature profile by brute force method like DEBAM
#     """
#     conduction = []
#     Tprofile_new = []


#     for layer in range(self.nlayers):
#         if layer == 0:
#             conduction.append(K[0]*self.Tprofile[0] - surftemp/self.layerh[0])
#         else:
#             dzl = 0.5*(self.layerh[layer]+self.layerh[layer-1])
#             layerconduct = 0.5/dzl*(K[layer]*self.layerh[layer]+K[layer-1]*self.layerh[layer-1])*(self.Tprofile[layer]-self.Tprofile[layer-1])/dzl
#             conduction.append(layerconduct)
#     for layer in range(self.nlayers-1):
#         if layer == 0:
#             dT = dt*2/eb_prms.Cp_ice/(self.density[0]+self.density[1])*(conduction[1]-Qm)/self.layerh[0]
#         else:
#             dT = dt*2/eb_prms.Cp_ice/(self.density[layer]+self.density[layer+1])*(conduction[layer+1]-conduction[layer])/self.layerh[layer]
#         Tprofile_new.append(self.Tprofile[layer] + dT)
#     return Tprofile_new
