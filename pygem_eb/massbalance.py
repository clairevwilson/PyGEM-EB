import numpy as np
import pandas as pd
from scipy.optimize import minimize
import pygem_eb.input as eb_prms
import pygem_eb.energybalance as eb
import pygem_eb.layers as eb_layers
import pygem_eb.albedo as eb_albedo
    
def main(layers,climateds,bin_idx,dt):
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
    if layers.types[0] in ['snow']:
        albedo = eb_prms.albedo_fresh_snow
        snow_timestamp = time_dt[0]
    elif layers.types[0] in ['firn']:
        albedo = eb_prms.albedo_firn
    elif layers.types[0] in ['ice']:
        albedo = eb_prms.albedo_ice

    # Initiate time loop
    time_idx = 0
    surftemp = 0 # initial guess, will be solved iteratively

    # Place to store melt for each month and for each timestep
    monthly_melt = []
    monthly_refreeze = []
    monthly_runoff = []
    running_melt = 0
    running_refreeze = 0
    running_runoff = 0

    # ===== ENTER TIME LOOP =====
    # index [12960:12964]] will start on a summer day (June 29)
    for time in time_dt[:26281]:
        # Initiate the energy balance to unpack climate data
        enbal = eb.energyBalance(climateds,time,bin_idx,dt)

        # Check if snowfall or rain occurred
        if enbal.prec > 0.01 and enbal.tempC <= eb_prms.tsnow_threshold:
            # set timestamp
            snow_timestamp = time
            rain = 0
            snowfall = enbal.prec #***** need to add in fresh snowfall
        elif enbal.tempC > eb_prms.tsnow_threshold:
            rain = enbal.prec
            snowfall = 0
        else:
            rain = 0
            snowfall = 0

        if time.hour < 1:
            # any daily happenings go here!!!
            days_since_snowfall = time.day - snow_timestamp.day
            #update albedo
            #albedo = eb_albedo.getAlbedo(BC,days_since_snowfall)

        # Calculate subsurface heating/melt from penetrating SW
        Snet_surf = enbal.getSW(enbal.surfrad,albedo)
        subsurf_melt = layers.subsurfaceMelt(Snet_surf,dt)
        layermelt = subsurf_melt  # distinguish from subsurface to store surface layer melt later

        # Calculate surface energy balance
        Qm = enbal.surfaceEB(surftemp,layers.depths,layers.types,days_since_snowfall,albedo)
            
        if Qm < 0:
            # If not melting, need to optimize surface temperature to force Qm to be 0
            result = minimize(enbal.surfaceEB,0, method = 'L-BFGS-B',bounds=((-50, 0),),tol=1e-2,
                              args=(layers.depths,layers.types,days_since_snowfall,albedo,True))
            surftemp = result.x
            surface_melt = 0
        elif Qm > 0:
            # If melting, calculate surface melt
            surface_melt = Qm/eb_prms.Lh_rf
            if surface_melt > layers.dry_masses[0]:
                # melt by surface energy balance completely melts surface layer, so check if it melts further layers
                fully_melted = np.where(np.array([np.sum(layers.dry_masses[:i+1]) for i in range(layers.nlayers)]) <= surface_melt)[0]

                # calculate how much additional melt will occur in the next layer down
                newsurface_melt = surface_melt - np.sum(layers.dry_masses[fully_melted])
                newsurface_idx = fully_melted[-1] + 1
                # it's possible to fully melt that layer too when combined with penetrating SW melt:
                if newsurface_melt + layermelt[newsurface_idx] > layers.dry_masses[newsurface_idx]:
                    fully_melted = np.append(fully_melted,newsurface_idx)
                    # push new surface to the next layer down
                    newsurface_melt += -1*layers.dry_masses[newsurface_idx]
                    newsurface_idx += 1

                # set melt amounts from surface melt into melt array
                layermelt[fully_melted] = layers.dry_masses[fully_melted] 
                layermelt[newsurface_idx] += newsurface_melt 
            else:
                # only surface layer is melting
                layermelt[0] = surface_melt

        # Percolate the meltwater and any liquid precipitation and remove layers that were fully melted
        runoff = layers.percolate(layermelt,rain)
        running_runoff += runoff
        running_melt += np.sum(layermelt)

        # Recalculate the temperature profile considering conduction
        Tprofile_new = layers.solveHeatEq(dt)
        layers.Tprofile = Tprofile_new

        # Calculate refreeze
        refreeze = layers.refreeze(layers.Tprofile)
        running_refreeze += refreeze

        # special time checks
        if time.hour < 1 and time.minute == 0:
            is_midnight = True
        else:
            is_midnight = False
        if time.is_month_start and is_midnight:
            # any monthly happenings go here!!
            monthly_melt.append(running_melt)
            monthly_runoff.append(running_runoff)
            monthly_refreeze.append(running_refreeze)
            running_melt = 0
            running_runoff = 0
            running_refreeze = 0

            print(f'Current Melt Energy: {Qm:.2f} W/m2')
            print('Temperatures',layers.Tprofile)
            print('Water content',layers.wprofile)
            if time.month == 10:
                print('Update glacier geometry!')

        time_idx +=1

    return monthly_melt,monthly_runoff,monthly_refreeze
    
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
#             dT = dt*2/eb_prms.Cp_ice/(self.pprofile[0]+self.pprofile[1])*(conduction[1]-Qm)/self.layerh[0]
#         else:
#             dT = dt*2/eb_prms.Cp_ice/(self.pprofile[layer]+self.pprofile[layer+1])*(conduction[layer+1]-conduction[layer])/self.layerh[layer]
#         Tprofile_new.append(self.Tprofile[layer] + dT)
#     return Tprofile_new
