"""
Surface class for PyGEM Energy Balance

@author: cvwilson
"""

import pygem_eb.input as eb_prms
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import sys, os
import yaml
import suncalc
sys.path.append(os.getcwd()+'/biosnicar-py/')

class Surface():
    """
    Surface scheme that tracks the accumulation of LAPs and calculates 
    albedo based on several switches.
    """ 
    def __init__(self,layers,time,args,climate):
        # Add args and climate to surface class
        self.args = args
        self.climate = climate

        # Initialize surface properties
        self.stemp = eb_prms.surftemp_guess
        self.days_since_snowfall = 0
        self.snow_timestamp = time[0]
        self.stype = layers.ltype[0]

        # Set initial albedo based on surface type
        self.albedo_dict = {'snow':eb_prms.albedo_fresh_snow,
                            'firn':eb_prms.albedo_firn,
                            'ice':args.a_ice}
        self.bba = self.albedo_dict[self.stype]
        self.albedo = [self.bba]
        self.vis_a = 1
        self.spectral_weights = np.ones(1)

        # Get shading df and initialize surrounding albedo
        self.shading_df = pd.read_csv(eb_prms.shading_fp,index_col=0)
        self.shading_df.index = pd.to_datetime(self.shading_df.index)
        self.albedo_surr = eb_prms.albedo_fresh_snow

        # Output spectral albedo 
        if eb_prms.store_bands:
            bands = np.arange(0,480).astype(str)
            self.albedo_df = pd.DataFrame(np.zeros((0,480)),columns=bands)

        # Update the underlying ice spectrum
        clean_ice = pd.read_csv(eb_prms.clean_ice_fp,names=[''])
        # Albedo of the base spectrum is in the filename
        albedo_string = eb_prms.clean_ice_fp.split('bba')[-1].split('.')[0]
        bba = int(albedo_string) / (10 ** len(albedo_string))
        # Scale the new spectrum by the ice albedo
        ice_point_spectrum = clean_ice * args.a_ice / bba
        # Name file for ice spectrum
        clean_ice_fn = eb_prms.clean_ice_fp.split('/')[-1]
        self.ice_spectrum_fp = eb_prms.clean_ice_fp.replace(clean_ice_fn,f'gulkana{args.site}_ice_spectrum_{args.task_id}.csv')
        # Store new spectrum
        df_spectrum = pd.DataFrame(ice_point_spectrum)
        df_spectrum.to_csv(self.ice_spectrum_fp, index=False, header=False)

        # Parallel runs need separate input files to access
        if args.task_id != -1:
            self.snicar_fn = os.getcwd() + f'/biosnicar-py/biosnicar/inputs_{args.task_id}{args.site}.yaml'
            if not os.path.exists(self.snicar_fn):
                self.reset_SNICAR(self.snicar_fn)
            try:
                with HiddenPrints():
                    from biosnicar import get_albedo
                    _,_ = get_albedo.get('adding-doubling',plot=False,validate=False)
            except:
                self.reset_SNICAR(self.snicar_fn)
        else:
            self.snicar_fn = eb_prms.snicar_input_fp
        self.tcc = 0.5
        return
    
    def daily_updates(self,layers,airtemp,surftemp,time):
        """
        Updates daily-evolving surface properties (grain size,
        surface type and days since snowfall)

        Parameters
        ----------
        layers
            class object from pygem_eb.layers
        time : pd.Datetime
            current timestep
        """
        self.stype = layers.ltype[0]
        if self.args.switch_melt == 2 and layers.nlayers > 2:
            layers.get_grain_size(airtemp,surftemp,time)
        self.days_since_snowfall = (time - self.snow_timestamp)/pd.Timedelta(days=1)
        self.get_surr_albedo(layers,time)
        return
    
    def get_surftemp(self,enbal,layers):
        """
        Solves energy balance equation for surface temperature. 
        Qm = SWnet + LWnet + Qs + Ql + Qp + Qg
        
        There are three cases:
        (1) LWout data is input : surftemp is derived from data
        (2) Qm is positive with surftemp = 0. : excess Qm is used 
                to warm layers to the melting point or melt layers,
                depending on layer temperatures
        (3) Qm is negative with surftemp = 0. : snowpack is cooling
                and surftemp is lowered to balance Qm. Two methods 
                are available (specified in input.py): 
                        - iterative (fast)
                        - minimization (slow) 
        
        Parameters
        ----------
        enbal
            class object from pygem_eb.energybalance
        layers
            class object from pygem_eb.layers
        """
        # CONSTANTS
        STEFAN_BOLTZMANN = eb_prms.sigma_SB
        HEAT_CAPACITY_ICE = eb_prms.Cp_ice
        dt = eb_prms.dt

        if not enbal.nanLWout:
            # CASE (1): surftemp from LW data
            self.stemp = np.power(np.abs(enbal.LWout_ds/STEFAN_BOLTZMANN),1/4)
            Qm = enbal.surface_EB(self.stemp,layers,self,self.days_since_snowfall)
        else:
            Qm_check = enbal.surface_EB(0,layers,self,self.days_since_snowfall)
            # If Qm>0 with surftemp=0, the surface is melting or warming.
            # If Qm<0 with surftemp=0, the surface is cooling.
            cooling = True if Qm_check < 0 else False
            if not cooling:
                # CASE (2): Energy toward the surface
                self.stemp = 0
                Qm = Qm_check
                if layers.ltemp[0] < 0.: 
                    # check heat with a surftemp of 0
                    Qm_check = enbal.surface_EB(self.stemp,layers,self,
                                               self.days_since_snowfall)
                    # heat the top layer
                    temp_change = Qm_check*dt/(HEAT_CAPACITY_ICE*layers.lice[0])
                    layers.ltemp[0] += temp_change

                    # temp change can raise layer above melting point
                    if layers.ltemp[0] > 0.:
                        # leave excess energy in the melt energy
                        Qm = layers.ltemp[0]*HEAT_CAPACITY_ICE*layers.lice[0]/dt
                        layers.ltemp[0] = 0.

                        # if that layer will be fully melted, warm the lower layer
                        if Qm*dt/eb_prms.Lh_rf > layers.lice[0] and layers.ltemp[1] < 0.:
                            leftover = Qm*dt/eb_prms.Lh_rf - layers.lice[0]
                            layers.ltemp[1] += leftover*dt/(HEAT_CAPACITY_ICE*layers.lice[1])
                    else:
                        Qm = 0

            elif cooling:
                # CASE (3): Energy away from surface
                if eb_prms.method_cooling in ['minimize']:
                    # Run minimization on EB function
                    result = minimize(enbal.surface_EB,self.stemp,
                                      method='L-BFGS-B',bounds=((-60,0),),tol=1e-3,
                                      args=(layers,self,self.days_since_snowfall,'optim'))
                    Qm = enbal.surface_EB(result.x[0],layers,self,self.days_since_snowfall)
                    # Check success and print warning 
                    if not result.success and abs(Qm) > 10:
                        print('Unsuccessful minimization, Qm = ',Qm)
                    else:
                        self.stemp = result.x[0]

                elif eb_prms.method_cooling in ['iterative']:
                    # Loop to iteratively calculate surftemp
                    loop = True
                    n_iters = 0
                    while loop:
                        n_iters += 1
                        # Initial check of Qm comparing to previous surftemp
                        Qm_check = enbal.surface_EB(self.stemp,layers,self,
                                                   self.days_since_snowfall)
                        
                        # Check direction of flux at that temperature and adjust
                        if Qm_check > 0.5:
                            self.stemp += 0.25
                        elif Qm_check < -0.5:
                            self.stemp -= 0.25
                        # surftemp cannot go below -60
                        self.stemp = max(-60,self.stemp)

                        # break loop if Qm is ~0 or after 10 iterations
                        if abs(Qm_check) < 0.5 or n_iters > 10:
                            # if temp is still bottoming out at -60, resolve minimization
                            if self.stemp == -60:
                                result = minimize(enbal.surface_EB,-50,method='L-BFGS-B',
                                                    bounds=((-60,0),),tol=1e-3,
                                                    args=(layers,self,
                                                          self.days_since_snowfall,'optim'))
                                if result.x > -60:
                                    self.stemp = result.x[0]
                            break

                # If cooling, Qm must be 0
                Qm = 0

        # Update surface balance terms with new surftemp
        enbal.surface_EB(self.stemp,layers,self,self.days_since_snowfall)
        self.Qm = Qm
        self.tcc = enbal.tcc
        return

    def get_albedo(self,layers,time):
        """
        Checks switches and gets albedo from corresponding method. If LAPs or grain size
        are tracked, albedo comes from SNICAR, otherwise it is parameterized by surface
        type or surface age.
        
        Parameters
        ----------
        layers
            class object from pygem_eb.layers
        """
        args = self.args

        # CONSTANTS
        ALBEDO_FIRN = eb_prms.albedo_firn
        ALBEDO_FRESH_SNOW = eb_prms.albedo_fresh_snow
        DEG_RATE = eb_prms.albedo_deg_rate
        SNOW_LIM = eb_prms.snicar_snow_limit
        
        # Update surface type
        self.stype = layers.ltype[0]
        snowdepth = np.sum(layers.lheight[layers.snow_idx])

        if self.stype == 'snow':
            if args.switch_melt == 0:
                if args.switch_LAPs == 0:
                    # SURFACE TYPE ONLY
                    self.albedo = self.albedo_dict[self.stype]
                    self.bba = self.albedo
                elif args.switch_LAPs == 1:
                    # LAPs ON, GRAIN SIZE OFF
                    albedo,sw = self.run_SNICAR(layers,time,override_grainsize=True)
                    self.albedo = albedo
                    self.spectral_weights = sw
            elif args.switch_melt == 1:
                # BASIC DEGRADATION RATE
                age = self.days_since_snowfall
                albedo_aging = (ALBEDO_FRESH_SNOW - ALBEDO_FIRN)*(np.exp(-age/DEG_RATE))
                self.albedo = max(ALBEDO_FIRN + albedo_aging,ALBEDO_FIRN)
                self.bba = self.albedo
            elif args.switch_melt == 2:
                if args.switch_LAPs == 0:
                    # LAPs OFF, GRAIN SIZE ON
                    albedo,sw = self.run_SNICAR(layers,time,override_LAPs=True)
                    self.albedo = albedo
                    self.spectral_weights = sw
                elif args.switch_LAPs == 1:
                    # LAPs ON, GRAIN SIZE ON
                    self.albedo,self.spectral_weights = self.run_SNICAR(layers,time)
        else:
            self.albedo = self.albedo_dict[self.stype]
            self.bba = self.albedo
        if '__iter__' not in dir(self.albedo):
            self.albedo = [self.albedo]

        if eb_prms.store_bands:
            if '__iter__' not in dir(self.albedo):
                self.albedo = np.ones(480) * self.albedo
            self.albedo_df.loc[time] = self.albedo.copy()
        return 
    
    def run_SNICAR(self,layers,time,nlayers=None,max_depth=None,
                  override_grainsize=False,override_LAPs=False):
        """
        Runs SNICAR model to retrieve broadband albedo. 

        Parameters
        ----------
        layers
            class object from pygem_eb.layers
        nlayers : int
            Number of layers to include in the calculation
        max_depth : float
            Maximum depth of layers to include in the calculation
            ** Specify nlayers OR max_depth **
        override_grainsize : Bool
            If True, use constant average grainsize specified in input.py
        override_LAPs: Bool
            If True, use constant LAP concentrations specified in input.py

        Returns
        -------
        albedo : np.ndarray
            Array containing spectral albedo
        spectral_weights : np.ndarray
            Array containing weights of each spectral band
        """
        with HiddenPrints():
            from biosnicar import get_albedo
        get_albedo.input_fp = self.snicar_fn

        # CONSTANTS
        AVG_GRAINSIZE = eb_prms.average_grainsize
        DIFFUSE_CLOUD_LIMIT = eb_prms.diffuse_cloud_limit
        DENSITY_FIRN = eb_prms.density_firn

        # Get layers to include in the calculation
        if not nlayers and max_depth:
            nlayers = np.where(layers.ldepth > max_depth)[0][0] + 1
        elif nlayers and not max_depth:
            nlayers = min(layers.nlayers,nlayers)
        elif not nlayers and not max_depth:
            # Default case if neither is specified: only includes top 1m or non-ice layers
            nlayers = np.where(layers.ldepth > 1)[0][0] + 1
            if layers.ldensity[nlayers-1] > DENSITY_FIRN:
                nlayers = np.where(layers.ltype != 'ice')[0][-1] + 1
        idx = np.arange(nlayers)

        # Unpack layer variables (need to be stored as lists)
        lheight = layers.lheight[idx].astype(float).tolist()
        ldensity = layers.ldensity[idx].astype(float).tolist()
        lgrainsize = layers.lgrainsize[idx].astype(int)
        lwater = layers.lwater[idx] / (layers.lice[idx]+layers.lwater[idx])

        # Grain size files are every 1um till 1500um, then every 500
        idx_1500 = lgrainsize>1500
        lgrainsize[idx_1500] = np.round(lgrainsize[idx_1500]/500) * 500
        lgrainsize[lgrainsize < 30] = 30
        lgrainsize = lgrainsize.tolist()

        # Convert LAPs from mass to concentration in ppb
        BC = layers.lBC[idx] / layers.lheight[idx] * 1e6
        OC = layers.lOC[idx] / layers.lheight[idx] * 1e6
        dust1 = layers.ldust[idx] / layers.lheight[idx] * 1e6 * eb_prms.ratio_DU_bin1
        dust2 = layers.ldust[idx] / layers.lheight[idx] * 1e6 * eb_prms.ratio_DU_bin2
        dust3 = layers.ldust[idx] / layers.lheight[idx] * 1e6 * eb_prms.ratio_DU_bin3
        dust4 = layers.ldust[idx] / layers.lheight[idx] * 1e6 * eb_prms.ratio_DU_bin4
        dust5 = layers.ldust[idx] / layers.lheight[idx] * 1e6 * eb_prms.ratio_DU_bin5

        # Convert arrays to lists for making input file
        lBC = (BC.astype(float)).tolist()
        lOC = (OC.astype(float)).tolist()
        ldust1 = (dust1.astype(float)).tolist()
        ldust2 = (dust2.astype(float)).tolist()
        ldust3 = (dust3.astype(float)).tolist()
        ldust4 = (dust4.astype(float)).tolist()
        ldust5 = (dust5.astype(float)).tolist()

        # Override options for switch runs
        if override_grainsize:
            lgrainsize = [AVG_GRAINSIZE for _ in idx]
        if override_LAPs:
            lBC = [eb_prms.BC_freshsnow*1e6 for _ in idx]
            lOC = [eb_prms.OC_freshsnow*1e6 for _ in idx]
            ldust1 = np.array([eb_prms.dust_freshsnow*1e6 for _ in idx]).tolist()
            ldust2 = ldust1.copy()
            ldust3 = ldust1.copy()
            ldust4 = ldust1.copy()
            ldust5 = ldust1.copy()

        # Open and edit yaml input file for SNICAR
        with open(self.snicar_fn) as f:
            list_doc = yaml.safe_load(f)

        # Update changing layer variables
        try:
            list_doc['IMPURITIES']['BC']['CONC'] = lBC
        except:
            self.reset_SNICAR(self.snicar_fn)
            with open(self.snicar_fn) as f:
                list_doc = yaml.safe_load(f)
            list_doc['IMPURITIES']['BC']['CONC'] = lBC
        list_doc['IMPURITIES']['OC']['CONC'] = lOC
        list_doc['IMPURITIES']['DUST1']['CONC'] = ldust1
        list_doc['IMPURITIES']['DUST2']['CONC'] = ldust2
        list_doc['IMPURITIES']['DUST3']['CONC'] = ldust3
        list_doc['IMPURITIES']['DUST4']['CONC'] = ldust4
        list_doc['IMPURITIES']['DUST5']['CONC'] = ldust5
        list_doc['ICE']['DZ'] = lheight
        list_doc['ICE']['RHO'] = ldensity
        list_doc['ICE']['RDS'] = lgrainsize
        if eb_prms.include_LWC_SNICAR:
            list_doc['ICE']['LAYER_TYPE'][0] = 4
            list_doc['ICE']['LWC'] = lwater.tolist()
        else:
            list_doc['ICE']['LWC'] = [0]*nlayers

        # The following variables are constants for the n layers
        ice_variables = ['LAYER_TYPE','SHP','HEX_SIDE','HEX_LENGTH',
                         'SHP_FCTR','WATER_COATING','AR','CDOM']
        # Option to change shape in inputs
        list_doc['ICE']['SHP'][0] = eb_prms.grainshape_SNICAR
        for var in ice_variables:
            list_doc['ICE'][var] = [list_doc['ICE'][var][0]] * nlayers

        # Filepath for ice albedo
        list_doc['PATHS']['SFC'] = self.ice_spectrum_fp.split('biosnicar-py/')[-1]

        # Solar zenith angle
        lat = self.climate.lat
        lon = self.climate.lon
        time_UTC = time - eb_prms.timezone
        altitude_angle = suncalc.get_position(time_UTC,lon,lat)['altitude']
        zenith = 180/np.pi * (np.pi/2 - altitude_angle) if altitude_angle > 0 else 89
        # zenith = np.round(zenith / 10) * 10
        list_doc['RTM']['SOLZEN'] = int(zenith)
        list_doc['RTM']['DIRECT'] = 0 if self.tcc > DIFFUSE_CLOUD_LIMIT else 1

        # Save SNICAR input file
        with open(self.snicar_fn, 'w') as f:
            yaml.dump(list_doc,f)
        
        # Run get_albedo from SNICAR
        with HiddenPrints():
            albedo,spectral_weights = get_albedo.get('adding-doubling',plot=False,validate=False)
        # I adjusted SNICAR code to return spectral weights, rather than BBA

        self.bba = np.sum(albedo * spectral_weights) / np.sum(spectral_weights)
        assert len(albedo) == len(eb_prms.wvs)
        vis_idx = np.where((eb_prms.wvs <= 0.75) & (eb_prms.wvs >= 0.4))[0]
        self.vis_a = np.sum(albedo[vis_idx] * spectral_weights[vis_idx]) / np.sum(spectral_weights[vis_idx])
        return albedo,spectral_weights
    
    def reset_SNICAR(self,fp):
        # remove old file if it exists
        if os.path.exists(fp):
            os.remove(fp)
        # open the base inputs file
        with open(eb_prms.snicar_input_fp, 'rb') as src_file:
            file_contents = src_file.read()
        # copy the base inputs file to fp
        with open(fp, 'wb') as dest_file:
            dest_file.write(file_contents)
    
    def get_surr_albedo(self,layers,time):
        ALBEDO_GROUND = eb_prms.albedo_ground
        ALBEDO_SNOW = eb_prms.albedo_fresh_snow
        # reset max snowdepth yearly
        if time.month + time.day + time.hour < 1:
            layers.max_snow = 0
        # check if max_snow has been exceeded
        current_snow = np.sum(layers.lice[layers.snow_idx])
        layers.max_snow = max(current_snow, layers.max_snow)
        # scale surrounding albedo based on snowdepth
        albedo_surr = np.interp(current_snow,
                                np.array([0, layers.max_snow]),
                                np.array([ALBEDO_GROUND,ALBEDO_SNOW]))
        self.albedo_surr = albedo_surr
        return

class HiddenPrints:
    """
    Class to hide prints when running SNICAR
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self,exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        return