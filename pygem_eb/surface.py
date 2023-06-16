import numpy as np
import pygem_eb.input as eb_prms
from scipy.optimize import minimize

class Surface():
    """
    Surface scheme that tracks the accumulation of LAPs and calculates albedo based on several switches.
    """ 
    def __init__(self,layers,time):
        # Set initial albedo based on surface type
        if layers.types[0] in ['snow']:
            self.albedo = eb_prms.albedo_fresh_snow
            self.snow_timestamp = time[0]
        elif layers.types[0] in ['firn']:
            self.albedo = eb_prms.albedo_firn
        elif layers.types[0] in ['ice']:
            self.albedo = eb_prms.albedo_ice

        # Initialize BC, dust, grain_size, etc.
        self.BC = 0
        self.dust = 0
        self.grain_size = 0
        self.temp = eb_prms.surftemp_guess
        self.Qm = 0
        self.days_since_snowfall = 0
        return
    
    def updateSurface(self):
        """
        Run every timestep to get properties that evolve with time. Keeps track of past surface in the case of fresh snowfall
        after significant melt.
        """
        self.getGrainSize()
        return
    
    def getSurfTemp(self,enbal,layers):
        Qm_check = enbal.surfaceEB(0,layers,self.albedo,self.days_since_snowfall)
        # If Qm is positive with a surface temperature of 0, the surface is either melting or warming to the melting point.
        # If Qm is negative with a surface temperature of 0, the surface temperature needs to be lowered to cool the snowpack.
        cooling = True if Qm_check < 0 else False
        if not cooling:
            # Energy toward the surface: either melting or top layer is heated to melting point
            self.temp = 0
            Qm = Qm_check
            if layers.snowtemp[0] < 0: # need to heat surface layer to 0 before it can start melting
                layers.snowtemp[0] += Qm_check*eb_prms.dt/(eb_prms.Cp_ice*layers.dry_spec_mass[0])
                if layers.snowtemp[0] > 0:
                    # if temperature rises above zero, leave excess energy in Qm
                    Qm = layers.snowtemp[0]*eb_prms.Cp_ice*layers.dry_spec_mass[0]/eb_prms.dt
                    layers.snowtemp[0] = 0
                else:
                    Qm = 0
        elif cooling:
            # Energy away from surface: need to change surface temperature to get 0 surface energy flux 
            result = minimize(enbal.surfaceEB,self.temp,method='L-BFGS-B',bounds=((-60,0),),tol=1e-3,
                            args=(layers,self.albedo,self.days_since_snowfall,'optim'))
            Qm = enbal.surfaceEB(result.x[0],layers,self.albedo,self.days_since_snowfall)
            if not result.success and abs(Qm) > 10:
                print('Unsuccessful minimization, Qm = ',Qm)
                # assert 1==0, 'Surface temperature was not lowered enough by minimization'
            else:
                self.temp = result.x[0]
        
        self.Qm = Qm
        return

    def getAlbedo(self):
        self.albedo = 0.85
        return 

    def getGrainSize(self):
        return 0
    
    def updatePrecip(self,type,amount):
        if type in 'snow' and amount > 1e-8:
            self.albedo = 0.85
        elif type in 'rain':
            self.albedo = 0.85
        return

