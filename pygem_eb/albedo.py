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
        self.getGrainSize()
        return
    
    def getSurfTemp(self,enbal,layers):
        Qm_init = enbal.surfaceEB(0,layers,self.days_since_snowfall)
        if Qm_init > 0:
            # Energy toward the surface: check if we're above or below past surface temp
            Qm_update = enbal.surfaceEB(self.surftemp,layers,self.days_since_snowfall)
            if Qm_update > 0: 
                self.temp = 0
                self.Qm = Qm_init
            elif Qm_init < 0: # not melting: cool surface
                self.Qm = 0
        elif Qm_init < 0:
            # Energy away from surface: need to change surface temperature to get 0 surface energy flux 
            result = minimize(enbal.surfaceEB,self.temp,method='L-BFGS-B',bounds=((-60,0),),tol=1e-3,
                            args=(layers,self.days_since_snowfall,'optim'))
            Qm_result = enbal.surfaceEB(result.x[0],layers,self.days_since_snowfall)
            if not result.success and Qm_result < 0:
                assert 1==0, 'Surface temperature was not lowered enough by minimization'
            else:
                self.temp = result.x[0]
                self.Qm = Qm_result

        else: # initial_Qm == 0, no need to update surface temperature
            self.Qm = 0
            


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

