import numpy as np
import pygem_eb.input as eb_prms

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
        return

    def getAlbedo(self,days_since_snowfall):
        self.albedo = 0.85
        return 

    def getGrainSize(self,days_since_snowfall):
        return 0
    
    def updatePrecip(self,type,amount):
        if type in 'snow' and amount > 1e-8:
            self.albedo = 0.85
        elif type in 'rain':
            self.albedo = 0.6
        return
