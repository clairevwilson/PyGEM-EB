class meltProfile():
    """
    Temperature and density profile function to distribute melt through vertical bins and define
    the content (snow or firn) of each bin.
    
    Attributes
    ----------
    switch_snow : Bool
        Switch to turn on/off snow-albedo feedback
    switch_melt : Bool
        Switch to turn on/off melt-albedo feedback
    switch_LAP : Bool
        Switch to turn on/off LAP-albedo feedback
    
    """ 
    def __init__(self,climateds):
        """
        Initialize T/D/rho profiles? Set options for parameterizations?
        """
        # self.vert_bins = data
    def EnergyMassBalance(self,climateds,dates_table):
        """
        Calculates the surface heat fluxes at each point on the glacier.
        """
    