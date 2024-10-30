
def get_bands():
    """
    Creates the elevation bins on which to find
    mean sky view factor and shaded array 
    """
    bands = [[1000,1500,2000],[2000,2500,3000]]
    return bands

def elev_band_search(bands):
    """
    Searches random cells within that elevation band to 
    """
    for band in bands:
        print('filter the dem by band[0] - band[-2]')
        # generate random indices within the length of the filtered list
        # loop through those
        # run the shading model
        # store the results as temporary files
        # find the median result
        # save under the median elevation of the result
    return