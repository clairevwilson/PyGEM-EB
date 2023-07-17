import cdsapi
import os
import pandas as pd
import xarray as xr
import numpy as np

lon_min = 134
lon_max = 180
lat_min = 50
lat_max = 72

all_vars = ['2m_temperature','total_precipitation','surface_pressure','2m_dewpoint_temperature',
            '10m_u_component_of_wind','10m_v_component_of_wind','surface_solar_radiation_downwards'] # uwind, vwind, SWin, TCC
var = '2m_dewpoint_temperature'
region = '08'
rgi_fp = '/home/claire/research/RGI/rgi60/00_rgi60_attribs/'

# Output information
file_format = 'netcdf'  # 'grib' or 'netcdf' (but ONLY netcdf is supported by the code below for mapping and time series extraction)
folder_out = '/home/claire/research/CDS'
downloaded_file = f'ERA5_{var}_lat_lon_hourly.nc'
downloaded_file = os.path.join(folder_out, downloaded_file)

# Set up time
start_year = 1980
end_year = 2021
years = [ str(start_year +i ) for i in range(end_year - start_year + 1)] 
start_day = 1
end_day = 31
days = [ str(start_day +i ).zfill(2) for i in range(end_day - start_day + 1)]

for fn in os.listdir(rgi_fp):
    if region in fn:
        rgi_fn = os.path.join(rgi_fp,fn)
main_glac_rgi = pd.read_csv(rgi_fn, encoding = 'ISO-8859-1',index_col='RGIId')

# Extract unique lat/lon cell where there are glaciers 
cells_df = pd.read_csv(folder_out+'/unique_cells.csv',index_col='index')
already_stored = cells_df[['lat','lon']].to_numpy()
data = xr.open_dataset(folder_out+'/GCMexample.nc')
region_coords = []

lat_nearidx = (np.abs(main_glac_rgi['CenLat'].values[:,np.newaxis] - 
                                data.coords['latitude'][:].values).argmin(axis=1))
lon_nearidx = (np.abs(main_glac_rgi['CenLon'].values[:,np.newaxis] - 
                                data.coords['longitude'][:].values).argmin(axis=1))
        
latlon_nearidx = list(zip(lat_nearidx, lon_nearidx))
latlon_nearidx_unique = list(set(latlon_nearidx))
for latlonidx in latlon_nearidx_unique:
    lat = data.coords['latitude'][latlonidx[0]].values
    lon = data.coords['longitude'][latlonidx[1]].values
    region_coords.append([lat,lon])
    if [lat,lon] not in already_stored:
        cells_df.loc[len(cells_df.index)] = [lat,lon,int(region)]
cells_df.to_csv(folder_out+'/unique_cells.csv')

# Select cells to download at a time
c = cdsapi.Client()
print(f'Downloading {len(region_coords)} cells in region {region}.')

for i,coord in enumerate( region_coords[0:5]):
    lat,lon = coord
    lat_min = lat-1e-3
    lat_max = lat+1e-3
    lon_min = lon-1e-3
    lon_max = lon+1e-3
    latlon_name = f'{lat}_{lon}'.replace('.','')
    downloaded_file = downloaded_file.replace('lat_lon',latlon_name)

    c.retrieve('reanalysis-era5-land',
        {
            'year': years,
            'variable': var,
            'month': [ '01', '02', '03',
                    '04', '05', '06',
                    '07', '08', '09',
                    '10', '11', '12',],
            'day': days,
            'time': [
                '00:00', '01:00', '02:00',
                '03:00', '04:00', '05:00',
                '06:00', '07:00', '08:00',
                '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00',
                '15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00',
                '21:00', '22:00', '23:00',
            ],
            'area': [ lat_min, lon_min, lat_max, lon_max ],
            'format': file_format,
        },
        downloaded_file)
    print('done',i)