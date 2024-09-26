"""
Contains functions to compare a model output dataset (ds)
to a field dataset. The field datasets included:

1. Seasonal mass balance from stake measurements
2. Surface height change (cumulative mass balance) from GNSS-IR
3. Snow temperatures from iButtons
4. Albedo timeseries from automatic weather station
5. Spectral albedo from FieldSpec

All functions use the following notation for the arguments:
    fp : str
        Filepath to the field data
    ds : xarray.Dataset
        Output dataset from PyGEM-EB
    method : str, default 'RMSE'
        Choose between 'RMSE','MAE'
    plot : Bool, default False
        Plot the result

@author: clairevwilson
"""
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

# User options
date_form = mpl.dates.DateFormatter('%b %d')
mpl.style.use('seaborn-v0_8-white')

# Objective function
def objective(model,data,method):
    if method == 'MSE':
        return np.mean(np.square(model - data))
    elif method == 'RMSE':
        return np.sqrt(np.mean(np.square(model - data)))
    elif method == 'MAE':
        return np.mean(np.abs(model - data))
    
# ========== 1. SEASONAL MASS BALANCE ==========
def seasonal_mass_balance(data_fp,ds,site='B',method='MAE',plot=False):
    """
    Compares seasonal mass balance measurements from
    USGS stake surveys to a model output.

    The stake data comes directly from the USGS Data
    Release (Input_Glaciological_Data.csv).

    ** Additional argument site : str
    Name of the USGS site.
    """
    # Load dataset
    mb_df = pd.read_csv(data_fp)
    mb_df = mb_df.loc[mb_df['site_name'] == site]
    mb_df.index = mb_df['Year']

    # Get overlapping years
    years_model = np.unique(pd.to_datetime(ds.time.values).year)
    years_measure = np.unique(mb_df.index)
    years = np.sort(list(set(years_model) & set(years_measure)))[1:]

    # Retrieve the model data
    mb_dict = {'bw':[],'bs':[],'w-':[],'s+':[]}
    for year in years:
        spring_date = str(year)+'-04-20 00:00'
        fall_date = str(year)+'-08-20 00:00'
        last_fall_date = str(year-1)+'-08-20 00:00'
        melt_dates = pd.date_range(spring_date,fall_date,freq='h')
        acc_dates = pd.date_range(last_fall_date,spring_date,freq='h')
        if pd.to_datetime(ds.time.values[0]).minute == 30:
            melt_dates = melt_dates + pd.Timedelta(minutes=30)
            acc_dates = acc_dates + pd.Timedelta(minutes=30)
        # sum mass balance
        try:
            wds = ds.sel(time=acc_dates).sum()
            sds = ds.sel(time=melt_dates).sum()
        except:
            if year == years[-1]:
                years = years[:-1]
                break
        winter_mb = wds.accum + wds.refreeze - wds.melt
        summer_mb = sds.accum + sds.refreeze - sds.melt
        mb_dict['bw'].append(winter_mb.values)
        mb_dict['bs'].append(summer_mb.values)
        mb_dict['w-'].append(wds.melt.values*-1)
        mb_dict['s+'].append(sds.accum.values)

    # Index mass balance data
    mb_df = mb_df.loc[years]
    winter_data = mb_df['bw']
    summer_data = mb_df['ba'] - mb_df['bw']
    wabl_data = mb_df['winter_ablation']
    sacc_data = mb_df['summer_accumulation']

    # Clean up arrays
    winter_model = np.array(mb_dict['bw'])
    summer_model = np.array(mb_dict['bs'])
    assert winter_model.shape == winter_data.shape
    assert summer_model.shape == summer_data.shape    

    # Assess error
    winter_error = objective(winter_model,winter_data,method) 
    summer_error = objective(summer_model,summer_data,method) 

    # Plot
    if plot:
        if plot == 'w-s+':
            fig,ax = plt.subplots()
            ax.plot(years,mb_dict['w-'],label='Winter Melt',color='turquoise',linewidth=2)
            ax.plot(years,mb_dict['s+'],label='Summer Acc.',color='orange',linewidth=2)
            ax.plot(years,wabl_data,color='turquoise',linestyle='--')
            ax.plot(years,sacc_data,color='orange',linestyle='--')
            ax.set_ylabel('Partitioned seasonal mass balance (m w.e.)',fontsize=14)
            ax.set_title(f'Summer accumulation and winter ablation at site {site}')
        else:
            fig,ax = plt.subplots()
            ax.plot(years,winter_model,label='Winter',color='turquoise',linewidth=2)
            ax.plot(years,summer_model,label='Summer',color='orange',linewidth=2)
            ax.plot(years,winter_data,color='turquoise',linestyle='--')
            ax.plot(years,summer_data,color='orange',linestyle='--')
            ax.axhline(0,color='grey',linewidth=0.5)
            min_all = np.min(np.array([winter_model,summer_model,winter_data,summer_data]))
            max_all = np.max(np.array([winter_model,summer_model,winter_data,summer_data]))
            ax.set_xticks(np.arange(years[0],years[-1],4))
            ax.set_yticks(np.arange(np.round(min_all,0),np.round(max_all,0)+1,1))
            ax.set_ylabel('Seasonal mass balance (m w.e.)',fontsize=14)
            ax.set_title(f'Summer {method} = {summer_error:.3f}   Winter {method} = {winter_error:.3f}')
        ax.plot(np.nan,np.nan,linestyle='--',color='grey',label='Data')
        ax.plot(np.nan,np.nan,color='grey',label='Modeled')
        ax.legend(fontsize=12,ncols=2)
        ax.tick_params(labelsize=12,length=5,width=1)
        ax.set_xlim(years[0],years[-1])
        ax.set_xticks(np.arange(years[0],years[-1],4))
        return fig,ax
    else:
        return winter_error, summer_error

# ========== 2. CUMULATIVE MASS BALANCE ==========
def cumulative_mass_balance(data_fp,ds,site='AB',method='MAE',plot=False):
    """
    Compares cumulative mass balance measurements from
    a stake to a model output. 

    The stake data should be formatted as a .csv with two 
    columns: 'Date' and 'CMB' where 'CMB' is the surface 
    height change in meters.
    """
    # Load dataset
    stake_df = pd.read_csv(data_fp)
    stake_df.index = pd.to_datetime(stake_df['Date'])

    # Retrieve the dates and index stake data
    start = stake_df.index[0]
    end = stake_df.index[-1]
    assert ds.time.values[0] < end, 'Model run begins after field date period'
    assert ds.time.values[-1] > start, 'Model run ends before field date period'
    stake_df['CMB'] -= stake_df['CMB'].iloc[0]

    # Index model data
    if start.minute != pd.to_datetime(ds.time.values[0]).minute:
        start += pd.Timedelta(minutes=30)
    if end.minute != pd.to_datetime(ds.time.values[0]).minute:
        end -= pd.Timedelta(minutes=30)
    ds = ds.sel(time=pd.date_range(start,end,freq='h'))
    ds = ds.dh.cumsum() - ds.dh.isel(time=0)
    ds = ds.sel(time=pd.date_range(start,end))

    # Clean up arrays
    model = ds.values
    data = stake_df['CMB'].values
    assert model.shape == data.shape

    # Assess error
    error = objective(model,data,method)

    # Plot
    if plot:
        fig,ax = plt.subplots()
        ax.plot(stake_df.index,stake_df['CMB'],label='Stake',linestyle='--',color='black')
        ax.plot(ds.time.values,ds.values,label='Model')
        ax.legend(fontsize=12)
        ax.xaxis.set_major_formatter(date_form)
        ax.set_xticks(pd.date_range(start,end,freq='MS'))
        ax.tick_params(labelsize=12,length=5,width=1)
        ax.set_xlim(start,end)
        ax.set_ylabel('Surface height change (m)',fontsize=14)
        ax.set_title(f'{method} = {error:.3e}')
        plt.show()

    return error

# ========== 3. SNOW TEMPERATURES ==========
def snow_temperature(data_fp,ds,method='RMSE',plot=False,plot_heights=[0.5]):
    """
    Compares modeled snow temperatures to measurements from
    iButton temperature sensors.

    The iButton data should be formatted as a .csv with 
    datetime in the first column and each sensor labeled 
    by its initial height above the ice in cm (eg."50").

    ** Additional option for method: 'ripe'
    Error returned is the difference in days between modeled 
    and measured ripening date (measured - modeled)

    ** Additional argument plot_heights:
    List of heights above the ice in meters to plot the timeseries.
    """
    # Load dataset
    temp_df = pd.read_csv(data_fp,index_col=0)
    temp_df.index = pd.to_datetime(temp_df.index)
    temp_df = temp_df.resample('30min').interpolate()
    h0 = np.array(temp_df.columns).astype(float)

    # Retrieve the dates and index stake data
    start = pd.to_datetime(temp_df.index[0])
    end = pd.to_datetime(temp_df.index[-1])
    assert ds.time.values[0] < end, 'Model run begins after field date period'
    assert ds.time.values[-1] > start, 'Model run ended before field data period'
    start = max(start,ds.time.values[0])
    end = min(end,ds.time.values[-1])
    time = pd.date_range(start,end,freq='h')
    if time[0].minute != 30:
        time += pd.Timedelta(minutes=30)
    temp_df = temp_df.loc[time]

    # Initialize time loop
    buried = np.arange(len(temp_df.columns))
    store = {'measured':[],'modeled':[],'measure_plot':[],'model_plot':[]}
    for j,hour in enumerate(time):
        current_meas = temp_df.loc[hour].values
        if np.any(current_meas > 0):
            n_exposed = len(np.where(current_meas > 0)[0])
            buried = buried[n_exposed:]
            if len(buried) < 1:
                break
        # get temperatures of buried iButtons
        temp_measure = np.flip(current_meas[buried])
        height_measure = np.flip(h0[buried])
        
        # get modeled temperatures
        # index snow layers
        dens_model = ds.sel(time=hour)['layerdensity'].values
        dens_model[np.where(np.isnan(dens_model))[0]] = 1e5
        snow = np.where(dens_model < 700)[0]
        # include one extra layer for interpolating (will index out when stored)
        snow = np.append(snow,snow[-1]+1).ravel()
        # get height above ice
        lheight = ds.sel(time=hour)['layerheight'].values[snow]
        icedepth = np.sum(lheight[:-1]) + lheight[-2] / 2
        # get property and absolute depth
        ldepth = np.array([np.sum(lheight[:i+1])-(lheight[i]/2) for i in range(len(lheight))])
        temp_model = np.flip(ds.sel(time=hour)['layertemp'].values[snow])
        height_model = np.flip(icedepth - ldepth)
        
        # interpolate measured temperatures to model heights
        temp_measure_interp = np.interp(height_model[:-1],height_measure,temp_measure)
        
        # store x height above the ice for plotting
        temp_measure_plot = np.interp(plot_heights,height_measure,temp_measure)
        temp_model_plot = np.interp(plot_heights,height_model,temp_model)

        # store
        store['measured'].append(temp_measure_interp)
        store['modeled'].append(temp_model[1:])
        store['measure_plot'].append(temp_measure_plot)
        store['model_plot'].append(temp_model_plot)
    
    # Clean up outputs
    time = time[:j]
    flatten = lambda xss: np.array([x for xs in xss for x in xs])
    data = flatten(store['measured'])
    model = flatten(store['modeled'])
    assert model.shape == data.shape

    # Error
    error = objective(model,data,method)

    # Plot
    if plot:
        # get plotting data
        measure_plot = np.array(store['measure_plot'])
        model_plot = np.array(store['model_plot'])

        # initialize plots
        n_plots = len(plot_heights)
        fig,axes = plt.subplots(n_plots,figsize=(6,1.5*n_plots),dpi=200,
                            layout='constrained',sharex=True,sharey=True)
        if n_plots == 1:
            axes = [axes]
        for i,ax in enumerate(axes):
            ax.plot(time,measure_plot[:,i],color='black',linestyle='--',label='iButtons')
            ax.plot(time,model_plot[:,i],label='Model')
            ax.xaxis.set_major_formatter(date_form)
            ax.tick_params(length=5,labelsize=11)
            ax.set_title(f'Initial height: {plot_heights[i]}',loc='right')
            ax.set_xlim(start,time[-1])
        ax.legend()
        fig.supylabel('Snow Temperature (C)',fontsize=12)
        fig.suptitle(f'{method} = {error:.3e}')
        plt.show()

    return error