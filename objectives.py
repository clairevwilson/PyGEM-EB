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
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import itertools

# User options
date_form = mpl.dates.DateFormatter('%b %d')
mpl.style.use('seaborn-v0_8-white')
GNSSIR_fp = '../MB_data/Stakes/gulkanaSITE24_GNSSIR.csv'
USGS_fp = '../MB_data/Gulkana/Input_Gulkana_Glaciological_Data.csv'

# Objective function
def objective(model,data,method):
    if method == 'MSE':
        return np.mean(np.square(model - data))
    elif method == 'RMSE':
        return np.sqrt(np.mean(np.square(model - data)))
    elif method == 'MAE':
        return np.mean(np.abs(model - data))
    elif method == 'ME':
        return np.mean(model - data)
    
# ========== 1. SEASONAL MASS BALANCE ==========
def seasonal_mass_balance(site,ds,method='MAE',plot=False):
    """
    Compares seasonal mass balance measurements from
    USGS stake surveys to a model output.

    The stake data comes directly from the USGS Data
    Release (Input_Glaciological_Data.csv).

    ** Additional argument site : str
    Name of the USGS site.
    """
    # Load dataset
    df_mb = pd.read_csv(USGS_fp)
    df_mb = df_mb.loc[df_mb['site_name'] == site]
    df_mb.index = df_mb['Year']

    # Get overlapping years
    years_model = np.unique(pd.to_datetime(ds.time.values).year)
    years_measure = np.unique(df_mb.index)
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
        internal_acc = ds.sel(time=melt_dates[-2]).cumrefreeze.values
        summer_mb = sds.accum + sds.refreeze - sds.melt - internal_acc
        mb_dict['bw'].append(winter_mb.values)
        mb_dict['bs'].append(summer_mb.values)
        mb_dict['w-'].append(wds.melt.values*-1)
        mb_dict['s+'].append(sds.accum.values)

    # Index mass balance data
    df_mb = df_mb.loc[years]
    winter_data = df_mb['bw']
    summer_data = df_mb['ba'] - df_mb['bw']
    wabl_data = df_mb['winter_ablation']
    sacc_data = df_mb['summer_accumulation']

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
            title = f'Summer {method} = {summer_error:.3f}   Winter {method} = {winter_error:.3f}'
            if method == 'MAE':
                winter_error = objective(winter_model,winter_data,'ME') 
                summer_error = objective(summer_model,summer_data,'ME')
                method = 'Mean Error' 
                title += f'\nSummer {method} = {summer_error:.3f}   Winter {method} = {winter_error:.3f}'
            ax.set_title(title)
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
def cumulative_mass_balance(site,ds,method='MAE',plot=False,plot_ax=False):
    """
    Compares cumulative mass balance measurements from
    a stake to a model output. 

    The stake data should be formatted as a .csv with two 
    columns: 'Date' and 'CMB' where 'CMB' is the surface 
    height change in meters.
    """
    # Update filepath
    fp_gnssir = GNSSIR_fp.replace('SITE',site)
    fp_stake = fp_gnssir.replace('GNSSIR','stake')

    # Load GNSSIR daily MB
    if os.path.exists(fp_gnssir):
        df_mb_daily = pd.read_csv(fp_gnssir)
        df_mb_daily.index = pd.to_datetime(df_mb_daily['Date'])
        df_mb_daily['CMB'] -= df_mb_daily['CMB'].iloc[0]
        df_mb_daily = df_mb_daily.sort_index()
    elif os.path.exists(fp_stake):
        df_mb_daily = pd.read_csv(fp_stake)
        df_mb_daily.index = pd.to_datetime(df_mb_daily['Date'])
        df_mb_daily = df_mb_daily.sort_index()

    # Load USGS seasonal MB
    if site not in ['ABB','BD']:
        year = pd.to_datetime(ds.time.values[0]).year
        df_mb = pd.read_csv(USGS_fp)
        df_mb = df_mb.loc[df_mb['site_name'] == site]
        mba = df_mb.loc[df_mb['Year'] == year,'ba'].values[0]
        mbw = df_mb.loc[df_mb['Year'] == year,'bw'].values[0]
        mbs_measured = mba - mbw

        # Retrieve modeled summer MB
        spring_date = str(year)+'-04-20 00:00'
        fall_date = str(year)+'-08-20 00:00'
        melt_dates = pd.date_range(spring_date,fall_date,freq='h')
        ds_summer = ds.sel(time=melt_dates)
        mbs_ds = ds_summer.accum + ds_summer.refreeze - ds_summer.melt
        internal_acc = ds.sel(time=melt_dates[-2]).cumrefreeze.values
        if internal_acc > 1e-5:
            print(f'Site {site} internal acc: {internal_acc:.5f} m w.e.')
        mbs_modeled = mbs_ds.sum().values - internal_acc

    if os.path.exists(fp_gnssir) or os.path.exists(fp_stake):
        # Retrieve the dates
        start = df_mb_daily.index[0]
        end = df_mb_daily.index[-1]
        assert ds.time.values[0] < end, 'Model run begins after field date period'
        assert ds.time.values[-1] > start, 'Model run ends before field date period'
        if start < ds.time.values[0]:
            start = pd.to_datetime(ds.time.values[0])
        if end > ds.time.values[-1]:
            end = pd.to_datetime(ds.time.values[-1])

        # Index state data
        df_mb_daily = df_mb_daily.loc[start:end]
        df_mb_daily.index = pd.to_datetime(df_mb_daily.index)
        idx_data = []
        for i,date in enumerate(pd.date_range(start,end)):
            date = pd.to_datetime(date.date())
            if date in df_mb_daily.index:
                if ~np.isnan(df_mb_daily.loc[date,'CMB']):
                    idx_data.append(i)
                
        # Index model data
        if pd.to_datetime(ds.time.values[0]).minute == 30:
            if start.minute != pd.to_datetime(ds.time.values[0]).minute:
                start += pd.Timedelta(minutes=30)
            if end.minute != pd.to_datetime(ds.time.values[0]).minute:
                end -= pd.Timedelta(minutes=30)
        ds = ds.sel(time=pd.date_range(start,end,freq='h'))

        # Accumulation area sites: need only dh above stake depth
        if site in ['D','T']:
            dh = []
            stake_depth = 9
            for hour in pd.date_range(start,end,freq='h'):
                ds_now = ds.sel(time=hour)
                lheight = ds_now.layerheight.values 
                ldepth = np.array([np.sum(lheight[:i+1])-(lheight[i]/2) for i in range(len(lheight))])
                layers = np.where(ldepth < stake_depth)[0]
                height_now = np.sum(lheight[layers])
                if hour == start:
                    height_before = height_now
                i = -1
                while np.abs(height_now - height_before) > 0.5:
                    height_now = np.sum(lheight[layers[:i]])
                    i -= 1
                    if len(layers[:i])<1:
                        break
                i = 1
                while np.abs(height_now - height_before) > 0.5:
                    height_now = np.sum(lheight[:layers[-1]+i])
                    i += 1
                    assert i < 20
                dh.append(height_now - height_before)
                height_before = height_now
                stake_depth += ds_now.dh.values
            ds['dh'].values = dh
        # Cumululative sum
        ds['dh'].values = ds.dh.cumsum().values - ds.dh.isel(time=0).values
        # Select data daily
        ds = ds.sel(time=pd.date_range(start,end)).dh

        # Clean up arrays
        model = ds.values[idx_data]
        data = df_mb_daily['CMB'].values
        if model.shape != data.shape:
            idx_data = np.where(~np.isnan(df_mb_daily['CMB'].values))[0]
            data = df_mb_daily.iloc[idx_data]
            data = data['CMB'].values
            model = ds.values[idx_data]
        assert model.shape == data.shape

        # Assess error
        error = objective(model,data,method)

        # Plot
        if plot:
            if not plot_ax:
                fig,ax = plt.subplots(figsize=(3,6))
            else:
                ax = plot_ax
            
            # Plot stake
            if os.path.exists(fp_stake):
                df_stake_daily = pd.read_csv(fp_stake.replace('GNSSIR','stake'),index_col=0)
                df_stake_daily.index = pd.to_datetime(df_stake_daily.index)
                df_stake_daily['CMB'] -= df_stake_daily['CMB'].iloc[0]
                df_stake_daily = df_stake_daily.sort_index().loc[start:end]
                ax.plot(df_stake_daily.index,df_stake_daily['CMB'],label='Stake',linestyle=':',color='gray')

            # Plot gnssir
            if os.path.exists(fp_gnssir):
                ax.plot(df_mb_daily.index,df_mb_daily['CMB'],label='GNSS-IR',linestyle='--',color='black')
                # error bounds
                lower = df_mb_daily['CMB'] - df_mb_daily['sigma']
                upper = df_mb_daily['CMB'] + df_mb_daily['sigma']
                ax.fill_between(df_mb_daily.index,lower,upper,alpha=0.2,color='gray')
            
            # Plot model and beautify plot
            ax.plot(ds.time.values,ds.values,label='Model',color=plt.cm.Dark2(0))
            ax.legend(fontsize=12)
            ax.xaxis.set_major_formatter(date_form)
            ax.set_xticks(pd.date_range(start,end,freq='MS'))
            ax.tick_params(labelsize=12,length=5,width=1)
            ax.set_xlim(start,end)
            ax.set_ylabel('Surface height change (m)',fontsize=14)
            if error < 1:
                ax_title = f'{method}: {error:.3f} m'
            else:
                ax_title = f'{method}: {error:.3e} m'
            if site not in ['ABB','BD']:
                mb_bias = mbs_modeled - mbs_measured
                # ax_title += f'\nMBMOD: {mbs_modeled:.3f} m w.e.\nMBMEAS: {mbs_measured:.3f} m w.e.'
                # ax_title.replace('MOD','$_{mod}$')
                # ax_title.replace('MEAS','$_{meas}$')
                direction = '' if mb_bias < 0 else '+'
                ax_title += f'\n{direction}{mb_bias:.3f} m w.e.'
            else:
                ax_title = ax_title + '\n' 
            ax.set_title(ax_title,y=1.03)
            if not plot_ax:
                return fig, ax
            else:
                return ax
        else:
            return error
    
    print(f'Modeled MB: {mbs_modeled} m w.e.\nMeasured MB: {mbs_measured} m w.e.')

# ========== 3. SNOW TEMPERATURES ==========
def snow_temperature(site,ds,method='RMSE',plot=False,plot_heights=[0.5]):
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
    year = pd.to_datetime(ds.time.values[0]).year
    data_fp = f'../Data/iButtons/iButtons_{year}_{site}.csv'
    temp_df = pd.read_csv(data_fp,index_col=0)
    temp_df.index = pd.to_datetime(temp_df.index)
    temp_df = temp_df.resample('30min').interpolate()
    h0 = np.array(temp_df.columns).astype(float)/100
    h0 = np.max(h0) - h0

    # Retrieve the dates and index stake data
    start = pd.to_datetime(temp_df.index[0])
    end = pd.to_datetime(temp_df.index[-1])
    assert ds.time.values[0] < end, 'Model run begins after field date period'
    assert ds.time.values[-1] > start, 'Model run ended before field data period'
    start = max(start,ds.time.values[0])
    end = min(end,ds.time.values[-1])
    time = pd.date_range(start,end,freq='h')
    if time[0].minute != 30 and pd.to_datetime(ds.time.values[0]).minute == 30:
        time += pd.Timedelta(minutes=30)
    temp_df = temp_df.loc[time]

    # Initialize time loop
    store = {'measured':[],'modeled':[],'measure_plot':[],'model_plot':[]}
    for j,hour in enumerate(time):
        buried = np.arange(len(temp_df.columns))
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
        n = len(plot_heights)
        fig,ax = plt.subplots(figsize=(6,4),dpi=200,
                            layout='constrained',sharex=True,sharey=True)
        norm = mpl.colors.Normalize(vmin=0, vmax=n)
        cmap = mpl.cm.get_cmap('viridis')
        for i in range(n):
            ax.plot(time,measure_plot[:,i],linestyle='--',color=cmap(norm(i)))
            ax.plot(time,model_plot[:,i],color=cmap(norm(i)))
            ax.xaxis.set_major_formatter(date_form)
            ax.tick_params(length=5,labelsize=11)
            # ax.set_title(f'Initial height: {plot_heights[i]}',loc='right')
            ax.set_xlim(start,time[-1])
            ax.plot(np.nan,np.nan,color=cmap(norm(i)),label=f'{plot_heights[i]}')
        ax.plot(np.nan,np.nan,linestyle='--',color='gray',label='Measured')
        ax.plot(np.nan,np.nan,color='gray',label='Modeled')
        ax.legend(title='Initial height above ice (m)')
        fig.supylabel('Snow Temperature ($^{\circ}$C)',fontsize=12)
        fig.suptitle(f'{method} = {error:.3e} '+'$^{\circ}$C')
        plt.show()

    return error

def grid_plot(params_dict,summer_result,winter_result):
    """
    Parameters
    ----------
    params_dict : dict
        dict formatted as 'param_name':[option_1,option_2,...]
    result_summer and reuslt_winter : np.array
        should be size (N1xN2x...Nn) where:
            n is the number of param sets
            N is the number of options in each set
    """
    # Parse parameters
    param_names = list(params_dict.keys())
    combos = list(itertools.combinations(param_names,2))
    n_combos = len(combos)
    mid_idx = n_combos / 2
    if mid_idx % 1 != 0:
        mid_idx = int(mid_idx)

    # Create plot
    fig = plt.figure(figsize=(3*n_combos, 5))
    ratios = np.append(np.array([1]*n_combos).flatten(),0.3)
    gs = fig.add_gridspec(2,n_combos+1, wspace=1,hspace=0.5, width_ratios=ratios)

    # Normalize loss values across all three plots for shared colorbar
    all_summer = summer_result[~np.isnan(summer_result)].flatten()
    all_winter = winter_result[~np.isnan(winter_result)].flatten()
    norm_summer = plt.Normalize(vmin=np.min(all_summer), vmax=np.max(all_summer))
    norm_winter = plt.Normalize(vmin=np.min(all_winter), vmax=np.max(all_winter))
    cmap = 'viridis_r'

    # Create the scatter plots
    for j,pair in enumerate(combos):
        # Unpack the parameters
        param_1 = params_dict[pair[0]]
        param_2 = params_dict[pair[1]]

        # If parameter is a string, need to parse differently
        ylabels = False
        xlabels = False
        if type(param_1[0]) == str:
            xlabels = param_1
            param_1 = np.arange(len(param_1))
        if type(param_2[0]) == str:
            ylabels = param_2
            param_2 = np.arange(len(param_2))
        
        # Create meshgrid
        x,y = np.meshgrid(param_1,param_2)
        for i,season in enumerate(['winter','summer']):
            ax = fig.add_subplot(gs[i,j])
    
            if n_combos > 1:
                slices = [slice(None)] * n_combos
                slices[np.flip(np.arange(len(combos)))[j]] = mid_idx
                if season == 'summer':
                    result = summer_result[tuple(slices)]
                    norm = norm_summer
                elif season == 'winter':
                    result = winter_result[tuple(slices)]
                    norm = norm_winter
            else:
                if season == 'summer':
                    result = summer_result
                    norm = norm_summer
                elif season == 'winter':
                    result = winter_result
                    norm = norm_winter

            ax.scatter(x,y,c=result.T,cmap=cmap,s=500,norm=norm)
            ax.set_xlabel(pair[0],fontsize=12)
            ax.set_ylabel(pair[1],fontsize=12)
            ax.set_xticks(param_1)
            ax.set_yticks(param_2)
            ax.grid(True)
            if type(xlabels) != bool:
                ax.set_xticklabels(xlabels)
            if type(ylabels) != bool:
                ax.set_yticklabels(ylabels)

    # Add colorbars to each row
    cax1 = fig.add_subplot(gs[0,n_combos])
    cax2 = fig.add_subplot(gs[1,n_combos])
    axes = np.array(fig.get_axes()).reshape(2,n_combos+1)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm_winter, cmap=cmap),ax=axes[0,:-1], orientation='vertical',cax=cax1)
    cbar.set_label('Winter MAE',loc='top',fontsize=12)

    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm_summer, cmap=cmap), ax=axes[1,:-1], orientation='vertical',cax=cax2)
    cbar.set_label('Summer MAE',loc='top',fontsize=12)
    return fig, axes