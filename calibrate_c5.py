import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from pygem_eb.processing.plotting_fxns import *

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
    
# import model
import pygem_eb.input as eb_prms
eb_prms.startdate = pd.to_datetime('2023-04-18 00:30')
eb_prms.enddate = eb_prms.startdate + pd.Timedelta(hours=2)
eb_prms.new_file = False
eb_prms.debug = False
with HiddenPrints():
    import run_simulation_eb as sim
    eb_prms.store_data = True

    # read command line args
    args = sim.get_args()
    args.startdate = pd.to_datetime('2023-04-18 00:30') # snow sample dates
    args.enddate = pd.to_datetime('2023-07-08 00:30')
    # initialize the model
    climate = sim.initialize_model(args.glac_no[0],args)

# ===== CALIBRATION DATA =====
# GULKANA
# surface height change
stake_df = pd.read_csv('~/research/MB_data/Stakes/gulkanaAB23_ALL.csv')
stake_df.index = pd.to_datetime(stake_df['Date'])
dates_index = pd.date_range(args.startdate,args.enddate) - pd.Timedelta(minutes=30)
stake_df = stake_df.loc[dates_index]
stake_df['CMB'] -= stake_df['CMB'].iloc[0]

# objective function
def objective(model,data):
    return np.mean(np.abs(model - data))

low = 3e-6
high = 3e-4
n_iters = 3
outputs = []
best_loss = np.inf

for guess in np.linspace(low,high,n_iters):
    eb_prms.Boone_c1 = guess
    with HiddenPrints():
        # run the model
        out = sim.run_model(climate,args,{'c5':str(guess)})
    result = out.dh.resample(time='d').sum().cumsum().values
    loss = objective(result.flatten(),stake_df['CMB'].values)

    # new best
    if loss < best_loss:
        best_loss = loss
        best_guess = guess
        
    outputs.append(out)

print(f'After {n_iters} iterations between {low} and {high} the best result was:')
print(f'      c5 = {guess:.3f}')
print(f'      mae = {best_loss:.3e}')

dh_vs_stake(stake_df,outputs,[args.startdate,args.enddate],labels=[str(i) for i in range(len(outputs))])
plt.savefig('/home/claire/research/dh_best.png',dpi=200)