import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from pygem_eb.processing.objectives import *
import pygem_eb.input as eb_prms
import run_simulation_eb as sim

# ===== DATA FILEPATHS =====
# A. Long run: seasonal mass balance data
base = '/home/claire/research/MB_data/'
data_fp_USGS = base + 'Gulkana/Input_Gulkana_Glaciological_Data.csv'
# B. 2023 run
data_fp_2023 = base + 'Stakes/gulkanaAB23_ALL.csv'
# C. 2024 run
data_fp_2024 = base + 'Stakes/gulkana24_ALL.csv'

# ===== OBJECTIVE FUNCTION =====
# def objective(parameter):
#     if param_name == 'a_ice':
#         args

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