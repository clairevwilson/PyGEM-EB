"""
This script processes grid search outputs in parallel.

@author: clairevwilson
"""
import pebsi.processing.gridsearch_processing as gsproc
from multiprocessing import Pool
import argparse
import time

n=120

for run_type in ['long','2024']:
    for site in gsproc.sitedict[run_type]:
        start_site = time.time()
        packed_vars = []
        date = gsproc.run_info[run_type]['date']
        idx = gsproc.run_info[run_type]['idx']
        fp = gsproc.base_fp+f'{date}_{site}_{idx}/'

        for i in range(n):
            fn = fp + f'grid_{date}_set{i}_run0_0.nc'
            packed_vars.append((run_type,fn))

        with Pool(n) as processes_pool:
            processes_pool.starmap(gsproc.process_runs,packed_vars)

        elapsed = time.time() - start_site
        print(f'Finished site {site} in {elapsed:.0f}')

print('Done!')