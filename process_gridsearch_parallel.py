"""
This script processes grid search outputs in parallel.

@author: clairevwilson
"""
import pebsi.processing.gridsearch_processing as gsproc
from multiprocessing import Pool

for run_type in ['long','2024']:
    packed_vars = []
    for site in gsproc.sitedict[run_type]:
        packed_vars.append((run_type,[site]))

    with Pool(4) as processes_pool:
        processes_pool.starmap(gsproc.process_runs,packed_vars)

print('Done!')