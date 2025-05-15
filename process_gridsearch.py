import pygem_eb.processing.gridsearch_processing as gsproc
from multiprocessing import Pool

for run_type in ['long','2024']:
    packed_vars = []
    for site in gsproc.sitedict[run_type]:
        packed_vars.append((run_type,[site]))

    with Pool(4) as processes_pool:
        processes_pool.starmap(gsproc.process_runs,packed_vars)

# gsproc.process_runs('2024', all=True)
print('Done!')