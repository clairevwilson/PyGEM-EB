#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --time=08:00:00

# Initialize conda
source ~/.bashrc
conda init bash
conda activate eb_env

# Run the Python script
cd ../research/PyGEM-EB/  
python param_set_parallel.py -n=128
# python gulkana_longruns.py -n 3
# python gulkana_2024runs.py -n 3