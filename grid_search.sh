#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --time=01:00:00

# Initialize conda
source ~/.bashrc
conda init bash
conda activate eb_env

# Run the Python script
cd ../research/PyGEM-EB/  
python run_param_set.py -n 128