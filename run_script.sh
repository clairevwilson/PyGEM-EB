#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=02:00:00

# Initialize conda
source ~/.bashrc
conda init bash
conda activate eb_env

# Run the Python script
cd ../research/PyGEM-EB/  
python gulkana_parallel.py