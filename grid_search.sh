#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --time=01:00:00

# Run the Python script
cd ../research/PyGEM-EB/  
python run_param_set.py -n 128