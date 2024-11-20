#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --time=08:00:00

# Initialize conda
source ~/.bashrc
conda init bash
conda activate eb_env

# Run the Python script
cd ../research/PyGEM-EB/  
python gulkana_longruns.py -n 3 -kp=2.7 -kw=2 -Boone_c5=0.028
# python gulkana_2024runs.py -n 3