#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --array=0-80
source ~/.bashrc
conda init bash
conda activate eb_env

# Define the argument file and delete if it exists
ARG_FILE="args.txt"

# If file doesn't exist, create list of args
if [[ ! -f $ARG_FILE ]]; then
    > $ARG_FILE
    for k_ice in 1.5 2 2.5; do
        for k_snow in 0.4 0.5 0.6; do
            for a_ice in 0.4 0.5 0.6; do
                for site in "AB" "B" "D"; do
                    echo "-k_ice=$k_ice -k_snow=$k_snow -a_ice=$a_ice -site=$site" >> $ARG_FILE
                done
            done
        done
    done
fi

# Run the job array with srun
srun bash -c '
    # Access the current task ID
    TASK_ID=$SLURM_ARRAY_TASK_ID
    
    # Get the unique argument for this task
    ARG=$(sed -n "$((TASK_ID + 1))p" args.txt)
    
    # Run the Python script with the unique argument
    cd ../research/PyGEM-EB/
    python grid_search.py $ARG -task_id $TASK_ID
'