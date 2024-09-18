#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --array=0-44
source ~/.bashrc
conda init bash
conda activate eb_env

# Define the argument file and check if it has the right number of rows
ARG_FILE="args.txt"
EXPECTED_ROWS = 45 # should match SBATCH --array above
if [[ -f "$ARG_FILE" ]]; then
    # Get the number of rows (lines) in the file
    ROW_COUNT=$(wc -l < "$ARG_FILE")

    # Check if the number of rows matches the expected count
    if [[ "$ROW_COUNT" -ne "$EXPECTED_ROWS" ]]; then
        echo "File $ARG_FILE has $ROW_COUNT rows, expected $EXPECTED_ROWS. Deleting the file..."
        rm "$ARG_FILE"  # Delete the file
    fi
fi

# If file doesn't exist, create list of args
if [[ ! -f $ARG_FILE ]]; then
    > $ARG_FILE
    for k_snow in "VanDusen" "Sturm" "Douville" "Jansson" "OstinAndersson"; do
        for a_ice in 0.4 0.5 0.6; do
            for site in "AB" "B" "D"; do
                echo "-k_ice=$k_ice -k_snow=$k_snow -a_ice=$a_ice -site=$site" >> $ARG_FILE
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