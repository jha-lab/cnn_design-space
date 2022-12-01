#!/bin/bash

#SBATCH --job-name=cnnbench_library		    # create a short name for your job
#SBATCH --nodes=1                  	    	# node count
#SBATCH --ntasks=1                  	    # total number of tasks across all nodes
#SBATCH --cpus-per-task=40                  # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=720G                          # memory per cpu-core (4G is default)
#SBATCH --time=144:00:00                    # total run time limit (HH:MM:SS)

module purge
module load anaconda3/2020.7
conda activate cnnbench

cd ..

export KMP_AFFINITY=disabled

python generate_library.py --modules_per_stack 4 --config_file ./tests/config_all-ops_super_reduced.yaml
