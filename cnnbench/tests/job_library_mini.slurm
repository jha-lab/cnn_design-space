#!/bin/bash

#SBATCH --job-name=cnnbench_library		    # create a short name for your job
#SBATCH --nodes=1                  	    	# node count
#SBATCH --ntasks=1                  	    # total number of tasks across all nodes
#SBATCH --cpus-per-task=20                  # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=160G                          # memory per cpu-core (4G is default)
#SBATCH --time=96:00:00                     # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all                     # send email
#SBATCH --mail-user=stuli@princeton.edu

module purge
module load anaconda3/2020.7
conda activate cnnbench

export KMP_AFFINITY=disabled

python generate_library_mini.py
