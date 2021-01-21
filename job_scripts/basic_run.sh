#!/bin/bash

## Basic run of the CNNBench tool. 
## Automated script to run what is explained in the README

module purge
module load anaconda3/2020.7
conda activate cnnbench

cd ../cnnbench/scripts

python generate_tf_records.py

cd ../../job_scripts

python generate_graphs_script.py \
	--module_vertices 2 \
	--output_file '../results/vertices_2/generated_graphs.json'

python run_evaluation_script.py \
	--module_vertices 2 \
	--models_file '../results/vertices_2/generated_graphs.json' \
	--output_dir '../results/vertices_2/evaluation' 

wait

python cleanup_script.py --cleanup_dir '../results/vertices_2/evaluation'

python generate_dataset_script.py --model_dir '../results/vertices_2'
