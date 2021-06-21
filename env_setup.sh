#!/bin/sh

# Script to install required packages in conda for GPU setup
# Author : Shikhar Tuli

module load anaconda3/2020.11

# Create a new conda environment
conda create --name cnnbench pytorch torchvision cudatoolkit=11.1 -c pytorch -c nvidia

conda activate cnnbench

# Install dependencies
conda install pyyaml
conda install scikit-learn
conda install -c conda-forge tqdm ipywidgets matplotlib
conda install -c conda-forge ray-tune

# Install pip dependencies
pip install HEBO
