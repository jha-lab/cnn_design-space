#!/bin/sh

# Script to install required packages in conda for GPU setup
# Author : Shikhar Tuli

module load anaconda3/2020.11
conda create --name cnnbench pytorch torchvision cudatoolkit=11.1 -c pytorch -c nvidia

conda activate cnnbench