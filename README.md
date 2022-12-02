# CNNBench: A CNN Design-Space Generation Tool and Benchmark

![Python Version](https://img.shields.io/badge/python-v3.6%20%7C%20v3.7%20%7C%20v3.8-blue)
![Conda](https://img.shields.io/badge/conda%7Cconda--forge-v4.8.3-blue)
![PyTorch](https://img.shields.io/badge/pytorch-v1.11.0-e74a2b)
![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FJHA-Lab%2Fcnn_design-space&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)

This repository contains the tool **CNNBench** which can be used to generate and evaluate different Convolutional Neural Network (CNN) architectures pertinent to the domain of Machine-Learning Accelerators. 
The tool can be used to search among a large set of CNN architectures.

## Table of Contents
- [Environment Setup](#environment-setup)
  - [Clone this repository](#clone-this-repository)
  - [Setup python environment](#setup-python-environment)
- [Basic run of the tool](#basic-run-of-the-tool)
  - [Download and prepare the CIFAR-10 dataset](#download-and-prepare-the-cifar\-10-dataset)
  - [Generate computational graphs](#generate-computational-graphs)
  - [Run evaluation over all generated graphs](#run-evaluation-over-all-generated-graphs)
  - [Generate the CNNBench dataset](#generate-the-cnnbench-dataset)
- [Developer](#developer)
- [Cite this work](#cite-this-work)
- [License](#license)
  

## Environment setup

### Clone this repository
```
git clone https://github.com/jha-lab/cnn_design-space.git
cd cnn_design-space
```
### Setup python environment  
* **PIP**
```
virtualenv cnnbench
source cnnbench/bin/activate
pip install -r requirements.txt
```  
* **CONDA**
```
conda env create -f environment.yaml
```

## Basic run of the tool

Running a basic version of the tool comprises of the following:
* CNNs with modules comprising of upto two vertices, each is one of the operations in `[MAXPOOL3X3, CONV1X1, CONV3X3]`
* Each module is stacked three times. A base stem of 3x3 convolution with 128 output channels is used. 
The stack of modules is followed by global average pooling and a final dense softmax layer.
* Training on the CIFAR-10 dataset.

### Download and prepare the CIFAR-10 dataset
```
cd cnnbench
python dataset_downloader.py
```

_To use another dataset (among CIFAR-10, CIFAR-100, MNIST, or ImageNet) use input arguments; check:_ `python dataset_downloader.py --help`.

### Generate computational graphs
```
python generate_library.py
```
This will create a `.json` file of all graphs at: `dataset/dataset.json` using the SHA-256 hashing algorithm and three modules per stack.

### Run BOSHNAS
```
python run_boshnas.py
```
All training scripts use bash and have been implemented using [SLURM](https://slurm.schedmd.com/documentation.html). This will have to be setup before running the experiments.

Other flags can be used to control the training procedure (check using `python run_boshnas.py --help`). This script uses the SLURM scheduler over mutiple compute nodes in a cluster (each cluster assumed to have 1 GPU, this can be changed in the script `job_scripts/job_train.sh`). SLURM can also be used in scenarios where distributed nodes are not available.

## Developer

[Shikhar Tuli](https://github.com/shikhartuli). For any questions, comments or suggestions, please reach me at [stuli@princeton.edu](mailto:stuli@princeton.edu).

## Cite this work

Cite our work using the following bitex entry:
```bibtex
@article{tuli2022codebench,
      title={{CODEBench}: A Neural Architecture and Hardware Accelerator Co-Design Framework}, 
      author={Tuli, Shikhar and Li, Chia-Hao and Sharma, Ritvik and Jha, Niraj K.},
      year={2022},
      eprint={----.-----},
      archivePrefix={arXiv},
      primaryClass={--.--}
}
```

## License

BSD-3-Clause. 
Copyright (c) 2022, Shikhar Tuli and Jha Lab.
All rights reserved.

See License file for more details.
