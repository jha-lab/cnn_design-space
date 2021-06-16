# Input pipeline for different datasets. Make sure that
# the dataset is downloaded using setup_dataset.py

# Author : Shikhar Tuli


import os
import sys

import numpy as np

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms

from utils import print_util as pu


DATASET_PRESETS = {
    'CIFAR10': {
        'horizontal_flip': True,
        'inception_crop': True,
        'normalize': ((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    },
    'CIFAR100': {
        'horizontal_flip': True,
        'inception_crop': True,
        'normalize': ((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    },
    'MNIST': {
        'horizontal_flip': False,
        'inception_crop': False,
        'normalize': ((0.1307,), (0.3081,))
    },
    'FashionMNIST': {
        'horizontal_flip': True,
        'inception_crop': False,
        'normalize': ((0.5,), (0.5,))
    },
    'ImageNet': {
        'horizontal_flip': True,
        'inception_crop': True,
        'normalize': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    },
}

VAL_PERCENTAGE = 0.1


def get_loader(config: dict, shuffle=True):
    """Gets the data loaders for the training and test sets
    
    Args:
        config (dict): configuration for the dataset
        shuffle (bool, optional): to shuffle before splitting training and validaton
        	sets.
    
    Returns:
        (train_loader, val_loader, test_loader, total_size, val_size): train, val and test DataLoaders,
        	total size of the training set and the size of the validation set
    """
    dataset = config['dataset']

    assert dataset in DATASET_PRESETS.keys(), f'Dataset needs to be in {__name__}.DATASET_PRESETS'

    train_transforms = [transforms.ToTensor(), transforms.Normalize(*DATASET_PRESETS[dataset]['normalize'])]
    test_transforms = [transforms.ToTensor(), transforms.Normalize(*DATASET_PRESETS[dataset]['normalize'])]

    if DATASET_PRESETS[dataset]['horizontal_flip']:
        train_transforms.insert(0, transforms.RandomHorizontalFlip())

    if DATASET_PRESETS[dataset]['inception_crop']:
        train_transforms.insert(0, transforms.RandomResizedCrop(config['image_size']))

        test_transforms.insert(0, transforms.CenterCrop(config['image_size']))
        test_transforms.insert(0, transforms.Resize(256))
    else:
        train_transforms.insert(0, transforms.Resize(config['image_size']))
        test_transforms.insert(0, transforms.Resize(config['image_size']))

    train_transforms = transforms.Compose(train_transforms)
    test_transforms = transforms.Compose(test_transforms)

    # Loading the dataset already downloaded
    if dataset != 'ImageNet':
        print(f'{pu.bcolors.OKBLUE}Loading Training dataset{pu.bcolors.ENDC}')
        train_dataset = eval(f"datasets.{dataset}(root='{config['data_dir']}', " \
            + "train=True, download=False, transform=train_transforms)")
        val_dataset = eval(f"datasets.{dataset}(root='{config['data_dir']}', " \
            + "train=True, download=False, transform=test_transforms)")
        print(train_dataset)

        print(f'{pu.bcolors.OKBLUE}Loading Test dataset{pu.bcolors.ENDC}')
        test_dataset = eval(f"datasets.{dataset}(root='{config['data_dir']}', " \
            + "train=False, download=False, transform=test_transforms)")
        print(test_dataset)
    else:
        print(f'{pu.bcolors.OKBLUE}Loading Training dataset{pu.bcolors.ENDC}')
        train_dataset = datasets.ImageFolder(root=os.path.join(config['data_dir'], dataset, 'train'),
            transform=train_transforms)
        val_dataset = datasets.ImageFolder(root=os.path.join(config['data_dir'], dataset, 'train'),
            transform=test_transforms)
        print(train_dataset)

        print(f'{pu.bcolors.OKBLUE}Loading Test dataset{pu.bcolors.ENDC}')
        test_dataset = datasets.ImageFolder(root=os.path.join(config['data_dir'], dataset, 'val'),
            transform=test_transforms)
        print(test_dataset)

    total_size, val_size = len(train_dataset), int(np.floor(len(train_dataset) * VAL_PERCENTAGE))
    
    indices = list(range(total_size))
    
    if shuffle:
        np.random.seed(0)
        np.random.shuffle(indices)

    train_indices, val_indices = indices[val_size:], indices[:val_size]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(train_dataset, batch_size=config['train_batch_size'],
                            sampler=train_sampler, pin_memory=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=config['train_batch_size'],
                            sampler=val_sampler, pin_memory=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=config['test_batch_size'],
                            shuffle=False, pin_memory=True, num_workers=8)

    return train_loader, val_loader, test_loader, total_size, val_size







