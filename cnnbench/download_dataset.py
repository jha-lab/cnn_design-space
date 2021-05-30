# Script to download and setup the required dataset

# Author : Shikhar Tuli


import os
import sys

import argparse
import shutil
import tarfile
from urllib import request

import torch
from torchvision import datasets

import print_util as pu


DATASET_PRESETS = {
    'CIFAR10': {
        'horizontal_flip': True,
        'inception_crop': False,
        'normalize': ((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    },
    'CIFAR100': {
        'horizontal_flip': True,
        'inception_crop': False,
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


def data_downloader(config: dict):
    """Downloads the required dataset into the data_dir directory
    
    Args:
        config (dict): dictionary of configuration of the dataset
    
    Raises:
        AssertionError: if dataset cannot be downloaded and manual_dir is not provided
    """

    assert config['dataset'] in DATASET_PRESETS.keys(), f'Dataset needs to be in {__name__}.DATASET_PRESETS'

    assert (config['dataset'] == 'ImageNet') == (config['manual_dir'] is not None), \
        'ImageNet is required to be manually downloaded'

    print(f"{pu.bcolors.OKBLUE}Setting up dataset:{pu.bcolors.ENDC} {config['dataset']}"\
        + f"{pu.bcolors.OKBLUE}, into directory:{pu.bcolors.ENDC} {config['data_dir']}")

    if config['manual_dir'] is None:
        # Download the dataset
        train_dataset = eval(f"datasets.{config['dataset']}(root='{config['data_dir']}', train=True, download=True)")
        test_dataset = eval(f"datasets.{config['dataset']}(root='{config['data_dir']}', train=False, download=True)")

        # Remove the .tar file
        for file in os.listdir(config['data_dir']):
            if file.endswith('.tar') or file.endswith('.tar.gz'):
                os.remove(os.path.join(config['data_dir'], file))

        if config['dataset'] == 'MNIST' or config['dataset'] == 'FasionMNIST':
            shutil.rmtree(os.path.join(config['data_dir'], config['dataset'], 'raw'))

        # if config['dataset'] == 'CIFAR10' or config['dataset'] == 'CIFAR100':
        #     curr_dir = 'cifar-10-batches-py' if config['dataset'] == 'CIFAR10' else 'cifar-100-python'
        #     os.rename(os.path.join(config['data_dir'], curr_dir), os.path.join(config['data_dir'], config['dataset']))
    else:
        assert config['dataset'] == 'ImageNet', 'Only ImageNet is supported for now'

        # Extract the dataset into PyTorch readable format
        os.mkdir(os.path.join(config['data_dir'], config['dataset']))
        os.mkdir(os.path.join(config['data_dir'], config['dataset'], 'train'))
        os.mkdir(os.path.join(config['data_dir'], config['dataset'], 'val'))

        train_tar = tarfile.open(os.path.join(config['manual_dir'], 'ILSVRC2012_img_train.tar'))
        train_tar.extractall(os.path.join(config['data_dir'], config['dataset'], 'train'))
        train_tar.close()

        imagenet_classes_tar = os.listdir(os.path.join(config['data_dir'], config['dataset'], 'train'))

        for file in imagenet_classes_tar:
            if file.endswith('.tar'):
                file_tar = tarfile.open(os.path.join(config['data_dir'], config['dataset'], 'train', file))
                file_tar.extractall(os.path.join(config['data_dir'], config['dataset'], 'train', file.split('.tar')[0]))
                file_tar.close()
                os.remove(os.path.join(config['data_dir'], config['dataset'], 'train', file))

        val_tar = tarfile.open(os.path.join(config['manual_dir'], 'ILSVRC2012_img_val.tar'))
        val_tar.extractall(os.path.join(config['data_dir'], config['dataset'], 'val'))
        val_tar.close()

        # Transfer images to respective folders (based on https://github.com/pytorch/examples/tree/master/imagenet)
        request.urlretrieve('https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh',
            os.path.join(config['data_dir'], config['dataset'], 'val', 'val_prep.sh'))

        os.chdir(os.path.join(config['data_dir'], config['dataset'], 'val'))
        os.system("source ./val_prep.sh")

    print(f'{pu.bcolors.OKGREEN}Done!{pu.bcolors.ENDC}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Input parameters for downloading or setting up the dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--dataset',
        metavar='',
        type=str,
        default='CIFAR10',
        help=f'Dataset name, one in: {DATASET_PRESETS.keys()}')
    parser.add_argument(
      '--data_dir',
      metavar='',
      type=str,
      default='../../datasets',
      help='Directory to download and extract dataset to')
    parser.add_argument(
      '--manual_dir',
      metavar='',
      type=str,
      default=None,
      help='Directory where dataset is already downloaded (in .tar format)')

    args = parser.parse_args()

    config = {'dataset': config['dataset'],
              'data_dir': config['data_dir'],
              'manual_dir': config['manual_dir']}
    
    data_downloader(config)
