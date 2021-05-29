# Script to donload and setup the required dataset
# Author : Shikhar Tuli


import os
import sys

import argparse
import shutil
import tarfile
from urllib import request

import torch
import torchvision


DATASET_PRESETS = {
    'CIFAR10': {
        'normalize': ((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    },
    'CIFAR100': {
        'normalize': ((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    },
    'MNIST': {
        'normalize': ((0.1307,), (0.3081,))
    },
    'FasionMNIST': {
        'normalize': ((0.5,), (0.5,))
    },
    'ImageNet': {
        'normalize': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    },
}


def main():
    """Downloads the required dataset into the data_dir directory
    
    Args:
        dataset (str): dataset name, one in DATASET_PRESETS.keys()
        data_dir (str): directory to download the dataset to
        manual_dir (Optional: str): directory where dataset is already downloaded

    Raises:
        AssertionError: if dataset cannot be downloaded and manual_dir is not provided
    """
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

    assert args.dataset in DATASET_PRESETS.keys(), f'Dataset needs to be in {__name__}.DATASET_PRESETS'

    assert (args.dataset == 'ImageNet') == (args.manual_dir is not None), 'ImageNet is required to be manually downloaded'

    print(f'Setting up dataset: {args.dataset}, into directory: {args.data_dir}')

    if args.manual_dir is None:
        # Download the dataset
        train_dataset = eval(f"torchvision.datasets.{args.dataset}(root='{args.data_dir}', train=True, download=True)")
        test_dataset = eval(f"torchvision.datasets.{args.dataset}(root='{args.data_dir}', train=False, download=True)")

        # Remove the .tar file
        for file in os.listdir(args.data_dir):
            if file.endswith('.tar') or file.endswith('.tar.gz'):
                os.remove(os.path.join(args.data_dir, file))

        if args.dataset == 'MNIST' or args.dataset == 'FasionMNIST':
            shutil.rmtree(os.path.join(args.data_dir, args.dataset, 'raw'))

        if args.dataset == 'CIFAR10' or args.dataset == 'CIFAR100':
            curr_dir = 'cifar-10-batches-py' if args.dataset == 'CIFAR10' else 'cifar-100-python'
            os.rename(os.path.join(args.data_dir, curr_dir), os.path.join(args.data_dir, args.dataset))
    else:
        assert args.dataset == 'ImageNet', 'Only ImageNet is supported for now'

        # Extract the dataset into PyTorch readable format
        os.mkdir(os.path.join(args.data_dir, args.dataset))
        os.mkdir(os.path.join(args.data_dir, args.dataset, 'train'))
        os.mkdir(os.path.join(args.data_dir, args.dataset, 'val'))

        train_tar = tarfile.open(os.path.join(args.manual_dir, 'ILSVRC2012_img_train.tar'))
        train_tar.extractall(os.path.join(args.data_dir, args.dataset, 'train'))
        train_tar.close()

        imagenet_classes_tar = os.listdir(os.path.join(args.data_dir, args.dataset, 'train'))

        for file in imagenet_classes_tar:
            if file.endswith('.tar'):
                file_tar = tarfile.open(os.path.join(args.data_dir, args.dataset, 'train', file))
                file_tar.extractall(os.path.join(args.data_dir, args.dataset, 'train', file.split('.tar')[0]))
                file_tar.close()
                os.remove(os.path.join(args.data_dir, args.dataset, 'train', file))

        val_tar = tarfile.open(os.path.join(args.manual_dir, 'ILSVRC2012_img_val.tar'))
        val_tar.extractall(os.path.join(args.data_dir, args.dataset, 'val'))
        val_tar.close()

        # Transfer images to respective folders (based on https://github.com/pytorch/examples/tree/master/imagenet)
        request.urlretrieve('https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh',
            os.path.join(args.data_dir, args.dataset, 'val', 'val_prep.sh'))

        os.chdir(os.path.join(args.data_dir, args.dataset, 'val'))
        os.system("source ./val_prep.sh")

    print('Done!')


if __name__ == '__main__':
  main()
