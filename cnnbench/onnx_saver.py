# Saves PyTorch model to ONNX format

# Author : Shikhar Tuli


import os
import sys

import torch
from torch.autograd import Variable

import argparse

from model_builder import CNNBenchModel
from manual_models import *
from library import Graph, GraphLib

from utils import print_util as pu


def save_onnx(ckpt_path: str, config: dict = None, model_name: str = None, onnx_file_path: str = None):
    """Saves ONNX file for the given PyTorch checkpoint
    
    Args:
        ckpt_path (str): checkpoint path for PyTorch model
        config (dict, optional): config dictionary
        model_name (str, optional): name of the manual model
        onnx_file_path (str, optional): path to save ONNX file. Defaults to
        	ckpt_path with different extension.
    """
    checkpoint = torch.load(ckpt_path)

    if not config:
    	config = checkpoint['config']

    if model_name:
    	graphObject = get_manual_graph(config, model_name)
    else:
    	graphObject = checkpoint['graphObject']

    model = CNNBenchModel(config, graphObject)

    model.load_state_dict(checkpoint['model_state_dict'])

    dummy_input = Variable(torch.rand(1, config['input_channels'], config['image_size'], config['image_size']))

    if onnx_file_path:
    	torch.onnx.export(model, dummy_input, onnx_file_path)
    else:
    	file_name = os.path.basename(ckpt_path).split('.')[0] + '.onnx'
    	onnx_file_path = os.path.join(os.path.dirname(ckpt_path), file_name)
    	torch.onnx.export(model, dummy_input, onnx_file_path)

    print(f'{pu.bcolors.OKGREEN}Saved ONNX model to:{pu.bcolors.ENDC} {onnx_file_path}')


if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description='Input parameters for saving ONNX model',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument(
		'--config',
		metavar='',
		type=str,
		default='./tests/config_test_tune.yaml',
		help=f'Path to the configuration file')
	parser.add_argument(
	  '--model_name',
	  metavar='',
	  type=str,
	  default='resnet18',
	  help=f'Name of the manual model. Should be in: {SUPPORTED_MODELS}')
	parser.add_argument(
	  '--ckpt_path',
	  metavar='',
	  type=str,
	  default='/scratch/gpfs/stuli/cnn_design-space/results/manual_models/resnet18_tune/model.pt',
	  help=f'Path to the model checkpoint')
	parser.add_argument(
	  '--onnx_file_path',
	  metavar='',
	  type=str,
	  default=None,
	  help='Path to save the ONNX model')

	args = parser.parse_args()

	with open(args.config) as config_file:
		try:
			config = yaml.safe_load(config_file)
		except yaml.YAMLError as exc:
			print(exc)

	save_onnx(args.ckpt_path, config, args.model_name, args.onnx_file_path)