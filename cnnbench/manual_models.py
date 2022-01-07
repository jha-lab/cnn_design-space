# Builds CNNBench specification graphs for popular networks

# Author : Shikhar Tuli


import os
import sys

import numpy as np
import yaml
import argparse
import torch
import math

from library import GraphLib, Graph
from model_trainer import worker
from utils import graph_util, print_util as pu


SUPPORTED_MODELS = ['lenet', 'alexnet', 'vgg11', 'vgg13',
					'vgg16', 'vgg19', 'resnet18', 'resnet34', 
					'resnet50', 'resnet101', 'resnet152',
					'shufflenet', 'mobilenet', 'googlenet',
					'inception', 'xception', 'efficientnet-b0',
					'efficientnet-b1', 'efficientnet-b2', 
					'efficientnet-b3', 'efficientnet-b4',
					'efficientnet-b5', 'efficientnet-b6',
					'efficientnet-b7', 'efficientnet-l2']
# TODO: add resnext, wide_resnet, inceptionv3, squeezenet, densenet


def get_manual_graph(config: dict, model_name: str):
	"""Get a manually defined CNN model
	
	Args:
		config (dict): dictionary of configuration
		model_name (str): name of the CNN model
	
	Returns:
		graphObject (Graph): Graph object for the specified model
	"""
	if config is None:
		hash_algo = 'md5' # the default hash algorithm
	else:
		hash_algo = config['hash_algo']

	assert model_name in SUPPORTED_MODELS, f'The model: {model_name} is not implemented yet.'

	if model_name == 'lenet':
		conv_module = (np.eye(7, k=1, dtype=np.int8),
			['input', 'conv5x5-c6-bn-relu', 'avgpool2x2', 'conv5x5-c16-bn-relu', 'avgpool2x2', 'conv5x5-c120-bn-relu', 'output'])

		head_module = (np.eye(5, k=1, dtype=np.int8),
			['input', 'flatten', 'dense-84-relu', 'dense_classes', 'output'])

		model_graph = [conv_module, head_module]

		graphObject = Graph(model_graph, graph_util.hash_graph(model_graph, hash_algo))

	elif model_name == 'alexnet':
		conv_module = (np.eye(10, k=1, dtype=np.int8),
			['input', 'conv11x11-c64-s4-p2-bn-relu', 'maxpool3x3-s2', 'conv5x5-c192-p2-bn-relu',
			 'maxpool3x3-s2', 'conv3x3-c384-p1-bn-relu', 'conv3x3-c256-p1-bn-relu', 
			 'conv3x3-c384-p1-bn-relu', 'maxpool3x3-s2', 'output'])

		head_module = (np.eye(8, k=1, dtype=np.int8),
			['input', 'flatten', 'dropout-p5', 'dense-4096-relu', 'dropout-p5', 'dense-4096-relu',
			 'dense_classes', 'output'])

		model_graph = [conv_module, head_module]

		graphObject = Graph(model_graph, graph_util.hash_graph(model_graph, hash_algo))

	elif model_name == 'vgg11':
		conv_module_1 = (np.eye(4, k=1, dtype=np.int8),
			['input', 'conv3x3-c64-p1-bn-relu', 'maxpool3x3-s2', 'output'])

		conv_module_2 = (np.eye(4, k=1, dtype=np.int8),
			['input', 'conv3x3-c128-p1-bn-relu', 'maxpool3x3-s2', 'output'])

		conv_module_3 = (np.eye(5, k=1, dtype=np.int8),
			['input', 'conv3x3-c256-p1-bn-relu', 'conv3x3-c256-p1-bn-relu', 
			 'maxpool3x3-s2', 'output'])

		conv_module_4 = (np.eye(5, k=1, dtype=np.int8),
			['input', 'conv3x3-c512-p1-bn-relu', 'conv3x3-c512-p1-bn-relu', 
			 'maxpool3x3-s2', 'output'])

		conv_module_5 = (np.eye(5, k=1, dtype=np.int8),
			['input', 'conv3x3-c512-p1-bn-relu', 'conv3x3-c512-p1-bn-relu', 
			 'maxpool3x3-s2', 'output'])

		head_module = (np.eye(8, k=1, dtype=np.int8),
			['input', 'flatten', 'dense-4096-relu', 'dropout-p5', 'dense-4096-relu', 'dropout-p5',
			 'dense_classes', 'output'])

		model_graph = [conv_module_1, conv_module_2, conv_module_3, conv_module_4, conv_module_5, 
			head_module]

		graphObject = Graph(model_graph, graph_util.hash_graph(model_graph, hash_algo))

	elif model_name == 'vgg13':
		conv_module_1 = (np.eye(5, k=1, dtype=np.int8),
			['input', 'conv3x3-c64-p1-bn-relu', 'conv3x3-c64-p1-bn-relu', 'maxpool3x3-s2', 'output'])

		conv_module_2 = (np.eye(5, k=1, dtype=np.int8),
			['input', 'conv3x3-c128-p1-bn-relu', 'conv3x3-c128-p1-bn-relu', 'maxpool3x3-s2', 'output'])

		conv_module_3 = (np.eye(5, k=1, dtype=np.int8),
			['input', 'conv3x3-c256-p1-bn-relu', 'conv3x3-c256-p1-bn-relu', 
			 'maxpool3x3-s2', 'output'])

		conv_module_4 = (np.eye(5, k=1, dtype=np.int8),
			['input', 'conv3x3-c512-p1-bn-relu', 'conv3x3-c512-p1-bn-relu', 
			 'maxpool3x3-s2', 'output'])

		conv_module_5 = (np.eye(5, k=1, dtype=np.int8),
			['input', 'conv3x3-c512-p1-bn-relu', 'conv3x3-c512-p1-bn-relu', 
			 'maxpool3x3-s2', 'output'])

		head_module = (np.eye(8, k=1, dtype=np.int8),
			['input', 'flatten', 'dense-4096-relu', 'dropout-p5', 'dense-4096-relu', 'dropout-p5',
			 'dense_classes', 'output'])

		model_graph = [conv_module_1, conv_module_2, conv_module_3, conv_module_4, conv_module_5, 
			head_module]

		graphObject = Graph(model_graph, graph_util.hash_graph(model_graph, hash_algo))

	elif model_name == 'vgg16':
		conv_module_1 = (np.eye(5, k=1, dtype=np.int8),
			['input', 'conv3x3-c64-p1-bn-relu', 'conv3x3-c64-p1-bn-relu', 'maxpool3x3-s2', 'output'])

		conv_module_2 = (np.eye(5, k=1, dtype=np.int8),
			['input', 'conv3x3-c128-p1-bn-relu', 'conv3x3-c128-p1-bn-relu', 'maxpool3x3-s2', 'output'])

		conv_module_3 = (np.eye(6, k=1, dtype=np.int8),
			['input', 'conv3x3-c256-p1-bn-relu', 'conv3x3-c256-p1-bn-relu', 'conv3x3-c256-p1-bn-relu',
			 'maxpool3x3-s2', 'output'])

		conv_module_4 = (np.eye(6, k=1, dtype=np.int8),
			['input', 'conv3x3-c512-p1-bn-relu', 'conv3x3-c512-p1-bn-relu', 'conv3x3-c512-p1-bn-relu',
			 'maxpool3x3-s2', 'output'])

		conv_module_5 = (np.eye(6, k=1, dtype=np.int8),
			['input', 'conv3x3-c512-p1-bn-relu', 'conv3x3-c512-p1-bn-relu', 'conv3x3-c512-p1-bn-relu',
			 'maxpool3x3-s2', 'output'])

		head_module = (np.eye(8, k=1, dtype=np.int8),
			['input', 'flatten', 'dense-4096-relu', 'dropout-p5', 'dense-4096-relu', 'dropout-p5',
			 'dense_classes', 'output'])

		model_graph = [conv_module_1, conv_module_2, conv_module_3, conv_module_4, conv_module_5, 
			head_module]

		graphObject = Graph(model_graph, graph_util.hash_graph(model_graph, hash_algo))

	elif model_name == 'vgg19':
		conv_module_1 = (np.eye(5, k=1, dtype=np.int8),
			['input', 'conv3x3-c64-p1-bn-relu', 'conv3x3-c64-p1-bn-relu', 'maxpool3x3-s2', 'output'])

		conv_module_2 = (np.eye(5, k=1, dtype=np.int8),
			['input', 'conv3x3-c128-p1-bn-relu', 'conv3x3-c128-p1-bn-relu', 'maxpool3x3-s2', 'output'])

		conv_module_3 = (np.eye(7, k=1, dtype=np.int8),
			['input', 'conv3x3-c256-p1-bn-relu', 'conv3x3-c256-p1-bn-relu', 'conv3x3-c256-p1-bn-relu',
			 'conv3x3-c256-p1-bn-relu', 'maxpool3x3-s2', 'output'])

		conv_module_4 = (np.eye(7, k=1, dtype=np.int8),
			['input', 'conv3x3-c512-p1-bn-relu', 'conv3x3-c512-p1-bn-relu', 'conv3x3-c512-p1-bn-relu',
			 'conv3x3-c512-p1-bn-relu', 'maxpool3x3-s2', 'output'])

		conv_module_5 = (np.eye(7, k=1, dtype=np.int8),
			['input', 'conv3x3-c512-p1-bn-relu', 'conv3x3-c512-p1-bn-relu', 'conv3x3-c512-p1-bn-relu',
			 'conv3x3-c512-p1-bn-relu', 'maxpool3x3-s2', 'output'])

		head_module = (np.eye(8, k=1, dtype=np.int8),
			['input', 'flatten', 'dense-4096-relu', 'dropout-p5', 'dense-4096-relu', 'dropout-p5',
			 'dense_classes', 'output'])

		model_graph = [conv_module_1, conv_module_2, conv_module_3, conv_module_4, conv_module_5, 
			head_module]

		graphObject = Graph(model_graph, graph_util.hash_graph(model_graph, hash_algo))

	elif model_name == 'resnet18':
		model_graph = []

		conv_module_1 = (np.eye(4, k=1, dtype=np.int8),
			['input', 'conv7x7-c64-p3-s2-bn-relu', 'maxpool3x3-p1-s2', 'output'])

		model_graph.append(conv_module_1)

		proj_mat = np.eye(4, k=1, dtype=np.int8)
		proj_mat[0, -1] = 1

		conv_module_2 = (proj_mat,
			['input', 'conv3x3-c64-p1-bn-relu', 'conv3x3-c64-p1-bn-relu', 'output'])

		model_graph.extend((conv_module_2,) * 2)

		conv_module_3 = (proj_mat,
			['input', 'conv3x3-c128-p1-s2-bn-relu', 'conv3x3-c128-p1-s2-bn-relu', 'output'])

		model_graph.extend((conv_module_3,) * 2)

		conv_module_4 = (proj_mat,
			['input', 'conv3x3-c256-p1-s2-bn-relu', 'conv3x3-c256-p1-s2-bn-relu', 'output'])

		model_graph.extend((conv_module_4,) * 2)

		conv_module_5 = (proj_mat,
			['input', 'conv3x3-c512-p1-s2-bn-relu', 'conv3x3-c512-p1-s2-bn-relu', 'output'])

		model_graph.extend((conv_module_5,) * 2)

		head_module = (np.eye(4, k=1, dtype=np.int8),
			['input', 'global-avg-pool', 'dense_classes', 'output'])

		model_graph.append(head_module)

		graphObject = Graph(model_graph, graph_util.hash_graph(model_graph, hash_algo))

	elif model_name == 'resnet34':
		model_graph = []

		conv_module_1 = (np.eye(4, k=1, dtype=np.int8),
			['input', 'conv7x7-c64-p3-s2-bn-relu', 'maxpool3x3-p1-s2', 'output'])

		model_graph.append(conv_module_1)

		proj_mat = np.eye(4, k=1, dtype=np.int8)
		proj_mat[0, -1] = 1

		conv_module_2 = (proj_mat,
			['input', 'conv3x3-c64-p1-bn-relu', 'conv3x3-c64-p1-bn-relu', 'output'])

		model_graph.extend((conv_module_2,) * 3)

		conv_module_3 = (proj_mat,
			['input', 'conv3x3-c128-p1-s2-bn-relu', 'conv3x3-c128-p1-s2-bn-relu', 'output'])

		model_graph.extend((conv_module_3,) * 4)

		conv_module_4 = (proj_mat,
			['input', 'conv3x3-c256-p1-s2-bn-relu', 'conv3x3-c256-p1-s2-bn-relu', 'output'])

		model_graph.extend((conv_module_4,) * 6)

		conv_module_5 = (proj_mat,
			['input', 'conv3x3-c512-p1-s2-bn-relu', 'conv3x3-c512-p1-s2-bn-relu', 'output'])

		model_graph.extend((conv_module_5,) * 3)

		head_module = (np.eye(4, k=1, dtype=np.int8),
			['input', 'global-avg-pool', 'dense_classes', 'output'])

		model_graph.append(head_module)

		graphObject = Graph(model_graph, graph_util.hash_graph(model_graph, hash_algo))

	elif model_name == 'resnet50':
		model_graph = []

		conv_module_1 = (np.eye(4, k=1, dtype=np.int8),
			['input', 'conv7x7-c64-p3-s2-bn-relu', 'maxpool3x3-p1-s2', 'output'])

		model_graph.append(conv_module_1)

		proj_mat = np.eye(5, k=1, dtype=np.int8)
		proj_mat[0, -1] = 1

		conv_module_2 = (proj_mat,
			['input', 'conv1x1-c64-bn-relu', 'conv3x3-c64-p1-bn-relu', 
			 'conv1x1-c256-bn-relu', 'output'])

		model_graph.extend((conv_module_2,) * 3)

		conv_module_3 = (proj_mat,
			['input', 'conv1x1-c128-bn-relu', 'conv3x3-c128-p1-s2-bn-relu', 
			 'conv1x1-c512-bn-relu', 'output'])

		model_graph.extend((conv_module_3,) * 4)

		conv_module_4 = conv_module_3 = (proj_mat,
			['input', 'conv1x1-c256-bn-relu', 'conv3x3-c256-p1-s2-bn-relu', 
			 'conv1x1-c1024-bn-relu', 'output'])

		model_graph.extend((conv_module_4,) * 6)

		conv_module_5 = (proj_mat,
			['input', 'conv1x1-c512-bn-relu', 'conv3x3-c512-p1-s2-bn-relu', 
			 'conv1x1-c2048-bn-relu', 'output'])

		model_graph.extend((conv_module_5,) * 3)

		head_module = (np.eye(4, k=1, dtype=np.int8),
			['input', 'global-avg-pool', 'dense_classes', 'output'])

		model_graph.append(head_module)

		graphObject = Graph(model_graph, graph_util.hash_graph(model_graph, hash_algo))

	elif model_name == 'resnet101':
		model_graph = []

		conv_module_1 = (np.eye(4, k=1, dtype=np.int8),
			['input', 'conv7x7-c64-p3-s2-bn-relu', 'maxpool3x3-p1-s2', 'output'])

		model_graph.append(conv_module_1)

		proj_mat = np.eye(5, k=1, dtype=np.int8)
		proj_mat[0, -1] = 1

		conv_module_2 = (proj_mat,
			['input', 'conv1x1-c64-bn-relu', 'conv3x3-c64-p1-bn-relu', 
			 'conv1x1-c256-bn-relu', 'output'])

		model_graph.extend((conv_module_2,) * 3)

		conv_module_3 = (proj_mat,
			['input', 'conv1x1-c128-bn-relu', 'conv3x3-c128-p1-s2-bn-relu', 
			 'conv1x1-c512-bn-relu', 'output'])

		model_graph.extend((conv_module_3,) * 4)

		conv_module_4 = conv_module_3 = (proj_mat,
			['input', 'conv1x1-c256-bn-relu', 'conv3x3-c256-p1-s2-bn-relu', 
			 'conv1x1-c1024-bn-relu', 'output'])

		model_graph.extend((conv_module_4,) * 23)

		conv_module_5 = (proj_mat,
			['input', 'conv1x1-c512-bn-relu', 'conv3x3-c512-p1-s2-bn-relu', 
			 'conv1x1-c2048-bn-relu', 'output'])

		model_graph.extend((conv_module_5,) * 3)

		head_module = (np.eye(4, k=1, dtype=np.int8),
			['input', 'global-avg-pool', 'dense_classes', 'output'])

		model_graph.append(head_module)

		graphObject = Graph(model_graph, graph_util.hash_graph(model_graph, hash_algo))

	elif model_name == 'resnet152':
		model_graph = []

		conv_module_1 = (np.eye(4, k=1, dtype=np.int8),
			['input', 'conv7x7-c64-p3-s2-bn-relu', 'maxpool3x3-p1-s2', 'output'])

		model_graph.append(conv_module_1)

		proj_mat = np.eye(5, k=1, dtype=np.int8)
		proj_mat[0, -1] = 1

		conv_module_2 = (proj_mat,
			['input', 'conv1x1-c64-bn-relu', 'conv3x3-c64-p1-bn-relu', 
			 'conv1x1-c256-bn-relu', 'output'])

		model_graph.extend((conv_module_2,) * 3)

		conv_module_3 = (proj_mat,
			['input', 'conv1x1-c128-bn-relu', 'conv3x3-c128-p1-s2-bn-relu', 
			 'conv1x1-c512-bn-relu', 'output'])

		model_graph.extend((conv_module_3,) * 8)

		conv_module_4 = conv_module_3 = (proj_mat,
			['input', 'conv1x1-c256-bn-relu', 'conv3x3-c256-p1-s2-bn-relu', 
			 'conv1x1-c1024-bn-relu', 'output'])

		model_graph.extend((conv_module_4,) * 36)

		conv_module_5 = (proj_mat,
			['input', 'conv1x1-c512-bn-relu', 'conv3x3-c512-p1-s2-bn-relu', 
			 'conv1x1-c2048-bn-relu', 'output'])

		model_graph.extend((conv_module_5,) * 3)

		head_module = (np.eye(4, k=1, dtype=np.int8),
			['input', 'global-avg-pool', 'dense_classes', 'output'])

		model_graph.append(head_module)

		graphObject = Graph(model_graph, graph_util.hash_graph(model_graph, hash_algo))

	elif model_name == 'shufflenet':
		model_graph = []

		conv_module_1 = (np.eye(4, k=1, dtype=np.int8),
			['input', 'conv3x3-c24-p1-bn-relu', 'maxpool3x3-p1-s2', 'output'])

		model_graph.append(conv_module_1)

		shuffle_matrix_1 = np.array([[0, 1, 0, 0, 0, 1, 0],
									 [0, 0, 1, 0, 0, 0, 0],
									 [0, 0, 0, 1, 0, 0, 0],
									 [0, 0, 0, 0, 1, 0, 0],
									 [0, 0, 0, 0, 0, 0, 1],
									 [0, 0, 0, 0, 0, 0, 1],
									 [0, 0, 0, 0, 0, 0, 0]], dtype=np.int8)

		shuffle_matrix_2 = np.array([[0, 1, 0, 0, 0, 1],
									 [0, 0, 1, 0, 0, 0],
									 [0, 0, 0, 1, 0, 0],
									 [0, 0, 0, 0, 1, 0],
									 [0, 0, 0, 0, 0, 1],
									 [0, 0, 0, 0, 0, 0]], dtype=np.int8)

		repetitions = [3, 7, 3]
		input_channels = [24, 384, 768]
		channels = [384, 768, 1536]

		for i in range(len(channels)):
			conv_module_2 = (shuffle_matrix_1,
				['input', f'conv1x1-c{channels[i]//4}-g8-bn-relu', 'channel_shuffle-g8', 
				f'conv3x3-c{channels[i]//4}-dw-p1-s2-bn-relu', 
				f'conv3x3-c{channels[i] - input_channels[i]}-g8-p1-bn-relu', 'avgpool3x3-p1-s2', 'output'])

			model_graph.append(conv_module_2)

			conv_module_3 = (shuffle_matrix_2,
				['input', f'conv1x1-c{channels[i]//4}-g8-bn-relu', 'channel_shuffle-g8', 
				f'conv3x3-c{channels[i]//4}-dw-p1-bn-relu', 
				f'conv3x3-c{channels[i]}-g8-p1-bn-relu', 'output'])

			model_graph.extend((conv_module_3,) * repetitions[i])

		head_module = (np.eye(4, k=1, dtype=np.int8),
			['input', 'global-avg-pool', 'dense_classes', 'output'])

		model_graph.append(head_module)

		graphObject = Graph(model_graph, graph_util.hash_graph(model_graph, hash_algo))

	elif model_name == 'mobilenet':
		model_graph = []

		conv_module_1 = (np.eye(3, k=1, dtype=np.int8),
			['input', 'conv3x3-c32-s2-bn-relu', 'output'])

		model_graph.append(conv_module_1)

		channels = [64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024]
		strides = [1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1]

		for i in range(len(channels)):
			if i == 0: input_channels = 32
			
			conv_temp = f'conv3x3-c{input_channels}-dw-p1-s{strides[i]}-bn-relu' if strides[i] != 1 else f'conv3x3-c{input_channels}-dw-p1-bn-relu'
			conv_module = (np.eye(4, k=1, dtype=np.int8),
				['input', conv_temp, 
				f'conv1x1-c{channels[i]}-bn-relu', 'output'])
			input_channels = channels[i]

			model_graph.append(conv_module)

		head_module = (np.eye(4, k=1, dtype=np.int8),
			['input', 'global-avg-pool', 'dense_classes', 'output'])

		model_graph.append(head_module)

		graphObject = Graph(model_graph, graph_util.hash_graph(model_graph, hash_algo))

	elif model_name == 'googlenet' or model_name == 'inception':
		model_graph = []

		conv_module_1 = (np.eye(7, k=1, dtype=np.int8),
			['input', 'conv7x7-c64-p3-s2-bn-relu', 'maxpool3x3-s2', 'conv1x1-c64-bn-relu',
			 'conv3x3-c192-p1-bn-relu', 'maxpool3x3-s2', 'output'])

		model_graph.append(conv_module_1)

		inception_matrix = np.array([[0, 1, 1, 0, 1, 0, 1, 0, 0],
									 [0, 0, 0, 0, 0, 0, 0, 0, 1],
									 [0, 0, 0, 1, 0, 0, 0, 0, 0],
									 [0, 0, 0, 0, 0, 0, 0, 0, 1],
									 [0, 0, 0, 0, 0, 1, 0, 0, 0],
									 [0, 0, 0, 0, 0, 0, 0, 0, 1],
									 [0, 0, 0, 0, 0, 0, 0, 1, 0],
									 [0, 0, 0, 0, 0, 0, 0, 0, 1],
									 [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int8)

		channels = [[64, 96, 128, 16, 32, 32],
					[128, 128, 192, 32, 96, 64],
					[192, 96, 208, 16, 48, 64],
					[160, 112, 224, 24, 64, 64],
					[128, 128, 256, 24, 64, 64],
					[112, 144, 288, 32, 64, 64],
					[256, 160, 320, 32, 128, 128],
					[256, 160, 320, 32, 128, 128],
					[384, 192, 384, 48, 128, 128]]

		for i in range(len(channels)):
			conv_module = (inception_matrix,
				['input', f'conv1x1-c{channels[i][0]}-bn-relu', f'conv1x1-c{channels[i][1]}-bn-relu', 
				f'conv3x3-c{channels[i][2]}-p1-bn-relu', f'conv1x1-c{channels[i][3]}-bn-relu',
				f'conv5x5-c{channels[i][4]}-p1-bn-relu', 'maxpool3x3-p1-s2', 
				f'conv1x1-c{channels[i][5]}-bn-relu', 'output'])

			model_graph.append(conv_module)

			if i == 1 or i == 6:
				max_pool_module = (np.eye(3, k=1, dtype=np.int8),
					['input', 'maxpool3x3-s2', 'output'])

				model_graph.append(max_pool_module)
		
		head_module = (np.eye(5, k=1, dtype=np.int8),
			['input', 'global-avg-pool', 'dropout-p4', 'dense_classes', 'output'])

		model_graph.append(head_module)

		graphObject = Graph(model_graph, graph_util.hash_graph(model_graph, hash_algo))

	elif model_name == 'xception':
		model_graph = []

		conv_module_1 = (np.eye(4, k=1, dtype=np.int8),
			['input', 'conv3x3-c32-s2-bn-relu', 'conv3x3-c64-bn-relu', 'output'])

		model_graph.append(conv_module_1)

		proj_mat = np.eye(7, k=1, dtype=np.int8)
		proj_mat[0, -1] = 1

		input_channels = 64
		output_channels = [128, 256, 728]

		for i in range(3):
			conv_module = (proj_mat,
				['input', f'conv3x3-c{input_channels}-dw-p1-bn-relu', f'conv1x1-c{output_channels[i]}-bn-relu',
				f'conv3x3-c{output_channels[i]}-dw-p1-bn-relu', f'conv1x1-c{output_channels[i]}-bn-relu',
				 'maxpool3x3-p1-s2', 'output'])
			input_channels = output_channels[i]
			model_graph.append(conv_module)

		proj_mat = np.eye(8, k=1, dtype=np.int8)
		proj_mat[0, -1] = 1

		for _ in range(8):
			conv_module = (proj_mat,
				['input', 'conv3x3-c728-dw-p1-bn-relu', 'conv1x1-c728-bn-relu',
				 'conv3x3-c728-dw-p1-bn-relu', 'conv1x1-c728-bn-relu',
				 'conv3x3-c728-dw-p1-bn-relu', 'conv1x1-c728-bn-relu',
				 'output'])
			model_graph.append(conv_module)

		proj_mat = np.eye(7, k=1, dtype=np.int8)
		proj_mat[0, -1] = 1

		conv_module = (proj_mat,
				['input', f'conv3x3-c728-dw-bn-relu', f'conv1x1-c728-bn-relu',
				f'conv3x3-c728-dw-bn-relu', f'conv1x1-c1024-bn-relu',
				 'maxpool3x3-p1-s2', 'output'])

		model_graph.append(conv_module)

		conv_module = (np.eye(6, k=1, dtype=np.int8),
			['input', 'conv3x3-c1024-dw-p1-bn-relu', 'conv1x1-c1536-bn-relu', 
			 'conv3x3-c1536-dw-p1-bn-relu', 'conv1x1-c2048-bn-relu', 'output'])

		model_graph.append(conv_module)

		head_module = (np.eye(5, k=1, dtype=np.int8),
			['input', 'global-avg-pool', 'dense-1024-relu', 'dense_classes', 'output'])

		model_graph.append(head_module)

		graphObject = Graph(model_graph, graph_util.hash_graph(model_graph, hash_algo))

	elif model_name.startswith('efficientnet'):

		block_args = [
			{'kernel_size': 3, 'num_repeat': 1, 'input_filters': 32, 'output_filters': 16,
				'expand_ratio': 1, 'id_skip': True, 'stride': 1, 'se_ratio': 0.25},
			{'kernel_size': 3, 'num_repeat': 2, 'input_filters': 16, 'output_filters': 24,
				'expand_ratio': 6, 'id_skip': True, 'stride': 2, 'se_ratio': 0.25},
			{'kernel_size': 5, 'num_repeat': 2, 'input_filters': 24, 'output_filters': 40,
				'expand_ratio': 6, 'id_skip': True, 'stride': 2, 'se_ratio': 0.25},
			{'kernel_size': 3, 'num_repeat': 3, 'input_filters': 40, 'output_filters': 80,
				'expand_ratio': 6, 'id_skip': True, 'stride': 2, 'se_ratio': 0.25},
			{'kernel_size': 5, 'num_repeat': 3, 'input_filters': 80, 'output_filters': 112,
				'expand_ratio': 6, 'id_skip': True, 'stride': 1, 'se_ratio': 0.25},
			{'kernel_size': 5, 'num_repeat': 4, 'input_filters': 112, 'output_filters': 192,
				'expand_ratio': 6, 'id_skip': True, 'stride': 2, 'se_ratio': 0.25},
			{'kernel_size': 3, 'num_repeat': 1, 'input_filters': 192, 'output_filters': 320,
				'expand_ratio': 6, 'id_skip': True, 'stride': 1, 'se_ratio': 0.25}
		]

		def _round_filters(filters, width_coefficient, depth_divisor):
			"""Round number of filters based on width multiplier."""

			filters *= width_coefficient
			new_filters = int(filters + depth_divisor / 2) // depth_divisor * depth_divisor
			new_filters = max(depth_divisor, new_filters)
			# Make sure that round down does not go down by more than 10%.
			if new_filters < 0.9 * filters:
				new_filters += depth_divisor
			return int(new_filters)


		def _round_repeats(repeats, depth_coefficient):
			"""Round number of repeats based on depth multiplier."""

			return int(math.ceil(depth_coefficient * repeats))

		def _mb_conv_block(block_args, input_filters, output_filters, stride, drop_rate=None):
			"""Mobile Inverted Residual Bottleneck."""

			has_se = (block_args['se_ratio'] is not None) and (0 < block_args['se_ratio'] <= 1)

			labels = ['input']

			# Expansion phase
			filters = input_filters * block_args['expand_ratio']
			if block_args['expand_ratio'] != 1:
				labels.append(f'conv1x1-c{filters}-bn-relu')

			# Depthwise Convolution
			padding = 1 if block_args["kernel_size"] == 3 else 2
			labels.append(f'conv{block_args["kernel_size"]}x{block_args["kernel_size"]}-s{stride}-dw-p{padding}-bn-silu')
			in_phase_idx = len(labels) - 1

			# Squeeze and Excitation phase
			if has_se:
				num_reduced_filters = max(1, int(input_filters * block_args['se_ratio']))

				# We use add operation instead of multiply. Addition would use interpolation
				# while creating the PyTorch model
				labels.append('global-avg-pool')
				labels.append(f'conv1x1-c{num_reduced_filters}-bn-silu')
				labels.append(f'conv1x1-c{filters}-bn-silu')

				se_idx = len(labels) - 1

			# Output phase
			labels.append(f'conv1x1-c{output_filters}-bn-silu')
			out_phase_idx = len(labels) - 1

			skip_connect = False

			if block_args['id_skip'] and stride == 1 \
				and input_filters == output_filters:

				if drop_rate and (drop_rate > 0):
					labels.append(f'dropout-p{round(drop_rate * 10)}')

				skip_connect = True

			labels.append('output')

			matrix = np.eye(len(labels), k=1, dtype=np.int8)

			if skip_connect: matrix[0, -1] = 1
			if has_se: matrix[in_phase_idx, out_phase_idx] = 1

			return (matrix, labels)
		
		# Adapted from the Keras implementation: 
		# https://github.com/qubvel/efficientnet/blob/master/efficientnet/model.py
		def _efficientnet(width_coefficient, 
						  depth_coefficient, 
						  resolution, 
						  dropout_rate=0.2, 
						  drop_connect_rate=0.2,
						  depth_divisor=8,
						  block_args=block_args): 
			model_graph = []

			filters = _round_filters(32, width_coefficient, depth_divisor)

			if resolution == config['image_size']:
				conv_module_1 = (np.eye(3, k=1, dtype=np.int8),
					['input', f'conv3x3-c{filters}-p1-s2-bn-silu', 'output'])
			else:
				assert resolution > config['image_size']
				conv_module_1 = (np.eye(4, k=1, dtype=np.int8),
					['input', f'upsample-s{resolution}', f'conv3x3-c{filters}-p1-s2-bn-silu', 'output'])

			model_graph.append(conv_module_1)

			num_blocks_total = sum(block_args['num_repeat'] for block_args in block_args)
			block_num = 0

			for idx, block_args in enumerate(block_args):
				# Update block input and output filters based on depth multiplier.
				input_filters = _round_filters(block_args['input_filters'], width_coefficient, depth_divisor)
				output_filters = _round_filters(block_args['output_filters'], width_coefficient, depth_divisor)
				num_repeat = _round_repeats(block_args['num_repeat'], depth_coefficient)

				# The first block needs to take care of stride and filter size increase.
				drop_rate = drop_connect_rate * float(block_num) / num_blocks_total

				model_graph.append(_mb_conv_block(block_args, input_filters, output_filters, 
					block_args['stride'], drop_rate))

				block_num += 1
				if num_repeat > 1:
					input_filters = output_filters

					for bidx in range(num_repeat - 1):
						drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
						model_graph.append(_mb_conv_block(block_args, input_filters, output_filters, 1, drop_rate))

						block_num += 1

			conv_module = (np.eye(3, k=1, dtype=np.int8),
				['input', f'conv1x1-c{_round_filters(1280, width_coefficient, depth_divisor)}-bn-relu', 'output'])

			model_graph.append(conv_module)

			head_module = (np.eye(5, k=1, dtype=np.int8),
				['input', 'global-avg-pool', f'dropout-p{round(drop_rate * 10)}', 'dense_classes', 'output'])

			model_graph.append(head_module)

			return model_graph

		if model_name == 'efficientnet-b0':
			model_graph = _efficientnet(1.0, 1.0, 224, 0.2)
		elif model_name == 'efficientnet-b1':
			model_graph = _efficientnet(1.0, 1.1, 240, 0.2)
		elif model_name == 'efficientnet-b2':
			model_graph = _efficientnet(1.1, 1.2, 260, 0.3)
		elif model_name == 'efficientnet-b3':
			model_graph = _efficientnet(1.2, 1.4, 300, 0.3)
		elif model_name == 'efficientnet-b4':
			model_graph = _efficientnet(1.4, 1.8, 380, 0.4)
		elif model_name == 'efficientnet-b5':
			model_graph = _efficientnet(1.6, 2.2, 465, 0.4)
		elif model_name == 'efficientnet-b6':
			model_graph = _efficientnet(1.8, 2.6, 528, 0.5)
		elif model_name == 'efficientnet-b7':
			model_graph = _efficientnet(2.0, 3.1, 600, 0.5)
		elif model_name == 'efficientnet-l2':
			model_graph = _efficientnet(4.3, 5.3, 800, 0.5)

		graphObject = Graph(model_graph, graph_util.hash_graph(model_graph, hash_algo))

	return graphObject


if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description='Input parameters for training a manually defined model',
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
	  default='alexnet',
	  help=f'Name of the manual model to train. Should be in: {SUPPORTED_MODELS}')
	parser.add_argument(
	  '--model_dir',
	  metavar='',
	  type=str,
	  default=None,
	  help=f'Name of the directory to save the model. Defaults to "../results/manual_models/<model_name>"')
	parser.add_argument(
	  '--auto_tune',
	  default=False,
	  action='store_true',
	  help=f'To tune the model training hyper-parameters')
	parser.add_argument(
	  '--device',
	  metavar='',
	  type=str,
	  default=None,
	  help='Device to train the model')

	args = parser.parse_args()

	with open(args.config) as config_file:
		try:
			config = yaml.safe_load(config_file)
		except yaml.YAMLError as exc:
			print(exc)

	model_graph = get_manual_graph(config, args.model_name)

	device = None
	if args.device is not None:
		device = torch.device(args.device)

	if args.model_dir is None:
		model_dir = os.path.join('../results/manual_models/', args.model_name)
	else:
		model_dir = args.model_dir
	
	worker(config, model_graph, device, model_dir, auto_tune=args.auto_tune)