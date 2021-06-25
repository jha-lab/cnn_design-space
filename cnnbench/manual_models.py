# Builds CNNBench specification graphs for popular networks

# Author : Shikhar Tuli


import os
import sys

import numpy as np
import yaml
import argparse
import torch

from library import GraphLib, Graph
from model_trainer import worker
from utils import graph_util, print_util as pu


SUPPORTED_MODELS = ['lenet', 'alexnet', 'vgg11', 'vgg13',
					'vgg16', 'vgg19', 'resnet18', 'resnet34', 
					'resnet50', 'resnet101', 'resnet152',
					'shufflenet', 'mobilenet', 'googlenet',
					'inception', 'xception']
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
		conv_module = (np.eye(5, k=1, dtype=np.int8),
			['input', 'conv3x3-c6-bn-relu', 'conv3x3-c16-bn-relu', 'avgpool3x3', 'output'])

		head_module = (np.eye(6, k=1, dtype=np.int8),
			['input', 'flatten', 'dense-120-relu', 'dense-84-relu', 'dense_classes', 'output'])

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
			['input', 'conv7x7-c64-s2-p3-bn-relu', 'maxpool3x3-s2-p1', 'output'])

		model_graph.append(conv_module_1)

		proj_mat = np.eye(4, k=1, dtype=np.int8)
		proj_mat[0, -1] = 1

		conv_module_2 = (proj_mat,
			['input', 'conv3x3-c64-p1-s1-bn-relu', 'conv3x3-c64-p1-s1-bn-relu', 'output'])

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
			['input', 'conv7x7-c64-s2-p3-bn-relu', 'maxpool3x3-s2-p1', 'output'])

		model_graph.append(conv_module_1)

		proj_mat = np.eye(4, k=1, dtype=np.int8)
		proj_mat[0, -1] = 1

		conv_module_2 = (proj_mat,
			['input', 'conv3x3-c64-p1-s1-bn-relu', 'conv3x3-c64-p1-s1-bn-relu', 'output'])

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
			['input', 'conv7x7-c64-s2-p3-bn-relu', 'maxpool3x3-s2-p1', 'output'])

		model_graph.append(conv_module_1)

		proj_mat = np.eye(5, k=1, dtype=np.int8)
		proj_mat[0, -1] = 1

		conv_module_2 = (proj_mat,
			['input', 'conv1x1-c64-bn-relu', 'conv3x3-c64-p1-s1-bn-relu', 
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
			['input', 'conv7x7-c64-s2-p3-bn-relu', 'maxpool3x3-s2-p1', 'output'])

		model_graph.append(conv_module_1)

		proj_mat = np.eye(5, k=1, dtype=np.int8)
		proj_mat[0, -1] = 1

		conv_module_2 = (proj_mat,
			['input', 'conv1x1-c64-bn-relu', 'conv3x3-c64-p1-s1-bn-relu', 
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
			['input', 'conv7x7-c64-s2-p3-bn-relu', 'maxpool3x3-s2-p1', 'output'])

		model_graph.append(conv_module_1)

		proj_mat = np.eye(5, k=1, dtype=np.int8)
		proj_mat[0, -1] = 1

		conv_module_2 = (proj_mat,
			['input', 'conv1x1-c64-bn-relu', 'conv3x3-c64-p1-s1-bn-relu', 
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
			['input', 'conv3x3-c24-p1-bn-relu', 'maxpool3x3-s2-p1', 'output'])

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
				f'conv3x3-c{channels[i]//4}-dw-s2-p1-bn-relu', 
				f'conv3x3-c{channels[i] - input_channels[i]}-p1-g8-bn-relu', 'avgpool3x3-s2-p1', 'output'])

			model_graph.append(conv_module_2)

			conv_module_3 = (shuffle_matrix_2,
				['input', f'conv1x1-c{channels[i]//4}-g8-bn-relu', 'channel_shuffle-g8', 
				f'conv3x3-c{channels[i]//4}-dw-s1-p1-bn-relu', 
				f'conv3x3-c{channels[i]}-p1-g8-bn-relu', 'output'])

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
			
			conv_module = (np.eye(4, k=1, dtype=np.int8),
				['input', f'conv3x3-c{input_channels}-dw-s{strides[i]}-p1-bn-relu', 
				f'conv1x1-c{channels[i]}-s1-bn-relu', 'output'])
			input_channels = channels[i]

			model_graph.append(conv_module)

		head_module = (np.eye(4, k=1, dtype=np.int8),
			['input', 'global-avg-pool', 'dense_classes', 'output'])

		model_graph.append(head_module)

		graphObject = Graph(model_graph, graph_util.hash_graph(model_graph, hash_algo))

	elif model_name == 'googlenet' or model_name == 'inception':
		model_graph = []

		conv_module_1 = (np.eye(7, k=1, dtype=np.int8),
			['input', 'conv7x7-c64-s2-p3-bn-relu', 'maxpool3x3-s2', 'conv1x1-c64-bn-relu',
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
				f'conv5x5-c{channels[i][4]}-p1-bn-relu', 'maxpool3x3-s2-p1', 
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
				 'maxpool3x3-s2-p1', 'output'])
			input_channels = output_channels[i]
			model_graph.append(conv_module)

		proj_mat = np.eye(8, k=1, dtype=np.int8)
		proj_mat[0, -1] = 1

		for _ in range(8):
			conv_module = (proj_mat,
				['input', 'conv3x3-c728-p1-dw-bn-relu', 'conv1x1-c728-bn-relu',
				 'conv3x3-c728-dw-p1-bn-relu', 'conv1x1-c728-bn-relu',
				 'conv3x3-c728-dw-p1-bn-relu', 'conv1x1-c728-bn-relu',
				 'output'])
			model_graph.append(conv_module)

		proj_mat = np.eye(7, k=1, dtype=np.int8)
		proj_mat[0, -1] = 1

		conv_module = (proj_mat,
				['input', f'conv3x3-c728-dw-bn-relu', f'conv1x1-c728-bn-relu',
				f'conv3x3-c728-dw-bn-relu', f'conv1x1-c1024-bn-relu',
				 'maxpool3x3-s2-p1', 'output'])

		model_graph.append(conv_module)

		conv_module = (np.eye(6, k=1, dtype=np.int8),
			['input', 'conv3x3-c1024-dw-p1-bn-relu', 'conv1x1-c1536-bn-relu', 
			 'conv3x3-c1536-dw-p1-bn-relu', 'conv1x1-c2048-bn-relu', 'output'])

		model_graph.append(conv_module)

		head_module = (np.eye(5, k=1, dtype=np.int8),
			['input', 'global-avg-pool', 'dense-1024-relu', 'dense_classes', 'output'])

		model_graph.append(head_module)

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