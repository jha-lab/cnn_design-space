# Builds PyTorch model for the given Graph Object

# Author : Shikhar Tuli


import os
import sys

import torch
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from inspect import getmembers
import json
import time

from input_pipeline import get_loader
from model_builder import CNNBenchModel

from library import GraphLib, Graph
from utils import print_util as pu

import matplotlib as mpl
mpl.use('Agg')

from matplotlib import pyplot as plt


LOG_INTERVAL = 10


def worker(config: dict, graphObject: 'Graph', device: torch.device = None, model_dir: str = None, save_fig=True):
	"""Trains a CNN model on a given device
	
	Args:
	    config (dict): dictionary of configuration
	    graphObject (Graph): Graph object
	    device (torch.device, optional): cuda device
	    model_dir (str, optional): directory to store the model and metrics
	    save_fig (bool, optional): to save the learning curves
	
	Raises:
	    ValueError: if an input parameter is not supported, or GPU device isn't found
	"""
	torch.manual_seed(0)

	if device is None:
		if torch.cuda.is_available():
			device = 'cuda'
		else:
			raise ValueError('No GPU device found!') 
	
	train_loader, test_loader = get_loader(config)

	model = CNNBenchModel(config, graphObject)

	model_params = model.get_params()

	if 'manual_models' in model_dir:
		model_name = model_dir.split('/')[-1]
	else:
		model_name = graphObject.hash

	if model_dir is None:
		if not os.path.exists(os.path.join(config['models_dir'], config['dataset'], graphObject.hash)):
			os.makedirs(os.path.join(config['models_dir'], config['dataset'], graphObject.hash))
		metrics_path = os.path.join(config['models_dir'], config['dataset'], graphObject.hash, 'metrics.json')
		model_path = os.path.join(config['models_dir'], config['dataset'], graphObject.hash, 'model.pt')
		fig_path = os.path.join(config['models_dir'], config['dataset'], graphObject.hash, 'curves.png')
	else:
		if not os.path.exists(model_dir):
			os.makedirs(model_dir)
		metrics_path = os.path.join(model_dir, 'metrics.json')
		model_path = os.path.join(model_dir, 'model.pt')
		fig_path = os.path.join(model_dir, 'curves.png')

	model.to(device)

	optims = [opt[0] for opt in getmembers(optim)]

	if 'optimizer' in config.keys():
		if config['optimizer'] not in optims:
			raise ValueError(f'Optimizer {config["optimizer"]} not supported in PyTorch')
		optimizer = eval(f'optim.{config["optimizer"]}(model.parameters(), **config["optimizer_args"])')
	else:
		optimizer = optim.SGD(model_parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

	shdlrs = [sh[0] for sh in getmembers(optim.lr_scheduler)]

	if 'scheduler' in config.keys():
		if config['scheduler'] not in shdlrs:
			raise ValueError(f'Scheduler {config["scheduler"]} not supported in PyTorch')
		scheduler = eval(f'optim.lr_scheduler.{config["scheduler"]}(optimizer, **config["scheduler_args"])')

	losses = []
	epochs = []
	learning_rates = []
	train_accuracies = []
	test_accuracies = []
	times = []

	start_time = time.time()

	batch_size = 0

	for epoch in range(config['epochs']):
		# Run training
		model.train()
		for batch_idx, (data, target) in enumerate(train_loader):
			data, target = data.to(device), target.to(device)

			if epoch == 0 and batch_idx == 0:
				batch_size = len(data)

			optimizer.zero_grad()
			output = model(data)
			loss = F.cross_entropy(output, target)
			loss.backward()
			optimizer.step()

			if 'scheduler' in config.keys() and config['scheduler'] == 'CosineAnnealingWarmRestarts':
				scheduler.step(epoch + batch_idx / len(train_loader))
			
			if batch_idx % LOG_INTERVAL == 0:
				print('Train Epoch: {} [{:6d}/{:6d} ({:.0f}%)]\tLearning Rate: {:.6f}\tLoss: {:.6f}'.format(
					epoch, batch_idx * batch_size, len(train_loader.dataset),
					100. * batch_idx / len(train_loader), optimizer.param_groups[0]['lr'], loss.item()))

		# Run validation
		model.eval()

		with torch.no_grad():
			train_loss = 0
			correct = 0
			for data, target in train_loader:
				data, target = data.to(device), target.to(device)
				output = model(data)
				train_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
				pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
				correct += pred.eq(target.view_as(pred)).sum().item()

			train_loss /= len(train_loader.dataset)

			print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
				train_loss, correct, len(train_loader.dataset),
				100. * correct / len(train_loader.dataset)))

			train_accuracies.append(100. * correct / len(train_loader.dataset))

			test_loss = 0
			correct = 0
			for data, target in test_loader:
				data, target = data.to(device), target.to(device)
				output = model(data)
				test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
				pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
				correct += pred.eq(target.view_as(pred)).sum().item()

			test_loss /= len(test_loader.dataset)

			print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
				test_loss, correct, len(test_loader.dataset),
				100. * correct / len(test_loader.dataset)))

			test_accuracies.append(100. * correct / len(test_loader.dataset))

		losses.append(train_loss)
		epochs.append(epoch)
		learning_rates.append(optimizer.param_groups[0]['lr'])

		if 'scheduler' in config.keys():
			if config['scheduler'] == 'ReduceLROnPlateau':
				scheduler.step(test_loss)
			elif config['scheduler'] != 'CosineAnnealingWarmRestarts':
				scheduler.step()

		times.append(time.time() - start_time)

		if save_fig:
			# Save figure of learning curves
			fig, ax1 = plt.subplots()
			ax2 = ax1.twinx()

			loss, = ax1.plot(epochs, losses, 'b-', label='Training Loss')
			train_acc, = ax2.plot(epochs, train_accuracies, 'r-', label='Training Accuracy')
			test_acc, = ax2.plot(epochs, test_accuracies, 'r--', label='Test Accuracy')

			ax1.set_xlabel('Epochs')
			ax1.set_ylabel('Loss')
			ax2.set_ylabel('Accuracy (%)')
			ax1.yaxis.label.set_color(loss.get_color())
			ax2.yaxis.label.set_color(train_acc.get_color())
			ax1.tick_params(axis='y', colors=loss.get_color())
			ax2.tick_params(axis='y', colors=train_acc.get_color())
			ax1.legend(handles=[loss, train_acc, test_acc], loc='center right')

			plt.title(f'Model: {model_name}. Params: {pu.human_format(model_params)}. Time: {times[-1]/3600 : 0.2f}h')

			plt.savefig(fig_path)

	# Save metrics to a json file
	with open(metrics_path, 'w', encoding='utf8') as json_file:
		json.dump({'model_hash': graphObject.hash,
				   'model_params': model_params,
				   'epochs': epochs,
				   'losses': losses,
				   'learning_rates': learning_rates,
				   'train_accuracies': train_accuracies,
				   'test_accuracies': test_accuracies,
				   'times': times}, 
				   json_file, ensure_ascii=True)
		print(f'{pu.bcolors.OKGREEN}Saved metrics to:{pu.bcolors.ENDC}\n{metrics_path}')		

	# Save trained model
	torch.save(model.state_dict(), model_path)

	print(f'{pu.bcolors.OKGREEN}Saved trained model to:{pu.bcolors.ENDC}\n{model_path}')
