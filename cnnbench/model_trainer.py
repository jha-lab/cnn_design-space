# Builds PyTorch model for the given Graph Object

# Author : Shikhar Tuli


import os
import sys

import torch
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from inspect import getmembers

from input_pipeline import get_loader
from model_builder import CNNBenchModel

from library import GraphLib, Graph
from utils import print_util as pu


LOG_INTERVAL = 10


def worker(config: dict, graphObject: 'Graph', device: 'torch.device' = None):
	"""Trains a CNN model on a given device
	
	Args:
		config (dict): dictionary of configuration
		graphObject (Graph): Graph object
		device (str, optional): cuda device
	"""
	if device is None:
		device = 'cuda' if torch.cuda.is_available() else 'cpu'
	
	train_loader, test_loader = get_loader(config)

	model = CNNBenchModel(config, graphObject)

	model.to(device)

	optims = [opt[0] for opt in getmembers(optim)]

	if 'optimizer' in config.keys():
		if config['optimizer'] not in optims:
			raise ValueError(f'Optimizer {config["optimizer"]} not supported in PyTorch')
		optimizer = eval(f'optim.{config["optimizer"]}(model.parameters(), lr=config["lr"])')
	else:
		if 'lr' in config.keys():
			optimizer = optim.SGD(model.parameters(), lr=config['lr'])
		else:
			# Use default learning rate
			optimizer = optim.SGD(model_parameters())

	shdlrs = [sh[0] for sh in getmembers(optim.lr_scheduler)]

	if 'scheduler' in config.keys():
		if config['scheduler'] not in shdlrs:
			raise ValueError(f'Scheduler {config["scheduler"]} not supported in PyTorch')
		scheduler = eval(f'optim.lr_scheduler.{config["scheduler"]}(optimizer, **config["scheduler_args"])')


	for epoch in range(config['epochs']):
		# Run training
		model.train()
		for batch_idx, (data, target) in enumerate(train_loader):
			data, target = data.to(device), target.to(device)

			optimizer.zero_grad()
			output = model(data)
			loss = F.cross_entropy(output, target)
			loss.backward()
			optimizer.step()

			if 'scheduler' in config.keys() and config['scheduler'] == 'CosineAnnealingWarmRestarts':
				scheduler.step(epoch + batch_idx / len(train_loader))
			
			if batch_idx % LOG_INTERVAL == 0:
				print('Train Epoch: {} [{:6d}/{:6d} ({:.0f}%)]\tLearning Rate: {:.6f}\tLoss: {:.6f}'.format(
					epoch, batch_idx * len(data), len(train_loader.dataset),
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

		if 'scheduler' in config.keys() and config['scheduler'] == 'ReduceLROnPlateau':
			scheduler.step(test_loss)
		elif 'scheduler' in config.keys():
			scheduler.step()

	if not os.path.exists(os.path.join(config['models_dir'], config['dataset'], graphObject.hash)):
		os.makedirs(os.path.join(config['models_dir'], config['dataset'], graphObject.hash))

	model_path = os.path.join(config['models_dir'],
									config['dataset'],
									graphObject.hash, 
									f'model.pt')
	torch.save(model.state_dict(), model_path)
	print(f'{pu.bcolors.OKGREEN}Saved trained model to:{pu.bcolors.ENDC}\n{model_path}')
