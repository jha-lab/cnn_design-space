# Builds PyTorch model for the given Graph Object

# Author : Shikhar Tuli


import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

os.environ['TUNE_DISABLE_AUTO_CALLBACK_LOGGERS'] = "1"

import numpy as np
from inspect import getmembers
from functools import partial
from copy import deepcopy
import shutil
import hashlib
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
NUM_SAMPLES = 4
CKPT_INTERVAL = 1


def worker(config: dict, 
		   graphObject: 'Graph', 
		   device: torch.device = None, 
		   model_dir: str = None, 
		   auto_tune = False,
		   save_fig = False):
	"""Trains or tunes a CNN model with different training recipes
	
	Args:
	    config (dict): dictionary of configuration
	    graphObject (Graph): Graph object
	    device (torch.device, optional): cuda device
	    model_dir (str, optional): directory to store the model and metrics
	    auto_tune (bool, optional): to use ray-tune for automatic tuning of hyper-parameter,
	    	else defaults to the first training recipe in config
	    save_fig (bool, optional): to save the learning curves
	
	Raises:
	    ValueError: if an input parameter is not supported, or GPU device isn't found
	"""
	torch.manual_seed(0)

	if model_dir is None:
		if not os.path.exists(os.path.join(config['models_dir'], config['dataset'], graphObject.hash)):
			os.makedirs(os.path.join(config['models_dir'], config['dataset'], graphObject.hash))
	else:
		if not os.path.exists(model_dir):
			os.makedirs(model_dir)

	if not auto_tune:
		hp_config = {}
		
		if 'optimizer' in config.keys():
			hp_config['optimizer'] = config['optimizer']
		
		if 'scheduler' in config.keys():
			hp_config['scheduler'] = config['scheduler']

		train(hp_config, config, graphObject, device, model_dir, auto_tune=False, 
			checkpointing=True)

		checkpoints = os.listdir(model_dir)
		checkpoints = [c for c in checkpoints if c.startswith('model')]

		sorted_checkpoints = sorted(checkpoints, key=lambda x: int(x.split('_')[1].split('.')[0]))

		# TODO: implement saving model from checkpoints with best validation accuracy

		best_checkpoint = torch.load(sorted_checkpoints[-1])

		if save_fig: plot_metrics(best_checkpoint, model_dir)

	else:
		# Implementing a basic hyper-parameter search space
		hp_config = {'optimizer':
						tune.choice(
						[{'Adam': 
							{'lr': tune.loguniform(1e-5, 1e-2),
							'betas': [tune.uniform(0.8, 0.95), tune.uniform(0.9, 0.999)],
							'weight_decay': tune.loguniform(1e-5, 1e-3)}},
						{'AdamW': 
							{'lr': tune.loguniform(1e-5, 1e-2),
							'betas': [tune.uniform(0.8, 0.95), tune.uniform(0.9, 0.999)],
							'weight_decay': tune.loguniform(1e-5, 1e-3)}}]),	
					 'scheduler':
					 	tune.choice(
					 	[{'CosineAnnealingLR':
					 		{'T_max': 200}},
					 	 {'ExponentialLR':
					 		{'gamma': tune.uniform(0.8, 0.99)}}])}

		scheduler = ASHAScheduler(
	        metric="val_loss",
	        mode="min",
	        max_t=config['epochs'],
	        grace_period=1,
	        reduction_factor=2)

		reporter = CLIReporter(parameter_columns=['optimizer', 'scheduler'],
			metric_columns=["val_loss", "val_accuracy", "training_iteration"])

		assert torch.cuda.device_count() > 1, 'More than one GPU is required for automatic tuning'

		trial_config = deepcopy(config)
		trial_config['epochs'] = 1
		trial_hp_config = {}
		if 'optimizer' in config.keys():
			trial_hp_config['optimizer'] = config['optimizer']
		if 'scheduler' in config.keys():
			trial_hp_config['scheduler'] = config['scheduler']

		print(f'{pu.bcolors.OKBLUE}Trial run to test batch size{pu.bcolors.ENDC}')

		# Robust batch sizing for different model sizes
		while (trial_config['train_batch_size'] >= 1) and (trial_config['test_batch_size'] >= 1):
			print(f'{pu.bcolors.OKBLUE}Running trial with batch size:{pu.bcolors.ENDC} {trial_config["train_batch_size"]}')
			try:
				# Do a trial run
				trial_device = torch.device('cuda:0')
				train(trial_hp_config, trial_config, graphObject, trial_device, model_dir, auto_tune=False, 
					checkpointing=False, gpuFrac=0.45)
			except RuntimeError as e:
				if 'CUDA out of memory' in str(e):
					trial_config['train_batch_size'] = trial_config['train_batch_size']//2
					trial_config['test_batch_size'] = trial_config['test_batch_size']//2
				else:
					raise e
			except Exception as e:
				raise e
			else:
				break
		
		print(f'{pu.bcolors.OKGREEN}Selected batch size:{pu.bcolors.ENDC} {trial_config["train_batch_size"]}')

		small_batch_config = deepcopy(config)
		small_batch_config['train_batch_size'] = trial_config['train_batch_size']
		small_batch_config['test_batch_size'] = trial_config['test_batch_size']

		print(f'{pu.bcolors.OKBLUE}Running automatic hyper-parameter tuning{pu.bcolors.ENDC}')

		result = tune.run(
	        partial(train, 
	        	main_config=small_batch_config, 
	        	graphObject=graphObject, 
	        	device=device, 
	        	model_dir=model_dir,
	        	auto_tune=True,
	        	checkpointing=True,
	        	gpuFrac=None),
	        local_dir=model_dir,
	        resources_per_trial={'gpu': 0.5},
	        config=hp_config,
	        num_samples=NUM_SAMPLES,
	        scheduler=scheduler,
	        progress_reporter=reporter)

		best_trial = result.get_best_trial(metric="val_loss", mode="min", scope="last")
		best_hp_config = best_trial.config

		print(f'{pu.bcolors.OKGREEN}Best hyper-parameter set:{pu.bcolors.ENDC}\n{best_hp_config}')

		trial_dirs = os.listdir(os.path.join(model_dir, 'auto_tune'))
		best_model_dir = [t for t in trial_dirs if t.endswith(get_hp_hash(best_hp_config))]
		best_model_dir = os.path.join(model_dir, 'auto_tune', best_model_dir[0])

		with open(os.path.join(model_dir, 'best_model_dir.txt'), 'w+') as f:
			f.write(best_model_dir)

		with open(os.path.join(model_dir, 'best_model_dir.txt'), 'r') as f:
			best_model_dir = f.read()

		checkpoints = os.listdir(best_model_dir)
		checkpoints = [c for c in checkpoints if c.startswith('model')]

		sorted_checkpoints = sorted(checkpoints, key=lambda x: int(x.split('_')[1].split('.')[0]))

		# TODO: implement saving model from checkpoints with best validation accuracy

		best_checkpoint = torch.load(os.path.join(best_model_dir, sorted_checkpoints[-1]))

		if save_fig: plot_metrics(best_checkpoint, model_dir)

		torch.save(best_checkpoint, os.path.join(model_dir, 'best_tuned_model.ckpt'))

		# Perform clean-up of ray-tune files
		ray_log_dir = [d for d in os.listdir(model_dir) if d.startswith('DEFAULT')]
		
		try:
			shutil.rmtree(os.path.join(model_dir, ray_log_dir[0]))
		except IndexError:
			pass


def get_hp_hash(hp_config: dict):
    """Returns the hash for a given hyper-parameter configuration
    
    Args:
        hp_config (dict): hyper-parameter configuration dictionary
    """
    return hashlib.shake_128(str(hp_config).encode('utf-8')).hexdigest(5)


def plot_metrics(checkpoint, model_dir):
	"""Plot and save metrics for the given checkpoint
	"""
	epochs = checkpoint['epochs']
	train_losses = checkpoint['train_losses']
	val_losses = checkpoint['val_losses']
	train_accuracies = checkpoint['train_accuracies']
	test_accuracies = checkpoint['test_accuracies']
	model_params = checkpoint['model_params']
	model_name = checkpoint['model_name']
	times = checkpoint['times']

	fig, ax1 = plt.subplots()
	ax2 = ax1.twinx()

	train_loss, = ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
	val_loss, = ax1.plot(epochs, val_losses, 'b--', label='Validation Loss')
	train_acc, = ax2.plot(epochs, train_accuracies, 'r-', label='Validation Accuracy')
	test_acc, = ax2.plot(epochs, test_accuracies, 'r--', label='Test Accuracy')

	ax1.set_xlabel('Epochs')
	ax1.set_ylabel('Loss')
	ax2.set_ylabel('Accuracy (%)')
	ax1.yaxis.label.set_color(train_loss.get_color())
	ax2.yaxis.label.set_color(train_acc.get_color())
	ax1.tick_params(axis='y', colors=train_loss.get_color())
	ax2.tick_params(axis='y', colors=train_acc.get_color())
	ax1.legend(handles=[train_loss, val_loss, train_acc, test_acc], loc='center right')

	plt.title(f'Model: {model_name}. Params: {pu.human_format(model_params)}. Time: {times[-1]/3600 : 0.2f}h')

	plt.savefig(os.path.join(model_dir, 'curves.png'))
	

def train(config, 
		  main_config, 
		  graphObject, 
		  device, 
		  model_dir, 
		  auto_tune, 
		  checkpointing = True,
		  gpuFrac = None):
	"""Trains a CNN model on a given device
	
	Args:
	    config (dict): dictionary of hyper-parameters of the training recipe
	    main_config (dict): dictionary of main configuration
	    graphObject (Graph): Graph object
	    device (torch.device, optional): cuda device
	    model_dir (str, optional): directory to store the model and metrics
	    auto_tune (bool): if ray-tune is running
	    checkpointing (bool, optional): to checkpoint trained models
	    gpuFrac (float): fraction of GPU memory to be alloted for training given model
	
	Raises:
	    ValueError: if no GPU is found
	"""
	print(f'{pu.bcolors.OKBLUE}Using hyper-parameters:{pu.bcolors.ENDC}\n{config}')

	model = CNNBenchModel(main_config, graphObject)

	model_params = model.get_params()

	train_loader, val_loader, test_loader, total_size, val_size = get_loader(main_config)
	train_size, test_size = int(total_size - val_size), len(test_loader.dataset)

	if gpuFrac:
		# GPU memory fraction only to be defined during trial run on a single GPU
		assert device is not None and device.index is not None
		torch.cuda.set_per_process_memory_fraction(float(gpuFrac), device.index)

	if device is None:
		if torch.cuda.is_available():
			device = 'cuda'
			if torch.cuda.device_count() > 1:
				model = nn.DataParallel(model)
		else:
			raise ValueError('No GPU device found!') 

	model.to(device)

	if 'manual_models' in model_dir:
		model_name = model_dir.split('/')[-1]
	else:
		model_name = graphObject.hash

	optims = [opt[0] for opt in getmembers(optim)]

	if 'optimizer' in config.keys():
		opt = list(config['optimizer'].keys())[0]
		if opt not in optims:
			raise ValueError(f'Optimizer {opt} not supported in PyTorch')
		optimizer = eval(f'optim.{opt}(model.parameters(), **config["optimizer"][opt])')
	else:
		opt = 'AdamW'
		optimizer = optim.AdamW(model_parameters(), lr=0.0001, weight_decay=0.0001)

	shdlrs = [sh[0] for sh in getmembers(optim.lr_scheduler)]

	if 'scheduler' in config.keys():
		schdlr = list(config['scheduler'].keys())[0]
		if schdlr not in shdlrs:
			raise ValueError(f'Scheduler {schdlr} not supported in PyTorch')
		scheduler = eval(f'optim.lr_scheduler.{schdlr}(optimizer, **config["scheduler"][schdlr])')
	else:
		schdlr = 'CosineAnnealingLR'
		scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(main_config['epochs']))

	train_losses = []
	val_losses = []
	epochs = []
	learning_rates = []
	train_accuracies = []
	val_accuracies = []
	test_accuracies = []
	times = []

	start_time = time.time()

	batch_size = 0

	for epoch in range(main_config['epochs']):
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

			if 'scheduler' in config.keys() and list(config['scheduler'].keys())[0] == 'CosineAnnealingWarmRestarts':
				scheduler.step(epoch + batch_idx / train_size)
			
			if batch_idx % LOG_INTERVAL == 0:
				print('Train Epoch: {} [{:6d}/{:6d} ({:.0f}%)]\tLearning Rate: {:.6f}\tLoss: {:.6f}'.format(
					epoch, batch_idx * batch_size, train_size,
					100. * batch_idx / len(train_loader), optimizer.param_groups[0]['lr'], loss.item()))
			
			if gpuFrac:
				# Dry run complete
				break

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

			train_loss /= train_size

			print('\nTrain set:\tAverage loss: {:.4f},\tAccuracy: {}/{} ({:.0f}%)'.format(
				train_loss, correct, train_size,
				100. * correct / train_size))

			train_accuracies.append(100. * correct / train_size)

			val_loss = 0
			correct = 0
			for data, target in val_loader:
				data, target = data.to(device), target.to(device)
				output = model(data)
				val_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
				pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
				correct += pred.eq(target.view_as(pred)).sum().item()

			val_loss /= val_size

			print('Val set:\tAverage loss: {:.4f},\tAccuracy: {}/{} ({:.0f}%)'.format(
				val_loss, correct, val_size,
				100. * correct / val_size))

			val_accuracies.append(100. * correct / val_size)

			test_loss = 0
			correct = 0
			for data, target in test_loader:
				data, target = data.to(device), target.to(device)
				output = model(data)
				test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
				pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
				correct += pred.eq(target.view_as(pred)).sum().item()

			test_loss /= test_size

			print('Test set:\tAverage loss: {:.4f},\tAccuracy: {}/{} ({:.0f}%)\n'.format(
				test_loss, correct, test_size,
				100. * correct / test_size))

			test_accuracies.append(100. * correct / test_size)

		train_losses.append(train_loss)
		val_losses.append(val_loss)
		epochs.append(epoch)
		learning_rates.append(optimizer.param_groups[0]['lr'])

		if checkpointing and epoch % CKPT_INTERVAL == 0:
			if auto_tune:
				checkpoint_dir = os.path.join('..', '..', 
						'auto_tune', 
						'_'.join([opt, schdlr, get_hp_hash(config)]))
			else:
				checkpoint_dir = model_dir

			if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
			
			ckpt_path = os.path.join(checkpoint_dir, f'model_{epoch}.ckpt')

			torch.save({'config': main_config,
						'hp_config': config,
						'model_name': model_name,
			   			'model_params': model_params,
						'model_state_dict': model.state_dict(),
						'optimizer_state_dict': optimizer.state_dict(),
						'scheduler_state_dict': scheduler.state_dict(),
				   		'epochs': epochs,
				   		'train_losses': train_losses,
				   		'val_losses': val_losses,
				   		'learning_rates': learning_rates,
				   		'train_accuracies': train_accuracies,
				   		'val_accuracies': val_accuracies,
				   		'test_accuracies': test_accuracies,
				   		'times': times}, ckpt_path)

			print(f'{pu.bcolors.OKGREEN}Saved checkpoint to:{pu.bcolors.ENDC} {ckpt_path}')

		if auto_tune:
			tune.report(val_loss=val_losses[-1], val_accuracy=val_accuracies[-1])

		if 'scheduler' in config.keys():
			if list(config['scheduler'].keys())[0] == 'ReduceLROnPlateau':
				scheduler.step(test_loss)
			elif list(config['scheduler'].keys())[0] != 'CosineAnnealingWarmRestarts':
				scheduler.step()

		times.append(time.time() - start_time)
