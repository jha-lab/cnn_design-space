# Run Bayesian Optimization using Second-order gradients and a Heteroscedastic
# surrogate model for Network Architecture Search (BOSHNAS) on all CNNs
# in CNNBench's design space.

# Author : Shikhar Tuli


import os
import sys

# Make sure that the boshnas directory exists outside the cnn_design-space directory.
sys.path.append('../../boshnas/boshnas/')

import argparse
import numpy as np
import yaml
import random
import tabulate
import subprocess
import time
import json

import torch

from boshnas import BOSHNAS
from acq import gosh_acq as acq

from library import GraphLib, Graph
from utils import print_util as pu


CONF_INTERVAL = 0.005 # Corresponds to 0.5% accuracy for 95% confidence interval
OVERLAP_THRESHOLD = 0.9 # Corresponds to the minimum overlap for model to be considered

DEBUG = False
ACCURACY_PATIENCE = 10 # Convergence criterion for accuracy
ALEATORIC_QUERIES = 10 # Number of queries to be run with aleatoric uncertainty
K = 10 # Number of parallel cold restarts for BOSHNAS
UNC_PROB = 0.1
DIV_PROB = 0.1


def worker(config_file: str,
	graphlib_file: str,
	models_dir: str,
	model_hash: str,
	chosen_neighbor_hash: str,
	autotune: bool):
	"""Worker to finetune or pretrain the given model
	
	Args:
		config_file (str): path to the configuration file
		graphlib_file (str): path the the graphLib dataset file
		models_dir (str): path to "models" directory containing "pretrained" sub-directory
		model_hash (str): hash of the given model
		chosen_neighbor_hash (str): hash of the chosen neighbor
		autotune (bool): to autotune the given model
	
	Returns:
		job_id, pretrain (str, bool): Job ID for the slurm scheduler and whether pretraining
		is being performed
	"""
	scratch = False

	print(f'Training model with hash: {model_hash}.')

	graphLib = GraphLib.load_from_dataset(graphlib_file)

	with open(config_file) as file:
		try:
			config = yaml.safe_load(file)
		except yaml.YAMLError as exc:
			raise exc

	chosen_neighbor_path = None
	if chosen_neighbor_hash is not None:
		# Load weights of current model using the finetuned neighbor that was chosen
		chosen_neighbor_path = os.path.join(models_dir, chosen_neighbor_hash, 'model.pt')
		print(f'Weights copied from neighbor model with hash: {chosen_neighbor_hash}.')
	else:
		scratch = True
		print('No neighbor found. Training model from scratch.')

	args = ['--dataset', config['dataset']]
	args.extend(['--autotune', '1' if autotune else '0'])
	args.extend(['--model_hash', model_hash])
	args.extend(['--model_dir', os.path.join(models_dir, model_hash)])
	args.extend(['--config_file', config_file])
	args.extend(['--graphlib_file', graphlib_file])

	if chosen_neighbor_path is not None:
		args.extend(['--neighbor_file', chosen_neighbor_path])
	
	slurm_stdout = subprocess.check_output(f'source ./job_scripts/job_train.sh {" ".join(args)}',
		shell=True, text=True)

	return slurm_stdout.split()[-1], scratch
		

def get_job_info(job_id: int):
	"""Obtain job info
	
	Args:
		job_id (int): job id
	
	Returns:
		start_time, elapsed_time, status (str, str, str): job details
	"""
	slurm_stdout = subprocess.check_output(f'slist {job_id}', shell=True, text=True)
	slurm_stdout = slurm_stdout.split('\n')[2].split()

	if len(slurm_stdout) > 7:
		start_time, elapsed_time, status = slurm_stdout[5], slurm_stdout[6], slurm_stdout[7]
		if start_time == 'Unknown': start_time = 'UNKNOWN'
	else:
		start_time, elapsed_time, status = 'UNKNOWN', 'UNKNOWN', 'UNKNOWN'

	return start_time, elapsed_time, status


def print_jobs(model_jobs: list):
	"""Print summary of all completed, pending and running jobs
	
	Args:
		model_jobs (list): list of jobs
	"""
	header = ['MODEL HASH', 'JOB ID', 'TRAIN TYPE', 'START TIME', 'ELAPSED TIME', 'STATUS']

	rows = []
	for job in model_jobs:
		start_time, elapsed_time, status = get_job_info(job['job_id'])
		rows.append([job['model_hash'], job['job_id'], job['train_type'], start_time, elapsed_time, status])

	print()
	print(tabulate.tabulate(rows, header))


def wait_for_jobs(model_jobs: list, running_limit: int = 4, patience: int = 1):
	"""Wait for current jobs in queue to complete
	
	Args:
		model_jobs (list): list of jobs
		running_limit (int, optional): nuber of running jobs to limit
		patience (int, optional): number of pending jobs to wait for
	"""
	print_jobs(model_jobs)

	completed_jobs = 0
	last_completed_jobs = 0
	running_jobs = np.inf
	pending_jobs = np.inf
	while running_jobs >= running_limit or pending_jobs > patience:
		completed_jobs, running_jobs, pending_jobs = 0, 0, 0
		for job in model_jobs:
			_, _, status = get_job_info(job['job_id'])
			if status == 'COMPLETED': 
				completed_jobs += 1
			elif status == 'PENDING':
				pending_jobs += 1
			elif status == 'RUNNING':
				running_jobs += 1
			elif status == 'FAILED':
				print_jobs(model_jobs)
				raise RuntimeError('Some jobs failed.')
		if last_completed_jobs != completed_jobs:
			print_jobs(model_jobs)
		last_completed_jobs = completed_jobs 
		time.sleep(1)


def update_dataset(graphLib: 'GraphLib', models_dir: str, dataset_file: str):
	"""Update the dataset with all finetuned models
	
	Args:
		graphLib (GraphLib): GraphLib opject to update
		models_dir (str): directory with all trained models
		dataset_file (str): path to the dataset file
	"""
	count = 0
	best_accuracy = 0
	for model_hash in os.listdir(models_dir):
		checkpoint_path = os.path.join(models_dir, model_hash, 'model.pt')
		if os.path.exists(checkpoint_path):
			model_checkpoint = torch.load(checkpoint_path)
			_, model_idx = graphLib.get_graph(model_hash=model_hash)
			graphLib.library[model_idx].accuracies['train'] = model_checkpoint['train_accuracies'][-1]
			graphLib.library[model_idx].accuracies['val'] = model_checkpoint['val_accuracies'][-1]
			graphLib.library[model_idx].accuracies['test'] = model_checkpoint['test_accuracies'][-1]
			if model_checkpoint['val_accuracies'][-1] > best_accuracy:
				best_accuracy = model_checkpoint['val_accuracies'][-1]
			count += 1

	graphLib.save_dataset(dataset_file)

	print()
	print(f'{pu.bcolors.OKGREEN}Trained points in dataset:{pu.bcolors.ENDC} {count}\n' \
		+ f'{pu.bcolors.OKGREEN}Best accuracy:{pu.bcolors.ENDC} {best_accuracy}')
	print()

	return best_accuracy


def convert_to_tabular(graphLib: 'GraphLib'):
	"""Convert the graphLib object to a tabular dataset from 
	input encodings to the output loss
	
	Args:
		graphLib (GraphLib): GraphLib object
	
	Returns:
		X, y (tuple): input embeddings and output loss
	"""
	X, y = [], []
	for graph in graphLib.library:
		if graph.accuracies['val']:
			X.append(graph.embedding)
			y.append(1 - graph.accuracies['val'])

	X, y = np.array(X), np.array(y)

	return X, y


def get_neighbor_hash(model: 'Graph', trained_hashes: list):
	chosen_neighbor_hash = None

	# Choose neighbor with max overlap given that it is trained
	for neighbor_hash in model.neighbors:
		if neighbor_hash in trained_hashes: 
			chosen_neighbor_hash = neighbor_hash
			break

	return chosen_neighbor_hash


def main():
	"""Run BOSHNAS to get the best architecture in the design space
	"""
	parser = argparse.ArgumentParser(
		description='Input parameters for generation of dataset library',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--graphlib_file',
		metavar='',
		type=str,
		help='path to load the graphlib dataset',
		default='./dataset/dataset_test.json')
	parser.add_argument('--dataset',
		metavar='',
		type=str,
		help='name of the dataset for training. Should match the one in the config file',
		default='CIFAR10')
	parser.add_argument('--config_file',
		metavar='',
		type=str,
		help='path to the configuration file',
		default='./tests/config_test_tune.yaml')
	parser.add_argument('--surrogate_model_dir',
		metavar='',
		type=str,
		help='path to save the surrogate model parameters',
		default='./dataset/surrogate_models/CIFAR10')
	parser.add_argument('--models_dir',
		metavar='',
		type=str,
		help='path to the directory where all models are trained',
		default='/home/stuli/cnn_design-space/results/CIFAR10')
	parser.add_argument('--num_init',
		metavar='',
		type=int,
		help='number of initial models to initialize the BOSHNAS model',
		default=10)
	parser.add_argument('--autotune',
		metavar='',
		type=int,
		help='to autotune models or not',
		default=0)
	parser.add_argument('--n_jobs',
		metavar='',
		type=int,
		help='number of parallel jobs for training BOSHNAS',
		default=8)

	args = parser.parse_args()

	random_seed = 0

	# Take global dataset and assign task
	graphLib = GraphLib.load_from_dataset(args.graphlib_file)

	# New dataset file
	new_dataset_file = args.graphlib_file.split('.json')[0] + '_trained.json'

	autotune = False
	if args.autotune == 1:
		autotune = True

	# 1. Pre-train randomly sampled models if number of pretrained models available 
	#   is less than num_init
	# 2. Fine-tune pretrained models for the given task
	# 3. Train BOSHNAS on initial models
	# 4. With prob 1 - epsilon - delta, train models with best acquisition function
	#   - add only those models to queue that have high overlap and finetune
	#   - if no model with high overlap neighbors, pretrain best predicted model(s)
	# 5. With prob epsilon: train models with max std, and prob delta: train random models
	#   - if neighbor pretrained with high overlap, finetune only; else pretrain
	# 6. Keep a dictionary of job id and model hash. Get accuracy from trained models.
	#   Wait for spawning more jobs
	# 7. Update the BOSHNAS model and put next queries in queue
	# 8. Optional: Use aleatoric uncertainty and re-finetune models if accuracy converges
	# 9. Stop training if a stopping criterion is reached
	
	# Initialize a dictionary mapping the model hash to its corresponding job_id
	model_jobs = []

	if not os.path.exists(args.models_dir):
		os.makedirs(args.models_dir)

	trained_hashes = os.listdir(args.models_dir)

	# Train randomly sampled models if total trained models is less than num_init
	# TODO: Add skopt.sampler.Sobol points instead
	while len(trained_hashes) < args.num_init:
		sample_idx = random.randint(0, len(graphLib)-1)
		model_hash = graphLib.library[sample_idx].hash

		if model_hash not in trained_hashes:
			trained_hashes.append(model_hash)

			model, model_idx = graphLib.get_graph(model_hash=model_hash)

			job_id, scratch = worker(config_file=args.config_file, graphlib_file=args.graphlib_file,
				models_dir=args.models_dir, model_hash=model_hash, chosen_neighbor_hash=None,
				autotune=autotune)
			assert scratch is True

			train_type = 'S' if scratch else 'WT'
			train_type += ' w/ A' if autotune else ''
			
			model_jobs.append({'model_hash': model_hash, 
				'job_id': job_id, 
				'train_type': train_type})

	# Wait for jobs to complete
	wait_for_jobs(model_jobs)

	# Update dataset with newly trained models
	old_best_accuracy = update_dataset(graphLib, args.models_dir, new_dataset_file)

	# Get entire dataset in embedding space
	X_ds = []
	for graph in graphLib.library:
		X_ds.append(graph.embedding)
	X_ds = np.array(X_ds)

	min_X, max_X = np.min(X_ds, axis=0), np.max(X_ds, axis=0)

	# Initialize the BOSHNAS model
	surrogate_model = BOSHNAS(input_dim=X_ds.shape[1],
							  bounds=(min_X, max_X),
							  trust_region=False,
							  second_order=True,
							  parallel=True if not DEBUG else False,
							  model_aleatoric=True,
							  save_path=args.surrogate_model_dir,
							  pretrained=False)

	# Get initial dataset after finetuning num_init models
	X, y = convert_to_tabular(graphLib)
	max_loss = np.amax(y)

	same_accuracy = 0
	method = 'optimization'

	while same_accuracy < ACCURACY_PATIENCE + ALEATORIC_QUERIES:
		prob = random.uniform(0, 1)
		if 0 <= prob <= (1 - UNC_PROB - DIV_PROB):
			method = 'optimization'
		elif 0 <= prob <= (1 - DIV_PROB):
			method = 'unc_sampling'
		else:
			method = 'div_sampling'

		# Get a set of trained models and models that are currently in the pipeline
		trained_hashes, pipeline_hashes = [], []
		for job in model_jobs:
			_, _, status = get_job_info(job['job_id'])
			if status == 'COMPLETED':
				trained_hashes.append(job['model_hash'])
			else:
				pipeline_hashes.append(job['model_hash'])

		new_queries = 0

		if method == 'optimization':
			print(f'{pu.bcolors.OKBLUE}Running optimization step{pu.bcolors.ENDC}')
			# Get current tabular dataset
			X, y = convert_to_tabular(graphLib)
			y = y/max_loss

			# Train BOSHNAS model
			train_error = surrogate_model.train(X, y)

			# Use aleatoric loss close to convergence to optimize training recipe
			if same_accuracy < ACCURACY_PATIENCE:
				# Architecture not converged yet. Use only epistemic uncertainty
				use_al = False
			else:
				# Use aleatoric uncertainty to optimize training recipe
				use_al = True

			# Get next queries
			query_indices = surrogate_model.get_queries(x=X_ds, k=K, explore_type='ucb', use_al=use_al)

			# Run queries
			for i in set(query_indices):
				model = graphLib.library[i]

				if not use_al and model.hash in trained_hashes + pipeline_hashes:
					# If aleatoric uncertainty is not considered, only consider models that are not 
					# already trained or in the pipeline
					continue

				chosen_neighbor_hash = get_neighbor_hash(model, trained_hashes)

				if chosen_neighbor_hash:
					# Finetune model with the chosen neighbor
					job_id, scratch = worker(config_file=args.config_file, graphlib_file=args.graphlib_file,
						models_dir=args.models_dir, model_hash=model.hash, autotune=autotune, 
						chosen_neighbor_hash=chosen_neighbor_hash)
					assert scratch is False
				else:
					# If no neighbor was found, proceed to next query model
					continue

				new_queries += 1

				train_type = 'S' if scratch else 'WT'
				train_type += ' w/ A' if autotune else ''
				
				model_jobs.append({'model_hash': model.hash, 
					'job_id': job_id, 
					'train_type': train_type})
			
			if new_queries == 0:
				# If no queries were found where weight transfer could be used, train the highest
				# predicted model from scratch
				query_embeddings = [X_ds[idx, :] for idx in query_indices]
				candidate_predictions = surrogate_model.predict(query_embeddings)

				best_prediction_index = query_indices[np.argmax(acq([pred[0] for pred in candidate_predictions],
																[pred[1][0] for pred in candidate_predictions],
																explore_type='ucb'))]

				model = graphLib.library[best_prediction_index]

				# Train model
				job_id, scratch = worker(config_file=args.config_file, graphlib_file=args.graphlib_file,
					models_dir=args.models_dir, model_hash=model.hash, autotune=autotune, 
					chosen_neighbor_hash=None)
				assert scratch is True

				train_type = 'S' if scratch else 'WT'
				train_type += ' w/ A' if autotune else ''
				
				model_jobs.append({'model_hash': model.hash, 
					'job_id': job_id, 
					'train_type': train_type})

		elif method == 'unc_sampling':
			print(f'{pu.bcolors.OKBLUE}Running uncertainty sampling{pu.bcolors.ENDC}')

			candidate_predictions = surrogate_model.predict(X_ds)

			# Get model index with highest epistemic uncertainty
			unc_prediction_idx = np.argmax([pred[1][0] for pred in candidate_predictions])

			# Sanity check: model with highest epistemic uncertainty should not be trained
			assert graphLib.library[unc_prediction_idx].hash not in trained_hashes

			if graphLib.library[unc_prediction_idx].hash in pipeline_hashes:
				print(f'{pu.bcolors.OKBLUE}Highest uncertainty model already in pipeline{pu.bcolors.ENDC}')
			else:
				model = graphLib.library[unc_prediction_idx]

				# Train model
				job_id, scratch = worker(config_file=args.config_file, graphlib_file=args.graphlib_file,
					models_dir=args.models_dir, model_hash=model.hash, autotune=autotune, 
					chosen_neighbor_hash=None)
				assert scratch is False

				new_queries += 1

				train_type = 'S' if scratch else 'WT'
				train_type += ' w/ A' if autotune else ''
				
				model_jobs.append({'model_hash': model_hash, 
					'job_id': job_id, 
					'train_type': train_type})

		else:
			print(f'{pu.bcolors.OKBLUE}Running diversity sampling{pu.bcolors.ENDC}')

			# Get randomly sampled model idx
			# TODO: Add skopt.sampler.Sobol points instead
			unc_prediction_idx = random.randint(0, len(graphLib))

			while graphLib.library[unc_prediction_idx].hash in trained_hashes + pipeline_hashes:
				unc_prediction_idx = random.randint(0, len(graphLib))

			model = graphLib.library[unc_prediction_idx]

			# Train sampled model
			job_id, scratch = worker(config_file=args.config_file, graphlib_file=args.graphlib_file,
				models_dir=args.models_dir, model_hash=model.hash, autotune=autotune, 
				chosen_neighbor_hash=None)
			assert scratch is False

			new_queries += 1

			train_type = 'S' if scratch else 'WT'
			train_type += ' w/ A' if autotune else ''
			
			model_jobs.append({'model_hash': model_hash, 
				'job_id': job_id, 
				'train_type': train_type})

		# Wait for jobs to complete
		wait_for_jobs(model_jobs)

		# Update dataset with newly trained models
		best_accuracy = update_dataset(graphLib, args.models_dir, new_dataset_file)

		# Update same_accuracy to check convergence
		if best_accuracy == old_best_accuracy and method == 'optimization':
			same_accuracy += 1

		old_best_accuracy = best_accuracy

	# Wait for jobs to complete
	wait_for_jobs(model_jobs, running_limit=0, patience=0)

	# Update dataset with newly trained models
	best_accuracy = update_dataset(graphLib, args.models_dir, new_dataset_file)

	print(f'{pu.bcolors.OKGREEN}Convergence criterion reached!{pu.bcolors.ENDC}')


if __name__ == '__main__':
	main()

