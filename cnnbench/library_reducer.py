# Reduces a library checkpoint by randomly sampling a subset of the modules.
# Removes graphs for fresh creation.
	 
# Author : Shikhar Tuli


import os
import sys

from random import sample
from six.moves import cPickle as pickle


CKPT_TEMP = '/scratch/gpfs/stuli/graphs_ckpt_temp.pkl'
MODULES_SAMPLE_SIZE = 50
HEADS_SAMPLE_SIZE = 50


if os.path.exists(CKPT_TEMP):
	print('Loading checkpoint...')
	ckpt = pickle.load(open(CKPT_TEMP, 'rb'))

	print('Reducing library...')
	sampled_keys = sample(ckpt['module_buckets'].keys(), MODULES_SAMPLE_SIZE)
	ckpt['module_buckets'] = {key: ckpt['module_buckets'][key] for key in sampled_keys}
	ckpt['total_modules'] = len(ckpt['module_buckets'])

	sampled_keys = sample(ckpt['head_buckets'].keys(), HEADS_SAMPLE_SIZE)
	ckpt['head_buckets'] = {key: ckpt['head_buckets'][key] for key in sampled_keys}
	ckpt['total_heads'] = len(ckpt['head_buckets'])

	ckpt['total_graphs'], ckpt['stacks_done'], ckpt['graph_buckets'] = 0, 0, {}

	print('Saving reduced library...')
	pickle.dump(ckpt, open(CKPT_TEMP, 'wb+'), pickle.HIGHEST_PROTOCOL)

	print('Done!')
else:
	print('Checkpoint not found!')
