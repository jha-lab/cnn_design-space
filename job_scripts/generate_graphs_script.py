# Generates graphs using graph generator
# Author :  Shikhar Tuli

import sys

if '../' not in sys.path:
	sys.path.append('../')

import os
import tensorflow as tf

from absl import logging 
from absl import flags
from absl import app

from cnnbench.scripts import generate_graphs as graph_generator

# Do not show warnings of deprecated functions
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.get_absl_handler().setFormatter(None)
logging.set_verbosity(logging.INFO)  # or any {DEBUG, INFO, WARN, ERROR, FATAL} 


FLAGS = flags.FLAGS

# Change default flag values
# Those flag values define in command line take precedence
# FLAGS.max_vertices = 3


def main(args):
	del args 

	if not FLAGS.output_file:
		if not os.path.exists(f'../results/vertices_{FLAGS.module_vertices}'):
			os.makedirs(f'../results/vertices_{FLAGS.module_vertices}')

		FLAGS.output_file = f'../results/vertices_{FLAGS.module_vertices}/generated_graphs.json'

	# Generate graphs
	graphs = graph_generator.main(1)


if __name__ == '__main__':
	app.run(main)
