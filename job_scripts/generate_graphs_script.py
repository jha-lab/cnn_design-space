# Test script to generates graphs
# Author :  Shikhar Tuli

import sys

if '../' not in sys.path:
	sys.path.append('../')

import os
import tensorflow as tf

from absl import logging 
from absl import flags
from absl import app

from cnnbench.scripts.generate_graphs import main as graph_generator

# Do not show warnings of deprecated functions
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.get_absl_handler().setFormatter(None)
logging.set_verbosity(logging.ERROR)  # or any {DEBUG, INFO, WARN, ERROR, FATAL} 


FLAGS = flags.FLAGS

# Change default flag values
# Those flag values define in command line take precedence
# FLAGS.max_vertices = 3

def main(args):
	del args 

	if not os.path.exists(f'../results/vertices_{FLAGS.max_vertices}'):
	    os.makedirs(f'../results/vertices_{FLAGS.max_vertices}')

	FLAGS.output_file = f'../results/vertices_{FLAGS.max_vertices}/generated_graphs.json'

	# Generate graphs
	app.run(graph_generator)


if __name__ == '__main__':
	app.run(main)
