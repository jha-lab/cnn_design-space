# Generates graphs and trains on multiple workers
# Author :  Shikhar Tuli

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

if '../' not in sys.path:
	sys.path.append('../')

import os
import tensorflow as tf

from absl import logging 
from absl import flags
from absl import app

from cnnbench.scripts import run_evaluation_new as run_evaluation

# Do not show warnings of deprecated functions
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.get_absl_handler().setFormatter(None)
logging.set_verbosity(logging.ERROR)  # or any {DEBUG, INFO, WARN, ERROR, FATAL} 


FLAGS = flags.FLAGS

# Change default flag values
# Those flag values define in command line take precedence
FLAGS.data_dir = '../datasets/'

def main(args):
	del args 

	worker_id = FLAGS.worker_id + FLAGS.worker_id_offset

	if not FLAGS.models_file:
		FLAGS.models_file = f'../results/vertices_{FLAGS.module_vertices}_test/generated_graphs.json'

	if not FLAGS.output_dir:
		FLAGS.output_dir = f'../results/vertices_{FLAGS.module_vertices}_test/evaluation'

	# Forcing evaluation on specified GPU (if GPU is available)
	gpus = tf.config.experimental.list_physical_devices('GPU')
	if gpus:
		tf.config.experimental.set_visible_devices(gpus[worker_id % len(gpus)], 'GPU')

	evaluator = run_evaluation.Evaluator(
	  models_file=FLAGS.models_file,
	  output_dir=FLAGS.output_dir,
	  worker_id=worker_id,
	  total_workers=FLAGS.total_workers,
	  model_id_regex=FLAGS.model_id_regex)

	evaluator.run_evaluation()

if __name__ == '__main__':
	app.run(main)
