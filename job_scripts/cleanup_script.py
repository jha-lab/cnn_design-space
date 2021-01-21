# Cleans up unused directories
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

# Do not show warnings of deprecated functions
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.set_verbosity(logging.INFO)  # or any {DEBUG, INFO, WARN, ERROR, FATAL} 

from cnnbench.lib import print_util

import shutil
import argparse

flags.DEFINE_string('cleanup_dir', '../results/vertices_2/evaluation',
                    'Directory to clean (also the model output directory)')

FLAGS = flags.FLAGS

def main(args):
	del args

	logging.get_absl_handler().setFormatter(None)
	for root, dirnames, filenames in os.walk(FLAGS.cleanup_dir):
		for dirname in dirnames:
			if dirname.startswith('eval'):
				logging.info(f'{print_util.bcolors.OKBLUE}Deleting directory{print_util.bcolors.ENDC}: {os.path.join(root, dirname)}')
				shutil.rmtree(os.path.join(root, dirname))

if __name__ == '__main__':
  app.run(main)
