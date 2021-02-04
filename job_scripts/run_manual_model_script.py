# Runs manually defined model 
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

from cnnbench.scripts import run_manual_model
from cnnbench.lib import evaluate
from cnnbench.lib import module_spec
from cnnbench.lib import config as _config

import csv
import time
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np

# Get defined models in run_manual_model
models = [model[7:-5] for model in dir(run_manual_model) if model.startswith('create')]

# Do not show warnings of deprecated functions
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.get_absl_handler().setFormatter(None)
logging.set_verbosity(logging.INFO)  # or any {DEBUG, INFO, WARN, ERROR, FATAL} 

flags.DEFINE_string('model_name', '', f'Model name in {models}')
flags.DEFINE_integer('epochs_per_eval', 1, 'Number of epochs after which'
					'an evaluation should be done of the trained model')

FLAGS = flags.FLAGS

# Change default flag values
# Those flag values define in command line take precedence
FLAGS.data_dir = '../datasets/'

def main(args):
	del args 

	if not FLAGS.model_dir:
		FLAGS.model_dir = f'../results/{FLAGS.model_name}_test'

	# The default settings in config are exactly what was used to generate the
	# dataset of models. However, given more epochs and a different learning rate
	# schedule, it is possible to get higher accuracy.
	config = _config.build_config()
	config['train_epochs'] = 3
	config['lr_decay_method'] = 'STEPWISE'
	config['train_seconds'] = -1      # Disable training time limit
	spec_list = eval(f'run_manual_model.create_{FLAGS.model_name}_spec(config)')

	# Forcing evaluation on specified GPU (if GPU is available)
	gpus = tf.config.experimental.list_physical_devices('GPU')
	if gpus:
		tf.config.experimental.set_visible_devices(gpus[FLAGS.worker_id % len(gpus)], 'GPU')

	# Run training
	start_time = time.time()
	data = evaluate.basic_train_and_evaluate(spec_list, config, FLAGS.model_dir, epochs_per_eval = FLAGS.epochs_per_eval)
	train_time = round(time.time() - start_time)

	logging.info(data)

	# Generate final outputresults file
	accuracy_list = []
	loss_list = []
	with open(os.path.join(FLAGS.model_dir, 'results_temp.csv'), mode = 'r') as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		for row in csv_reader:
			accuracy_list.append(row[0])
			loss_list.append(row[1])

	with open(os.path.join(FLAGS.model_dir, 'results.csv'), mode = 'w') as csv_file:
		csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		csv_writer.writerow(['Epoch', 'Accuracy', 'Loss', 'Model', 'Trainable parameters', 'Training time (s)'])
		for epoch in range(len(accuracy_list)):
			if epoch == 0:
				csv_writer.writerow([(epoch+1)*FLAGS.epochs_per_eval, 
					accuracy_list[epoch], loss_list[epoch], FLAGS.model_name, data['trainable_params'], train_time])
			else:
				csv_writer.writerow([(epoch+1)*FLAGS.epochs_per_eval, 
					accuracy_list[epoch], loss_list[epoch]])

	# Plot and save performance figure
	results = pd.read_csv(os.path.join(FLAGS.model_dir, 'results.csv'))
	epochs = results['Epoch'].tolist()
	accuracies = results['Accuracy'].tolist()
	accuracies = [a*100 for a in accuracies]
	losses = results['Loss'].tolist()

	logging.info(f'Epochs: {epochs}')
	logging.info(f'Accuracy: {accuracies}')
	logging.info(f'Losses: {losses}')

	fig, ax_acc = plt.subplots()
	ax_loss = ax_acc.twinx()

	ax_acc.plot(epochs, accuracies, color = 'blue', label = 'Accuracy')
	ax_loss.plot(epochs, losses, color = 'orange', label = 'Loss')

	lines, labels = ax_acc.get_legend_handles_labels()
	lines2, labels2 = ax_loss.get_legend_handles_labels()
	ax_loss.legend(lines + lines2, labels + labels2)
	ax_acc.ticklabel_format(useOffset=False)
	ax_acc.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
	ax_loss.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
	ax_acc.set_xticks(np.arange(int(epochs[0]), int(epochs[-1]+ 1), 1))

	ax_acc.set_ylabel('Accuracy (%)')
	ax_loss.set_ylabel('Loss')
	ax_acc.set_xlabel('Epochs')
	ax_acc.set_title(f'Training performance: {FLAGS.model_name}')
	fig.tight_layout()
	ax_acc.grid()
	plt.savefig(os.path.join(FLAGS.model_dir, 'results.pdf'), bbox_inches = 'tight')

if __name__ == '__main__':
	app.run(main)
