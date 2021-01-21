# Adds cnnbench.tfrecord dataset file
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

import json
import numpy as np

from cnnbench.lib import config as _config
from cnnbench.lib import print_util

# Do not show warnings of deprecated functions
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.get_absl_handler().setFormatter(None)
logging.set_verbosity(logging.INFO)  # or any {DEBUG, INFO, WARN, ERROR, FATAL} 

DEBUG = False

flags.DEFINE_string('model_dir', '../results/vertices_2', 'Model directory containing "generated_graphs.json" and "evaluation" directory')

FLAGS = flags.FLAGS

def main(args):
  del args

  model_dir = FLAGS.model_dir

  config = _config.build_config()
  dataset = None

  with open(os.path.join(model_dir, 'generated_graphs.json')) as f:
    models = json.load(f)

  if DEBUG: print(f'All hashes: models.keys()')    

  with tf.compat.v1.python_io.TFRecordWriter(os.path.join(model_dir, 'cnnbench.tfrecord')) as writer:
    for dr in os.listdir(os.path.join(model_dir, 'evaluation')):
      if dr.startswith('_recovery'): continue
      for model in os.listdir(os.path.join(model_dir, 'evaluation', dr)):
        for repeat in os.listdir(os.path.join(model_dir, 'evaluation', dr, model)):
          with open(os.path.join(model_dir, 'evaluation', dr, model, repeat, 'results.json')) as f:
            result = json.load(f)
            
            adjacency_flattened = []
            vertices = []
            string_labels = ''
            for module in models[model]:
                raw_adjacency = module[0]
                raw_labels = module[1]
                adjacency = np.array(raw_adjacency)
                adjacency_flattened.extend(adjacency.flatten())
                vertices.append(np.shape(adjacency[-1])[0])
                labels = (['input'] + [config['available_ops'][lab] for lab in raw_labels[1:-1]] + ['output'])
                for label in labels:
                  string_labels = string_labels + label + ','
                
                if DEBUG:
                  print(f'Module adjacency: \n{adjacency}')
                  print(f'Module operations: {labels}')
                  print(f'Trainable parameters: {result["trainable_params"]}')
                  print(f'Total training time: {result["total_time"]}')
                  print('Evaluation results:')
                  for ckpt in result['evaluation_results']:
                        print(f'Epoch: {ckpt["epochs"]}')
                        print(f'\tTraining time: {ckpt["training_time"]}')
                        print(f'\tTrain accuracy: {ckpt["train_accuracy"]}')
                        print(f'\tValidation accuracy: {ckpt["validation_accuracy"]}')
                        print(f'\tTest accuracy: {ckpt["test_accuracy"]}')
                  print() 
                                

            example = tf.train.Example(features = tf.train.Features(feature = {
              'graph_adjacencies': tf.train.Feature(int64_list = tf.train.Int64List(value = adjacency_flattened)),
              'graph_vertices': tf.train.Feature(int64_list = tf.train.Int64List(value = vertices)),
              'graph_operations': tf.train.Feature(bytes_list = tf.train.BytesList(value = [bytes(string_labels, 'utf-8') ])),
              'trainable_parameters': tf.train.Feature(int64_list = tf.train.Int64List(value = [result["trainable_params"]])),
              'training_time': tf.train.Feature(float_list=tf.train.FloatList(value=[result["total_time"]])),
              'train_accuracy': tf.train.Feature(float_list=tf.train.FloatList(value=[result['evaluation_results'][-1]['train_accuracy']])),
              'validation_accuracy': tf.train.Feature(float_list=tf.train.FloatList(value=[result['evaluation_results'][-1]['validation_accuracy']])),
              'test_accuracy': tf.train.Feature(float_list=tf.train.FloatList(value=[result['evaluation_results'][-1]['test_accuracy']]))}))

            if DEBUG: print(example)

            writer.write(example.SerializeToString())

  logging.info(f'{print_util.bcolors.OKGREEN}Saved dataset{print_util.bcolors.ENDC} to {os.path.join(model_dir, "cnnbench.tfrecord")}')

if __name__ == '__main__':
    app.run(main)
