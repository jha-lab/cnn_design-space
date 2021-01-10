# Adds cnnbench.tfrecord dataset file
# Author :  Shikhar Tuli

import sys

if '../' not in sys.path:
	sys.path.append('../')

# Do not show warnings of deprecated functions
import os
import json
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # or any {DEBUG, INFO, WARN, ERROR, FATAL} 

from cnnbench.lib import config as _config

DEBUG = False

assert len(sys.argv) == 2, "Takes exactly one argument [model directory]"
# model_dir = '../results/vertices_2/'
model_dir = sys.argv[1]

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

                    raw_adjacency = models[model][0]
                    raw_labels = models[model][1]
                    adjacency = np.array(raw_adjacency)
                    labels = (['input'] + [self.config['available_ops'][lab] for lab in raw_labels[1:-1]] + ['output'])
                    string_labels = ''
                    for label in labels:
                        string_labels = string_labels + label + ','
                    string_labels = string_labels[:-1]

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
                        'module_adjacency': tf.train.Feature(int64_list = tf.train.Int64List(value = [a for a in list(adjacency.flatten())])),
                        'module_operations': tf.train.Feature(bytes_list = tf.train.BytesList(value = [bytes(string_labels, 'utf-8') ])),
                        'trainable_parameters': tf.train.Feature(int64_list = tf.train.Int64List(value = [result["trainable_params"]])),
                        'training_time': tf.train.Feature(float_list=tf.train.FloatList(value=[result["total_time"]])),
                        'train_accuracy': tf.train.Feature(float_list=tf.train.FloatList(value=[result['evaluation_results'][-1]['train_accuracy']])),
                        'validation_accuracy': tf.train.Feature(float_list=tf.train.FloatList(value=[result['evaluation_results'][-1]['validation_accuracy']])),
                        'test_accuracy': tf.train.Feature(float_list=tf.train.FloatList(value=[result['evaluation_results'][-1]['test_accuracy']]))}))
                    
                    if DEBUG: print(example)
                    writer.write(example.SerializeToString())
