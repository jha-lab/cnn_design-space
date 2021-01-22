# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for lib/model_builder.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os

from absl import flags
from absl import logging

import warnings

warnings.filterwarnings("ignore")

if os.path.abspath(os.path.join(sys.path[0], '../..')) not in sys.path:
  sys.path.append(os.path.abspath(os.path.join(sys.path[0], '../..')))

from cnnbench.lib import base_ops
from cnnbench.lib import graph_util
from cnnbench.lib import model_builder 
from cnnbench.lib import module_spec
from cnnbench.lib import config as _config

import random
import numpy as np
import tensorflow as tf
import copy

# Do not show warnings of deprecated functions
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.get_absl_handler().setFormatter(None)
logging.set_verbosity(logging.ERROR)  # or any {DEBUG, INFO, WARN, ERROR, FATAL} 


def get_basic_model(graph):
  config = _config.build_config()

  is_training = True

  if config['data_format'] == 'channels_last':
    channel_axis = 3
  elif config['data_format'] == 'channels_first':
    # Currently this is not well supported
    channel_axis = 1
  else:
    raise ValueError('invalid data_format')

  matrices_list = []
  labels_list = []
  spec_list = []

  for module in graph:
    matrix = np.array(module[0])

    # Re-label to config['available_ops']
    labels = (['input'] +
            [config['available_ops'][lab] for lab in module[1][1:-1]] +
            ['output'])

    spec = module_spec.ModuleSpec(matrix, labels, config['hash_algo'])

    assert spec.valid_spec
    assert np.sum(spec.matrix) <= config['max_edges'] 

    matrices_list.append(matrix)
    labels_list.append(labels)
    spec_list.append(spec)

  input = tf.keras.layers.Input(shape=(224, 224, 3))

  if config['run_nasbench']:
    # Initial stem convolution
    net = base_ops.conv_bn_relu(
        input, 3, config['stem_filter_size'],
        is_training, config['data_format'])

    channels = net.get_shape()[channel_axis]

    for module_num in range(len(graph)):
      spec = spec_list[module_num]
      net = model_builder.build_module(
        spec,
        inputs=net,
        channels=channels,
        is_training=is_training)

    net = tf.keras.layers.GlobalAvgPool2D()(net)

    output = tf.keras.layers.Dense(1000, activation='softmax')(net)


  model = tf.keras.Model(input, output)
  return model


def test_models_with_same_hash():
  """Tests graphs with same recursive hash"""

  warnings.filterwarnings("ignore")
  
  matrix1 = np.array([[0, 1, 0],
                       [0, 0, 1],
                       [0, 0, 0]])
  matrix2 = copy.deepcopy(matrix1)
  label1 = [-1, 0, -2]

  graph1 = [(matrix1.astype(int).tolist(), label1), (matrix2.astype(int).tolist(), label1)]

  matrix3 = np.array([[0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [0, 0, 0, 0]])
  label2 = [-1, 0, 0, -2]

  graph2 = [(matrix3.astype(int).tolist(), label2)]

  model1 = get_basic_model(graph1)
  model2 = get_basic_model(graph2)

  assert model1.count_params() == model2.count_params()


def test_compute_vertex_channels_linear():
  """Tests modules with no branching."""
  matrix1 = np.array([[0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1],
                      [0, 0, 0, 0]])
  vc1 = model_builder.compute_vertex_channels(8, 8, matrix1)
  assert vc1 == [8, 8, 8, 8]

  vc2 = model_builder.compute_vertex_channels(8, 16, matrix1)
  assert vc2 == [8, 16, 16, 16]

  vc3 = model_builder.compute_vertex_channels(16, 8, matrix1)
  assert vc3 == [16, 8, 8, 8]

  matrix2 = np.array([[0, 1],
                      [0, 0]])
  vc4 = model_builder.compute_vertex_channels(1, 1, matrix2)
  assert vc4 == [1, 1]

  vc5 = model_builder.compute_vertex_channels(1, 5, matrix2)
  assert vc5 == [1, 5]

  vc5 = model_builder.compute_vertex_channels(5, 1, matrix2)
  assert vc5 == [5, 1]

def test_compute_vertex_channels_no_output_branch():
  """Tests modules that branch but not at the output vertex."""
  matrix1 = np.array([[0, 1, 1, 0, 0],
                      [0, 0, 0, 1, 0],
                      [0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0]])
  vc1 = model_builder.compute_vertex_channels(8, 8, matrix1)
  assert vc1 == [8, 8, 8, 8, 8]

  vc2 = model_builder.compute_vertex_channels(8, 16, matrix1)
  assert vc2 == [8, 16, 16, 16, 16]

  vc3 = model_builder.compute_vertex_channels(16, 8, matrix1)
  assert vc3 == [16, 8, 8, 8, 8]

def test_compute_vertex_channels_output_branching():
  """Tests modules that branch at output."""
  matrix1 = np.array([[0, 1, 1, 0],
                      [0, 0, 0, 1],
                      [0, 0, 0, 1],
                      [0, 0, 0, 0]])
  vc1 = model_builder.compute_vertex_channels(8, 8, matrix1)
  assert vc1 == [8, 4, 4, 8]

  vc2 = model_builder.compute_vertex_channels(8, 16, matrix1)
  assert vc2 == [8, 8, 8, 16]

  vc3 = model_builder.compute_vertex_channels(16, 8, matrix1)
  assert vc3 == [16, 4, 4, 8]

  vc4 = model_builder.compute_vertex_channels(8, 15, matrix1)
  assert vc4 == [8, 8, 7, 15]

  matrix2 = np.array([[0, 1, 1, 1, 0],
                      [0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0]])
  vc5 = model_builder.compute_vertex_channels(8, 8, matrix2)
  assert vc5 == [8, 3, 3, 2, 8]

  vc6 = model_builder.compute_vertex_channels(8, 15, matrix2)
  assert vc6 == [8, 5, 5, 5, 15]

def test_compute_vertex_channels_max():
  """Tests modules where some vertices take the max channels of neighbors."""
  matrix1 = np.array([[0, 1, 0, 0, 0],
                      [0, 0, 1, 1, 0],
                      [0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0]])
  vc1 = model_builder.compute_vertex_channels(8, 8, matrix1)
  assert vc1 == [8, 4, 4, 4, 8]

  vc2 = model_builder.compute_vertex_channels(8, 9, matrix1)
  assert vc2 == [8, 5, 5, 4, 9]

  matrix2 = np.array([[0, 1, 0, 1, 0],
                      [0, 0, 1, 0, 1],
                      [0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0]])

  vc3 = model_builder.compute_vertex_channels(8, 8, matrix2)
  assert vc3 == [8, 4, 4, 4, 8]

  vc4 = model_builder.compute_vertex_channels(8, 15, matrix2)
  assert vc4 == [8, 8, 7, 7, 15]

def test_covariance_matrix_against_numpy():
  """Tests that the TF implementation of covariance matrix matchs np.cov."""

  # Randomized test 100 times
  for _ in range(100):
    batch = np.random.randint(50, 150)
    features = np.random.randint(500, 1500)
    matrix = np.random.random((batch, features))

    tf_matrix = tf.constant(matrix, dtype=tf.float32)
    tf_cov_tensor = model_builder._covariance_matrix(tf_matrix)

    np_cov = np.cov(matrix)
    np.testing.assert_array_almost_equal(tf_cov_tensor, np_cov)

