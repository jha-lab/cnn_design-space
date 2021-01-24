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

"""Tests for lib/graph_util.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os

from absl import flags

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

FLAGS = flags.FLAGS

FLAGS(sys.argv)

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
        input, 3, config['stem_filter_size'], (1, 1),
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

  assert graph_util.hash_graph_simple(graph1) != graph_util.hash_graph_simple(graph2)

  assert graph_util.hash_graph(graph1) == graph_util.hash_graph(graph2)

  assert graph_util.compare_graphs(graph1, graph2)


def test_gen_is_edge():
  """Tests gen_is_edge generates correct graphs."""
  fn = graph_util.gen_is_edge_fn(0)     # '000'
  arr = np.fromfunction(fn, (3, 3), dtype=np.int8)
  assert (np.array_equal(arr,
                                 np.array([[0, 0, 0],
                                           [0, 0, 0],
                                           [0, 0, 0]])))

  fn = graph_util.gen_is_edge_fn(3)     # '011'
  arr = np.fromfunction(fn, (3, 3), dtype=np.int8)
  assert (np.array_equal(arr,
                                 np.array([[0, 1, 1],
                                           [0, 0, 0],
                                           [0, 0, 0]])))

  fn = graph_util.gen_is_edge_fn(5)     # '101'
  arr = np.fromfunction(fn, (3, 3), dtype=np.int8)
  assert (np.array_equal(arr,
                                 np.array([[0, 1, 0],
                                           [0, 0, 1],
                                           [0, 0, 0]])))

  fn = graph_util.gen_is_edge_fn(7)     # '111'
  arr = np.fromfunction(fn, (3, 3), dtype=np.int8)
  assert (np.array_equal(arr,
                                 np.array([[0, 1, 1],
                                           [0, 0, 1],
                                           [0, 0, 0]])))

  fn = graph_util.gen_is_edge_fn(7)     # '111'
  arr = np.fromfunction(fn, (4, 4), dtype=np.int8)
  assert (np.array_equal(arr,
                                 np.array([[0, 1, 1, 0],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 0],
                                           [0, 0, 0, 0]])))

  fn = graph_util.gen_is_edge_fn(18)     # '010010'
  arr = np.fromfunction(fn, (4, 4), dtype=np.int8)
  assert (np.array_equal(arr,
                                 np.array([[0, 0, 1, 0],
                                           [0, 0, 0, 1],
                                           [0, 0, 0, 0],
                                           [0, 0, 0, 0]])))

  fn = graph_util.gen_is_edge_fn(35)     # '100011'
  arr = np.fromfunction(fn, (4, 4), dtype=np.int8)
  assert (np.array_equal(arr,
                                 np.array([[0, 1, 1, 0],
                                           [0, 0, 0, 0],
                                           [0, 0, 0, 1],
                                           [0, 0, 0, 0]])))

def test_is_full_dag():
  """Tests is_full_dag classifies DAGs."""
  assert (graph_util.is_full_dag(np.array(
      [[0, 1, 0],
       [0, 0, 1],
       [0, 0, 0]])))

  assert (graph_util.is_full_dag(np.array(
      [[0, 1, 1],
       [0, 0, 1],
       [0, 0, 0]])))

  assert (graph_util.is_full_dag(np.array(
      [[0, 1, 1, 0],
       [0, 0, 0, 1],
       [0, 0, 0, 1],
       [0, 0, 0, 0]])))

  # vertex 1 not connected to input
  assert not (graph_util.is_full_dag(np.array(
      [[0, 0, 1],
       [0, 0, 1],
       [0, 0, 0]])))

  # vertex 1 not connected to output
  assert not (graph_util.is_full_dag(np.array(
      [[0, 1, 1],
       [0, 0, 0],
       [0, 0, 0]])))

  # 1, 3 are connected to each other but disconnected from main path
  assert not (graph_util.is_full_dag(np.array(
      [[0, 0, 1, 0, 0],
       [0, 0, 0, 1, 0],
       [0, 0, 0, 0, 1],
       [0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0]])))

  # no path from input to output
  assert not (graph_util.is_full_dag(np.array(
      [[0, 0, 1, 0],
       [0, 0, 0, 1],
       [0, 0, 0, 0],
       [0, 0, 0, 0]])))

  # completely disconnected vertex
  assert not (graph_util.is_full_dag(np.array(
      [[0, 1, 0, 0],
       [0, 0, 0, 1],
       [0, 0, 0, 0],
       [0, 0, 0, 0]])))

def test_hash_module():
  # Diamond graph with label permutation
  matrix1 = np.array(
      [[0, 1, 1, 0,],
       [0, 0, 0, 1],
       [0, 0, 0, 1],
       [0, 0, 0, 0]])
  label1 = [-1, 1, 2, -2]
  label2 = [-1, 2, 1, -2]

  hash1 = graph_util.hash_module(matrix1, label1)
  hash2 = graph_util.hash_module(matrix1, label2)
  assert hash1 == hash2

  # Simple graph with edge permutation
  matrix1 = np.array(
      [[0, 1, 1, 0, 0],
       [0, 0, 0, 0, 1],
       [0, 0, 0, 1, 0],
       [0, 0, 0, 0, 1],
       [0, 0, 0, 0, 0]])
  label1 = [-1, 1, 2, 3, -2]

  matrix2 = np.array(
      [[0, 1, 0, 1, 0],
       [0, 0, 1, 0, 0],
       [0, 0, 0, 0, 1],
       [0, 0, 0, 0, 1],
       [0, 0, 0, 0, 0]])
  label2 = [-1, 2, 3, 1, -2]

  matrix3 = np.array(
      [[0, 1, 1, 0, 0],
       [0, 0, 0, 1, 0],
       [0, 0, 0, 0, 1],
       [0, 0, 0, 0, 1],
       [0, 0, 0, 0, 0]])
  label3 = [-1, 2, 1, 3, -2]

  hash1 = graph_util.hash_module(matrix1, label1)
  hash2 = graph_util.hash_module(matrix2, label2)
  hash3 = graph_util.hash_module(matrix3, label3)
  assert hash1 == hash2
  assert hash2 == hash3

  hash4 = graph_util.hash_module(matrix1, label2)
  assert hash4 != hash1

  hash5 = graph_util.hash_module(matrix1, label3)
  assert hash5 != hash1

  # Connected non-isomorphic regular graphs on 6 interior vertices (8 total)
  matrix1 = np.array(
      [[0, 1, 0, 0, 0, 0, 0, 0],
       [0, 0, 1, 1, 0, 0, 1, 0],
       [0, 0, 0, 0, 1, 1, 0, 0],
       [0, 0, 0, 0, 1, 1, 0, 0],
       [0, 0, 0, 0, 0, 0, 1, 0],
       [0, 0, 0, 0, 0, 0, 1, 0],
       [0, 0, 0, 0, 0, 0, 0, 1],
       [0, 0, 0, 0, 0, 0, 0, 0]])
  matrix2 = np.array(
      [[0, 1, 0, 0, 0, 0, 0, 0],
       [0, 0, 1, 1, 0, 1, 0, 0],
       [0, 0, 0, 0, 1, 0, 1, 0],
       [0, 0, 0, 0, 1, 1, 0, 0],
       [0, 0, 0, 0, 0, 0, 1, 0],
       [0, 0, 0, 0, 0, 0, 1, 0],
       [0, 0, 0, 0, 0, 0, 0, 1],
       [0, 0, 0, 0, 0, 0, 0, 0]])
  label1 = [-1, 1, 1, 1, 1, 1, 1, -2]

  hash1 = graph_util.hash_module(matrix1, label1)
  hash2 = graph_util.hash_module(matrix2, label1)
  assert hash1 != hash2

  # Non-isomorphic tricky case (breaks if you don't include )
  hash1 = graph_util.hash_module(
      np.array([[0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0]]),
      [-1, 1, 0, 0, -2])

  hash2 = graph_util.hash_module(
      np.array([[0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0]]),
      [-1, 0, 0, 1, -2])
  assert hash1 != hash2

  # Non-isomorphic tricky case (breaks if you don't use directed edges)
  hash1 = graph_util.hash_module(
      np.array([[0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 0]]),
      [-1, 1, 0, -2])

  hash2 = graph_util.hash_module(
      np.array([[0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 0]]),
      [-1, 0, 1, -2])
  assert hash1 != hash2

  # Non-isomorphic tricky case (breaks if you only use out-neighbors and )
  hash1 = graph_util.hash_module(np.array([[0, 1, 1, 1, 1, 0, 0],
                                           [0, 0, 1, 0, 0, 0, 0],
                                           [0, 0, 0, 0, 0, 0, 1],
                                           [0, 0, 0, 0, 0, 1, 0],
                                           [0, 0, 0, 0, 0, 1, 0],
                                           [0, 0, 0, 0, 0, 0, 1],
                                           [0, 0, 0, 0, 0, 0, 0]]),
                                 [-1, 1, 0, 0, 0, 0, -2])
  hash2 = graph_util.hash_module(np.array([[0, 1, 1, 1, 1, 0, 0],
                                           [0, 0, 1, 0, 0, 0, 0],
                                           [0, 0, 0, 0, 0, 0, 1],
                                           [0, 0, 0, 0, 0, 1, 0],
                                           [0, 0, 0, 0, 0, 1, 0],
                                           [0, 0, 0, 0, 0, 0, 1],
                                           [0, 0, 0, 0, 0, 0, 0]]),
                                 [-1, 0, 0, 0, 1, 0, -2])
  assert hash1 != hash2

def test_permute_module():
  # Does not have to be DAG
  matrix = np.array([[1, 1, 0],
                     [0, 0, 1],
                     [1, 0, 1]])
  labels = ['a', 'b', 'c']

  p1, l1 = graph_util.permute_module(matrix, labels, [2, 0, 1])
  assert (np.array_equal(p1,
                   np.array([[0, 1, 0],
                             [0, 1, 1],
                             [1, 0, 1]])))
  assert l1 == ['b', 'c', 'a']

  p1, l1 = graph_util.permute_module(matrix, labels, [0, 2, 1])
  assert (np.array_equal(p1,
                   np.array([[1, 0, 1],
                             [1, 1, 0],
                             [0, 1, 0]])))
  assert l1 == ['a', 'c', 'b']

def test_is_isomorphic():
  # Reuse some tests from hash_module
  matrix1 = np.array(
      [[0, 1, 1, 0,],
       [0, 0, 0, 1],
       [0, 0, 0, 1],
       [0, 0, 0, 0]])
  label1 = [-1, 1, 2, -2]
  label2 = [-1, 2, 1, -2]

  assert (graph_util.is_isomorphic((matrix1, label1),
                                           (matrix1, label2)))

  # Simple graph with edge permutation
  matrix1 = np.array(
      [[0, 1, 1, 0, 0],
       [0, 0, 0, 0, 1],
       [0, 0, 0, 1, 0],
       [0, 0, 0, 0, 1],
       [0, 0, 0, 0, 0]])
  label1 = [-1, 1, 2, 3, -2]

  matrix2 = np.array(
      [[0, 1, 0, 1, 0],
       [0, 0, 1, 0, 0],
       [0, 0, 0, 0, 1],
       [0, 0, 0, 0, 1],
       [0, 0, 0, 0, 0]])
  label2 = [-1, 2, 3, 1, -2]

  matrix3 = np.array(
      [[0, 1, 1, 0, 0],
       [0, 0, 0, 1, 0],
       [0, 0, 0, 0, 1],
       [0, 0, 0, 0, 1],
       [0, 0, 0, 0, 0]])
  label3 = [-1, 2, 1, 3, -2]

  assert (graph_util.is_isomorphic((matrix1, label1),
                                           (matrix2, label2)))
  assert (graph_util.is_isomorphic((matrix1, label1),
                                           (matrix3, label3)))
  assert not (graph_util.is_isomorphic((matrix1, label1),
                                            (matrix2, label1)))

  # Connected non-isomorphic regular graphs on 6 interior vertices (8 total)
  matrix1 = np.array(
      [[0, 1, 0, 0, 0, 0, 0, 0],
       [0, 0, 1, 1, 0, 0, 1, 0],
       [0, 0, 0, 0, 1, 1, 0, 0],
       [0, 0, 0, 0, 1, 1, 0, 0],
       [0, 0, 0, 0, 0, 0, 1, 0],
       [0, 0, 0, 0, 0, 0, 1, 0],
       [0, 0, 0, 0, 0, 0, 0, 1],
       [0, 0, 0, 0, 0, 0, 0, 0]])
  matrix2 = np.array(
      [[0, 1, 0, 0, 0, 0, 0, 0],
       [0, 0, 1, 1, 0, 1, 0, 0],
       [0, 0, 0, 0, 1, 0, 1, 0],
       [0, 0, 0, 0, 1, 1, 0, 0],
       [0, 0, 0, 0, 0, 0, 1, 0],
       [0, 0, 0, 0, 0, 0, 1, 0],
       [0, 0, 0, 0, 0, 0, 0, 1],
       [0, 0, 0, 0, 0, 0, 0, 0]])
  label1 = [-1, 1, 1, 1, 1, 1, 1, -2]

  assert not (graph_util.is_isomorphic((matrix1, label1),
                                            (matrix2, label1)))

  # Connected isomorphic regular graphs on 8 total vertices (bipartite)
  matrix1 = np.array(
      [[0, 0, 0, 0, 1, 1, 1, 0],
       [0, 0, 0, 0, 1, 1, 0, 1],
       [0, 0, 0, 0, 1, 0, 1, 1],
       [0, 0, 0, 0, 0, 1, 1, 1],
       [1, 1, 1, 0, 0, 0, 0, 0],
       [1, 1, 0, 1, 0, 0, 0, 0],
       [1, 0, 1, 1, 0, 0, 0, 0],
       [0, 1, 1, 1, 0, 0, 0, 0]])
  matrix2 = np.array(
      [[0, 1, 0, 1, 1, 0, 0, 0],
       [1, 0, 1, 0, 0, 1, 0, 0],
       [0, 1, 0, 1, 0, 0, 1, 0],
       [1, 0, 1, 0, 0, 0, 0, 1],
       [1, 0, 0, 0, 0, 1, 0, 1],
       [0, 1, 0, 0, 1, 0, 1, 0],
       [0, 0, 1, 0, 0, 1, 0, 1],
       [0, 0, 0, 1, 1, 0, 1, 0]])
  label1 = [1, 1, 1, 1, 1, 1, 1, 1]

  # Sanity check: manual permutation
  perm = [0, 5, 7, 2, 4, 1, 3, 6]
  pm1, pl1 = graph_util.permute_module(matrix1, label1, perm)
  assert (np.array_equal(matrix2, pm1))
  assert pl1 == label1

  assert (graph_util.is_isomorphic((matrix1, label1),
                                           (matrix2, label1)))

  label2 = [1, 1, 1, 1, 2, 2, 2, 2]
  label3 = [1, 2, 1, 2, 2, 1, 2, 1]

  assert (graph_util.is_isomorphic((matrix1, label2),
                                           (matrix2, label3)))

def test_random_isomorphism_hashing():
  # Tests that hash_module always provides the same hash for randomly
  # generated isomorphic graphs.
  for _ in range(1000):
    # Generate random graph. Note: the algorithm works (i.e. same hash ==
    # isomorphic graphs) for all directed graphs with coloring and does not
    # require the graph to be a DAG.
    size = random.randint(3, 20)
    matrix = np.random.randint(0, 2, [size, size])
    labels = [random.randint(0, 10) for _ in range(size)]

    # Generate permutation of matrix and labels.
    perm = np.random.permutation(size).tolist()
    pmatrix, plabels = graph_util.permute_module(matrix, labels, perm)

    # Hashes should be identical.
    hash1 = graph_util.hash_module(matrix, labels)
    hash2 = graph_util.hash_module(pmatrix, plabels)
    assert hash1 == hash2

def test_counterexample_bipartite():
  # This is a counter example that shows that the hashing algorithm is not
  # perfectly identifiable (i.e. there are non-isomorphic graphs with the same
  # hash). If this tests fails, it means the algorithm must have been changed
  # in some way that allows it to identify these graphs as non-isomoprhic.
  matrix1 = np.array(
      [[0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
       [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
       [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

  matrix2 = np.array(
      [[0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
       [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

  labels = [-1, 1, 1, 1, 1, 2, 2, 2, 2, -2]

  # This takes far too long to run so commenting it out. The graphs are
  # non-isomorphic fairly obviously from visual inspection.
  # assert not (graph_util.is_isomorphic((matrix1, labels),
  #                                           (matrix2, labels)))
  assert graph_util.hash_module(matrix1, labels) == graph_util.hash_module(matrix2, labels)

