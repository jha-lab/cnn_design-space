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

"""Generate all graphs up to structure and label isomorphism.

The goal is to generate all unique computational graphs up to some number of
vertices and edges. Computational graphs can be represented by directed acyclic
graphs with all components connected along some path from a specially-labeled
input to output. The pseudocode for generating these is:

for V in [2, ..., module_vertices]:    # V includes input and output vertices
  generate all bitmasks of length V*(V-1)/2   # num upper triangular entries
  for each bitmask:
    convert bitmask to adjacency matrix
    if adjacency matrix has disconnected vertices from input/output:
      discard and continue to next matrix
    generate all labelings of ops to vertices
    for each labeling:
      compute graph hash from matrix and labels
      if graph hash has not been seen before:
        output graph (adjacency matrix + labeling)

This script uses a modification on Weisfeiler-Lehman color refinement
(https://ist.ac.at/mfcs13/slides/gi.pdf) for graph hashing, which is very
loosely similar to the hashing approach described in
https://arxiv.org/pdf/1606.00001.pdf. The general idea is to assign each vertex
a hash based on the in-degree, out-degree, and operation label then iteratively
hash each vertex with the hashes of its neighbors.

In more detail, the iterative update involves repeating the following steps a
number of times greater than or equal to the diameter of the graph:
  1) For each vertex, sort the hashes of the in-neighbors.
  2) For each vertex, sort the hashes of the out-neighbors.
  3) For each vertex, concatenate the sorted hashes from (1), (2) and the vertex
     operation label.
  4) For each vertex, compute the MD5 hash of the concatenated values in (3).
  5) Assign the newly computed hashes to each vertex.

Finally, sort the hashes of all the vertices and concat and hash one more time
to obtain the final graph hash. This hash is a graph invariant as all operations
are invariant under isomorphism, thus we expect no false negatives (isomorphic
graphs hashed to different values).

We have empirically verified that, for graphs up to 7 vertices, 9 edges, 3 ops,
this algorithm does not cause "false positives" (graphs that hash to the same
value but are non-isomorphic). For such graphs, this algorithm yields 423,624
unique computation graphs, which is roughly 1/3rd of the total number of
connected DAGs before de-duping using this hash algorithm.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import json
import sys

if '../../' not in sys.path:
  sys.path.append('../../')

from absl import app
from absl import flags
from absl import logging

from cnnbench.lib import graph_util
from cnnbench.lib import print_util
from cnnbench.lib import config as _config
import numpy as np
import tensorflow as tf   # For gfile

flags.DEFINE_string('output_file', '/tmp/generated_graphs.json',
                    'Output file name.')
flags.DEFINE_integer('num_ops', 3, 'Number of operation labels.')
flags.DEFINE_boolean('verify_isomorphism', True,
                     'Exhaustively verifies that each detected isomorphism'
                     ' is truly an isomorphism. This operation is very'
                     ' expensive.')
flags.DEFINE_integer('max_modules', 1, 'Maximum number of modules comprising of'
                    ' pairs of adjacency matrices and operation labels')

FLAGS = flags.FLAGS


def main(_):
  HASH_SIMPLE = False
  ALLOW_2_V = False

  config = _config.build_config()

  if FLAGS.run_nasbench:
    ALLOW_2_V = True

  total_modules = 0    # Total number of modules (including isomorphisms)
  total_graphs = 0    # Total number of graphs (including isomorphisms)

  # hash --> (matrix, label) for the canonical graph associated with each hash
  module_buckets = {}
  graph_buckets = {}

  logging.get_absl_handler().setFormatter(None)
  logging.info(f'{print_util.bcolors.HEADER}Generating modules{print_util.bcolors.ENDC}')
  if not ALLOW_2_V: 
    logging.info(f'{print_util.bcolors.HEADER}Neglecting 2 vertex modules{print_util.bcolors.ENDC}')
  logging.info(f'{print_util.bcolors.HEADER}Using {FLAGS.module_vertices} vertices, {FLAGS.num_ops} op labels, max {FLAGS.max_edges} edges{print_util.bcolors.ENDC}')

  if FLAGS.run_nasbench == False:
    if not ALLOW_2_V and FLAGS.module_vertices < 3: 
      logging.error(f'{print_util.bcolors.FAIL}ERROR: "module_vertices" should be 3 or greater{print_util.bcolors.ENDC}')
      sys.exit()

  # Generate all possible martix-label pairs (or modules)
  for vertices in range(2 if ALLOW_2_V else 3, FLAGS.module_vertices+1):
    for bits in range(2 ** (vertices * (vertices-1) // 2)):
      # Construct adj matrix from bit string
      matrix = np.fromfunction(graph_util.gen_is_edge_fn(bits),
                               (vertices, vertices),
                               dtype=np.int8)

      # Discard any modules which can be pruned or exceed constraints
      if (not graph_util.is_full_dag(matrix) or
          graph_util.num_edges(matrix) > FLAGS.max_edges):
        continue

      # Iterate through all possible labelings
      for labeling in itertools.product(*[range(FLAGS.num_ops)
                                          for _ in range(vertices-2)]):
        total_modules += 1
        labeling = [-1] + list(labeling) + [-2]
        module_fingerprint = graph_util.hash_module(matrix, labeling, FLAGS.hash_algo)

        if module_fingerprint not in module_buckets:
          # No need for modules with 2 vertices in expanded space
          if vertices != 2 or ALLOW_2_V: module_buckets[module_fingerprint] = (matrix.tolist(), labeling)

        # Module-level isomorphism check -
        # This catches the "false positive" case of two modules which are not isomorphic
        elif FLAGS.verify_isomorphism:
          canonical_matrix = module_buckets[module_fingerprint]
          if not graph_util.is_isomorphic(
              (matrix.tolist(), labeling), canonical_matrix):
            logging.error('Matrix:\n%s\nLabel: %s\nis not isomorphic to'
                          ' canonical matrix:\n%s\nLabel: %s',
                          str(matrix), str(labeling),
                          str(canonical_matrix[0]),
                          str(canonical_matrix[1]))
            sys.exit()

    logging.info('Up to %d vertices: %d modules (%d without hashing)',
                 vertices, len(module_buckets), total_modules)

  logging.info('')
  logging.info(f'{print_util.bcolors.HEADER}Generating graphs{print_util.bcolors.ENDC}')

  # Generate graphs using modules
  # Permute over modules to generate graphs with upto FLAGS.max_modules
  if not FLAGS.run_nasbench:
    logging.info(f'{print_util.bcolors.HEADER}Using max {FLAGS.max_modules} modules{print_util.bcolors.ENDC}')
    for modules in range(1, FLAGS.max_modules+1):
      for module_fingerprints in itertools.product(*[module_buckets.keys()
                                                  for _ in range(modules)]): 

        modules_selected = [module_buckets[fingerprint] for fingerprint in module_fingerprints]
        merged_modules = graph_util.generate_merged_modules(modules_selected)

        if HASH_SIMPLE:
          graph_fingerprint = graph_util.hash_graph_simple(modules_selected, FLAGS.hash_algo)
        else:
          graph_fingerprint = graph_util.hash_graph(modules_selected, FLAGS.hash_algo)

        if graph_fingerprint not in graph_buckets:
          total_graphs += 1
          if HASH_SIMPLE:
            graph_buckets[graph_fingerprint] = modules_selected
          else:
            graph_buckets[graph_fingerprint] = merged_modules

        # Graph-level isomorphism check -
        # This catches the "false positive" case of two graphs which are not isomorphic
        elif FLAGS.verify_isomorphism:
          if not graph_util.compare_graphs(merged_modules, graph_buckets[graph_fingerprint]) or HASH_SIMPLE:
            logging.error(f'{print_util.bcolors.FAIL}ERROR: two graphs found with same hash - {graph_fingerprint}{print_util.bcolors.ENDC}')

            count = 0
            for fingerprint in module_fingerprints:
              count += 1
              logging.error(f'{print_util.bcolors.FAIL}Module no.: {count}{print_util.bcolors.ENDC}')
              logging.error(f'{print_util.bcolors.FAIL}Module Matrix: \n{np.array(module_buckets[fingerprint][0])}{print_util.bcolors.ENDC}')
              logging.error(f'{print_util.bcolors.FAIL}Operations: {module_buckets[fingerprint][1]}{print_util.bcolors.ENDC}')

            sys.exit()

      logging.info('Upto %d modules: %d graphs',
                    modules, total_graphs)
  else:
    assert FLAGS.max_modules == 1, 'The flag "max_modules" must be 1 if "run_nasbench" is True.'
    logging.info(f'{print_util.bcolors.HEADER}Using {len(module_buckets)} modules to generate NASBench-like graphs{print_util.bcolors.ENDC}')
    for module_fingerprint in module_buckets:

        total_graphs += 1
        graph_fingerprint = module_fingerprint

        if graph_fingerprint not in graph_buckets:
          graph_buckets[graph_fingerprint] = [module_buckets[module_fingerprint] 
                                              for _ in range(config['num_stacks']*config['num_modules_per_stack'])]

        # No need for Graph-level isomorphism check

    logging.info('Upto %d modules: %d graphs',
                    1, total_graphs)

  with tf.io.gfile.GFile(FLAGS.output_file, 'w') as f:
    json.dump(graph_buckets, f, sort_keys=True)

  return total_graphs


if __name__ == '__main__':
  app.run(main)
