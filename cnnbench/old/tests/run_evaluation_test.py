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

"""Unit tests for scripts/run_evaluation.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os

from absl import flags
from absl import logging

import pytest
import warnings

warnings.filterwarnings("ignore")

if os.path.abspath(os.path.join(sys.path[0], '../..')) not in sys.path:
  sys.path.append(os.path.abspath(os.path.join(sys.path[0], '../..')))

if os.path.abspath(os.path.join(sys.path[0], '../../job_scripts')) not in sys.path:
  sys.path.append(os.path.abspath(os.path.join(sys.path[0], '../../job_scripts')))

from cnnbench.scripts import generate_tfrecords
import generate_dataset_script
import generate_graphs_script
import run_evaluation_script
import cleanup_script

# Do not show warnings of deprecated functions
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.get_absl_handler().setFormatter(None)
logging.set_verbosity(logging.ERROR)  # or any {DEBUG, INFO, WARN, ERROR, FATAL} 

FLAGS = flags.FLAGS

FLAGS(sys.argv)


def test_basic_model_run():
  """Testing basic model run for two vertices and one repeat"""
  # Takes a lot of time to run

  warnings.filterwarnings("ignore")

  FLAGS.module_vertices = 2
  FLAGS.num_repeats = 1
  FLAGS.num_stacks = 1
  FLAGS.num_modules_per_stack = 1
  FLAGS.run_nasbench = True
  FLAGS.output_file = './results/vertices_2_test/generated_graphs.json'
  FLAGS.models_file = './results/vertices_2_test/generated_graphs.json' 
  FLAGS.output_dir = './results/vertices_2_test/evaluation' 
  FLAGS.cleanup_dir = './results/vertices_2_test/evaluation'
  FLAGS.model_dir = './results/vertices_2_test'
  FLAGS.data_dir = './datasets_test'
  
  generate_tfrecords.main('cifar10', FLAGS.data_dir, None)
  generate_graphs_script.main(1)
  run_evaluation_script.main(1)
  # cleanup_script.main(1) Taken into account by Github Action

