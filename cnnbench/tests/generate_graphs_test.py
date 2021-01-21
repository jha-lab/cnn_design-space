# Script to test graph generation
# Author :  Shikhar Tuli

import sys
import os

from absl import flags

import warnings

warnings.filterwarnings("ignore")

if os.path.abspath(os.path.join(sys.path[0], '../..')) not in sys.path:
  sys.path.append(os.path.abspath(os.path.join(sys.path[0], '../..')))


from cnnbench.scripts import generate_graphs as graph_generator

FLAGS = flags.FLAGS

FLAGS(sys.argv)


def test_graph_generation():

    FLAGS.output_file = 'g_test.json'
    FLAGS.module_vertices = 4
    FLAGS.num_ops = 1
    FLAGS.max_modules = 3
    FLAGS.run_nasbench = False

    graphs = graph_generator.main(1)

    os.remove(FLAGS.output_file)

    assert graphs == 1836

def test_graph_generation_nasbench():

    FLAGS.output_file = 'g_test.json'
    FLAGS.module_vertices = 4
    FLAGS.num_ops = 1
    FLAGS.max_modules = 1
    FLAGS.run_nasbench = True

    graphs = graph_generator.main(1)

    os.remove(FLAGS.output_file)

    assert graphs == 13
