# Generate a library of all graphs up to structure and label isomorphism.

# Code built upon https://github.com/google-research/nasbench/blob/
#   master/nasbench/scripts/generate_graphs.py
	 
# The goal is to generate all unique computational graphs up to some number of
# vertices and edges. Computational graphs can be represented by directed acyclic
# graphs with all components connected along some path from a specially-labeled
# input to output. The pseudocode for generating these is:

# for V in [2, ..., module_vertices]:    # V includes input and output vertices
#   generate all bitmasks of length V*(V-1)/2   # num upper triangular entries
#   for each bitmask:
#     convert bitmask to adjacency matrix
#     if adjacency matrix has disconnected vertices from input/output:
#       discard and continue to next matrix
#     generate all labelings of ops to vertices
#     for each labeling:
#       compute graph hash from matrix and labels
#       if graph hash has not been seen before:
#         output graph (adjacency matrix + labeling)

# This script uses a modification on Weisfeiler-Lehman color refinement
# (https://ist.ac.at/mfcs13/slides/gi.pdf) for graph hashing, which is very
# loosely similar to the hashing approach described in
# https://arxiv.org/pdf/1606.00001.pdf. The general idea is to assign each vertex
# a hash based on the in-degree, out-degree, and operation label then iteratively
# hash each vertex with the hashes of its neighbors.

# In more detail, the iterative update involves repeating the following steps a
# number of times greater than or equal to the diameter of the graph:
#   1) For each vertex, sort the hashes of the in-neighbors.
#   2) For each vertex, sort the hashes of the out-neighbors.
#   3) For each vertex, concatenate the sorted hashes from (1), (2) and the vertex
#      operation label.
#   4) For each vertex, compute the MD5 hash of the concatenated values in (3).
#   5) Assign the newly computed hashes to each vertex.

# Finally, sort the hashes of all the vertices and concat and hash one more time
# to obtain the final graph hash. This hash is a graph invariant as all operations
# are invariant under isomorphism, thus we expect no false negatives (isomorphic
# graphs hashed to different values).

# Author : Shikhar Tuli


import os
import sys
import yaml
import numpy as np
import itertools
import json

from utils import graph_util, print_util as pu


HASH_SIMPLE = False
ALLOW_2_V = False


def main(config, check_isomorphism = False):

	total_modules = 0    # Total number of modules (including isomorphisms)
	total_graphs = 0    # Total number of graphs (including isomorphisms)

	# hash --> (matrix, label) for the canonical graph associated with each hash
	module_buckets = {}
	graph_buckets = {}

	print(f'{pu.bcolors.HEADER}Generating modules...{pu.bcolors.ENDC}')
	print(f"{pu.bcolors.HEADER}Using {config['module_vertices']} vertices, {len(config['base_ops'])} labels, " \
		+ f"max {config['max_edges']} edges{pu.bcolors.ENDC}")
	print()

	if not ALLOW_2_V and config['module_vertices'] < 3: 
		print(f'{pu.bcolors.FAIL}Check config file. "module_vertices" should be 3 or greater{pu.bcolors.ENDC}')
		sys.exit()

	# Generate all possible martix-label pairs (or modules)
	for vertices in range(2 if ALLOW_2_V else 3, config['module_vertices'] + 1):
		for bits in range(2 ** (vertices * (vertices-1) // 2)):
			# Construct adj matrix from bit string
			matrix = np.fromfunction(graph_util.gen_is_edge_fn(bits),
									 (vertices, vertices),
									 dtype=np.int8)

			# Discard any modules which can be pruned or exceed constraints
			if (not graph_util.is_full_dag(matrix) or
					graph_util.num_edges(matrix) > config['max_edges']):
				continue

			# Iterate through all possible labelings
			for labels in itertools.product(*[list(config['base_ops'])
												for _ in range(vertices-2)]):
				total_modules += 1
				labels = ['input'] + list(labels) + ['output']
				module_fingerprint = graph_util.hash_module(matrix, labels, config['hash_algo'])

				if module_fingerprint not in module_buckets:
					if vertices != 2 or ALLOW_2_V: module_buckets[module_fingerprint] = (matrix, labels)
				# Module-level isomorphism check -
				elif check_isomorphism:
					canonical_matrix = module_buckets[module_fingerprint]
					if not graph_util.compare_modules(
							(matrix.tolist(), labeling), canonical_matrix):
						print(f'{pu.bcolors.FAIL}Matrix:\n{matrix}\nLabels: {labels}\nis not isomorphic to' \
								+ f' canonical matrix:\n{canonical_matrix[0]}\nLabels: {canonical_matrix[1]}{pu.bcolors.ENDC}')
						sys.exit()

		print(f'{pu.bcolors.OKGREEN}Generated up to {vertices} vertices: {len(module_buckets)} modules ' \
			+ f'({total_modules} without hashing){pu.bcolors.ENDC}')

	print()
	print(f'{pu.bcolors.HEADER}Generating graphs...{pu.bcolors.ENDC}')
	print(f'{pu.bcolors.HEADER}Using max {config['max_modules']} modules{pu.bcolors.ENDC}')
	print()

	# Generate graphs using modules
	# TODO: add support for dense_ops in the last module
	for modules in range(1, config['max_modules'] + 1):
		for module_fingerprints in itertools.product(*[module_buckets.keys()
														for _ in range(modules)]): 

			modules_selected = [module_buckets[fingerprint] for fingerprint in module_fingerprints]
			merged_modules = graph_util.generate_merged_modules(modules_selected)

			if HASH_SIMPLE:
				graph_fingerprint = graph_util.hash_graph_simple(modules_selected, config['hash_algo'])
			else:
				graph_fingerprint = graph_util.hash_graph(modules_selected, config['hash_algo'])

			if graph_fingerprint not in graph_buckets:
				total_graphs += 1
				if HASH_SIMPLE:
					graph_buckets[graph_fingerprint] = modules_selected
				else:
					graph_buckets[graph_fingerprint] = merged_modules

			# Graph-level isomorphism check -
			elif check_isomorphism:
				if not graph_util.compare_graphs(merged_modules, graph_buckets[graph_fingerprint]) or HASH_SIMPLE:
					print(f'{pu.bcolors.FAIL}Two graphs found with same hash - {graph_fingerprint}{pu.bcolors.ENDC}')
					count = 0
					for fingerprint in module_fingerprints:
						count += 1
						print(f'{pu.bcolors.FAIL}Module no.: {count}{pu.bcolors.ENDC}')
						print(f'{pu.bcolors.FAIL}Module Matrix: \n{np.array(module_buckets[fingerprint][0])}{pu.bcolors.ENDC}')
						print(f'{pu.bcolors.FAIL}Operations: {module_buckets[fingerprint][1]}{pu.bcolors.ENDC}')

					sys.exit()

		print(f'{pu.bcolors.OKGREEN}Generated up to {modules} modules: {total_graphs} graphs{pu.bcolors.ENDC}')

	return graph_buckets

