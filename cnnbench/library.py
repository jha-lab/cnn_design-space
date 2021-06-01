# Generate a library of all graphs up to structure and label isomorphism.

# Code built upon https://github.com/google-research/nasbench/blob/
#   master/nasbench/scripts/generate_graphs.py
	 
# Author : Shikhar Tuli


import os
import sys
import yaml
import numpy as np
import itertools
from tqdm.contrib.itertools import product
import json

from utils import graph_util, print_util as pu


HASH_SIMPLE = False
ALLOW_2_V = False


class GraphLib(object):
	"""Graph Library containing all possible graphs within the design space
	
	Attributes:
		config (dict): dictionary of design space hyper-parameters
		library (list[Graph]): list of all possible graphs within the design space
	"""
	
	def __init__(self, config=None):
		"""Init GraphLib instance with config file
		
		Args:
			config (str): path to yaml file containing range of hyper-parameters in
				the design space

		Raises:
			AssertionError: if a sanity check fails
		"""
		if config:
			with open(config) as config_file:
				try:
					self.config = yaml.safe_load(config_file)
				except yaml.YAMLError as exc:
					print(exc)
				self.library = []
		else:
			self.config = {}
			self.library = []

	def __len__(self):
		"""Computes the number of graphs in the library

		Returns:
			len (int): number of graphs in the library
		"""
		return len(self.library)

	def __repr__(self):
		"""Representation of the GraphLib"""
		return f'{pu.bcolors.HEADER}Graph Library with configurations:{pu.bcolors.ENDC}\n{self.config}' \
			+ f'\n{pu.bcolors.HEADER}Number of graphs:{pu.bcolors.ENDC} {len(self.library)}'

	def build_library(self, check_isomorphism=True):
		"""Build the GraphLib library
		
		Args:
			check_isomorphism (bool, optional): if True, isomorphism is checked 
				for every graph. Default is False, to save compute time
		"""
		graph_buckets = generate_graphs(self.config, check_isomorphism=check_isomorphism)

		for graph_hash, graph in graph_buckets.items():
			self.library.append(Graph(graph, graph_hash))

		print(f'{pu.bcolors.OKGREEN}Graph library created!{pu.bcolors.ENDC} ' \
			+ f'\n{len(self.library)} graphs within the design space.')

	def build_embeddings(self, embedding_size: int, algo='MDS', kernel='WeisfeilerLehman', neighbors=100, n_jobs=8):
		"""Build the embeddings of all Graphs in GraphLib using MDS
		
		Args:
			embedding_size (int): size of the embedding
			algo (str): algorithm to use for generating embeddings. Can be any
				of the following:
					- 'MDS'
					- 'GD'
			kernel (str, optional): the kernel to be used for computing the dissimilarity 
				matrix. Can be any of the following:
					- 'WeisfeilerLehman'
					- 'NeighborhoodHash'
					- 'RandomWalkLabeled'
					- 'GraphEditDistance'
				The default value is 'WeisfeilerLehman'
			neighbors (int, optional): number of nearest neighbors to save for every graph
			n_jobs (int, optional): number of parrallel jobs for joblib
		"""
		print('Building embeddings for the Graph library')

		# Create list of graphs (tuples of adjacency matrices and ops)
		graph_list = [self.library[i].graph for i in range(len(self))]

		# Generate dissimilarity_matrix using the specified kernel
		diss_mat = graph_util.generate_dissimilarity_matrix(graph_list, kernel=kernel, n_jobs=n_jobs)

		# Generate embeddings using MDS or GD
		if algo == 'MDS':
			embeddings = embedding_util.generate_mds_embeddings(diss_mat, embedding_size=embedding_size, n_jobs=n_jobs)
		else:
			embeddings = embedding_util.generate_grad_embeddings(diss_mat, embedding_size=embedding_size, silent=True)

		# Get neighboring graph in the embedding space, for all Graphs
		neighbor_idx = embedding_util.get_neighbors(embeddings, neighbors)

		# Update embeddings and neighbors of all Graphs in GraphLib
		for i in range(len(self)):
			self.library[i].embedding = embeddings[i, :]
			self.library[i].neighbors = [self.library[int(neighbor_idx[i, n])].hash for n in range(neighbors)]

		self.num_neighbors = neighbors

		print(f'{pu.bcolors.OKGREEN}Embeddings generated, of size: {embedding_size}{pu.bcolors.ENDC}')

	def get_graph(self, model_hash: str) -> 'Graph':
		"""Return a Graph object in the library from hash
		
		Args:
			model_hash (str): hash of the graph in
				the library
		
		Returns:
			Graph object, model index
		"""
		hashes = [graph.hash for graph in self.library]
		model_idx = hashes.index(model_hash)
		return self.library[model_idx], model_idx

	def save_dataset(self, file_path: str):
		"""Saves dataset of all CNNs in the design space
		
		Args:
			file_path (str): file path to save dataset
		"""
		matrices_list = None
		labels_list = None
		hashes_list = None
		neighbors_list = None
		accuracies_list = None
		embeddings_list = [None for graph in self.library]

		if self.library:
			matrices_list = []
			labels_list = []
			for graph in self.library:
				matrices_list.append([matrix.tolist() for matrix, label in graph.graph])
				labels_list.append([label for matrix, label in graph.graph])
			hashes_list = [graph.hash for graph in self.library]
			neighbors_list = [graph.neighbors for graph in self.library]
			accuracies_list = [graph.accuracies for graph in self.library]

		if self.library and self.library[0].embedding is not None:
			embeddings_list = [graph.embedding.tolist() for graph in self.library]

		with open(file_path, 'w', encoding ='utf8') as json_file:
			json.dump({'config': self.config,
						'matrices': matrices_list,
						'labels': labels_list,
						'hashes': hashes_list,
						'neighbors': neighbors_list,
						'accuracies': accuracies_list,
						'embeddings': embeddings_list}, 
						json_file, ensure_ascii = True)

		print(f'{pu.bcolors.OKGREEN}Dataset saved to:{pu.bcolors.ENDC} {file_path}')

	@staticmethod
	def load_from_dataset(file_path: str) -> 'GraphLib':
		"""Summary
		
		Args:
			file_path (str): file path to load dataset
		
		Returns:
			GraphLib: a GraphLib object
		"""
		graphLib = GraphLib()

		with open(file_path, 'r', encoding ='utf8') as json_file:
			dataset_dict = json.load(json_file)

			graphLib.config = dataset_dict['config']

			if dataset_dict['matrices'] is None:
				pass
			else:
				for i in range(len(dataset_dict['matrices'])):
					model_graph = []
					for matrix, labels in zip(dataset_dict['matrices'][i], dataset_dict['labels'][i]):
						model_graph.append((np.array(matrix, dtype=np.int8), labels))
					graph_hash = dataset_dict['hashes'][i]
					embedding = np.array(dataset_dict['embeddings'][i])
					neighbors = dataset_dict['neighbors'][i]
					accuracies = dataset_dict['accuracies'][i]
					graph = Graph(model_graph, graph_hash, embedding, neighbors, accuracies)

					graphLib.library.append(graph)

		return graphLib


class Graph(object):
	"""Graph class to represent a computational graph in the design space
	
	Attributes:
		graph (list[tuple(np.ndarray, list[str])]): model graph as a list of tuples of adjacency 
			matrices and lists of operations
		hash (str): hash for current graph to check isomorphism
		model_params (int): number of model parameters after pytorch model has been initialized
		embedding (np.ndarray): embedding for every graph in the design space
		neighbors (list[str]): hashes of the nearest neighbors for this graph in order of
			nearest to farther neighbors
		accuraies (dict): value of accuracy for the given dataset in consideration as
			per the config file.
	"""
	def __init__(self, graph: list, graph_hash: str, embedding=None, neighbors=None, accuracies=None):
		"""Init a Graph instance from model_dict
		
		Args:
			graph (list[tuple(np.ndarray, list[str])]): model graph as a list of tuples of 
				adjacency matrices and lists of operations
			graph_hash (str): hash of the current graph
		"""
		# Initialize model graph
		self.graph = graph

		# Initialize model hash
		self.hash = graph_hash

		# Instantiate number of model parameters
		self.model_params = None

		# Initialize embedding
		self.embedding = embedding

		# Initialize the nearest neighboring graph
		self.neighbors = neighbors

		# Initialize accuracies for all datasets
		if accuracies is None:
			self.accuracies = {'train': None, 'val': None, 'test': None}
		else:
			self.accuracies = accuracies

	def __repr__(self):
		"""Representation of the Graph"""
		return f'{pu.bcolors.HEADER}Model parameters:{pu.bcolors.ENDC} {self.model_params}\n' \
			+ f'{pu.bcolors.HEADER}Accuracies:{pu.bcolors.ENDC} {self.accuracies}\n' \
			+ f'{pu.bcolors.HEADER}Embedding:{pu.bcolors.ENDC} {self.embedding}\n' \
			+ f'{pu.bcolors.HEADER}Hash:{pu.bcolors.ENDC} {self.hash}\n' \
			+ ''.join([f'{pu.bcolors.OKCYAN}Module:{pu.bcolors.ENDC}\n{matrix}\n' \
					+ f'{pu.bcolors.OKCYAN}Labels:{pu.bcolors.ENDC}{labels}\n' for matrix, labels in self.graph])


def generate_graphs(config, check_isomorphism = True):

	total_modules = 0	# Total number of modules (including isomorphisms)
	total_heads = 0 	# Total number of heads
	total_graphs = 0    # Total number of graphs (including isomorphisms)

	# hash --> (matrix, label) for the canonical graph associated with each hash
	module_buckets = {}
	head_buckets = {}
	graph_buckets = {}

	if not ALLOW_2_V and config['module_vertices'] < 3: 
		print(f'{pu.bcolors.FAIL}Check config file. "module_vertices" should be 3 or greater{pu.bcolors.ENDC}')
		sys.exit()

	if config['head_vertices'] < 4:
		print(f'{pu.bcolors.FAIL}Check config file. "head_vertices" should be 4 or greater{pu.bcolors.ENDC}')
		sys.exit()		

	print(f'{pu.bcolors.HEADER}Generating modules...{pu.bcolors.ENDC}')
	print(f"{pu.bcolors.HEADER}Using {config['module_vertices']} vertices, {len(config['base_ops'])} labels, " \
		+ f"max {config['max_edges']} edges{pu.bcolors.ENDC}")

	# Generate all possible martix-label pairs (or modules)
	for vertices in range(2 if ALLOW_2_V else 3, config['module_vertices'] + 1):
		for bits in range(2 ** (vertices * (vertices - 1) // 2)):
			# Construct adj matrix from bit string
			matrix = np.fromfunction(graph_util.gen_is_edge_fn(bits),
									 (vertices, vertices),
									 dtype=np.int8)

			# Discard any modules which can be pruned or exceed constraints
			if (not graph_util.is_full_dag(matrix) or
					graph_util.num_edges(matrix) > config['max_edges']):
				continue

			# Iterate through all possible labels
			for labels in itertools.product(*[list(config['base_ops'])
												for _ in range(vertices - 2)]):
				total_modules += 1
				labels = ['input'] + list(labels) + ['output']
				module_fingerprint = graph_util.hash_module(matrix, labels, config['hash_algo'])

				if module_fingerprint not in module_buckets:
					if vertices != 2 or ALLOW_2_V: module_buckets[module_fingerprint] = (matrix, labels)
				# Module-level isomorphism check -
				elif check_isomorphism:
					canonical_matrix = module_buckets[module_fingerprint]
					if not graph_util.compare_modules(
							(matrix, labels), canonical_matrix):
						print(f'{pu.bcolors.FAIL}Matrix:\n{matrix}\nLabels: {labels}\nis not isomorphic to' \
								+ f' canonical matrix:\n{canonical_matrix[0]}\nLabels: {canonical_matrix[1]}{pu.bcolors.ENDC}')
						sys.exit()

		print(f'\t{pu.bcolors.OKGREEN}Generated up to {vertices} vertices: {len(module_buckets)} modules ' \
			+ f'({total_modules} without hashing){pu.bcolors.ENDC}')


	print()
	print(f'{pu.bcolors.HEADER}Generating heads...{pu.bcolors.ENDC}')
	print(f"{pu.bcolors.HEADER}Using {config['head_vertices']} vertices, " \
		+ f"{len(config['flatten_ops']) + len(config['dense_ops'])} labels{pu.bcolors.ENDC}")

	for vertices in range(4, config['head_vertices'] + 1):
		for flatten_label in config['flatten_ops']:
			for dense_labels in itertools.product(*[list(config['dense_ops']) 
													for _ in range(vertices - 4)]):
				total_heads += 1

				# Create labels list
				labels = ['input'] + [flatten_label] + list(dense_labels) + ['dense_classes', 'output']

				# Construct adjacency matrix
				matrix = np.eye(vertices, k=1, dtype=np.int8)

				head_fingerprint = graph_util.hash_module(matrix, labels, config['hash_algo'])

				if head_fingerprint not in head_buckets:
					head_buckets[head_fingerprint] = (matrix, labels)
				elif check_isomorphism:
					canonical_matrix = head_buckets[head_fingerprint]
					if not graph_util.compare_modules(
							(matrix, labels), canonical_matrix):
						print(f'{pu.bcolors.FAIL}Matrix:\n{matrix}\nLabels: {labels}\nis not isomorphic to' \
								+ f' canonical matrix:\n{canonical_matrix[0]}\nLabels: {canonical_matrix[1]}{pu.bcolors.ENDC}')
						sys.exit()

		print(f'\t{pu.bcolors.OKGREEN}Generated up to {vertices} vertices: {len(head_buckets)} heads ' \
			+ f'({total_heads} without hashing){pu.bcolors.ENDC}')


	print()
	print(f'{pu.bcolors.HEADER}Generating graphs...{pu.bcolors.ENDC}')
	print(f"{pu.bcolors.HEADER}Using max {config['max_modules']} modules{pu.bcolors.ENDC}")

	# Generate graphs using modules and heads
	for modules in range(1, config['max_modules'] + 1):
		for module_fingerprints in product(*[module_buckets.keys()
										for _ in range(modules)], \
										desc=f'Generating CNNs with {modules} module(s)'): 
			for head_fingerprint in head_buckets.keys():
				modules_selected = [module_buckets[fingerprint] for fingerprint in module_fingerprints]
				merged_modules = graph_util.generate_merged_modules(modules_selected)

				# Add head
				modules_selected.append(head_buckets[head_fingerprint])
				merged_modules.append(head_buckets[head_fingerprint])

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
						print(f'{pu.bcolors.FAIL}Current graph:{pu.bcolors.ENDC}')
						count = 0
						for module in merged_modules:
							count += 1
							print(f'{pu.bcolors.FAIL}Module no. {count}:{pu.bcolors.ENDC}')
							print(f'{pu.bcolors.FAIL}Module Matrix: \n{np.array(module[0])}{pu.bcolors.ENDC}')
							print(f'{pu.bcolors.FAIL}Operations: {module[1]}{pu.bcolors.ENDC}')
						print(f'{pu.bcolors.FAIL}Graph in library:{pu.bcolors.ENDC}')
						count = 0
						for module in graph_buckets[graph_fingerprint]:
							count += 1
							print(f'{pu.bcolors.FAIL}Module no. {count}:{pu.bcolors.ENDC}')
							print(f'{pu.bcolors.FAIL}Module Matrix: \n{np.array(module[0])}{pu.bcolors.ENDC}')
							print(f'{pu.bcolors.FAIL}Operations: {module[1]}{pu.bcolors.ENDC}')

						sys.exit()

		print(f'\t{pu.bcolors.OKGREEN}Generated up to {modules} modules: {total_graphs} graphs{pu.bcolors.ENDC}')

	return graph_buckets