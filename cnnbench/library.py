# Generate a library of all graphs up to structure and label isomorphism 
# and add embeddings and neighbours for every graph in the library.
	 
# Author : Shikhar Tuli


import os
import sys

import yaml
import numpy as np
import json

import itertools
from tqdm import tqdm
from tqdm.contrib.itertools import product
from scipy.stats import zscore
from copy import deepcopy
from six.moves import cPickle as pickle
import multiprocessing as mp
from functools import partial

from model_builder import CNNBenchModel
from utils import graph_util, embedding_util, print_util as pu


HASH_SIMPLE = True
ALLOW_2_V = False
SPEED_RUN = True
PARALLEL = True

CKPT_TEMP = '/scratch/gpfs/stuli/graphs_ckpt_temp.pkl'


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

	def build_library(self, modules_per_stack=1, check_isomorphism=True, create_graphs=True):
		"""Build the GraphLib library
		
		Args:
			modules_per_stack: number of modules in a stack
		    check_isomorphism (bool, optional): if True, isomorphism is checked 
		    	for every graph. If False, saves compute time
		    create_graphs (bool, optional): if True, graphs are created and added
		   		to the library 
		"""
		graph_buckets = generate_graphs(self.config, modules_per_stack=modules_per_stack,
			check_isomorphism=check_isomorphism, create_graphs=create_graphs)

		for graph_hash, graph in graph_buckets.items():
			graph = Graph(graph, graph_hash)
			try:
				model = CNNBenchModel(self.config, graph)

				# Only add graph to library if correponding CNN model works
				self.library.append(graph)
			except:
				pass

		self.modules_per_stack = modules_per_stack

		print(f'{pu.bcolors.OKGREEN}Graph library created!{pu.bcolors.ENDC} ' \
			+ f'\n{len(self.library)} graphs within the design space.')

	def get_interpolants(self,
	                     graph1: 'Graph',
	                     graph2: 'Graph',
	                     old_modules_per_stack: int,
	                     new_modules_per_stack: int,
	                     check_isomorphism=True):
	    """Interpolates between two neighbors with finer grained stacks

	        Args:
	            graph1 (Graph): first graph in the library
	            graph2 (Graph): second graph in the library
	            old_modules_per_stack (int): old modules per stack
	            new_modules_per_stack (int): new modules per stack
	            check_isomorphism (bool, optional): if True, isomorphism is checked 
	                for every graph

	        Returns:
	            interpolants (list): list of Graph objects between graph1 and graph2
	    """
	    assert new_modules_per_stack <= old_modules_per_stack and old_modules_per_stack % new_modules_per_stack == 0, \
	        'Old number of modules per stack should be divisible by new number of modules per stack'

	    interpolants = []

	    stack_mult = old_modules_per_stack // new_modules_per_stack

	    smaller_length = min(len(graph1.graph) - 1, len(graph2.graph) - 1)
	    larger_length = max(len(graph1.graph) - 1, len(graph2.graph) - 1)
	    different_lengths = smaller_length != larger_length

	    neighbor_config = deepcopy(self.config)

	    flatten_ops, dense_ops = [], []

	    flatten_ops.append(graph1.graph[-1][1][1])
	    dense_ops.extend(graph1.graph[-1][1][2:-1])

	    flatten_ops.append(graph2.graph[-1][1][1])
	    dense_ops.extend(graph2.graph[-1][1][2:-1])

	    neighbor_config['flatten_ops'] = list(set(flatten_ops))
	    neighbor_config['dense_ops'] = list(set(dense_ops))

	    graphs_stack = []

	    for stack in range(smaller_length//old_modules_per_stack):
	        base_ops = graph1.graph[stack * old_modules_per_stack][1][1:-1] \
	            + graph2.graph[stack * old_modules_per_stack][1][1:-1]
	        neighbor_config['base_ops'] = list(set(base_ops))

	        neighbor_config['max_modules'] = old_modules_per_stack

	        if stack == smaller_length//old_modules_per_stack - 1:
	            add_head = True
	        else:
	            add_head = False

	        graph_buckets = generate_graphs(neighbor_config, modules_per_stack=new_modules_per_stack,
	            check_isomorphism=check_isomorphism, create_graphs=True, add_head=add_head)

	        graphs_stack.append([Graph(graph, graph_hash) for graph_hash, graph in graph_buckets.items()])

	    for stacks in itertools.product(*graphs_stack):
	        graph = []
	        for stack in stacks:
	            graph.extend(stack.graph)

	        if HASH_SIMPLE:
	            graph_hash = graph_util.hash_graph_simple(graph, self.config['hash_algo'])
	        else:
	            graph_hash = graph_util.hash_graph(graph, self.config['hash_algo'])

	        interpolants.append(Graph(graph, graph_hash))
	        
	    if different_lengths:
	        larger_graph = graph1 if len(graph1.graph) - 1 == larger_length else graph2
	        larger_interpolants = []
	        
	        for smaller_graph in interpolants:
	            graph = smaller_graph.graph[:-1] + larger_graph.graph[smaller_length:]
	            
	            if HASH_SIMPLE:
	                graph_hash = graph_util.hash_graph_simple(graph, self.config['hash_algo'])
	            else:
	                graph_hash = graph_util.hash_graph(graph, self.config['hash_algo'])
	                
	            larger_interpolants.append(Graph(graph, graph_hash))
	    
	        interpolants.extend(larger_interpolants)

	    return interpolants

	def build_embeddings(self, embedding_size: int, 
						 algo='GD', 
						 kernel='GraphEditDistance', 
						 zscore_emb=True, 
						 nbr_method='biased', 
						 neighbors=100, 
						 n_jobs=8):
		"""Build the embeddings of all Graphs in GraphLib using MDS
		
		Args:
		    embedding_size (int): size of the embedding
		    algo (str): algorithm to use for generating embeddings. Can be any
		    	of the following:
		    		- 'GD'
		    		- 'MDS'
		    	The default value is 'GD'
		    kernel (str, optional): the kernel to be used for computing the dissimilarity 
		    	matrix. Can be any of the following:
		    		- 'GraphEditDistance'
		    		- 'WeisfeilerLehman'
		    		- 'NeighborhoodHash'
		    		- 'RandomWalkLabeled'
		    	The default value is 'GraphEditDistance'
		    zscore_emb (bool, optional): if True, embeddings are z-scored
		    nbr_method (str, optional): method to use for finding the neighbors. Can be
				any of the following:
					- 'biased'
					- 'distance'
				The default value is 'biased'
		    neighbors (int, optional): number of nearest neighbors to save for every graph
		    n_jobs (int, optional): number of parrallel jobs for joblib
		
		Raises:
		    NotImplementedError: Description
		"""
		print(f'{pu.bcolors.HEADER}Building embeddings for the Graph library...{pu.bcolors.ENDC}')

		# Create list of graphs (tuples of adjacency matrices and ops)
		graph_list = [self.library[i].graph for i in range(len(self))]

		# Generate dissimilarity_matrix using the specified kernel
		diss_mat = graph_util.generate_dissimilarity_matrix(graph_list, self.config, kernel=kernel, n_jobs=n_jobs)

		# Generate embeddings using MDS or GD
		if algo == 'MDS':
			embeddings = embedding_util.generate_mds_embeddings(diss_mat, embedding_size=embedding_size, n_jobs=n_jobs)
		elif algo == 'GD':
			embeddings = embedding_util.generate_grad_embeddings(diss_mat, embedding_size=embedding_size, n_jobs=n_jobs, 
				silent=True)
		else:
			raise NotImplementedError(f'Embedding algorithm: {algo} is not supported')

		if zscore_emb:
			# Z-score embeddings for efficient training on a neural network
			embeddings = zscore(embeddings, axis=0)

		# Get neighboring graph in the embedding space, for all Graphs
		neighbor_idx = embedding_util.get_neighbors(embeddings, method=nbr_method, graph_list=graph_list, neighbors=neighbors)

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


def generate_graphs(config, modules_per_stack=1, check_isomorphism=True, create_graphs=True, add_head=True):

	# Code built upon https://github.com/google-research/nasbench/blob/
	# master/nasbench/scripts/generate_graphs.py
	
	assert config['max_modules'] % modules_per_stack == 0, "'max_modules' in config should be divisible by 'modules_per_stack'"

	total_modules = 0	# Total number of modules (including isomorphisms)
	total_heads = 0 	# Total number of heads
	total_graphs = 0    # Total number of graphs (including isomorphisms)

	# hash --> (matrix, label) for the canonical graph associated with each hash
	module_buckets = {}
	head_buckets = {}
	graph_buckets = {}

	if os.path.exists(CKPT_TEMP):
		ckpt = pickle.load(open(CKPT_TEMP, 'rb'))
		total_modules = ckpt['total_modules']
		total_heads = ckpt['total_heads']
		total_graphs = ckpt['total_graphs']
		module_vertices_done = ckpt['module_vertices_done']
		head_vertices_done = ckpt['head_vertices_done']
		stacks_done = ckpt['stacks_done']
		module_buckets = ckpt['module_buckets']
		head_buckets = ckpt['head_buckets']
		graph_buckets = ckpt['graph_buckets']

		print(f'{pu.bcolors.OKGREEN}Loaded checkpoint with:{pu.bcolors.ENDC}' \
			+ f'\n\t{len(module_buckets)} modules, {len(head_buckets)} heads and {len(graph_buckets)} graphs' \
			+ f'\n\t{module_vertices_done} module vertices, {head_vertices_done} head vertices and {stacks_done} stacks are done\n')
	else:
		module_vertices_done, head_vertices_done, stacks_done = 0, 0, 0
		pickle.dump({'total_modules': total_modules, 'total_heads': total_heads, 'total_graphs': total_graphs,
			'module_vertices_done': module_vertices_done, 'head_vertices_done': head_vertices_done, 'stacks_done': stacks_done,
			'module_buckets': module_buckets, 'head_buckets': head_buckets, 'graph_buckets': graph_buckets},
			open(CKPT_TEMP, 'wb+'), pickle.HIGHEST_PROTOCOL)

	if not ALLOW_2_V and config['module_vertices'] < 3: 
		print(f'{pu.bcolors.FAIL}Check config file. "module_vertices" should be 3 or greater{pu.bcolors.ENDC}')
		sys.exit()

	if config['head_vertices'] < 4:
		print(f'{pu.bcolors.FAIL}Check config file. "head_vertices" should be 4 or greater{pu.bcolors.ENDC}')
		sys.exit()	

	if isinstance(config['max_edges'], int):
		max_edges = config['max_edges'] 
		extra_edges = 0
	else:
		max_edges = 0
		extra_edges = int(config['max_edges'].split('+')[-1])

	print(f'{pu.bcolors.HEADER}Generating modules...{pu.bcolors.ENDC}')
	print(f"{pu.bcolors.HEADER}Using {config['module_vertices']} vertices and {len(config['base_ops'])} labels{pu.bcolors.ENDC}")

	if PARALLEL:
		pool = mp.Pool(32)

	# Generate all possible martix-label pairs (or modules)
	for vertices in range(2 if ALLOW_2_V else 3, config['module_vertices'] + 1):
		if vertices <= module_vertices_done: continue

		if SPEED_RUN and PARALLEL:
			modules = pool.map(
				partial(_get_modules, vertices=vertices, max_edges=max_edges, extra_edges=extra_edges, config=config), 
				tqdm(range(2 ** (vertices * (vertices - 1) // 2)), desc=f'Generating modules with {vertices} vertices'))

			# Flatten modules list
			modules = [item for sublist in modules for item in sublist]
			total_modules += len(modules)

			for module_fingerprint, module in modules:
				module_buckets[module_fingerprint] = module
		else:
			for bits in tqdm(range(2 ** (vertices * (vertices - 1) // 2)), desc=f'Generating modules with {vertices} vertices'):
				# Construct adj matrix from bit string
				matrix = np.fromfunction(graph_util.gen_is_edge_fn(bits),
										 (vertices, vertices),
										 dtype=np.int8)

				# Discard any modules which can be pruned or exceed constraints
				if max_edges == 0:
					edges_limit = vertices + extra_edges
				else:
					edges_limit = max_edges 
				if (not graph_util.is_full_dag(matrix) or
						graph_util.num_edges(matrix) > edges_limit):
					continue

				# Iterate through all possible labels
				for labels in itertools.product(*[list(config['base_ops'])
													for _ in range(vertices - 2)]):
					total_modules += 1
					labels = ['input'] + list(labels) + ['output']
					module_fingerprint = graph_util.hash_module(matrix, labels, config['hash_algo'])

					if SPEED_RUN:
						# Skip checking if module already in buckets. Overwrite old module if hash matches
						module_buckets[module_fingerprint] = (matrix, labels)
						continue

					if module_fingerprint not in module_buckets:
						module_buckets[module_fingerprint] = (matrix, labels)
					# Module-level isomorphism check -
					elif check_isomorphism:
						canonical_matrix = module_buckets[module_fingerprint]
						if not graph_util.compare_modules(
								(matrix, labels), canonical_matrix):
							print(f'{pu.bcolors.FAIL}Matrix:\n{matrix}\nLabels: {labels}\nis not isomorphic to' \
									+ f' canonical matrix:\n{canonical_matrix[0]}\nLabels: {canonical_matrix[1]}{pu.bcolors.ENDC}')
							sys.exit()

		print(f'\t{pu.bcolors.OKGREEN}Generated up to {vertices} vertices: {len(module_buckets)} module(s) ' \
			+ f'({total_modules} without hashing){pu.bcolors.ENDC}')

		module_vertices_done = vertices
		pickle.dump({'total_modules': total_modules, 'total_heads': total_heads, 'total_graphs': total_graphs,
			'module_vertices_done': module_vertices_done, 'head_vertices_done': head_vertices_done, 'stacks_done': stacks_done,
			'module_buckets': module_buckets, 'head_buckets': head_buckets, 'graph_buckets': graph_buckets},
			open(CKPT_TEMP, 'wb+'), pickle.HIGHEST_PROTOCOL)


	print()
	print(f'{pu.bcolors.HEADER}Generating heads...{pu.bcolors.ENDC}')
	print(f"{pu.bcolors.HEADER}Using {config['head_vertices']} vertices, " \
		+ f"{len(config['flatten_ops']) + len(config['dense_ops'])} labels{pu.bcolors.ENDC}")

	for vertices in range(4, config['head_vertices'] + 1):
		if vertices <= head_vertices_done: continue
		for flatten_label in config['flatten_ops']:
			for dense_labels in itertools.product(*[list(config['dense_ops']) 
													for _ in range(vertices - 4)]):
				total_heads += 1

				# Create labels list
				labels = ['input'] + [flatten_label] + list(dense_labels) + ['dense_classes', 'output']

				# Construct adjacency matrix
				matrix = np.eye(vertices, k=1, dtype=np.int8)

				head_fingerprint = graph_util.hash_module(matrix, labels, config['hash_algo'])

				if SPEED_RUN:
					head_buckets[head_fingerprint] = (matrix, labels)
					continue

				if head_fingerprint not in head_buckets:
					head_buckets[head_fingerprint] = (matrix, labels)
				elif check_isomorphism:
					canonical_matrix = head_buckets[head_fingerprint]
					if not graph_util.compare_modules(
							(matrix, labels), canonical_matrix):
						print(f'{pu.bcolors.FAIL}Matrix:\n{matrix}\nLabels: {labels}\nis not isomorphic to' \
								+ f' canonical matrix:\n{canonical_matrix[0]}\nLabels: {canonical_matrix[1]}{pu.bcolors.ENDC}')
						sys.exit()

		print(f'\t{pu.bcolors.OKGREEN}Generated up to {vertices} vertices: {len(head_buckets)} head(s) ' \
			+ f'({total_heads} without hashing){pu.bcolors.ENDC}')

		head_vertices_done = vertices
		pickle.dump({'total_modules': total_modules, 'total_heads': total_heads, 'total_graphs': total_graphs,
			'module_vertices_done': module_vertices_done, 'head_vertices_done': head_vertices_done, 'stacks_done': stacks_done,
			'module_buckets': module_buckets, 'head_buckets': head_buckets, 'graph_buckets': graph_buckets},
			open(CKPT_TEMP, 'wb+'), pickle.HIGHEST_PROTOCOL)


	print()
	print(f'{pu.bcolors.HEADER}Generating graphs...{pu.bcolors.ENDC}')
	print(f"{pu.bcolors.HEADER}Using max {config['max_modules']} modules with " \
		+ f"{modules_per_stack} module(s) per stack{pu.bcolors.ENDC}")

	# Generate graphs using modules and heads
	for stacks in range(1, config['max_modules']//modules_per_stack + 1):
		if stacks <= stacks_done: continue
		for module_fingerprints in product(*[module_buckets.keys()
										for _ in range(stacks)], \
										desc=f'Generating CNNs with {stacks} stack(s)'): 
			head_added = False
			for head_fingerprint in head_buckets.keys():
				if not create_graphs:
					total_graphs += 1
					continue

				if head_added:
					continue

				modules_selected = _get_stack([module_buckets[fingerprint] for fingerprint in module_fingerprints],
					repeat=modules_per_stack)
				merged_modules = graph_util.generate_merged_modules(modules_selected)

				if add_head:
					# Add head
					modules_selected.append(head_buckets[head_fingerprint])
					merged_modules.append(head_buckets[head_fingerprint])
					head_added = True

				if HASH_SIMPLE:
					graph_fingerprint = graph_util.hash_graph_simple(modules_selected, config['hash_algo'])
				else:
					graph_fingerprint = graph_util.hash_graph(modules_selected, config['hash_algo'])

				if SPEED_RUN:
					total_graphs += 1
					if HASH_SIMPLE:
						graph_buckets[graph_fingerprint] = modules_selected
					else:
						graph_buckets[graph_fingerprint] = merged_modules
					continue

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

		print(f'\t{pu.bcolors.OKGREEN}Generated up to {stacks} stack(s) ({modules_per_stack * stacks} module(s)): '\
			+ f'{total_graphs} graphs{pu.bcolors.ENDC}')

		stacks_done = stacks
		pickle.dump({'total_modules': total_modules, 'total_heads': total_heads, 'total_graphs': total_graphs,
			'module_vertices_done': module_vertices_done, 'head_vertices_done': head_vertices_done, 'stacks_done': stacks_done,
			'module_buckets': module_buckets, 'head_buckets': head_buckets, 'graph_buckets': graph_buckets},
			open(CKPT_TEMP, 'wb+'), pickle.HIGHEST_PROTOCOL)


	return graph_buckets


def _get_stack(lst: list, repeat: int):
	return list(itertools.chain.from_iterable(itertools.repeat(x, repeat) for x in lst))

def _get_modules(bits, vertices, max_edges, extra_edges, config):
	modules_list = []

	# Construct adj matrix from bit string
	matrix = np.fromfunction(graph_util.gen_is_edge_fn(bits),
							 (vertices, vertices),
							 dtype=np.int8)

	# Discard any modules which can be pruned or exceed constraints
	if max_edges == 0:
		edges_limit = vertices + extra_edges
	else:
		edges_limit = max_edges 
	if (not graph_util.is_full_dag(matrix) or graph_util.num_edges(matrix) > edges_limit):
		return []

	# Iterate through all possible labels
	for labels in itertools.product(*[list(config['base_ops'])
										for _ in range(vertices - 2)]):
		labels = ['input'] + list(labels) + ['output']
		module_fingerprint = graph_util.hash_module(matrix, labels, config['hash_algo'])

		modules_list.append((module_fingerprint, (matrix, labels)))

	return modules_list