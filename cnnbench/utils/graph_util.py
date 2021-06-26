# Utility functions for graph generation and distance computation.

# Author : Shikhar Tuli


import hashlib
import itertools

import numpy as np
import networkx as nx
import re

from tqdm.notebook import tqdm
from itertools import combinations
from joblib import Parallel, delayed

from utils import print_util as pu


def gen_is_edge_fn(bits):
    """Generate a boolean function for the edge connectivity.

    Given a bitstring FEDCBA and a 4x4 matrix, the generated matrix is
    [[0, A, B, D],
     [0, 0, C, E],
     [0, 0, 0, F],
     [0, 0, 0, 0]]

    Note that this function is agnostic to the actual matrix dimension due to
    order in which elements are filled out (column-major, starting from least
    significant bit). For example, the same FEDCBA bitstring (0-padded) on a 5x5
    matrix is
    [[0, A, B, D, 0],
     [0, 0, C, E, 0],
     [0, 0, 0, F, 0],
     [0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0]]

    Args:
    bits: integer which will be interpreted as a bit mask.

    Returns:
    vectorized function that returns True when an edge is present.
    """
    def is_edge(x, y):
        """Is there an edge from x to y (0-indexed)?"""
        if x >= y:
          return 0
        # Map x, y to index into bit string
        index = x + (y * (y - 1) // 2)
        return (bits >> index) % 2 == 1

    return np.vectorize(is_edge)


def is_full_dag(matrix):
    """Full DAG == all vertices on a path from vert 0 to (V-1).

    i.e. no disconnected or "hanging" vertices.

    It is sufficient to check for:
    1) no rows of 0 except for row V-1 (only output vertex has no out-edges)
    2) no cols of 0 except for col 0 (only input vertex has no in-edges)

    Args:
    matrix: V x V upper-triangular adjacency matrix

    Returns:
    True if the there are no dangling vertices.
    """
    shape = np.shape(matrix)

    rows = matrix[:shape[0]-1, :] == 0
    rows = np.all(rows, axis=1)     # Any row with all 0 will be True
    rows_bad = np.any(rows)

    cols = matrix[:, 1:] == 0
    cols = np.all(cols, axis=0)     # Any col with all 0 will be True
    cols_bad = np.any(cols)

    return (not rows_bad) and (not cols_bad)


def num_edges(matrix):
    """Computes number of edges in adjacency matrix."""
    return np.sum(matrix)


def hash_module(matrix, labeling, algo='md5'):
    """Computes a module-invariance hash of the matrix and label pair.

    Args:
    matrix: np.ndarray square upper-triangular adjacency matrix.
    labeling: list of int labels of length equal to both dimensions of
      matrix.
    algo: hash algorithm among ["md5", "sha256", "sha512"]

    Returns:
        Hash of the matrix and labeling based on algo.
    """
    vertices = np.shape(matrix)[0]
    in_edges = np.sum(matrix, axis=0).tolist()
    out_edges = np.sum(matrix, axis=1).tolist()

    assert len(in_edges) == len(out_edges) == len(labeling)
    hashes = list(zip(out_edges, in_edges, labeling))
    hashes = [hash_func(str(h).encode('utf-8'), algo).hexdigest() for h in hashes]

    # Computing this up to the diameter is probably sufficient but since the
    # operation is fast, it is okay to repeat more times.
    for _ in range(vertices):
        new_hashes = []
        for v in range(vertices):
            in_neighbors = [hashes[w] for w in range(vertices) if matrix[w, v]]
            out_neighbors = [hashes[w] for w in range(vertices) if matrix[v, w]]
            new_hashes.append(hash_func(
                (''.join(sorted(in_neighbors)) + '|' +
                ''.join(sorted(out_neighbors)) + '|' +
                hashes[v]).encode('utf-8'), algo).hexdigest())
        hashes = new_hashes
    fingerprint = hash_func(str(sorted(hashes)).encode('utf-8'), algo).hexdigest()

    return fingerprint


def hash_graph_simple(modules, algo='md5'):
    """Computes a simple hash of the graph using the modules, a list of (matrix, label) pairs.
    This method may lead to different hashes for the same graph. Use hash_graph() instead

    Args:
    modules:list of tuples - (matrix, label) pairs
    algo: hash algorithm among ["md5", "sha256", "sha512"]

    Returns:
    Hash of the graph based on algo.
    """
    module_hashes = []
    for module in modules:
        module_hashes.append(hash_module(np.array(module[0]), module[1], algo))

    graph_fingerprint = hash_func('|'.join(module_hashes).encode('utf-8'), algo).hexdigest()

    return graph_fingerprint
    

def hash_graph(modules, algo='md5'):
    """Computes a hash of the graph using the modules, a list of (matrix, label) pairs.

    Args:
    modules:list of tuples - (matrix, label) pairs
    algo: hash algorithm among ["md5", "sha256", "sha512"]

    Returns:
    Hash of the graph based on algo.
    """     
    merged_modules = generate_merged_modules(modules)
      
    module_hashes = []
    for module in merged_modules:
        assert is_full_dag(np.array(module[0]))
        module_hashes.append(hash_module(np.array(module[0]), module[1], algo))

    graph_fingerprint = hash_func('|'.join(module_hashes).encode('utf-8'), algo).hexdigest()

    return graph_fingerprint


def generate_merged_modules(modules):
    """Merge modules if they are mergeable"""
    merged_modules = [modules[0]]

    for i in range(1, len(modules)):
        if mergeable(merged_modules[-1], modules[i]):
            merged_modules[-1] = merge_modules(merged_modules[-1], modules[i])
        else:
            merged_modules.append(modules[i])

    return merged_modules


def hash_func(str, algo='md5'):
    """Outputs has based on algorithm defined."""
    return eval(f"hashlib.{algo}(str)")


def mergeable(module1, module2):
    """Checks if modules can be merged."""
    return np.sum(np.array(module1[0])[:, -1]) == 1 and np.sum(np.array(module2[0])[:, -1]) == 1


def merge_modules(module1, module2):
    """Merges two modules when output vertex of one and input vertex of second do not have more than 
    one neighbor"""
    size_1 = np.shape(np.array(module1[0]))[0]
    size_2 = np.shape(np.array(module2[0]))[0]
    size_new = size_1 + size_2 - 2
    new_matrix = np.zeros((size_new, size_new))
    new_matrix[:size_1, :size_1] = np.array(module1[0])

    for i, j in itertools.product(range(size_2), range(size_2)):
        new_matrix[i+size_1-2, j+size_1-2] = np.array(module2[0])[i, j] 

    new_module = (new_matrix.astype(np.int8), module1[1][:-1] + module2[1][1:])

    return new_module


def compare_graphs(graph1, graph2):
    """Exhaustively checks if two graphs are equal."""
    merged_modules1 = generate_merged_modules(graph1)
    merged_modules2 = generate_merged_modules(graph2)

    if len(merged_modules1) != len(merged_modules2): return False

    flag = False
    for i in range(len(merged_modules1)):
        if not np.all(np.array(merged_modules1[i][0]).shape == np.array(merged_modules2[i][0]).shape) \
            or not np.all(np.array(merged_modules1[i][1]).shape == np.array(merged_modules2[i][1]).shape):
            return False
        elif np.all(np.array(merged_modules1[i][0]) == np.array(merged_modules2[i][0])) \
            and np.all(np.array(merged_modules1[i][1]) == np.array(merged_modules2[i][1])):
            flag = True
        else:
            return False

    return flag


def permute_module(matrix, label, permutation):
    """Permutes the matrix and labels based on permutation.

    Args:
    matrix: np.ndarray adjacency matrix.
    label: list of labels of same length as matrix dimensions.
    permutation: a permutation list of ints of same length as matrix dimensions.

    Returns:
    np.ndarray where vertex permutation[v] is vertex v from the original matrix
    """
    # vertex permutation[v] in new matrix is vertex v in the old matrix
    forward_perm = zip(permutation, list(range(len(permutation))))
    inverse_perm = [x[1] for x in sorted(forward_perm)]
    edge_fn = lambda x, y: matrix[inverse_perm[x], inverse_perm[y]] == 1

    new_matrix = np.fromfunction(np.vectorize(edge_fn),
                                   (len(label), len(label)),
                                   dtype=np.int8)
    new_label = [label[inverse_perm[i]] for i in range(len(label))]

    return new_matrix, new_label


def compare_modules(matrix1, matrix2):
    """Exhaustively checks if 2 modules are isomorphic."""
    matrix1, label1 = np.array(matrix1[0]), matrix1[1]
    matrix2, label2 = np.array(matrix2[0]), matrix2[1]
    assert np.shape(matrix1) == np.shape(matrix2)
    assert len(label1) == len(label2)

    vertices = np.shape(matrix1)[0]
    # Note: input and output in our constrained matrices always map to themselves
    # but this script does not enforce that.
    for perm in itertools.permutations(range(0, vertices)):
        pmatrix1, plabel1 = permute_module(matrix1, label1, perm)
        if np.array_equal(pmatrix1, matrix2) and plabel1 == label2:
            return True

    return False


def generate_dissimilarity_matrix(graph_list: list, config: dict, kernel='GraphEditDistance', n_jobs=8, approx=1):
    """Generate the dissimilarity matrix which is N x N, for N graphs 
    in the design space
    
    Args:
        graph_list (list): list of graphs, which are lists of
            tuples of adjacency matrix and ops
        config (dict): the configuration dictionary with the allowed operations
        kernel (str, optional): the kernel to be used for computing the dissimilarity matrix. The
            default value is 'GraphEditDistance'
        n_jobs (int, optional): number of parrallel jobs for joblib
        approx (int, optional): number of approximations to be implemented. Used when kernel = 'GraphEditDistance'.
    
    Returns:
        dissimilarity_matrix (np.ndarray): dissimilarity matrix
    
    Raises:
        NotImplementedError: if kernel provided is not implemented
    """
    if kernel != 'GraphEditDistance':
        raise NotImplementedError('Kernels other than GraphEditDistance are not implemented yet')

    def get_nx_graph_list(graph_list):
        nx_graph_list = []

        # Convert modules to one adjacency matrix, labels pair
        for graph in graph_list:
            total_ops = sum(len(module[1]) for module in graph)

            matrix = np.zeros((total_ops, total_ops))
            labels = []
            op_idx = 0
            
            for module in graph:
                labels.extend(module[1])
                num_ops = len(module[1])

                matrix[op_idx:(op_idx+num_ops), op_idx:(op_idx+num_ops)] = module[0]
                
                try:
                    # A connection from output of previous module to input of next module
                    matrix[(op_idx+num_ops-1), (op_idx+num_ops)] = 1
                except:
                    pass
                
                op_idx += num_ops
            
            # Create networkx graph
            nx_graph = nx.DiGraph(matrix)
            nx.set_node_attributes(nx_graph, {i:label for i, label in enumerate(labels)}, 'label')
            nx_graph_list.append(nx_graph)

        return nx_graph_list

    def get_ops_weights(config):
        input_channels = config['default_channels']

        ops_list = []
        ops_weights = []

        ops_list.extend(['input', 'output'])
        ops_list.extend(config['base_ops'])
        ops_list.extend(config['flatten_ops'])
        ops_list.extend(config['dense_ops'])
        ops_list.extend(['dense_classes'])

        input_channels = config['default_channels']
    
        for op in ops_list:
            if op == 'input': 
                ops_weights.append(1)
            elif op == 'output': 
                ops_weights.append(5)
            elif op.startswith('conv'):
                kernel_size = re.search('([0-9]+)x([0-9]+)', op)
                assert kernel_size is not None
                kernel_size = kernel_size.group(0).split('x')
                
                output_channels = re.search('-c([0-9]+)', op)
                output_channels = config['default_channels'] if output_channels is None \
                    else int(channels_conv.group(0)[2:])
                groups = re.search('-g([0-9]+)', op)
                groups = 1 if groups is None else int(groups.group(0)[2:])
                
                # Group correction
                while output_channels % groups != 0 or input_channels % groups != 0:
                    groups -= 1
                    
                ops_weights.append(input_channels * output_channels * int(kernel_size[0]) * int(kernel_size[1]) // groups)
            elif 'pool' in op:
                if 'max' in op: 
                    ops_weights.append(1) 
                elif 'avg' in op:
                     # avg-pool takes slightly more compute
                    if 'global' in op:
                        ops_weights.append(10)
                    else:
                        ops_weights.append(5)
            elif op == 'flatten':
                ops_weights.append(1)
            elif op == 'channel_suffle':
                ops_weights.append(5)
            elif op == 'upsample':
                ops_weights.append(5)
            elif op.startswith('dense'):
                size = re.search('([0-9]+)', op)
                size = config['classes'] if size is None else int(size.group(0))
                ops_weights.append(size)
            elif op.startswith('dropout'):
                ops_weights.append(1)
            else:
                print(f'Provided operation: {op} in the given configuration is not interpretable')
                sys.exit(1)
                
        return ops_list, ops_weights
        

    def get_ged(i, j, dissimilarity_matrix, nx_graph_list, ops_list, ops_weights, dist_weight=0.1, approx=approx):

        def node_subst_cost(node1, node2):
            node1_idx, node2_idx = ops_list.index(node1['label']), ops_list.index(node2['label'])
            return (1 + dist_weight * abs(node1_idx - node2_idx)) * \
                abs(ops_weights[node1_idx] - ops_weights[node2_idx])
            
        def node_cost(node):
            node_idx = ops_list.index(node['label'])
            return ops_weights[node_idx]

        def edge_cost(edge):
            return 0.1

        if approx == 0:
            dissimilarity_matrix[i, j] = nx.graph_edit_distance(nx_graph_list[i], nx_graph_list[j],
                                                                node_subst_cost=node_subst_cost, 
                                                                node_del_cost=node_cost, 
                                                                node_ins_cost=node_cost, 
                                                                edge_del_cost=edge_cost,
                                                                edge_ins_cost=edge_cost,
                                                                timeout=10)
        else:
            count = 0
            approx_dist = 0
            for dist in nx.optimize_graph_edit_distance(nx_graph_list[i], nx_graph_list[j],
                                                        node_subst_cost=node_subst_cost, 
                                                        node_del_cost=node_cost, 
                                                        node_ins_cost=node_cost, 
                                                        edge_del_cost=edge_cost,
                                                        edge_ins_cost=edge_cost):
                approx_dist = dist
                count += 1
                if count == approx: break

            dissimilarity_matrix[i, j] = approx_dist

    ops_list, ops_weights = get_ops_weights(config)  

    print(f'{pu.bcolors.OKGREEN}Generated operation weights{pu.bcolors.ENDC}')

    nx_graph_list = get_nx_graph_list(graph_list)

    print(f'{pu.bcolors.OKGREEN}Generated networkx graphs{pu.bcolors.ENDC}')

    dissimilarity_matrix = np.zeros((len(graph_list), len(graph_list)))

    Parallel(n_jobs=n_jobs, prefer='threads', require='sharedmem')(
        delayed(get_ged)(i, j, dissimilarity_matrix, nx_graph_list, ops_list, ops_weights) \
            for i, j in tqdm(list(combinations(range(len(graph_list)), 2)), desc='Generating dissimilarity matrix'))

    dissimilarity_matrix = dissimilarity_matrix + np.transpose(dissimilarity_matrix)

    return dissimilarity_matrix
