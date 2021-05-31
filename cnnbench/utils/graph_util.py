# Utility functions for graph generation

# Author : Shikhar Tuli


import hashlib
import itertools

import numpy as np


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

    new_module = (new_matrix.astype(int).tolist(), module1[1][:-1] + module2[1][1:])

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
