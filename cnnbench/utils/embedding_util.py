# Utility methods for generating embeddings

# Author : Shikhar Tuli

import numpy as np
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


def generate_mds_embeddings(dissimilarity_matrix, embedding_size: int, n_init=4, max_iter=1000, n_jobs=8):
    """Generate embeddings using Multi-Dimensional Scaling (SMACOF algorithm)
    
    Args:
        dissimilarity_matrix (np.ndarry): input dissimilarity_matrix
        embedding_size (int): size of the embedding
        n_init (int, optional): number of times the SMACOF algorithm will be run with 
        	different initializations. The final results will be the best output of 
        	the runs, determined by the run with the smallest final stress
        max_iter (int, optional): maximum number of iterations of the SMACOF algorithm 
        	for a single run.
        n_jobs (int, optional): number of parrallel jobs for joblib
    
    Returns:
        embeddings (np.ndarray): ndarray of embeddings of shape (len(dissimilarity_matrix), embedding_size)
    """
    #  Instantiate embedding function
    embedding_func = MDS(n_components=embedding_size, n_init=n_init, max_iter=max_iter, 
    	dissimilarity='precomputed', eps=1e-10, n_jobs=n_jobs)

    # Fit the embedding
    embeddings = embedding_func.fit_transform(dissimilarity_matrix)

    return embeddings


def generate_grad_embeddings(dissimilarity_matrix, embedding_size: int, epochs: int = 10, 
        batch_size: int = 1024, lr = 0.1, n_jobs = 8, silent: bool = False):
    """Generate embeddings using Gradient Descent on GPU
    
    Args:
        dissimilarity_matrix (np.ndarray): input dissimilarity matrix
        embedding_size (int): size of the embedding
        epochs (int, optional): number of epochs
        batch_size (int, optional): batch size for the number of pairs to consider
        lr (float, optional): learning rate
        n_jobs (int, optional): number of parallel jobs 
        silent (bool, optional): whether to suppress output
    
    Returns:
        embeddings (np.ndarray): ndarray of embeddings of shape (len(dissimilarity_matrix), embedding_size)
    """
    torch.manual_seed(0)
    
    # Create the model which learns graph embeddings
    class GraphEmbeddingModel(nn.Module):
        def __init__(self, num_graphs, embedding_size):
            super().__init__()
            self.embeddings = nn.Embedding(num_graphs, embedding_size)

        def forward(self, model_idx_pairs):
            embeddings_first = self.embeddings(model_idx_pairs[:, 0])
            embeddings_second = self.embeddings(model_idx_pairs[:, 1])
            distances = F.pairwise_distance(embeddings_first, embeddings_second)
            return distances

    # Create distance dataset
    class DistanceDataset(Dataset):
        def __init__(self, distance_matrix):
            super().__init__()
            self.num_pairs = int(0.5 * len(distance_matrix) * (len(distance_matrix) - 1))
            self.graph_pairs = torch.zeros([self.num_pairs, 2], dtype=torch.int64)
            self.labels = torch.zeros(self.num_pairs)

            count = 0
            for i in range(len(distance_matrix)):
                for j in range(i+1, len(distance_matrix)):
                    self.graph_pairs[count, :] = torch.Tensor([i, j])
                    self.labels[count] = distance_matrix[i, j]
                    count += 1 

        def __getitem__(self, pair_idx):
            return {'graph_pairs': self.graph_pairs[pair_idx, :], 
                    'labels': self.labels[pair_idx]}

        def __len__(self):
            return self.num_pairs

    # Create dataloader
    train_loader = DataLoader(DistanceDataset(dissimilarity_matrix), 
        batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=n_jobs)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Instantiate the model
    model = GraphEmbeddingModel(len(dissimilarity_matrix), embedding_size)
    model.to(device)
    model.train()

    # Instantiate the optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Instantiate the scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    losses = []
    report_interval = 1

    # Run training
    for epoch in tqdm(range(epochs), desc='Training embeddings'):
        if not silent: print(f'Epoch: {epoch}')

        for i, data_batch in enumerate(train_loader):
            graph_pairs = data_batch['graph_pairs'].to(device, non_blocking=True)
            labels = data_batch['labels'].to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(graph_pairs)
            loss = F.mse_loss(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step(epoch + i/len(train_loader))

            losses.append(loss.item())

            if not silent and i > 0 and i % report_interval == 0:
                print(f'\t[{i: 6d}/{len(train_loader): 6d}] Loss: {np.mean(losses[-report_interval:]): 0.6f}')

    embeddings = np.zeros((len(dissimilarity_matrix), embedding_size))

    model.to("cpu")
    with torch.no_grad():
        for i in range(len(dissimilarity_matrix)):
            embeddings[i, :] = model.embeddings.weight[i].numpy()

    return embeddings


def get_neighbors(embeddings, method: str, graph_list: list, neighbors: int):
    """Get neighbor indices for all graphs from embeddings
    
    Args:
        embeddings (np.ndarray): embeddings array from generate_embeddings() or
            generate_naive_embeddings()
        method (str): one in 'distance' or 'biased'
        graph_list (list): list of graphs in the library. Required if
            method = 'biased'
        neighbors (int): number of neighbors
    
    Returns:
        neighbor_idx (np.ndarray): neighbor indices according to the order in the
            embeddings array
    """
    if method == 'distance':
        # Generate distance matrix in the embeddings space
        distance_matrix = pairwise_distances(embeddings, embeddings, metric='euclidean')

        # Argmin should not return the same graph index
        np.fill_diagonal(distance_matrix, np.inf)

        # Find neighbors greedily
        neighbor_idx = np.argsort(distance_matrix, axis=1)[:, :neighbors]
    elif method == 'biased':
        # graph_list is required
        assert graph_list is not None, 'graph_list is required with method = "biased"'

        # Get biased overlap between graphs
        def _get_overlap(curr_graph, query_graph):
            overlap = 0
            for i in range(len(curr_graph)):
                # Basic tests
                if i >= len(query_graph): break
                if len(curr_graph[i][1]) != len(query_graph[i][1]): break

                # Check if modules are the same
                if (curr_graph[i][0] == query_graph[i][0]).all() and curr_graph[i][1] == query_graph[i][1]:
                    overlap += 1
                else:
                    break

            return overlap

        neighbor_idx = np.zeros((embeddings.shape[0], neighbors))
        ids = list(range(len(graph_list)))

        for i in range(len(graph_list)):
            # Sort indices based on overlap. Break ties with embedding distance
            neighbor_idx[i, :] = sorted(ids, key=lambda idx: (-1*_get_overlap(graph_list[i], graph_list[idx]), \
                float(pairwise_distances(embeddings[i, :].reshape(1, -1), embeddings[idx, :].reshape(1, -1)))))[1:neighbors+1]

    return neighbor_idx