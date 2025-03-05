import random
import numpy as np
import scipy.sparse as sp
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


class GraphDataset(Dataset):
    def __init__(self, args, filter_labels=[]):
        data = np.load(args.data_path)
        self.data_x = data['x']
        self.x = (self.data_x - self.data_x.mean(0)) / self.data_x.std(0)
        self.y = data['y']
        self.edge_index = data['edge_index']
        self.edge_type = data['edge_type']
        self.train_mask = data['train_mask']
        self.test_mask = data['test_mask']
        self.edge_timestamp = data['edge_timestamp']

        # filter labels
        print(f"train_mask.shape: {self.train_mask.shape}")
        print(f"test_mask.shape: {self.test_mask.shape}")
        if filter_labels:
            mask = np.isin(self.y, filter_labels)
            self.train_mask = np.array([index for index in self.train_mask if mask[index]])
            self.test_mask = np.array([index for index in self.test_mask if mask[index]])
            num_true = np.count_nonzero(mask)
            num_false = mask.size - num_true

            print(f"True: {num_true}")
            print(f"False: {num_false}")
            print(f"train_mask.shape: {self.train_mask.shape}")
            print(f"test_mask.shape: {self.test_mask.shape}")

        self.sample_weight = np.array([20 if i == 1 else 1 for i in self.x.tolist()])

        self.train_index = np.where(self.y != -100)[0]
        self.length = len(self.x)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        index = item
        x = self.x[index]
        y = self.y[index]
        edge_start_index = np.where(self.edge_index[:, 0] == index)
        edge_start_index = sorted(edge_start_index, key=lambda e_s: self.edge_timestamp[e_s])
        back_x = self.x[self.edge_index[edge_start_index][:, :, 1]].squeeze(0)
        edge_end_index = np.where(self.edge_index[:, 1] == index)
        edge_end_index = sorted(edge_end_index, key=lambda e_s: self.edge_timestamp[e_s])
        front_x = self.x[self.edge_index[edge_end_index][:, :, 0]].squeeze(0)
        edge_start_type = self.edge_type[edge_start_index]
        edge_end_type = self.edge_type[edge_end_index]
        start_edge_timestamp = self.edge_timestamp[edge_start_index]
        end_edge_timestamp = self.edge_timestamp[edge_end_index]
        back_x_attention_mask = [0] * back_x.shape[0]
        front_x_attention_mask = [0] * front_x.shape[0]
        return {
            'x': x.tolist(),
            'y': y.tolist(),
            'start_edge_timestamp': start_edge_timestamp.squeeze(0).tolist(),
            'end_edge_timestamp': end_edge_timestamp.squeeze(0).tolist(),
            'edge_start_type': edge_start_type.squeeze(0).tolist(),
            'edge_end_type': edge_end_type.squeeze(0).tolist(),
            'back_x': back_x.tolist(),
            'front_x': front_x.tolist(),
            'back_x_attention_mask': back_x_attention_mask,
            'front_x_attention_mask': front_x_attention_mask
        }

    def split_data(self):
        all_indices = list(range(len(self.x)))
        train_size = int(len(self.x) * 5 / 6)

        train_indices = random.sample(all_indices, train_size)
        test_indices = list(set(all_indices) - set(train_indices))

        self.train_new_mask = np.array(train_indices)
        self.test_new_mask = np.array(test_indices)

        print("train_new_mask.shape:", self.train_new_mask.shape)
        print("test_new_mask.shape:", self.test_new_mask.shape)

    def merge_predictions_to_train_set(self, predictions):
        self.train_new_mask = np.copy(self.train_mask)
        self.test_new_mask = np.copy(self.test_mask)
        for index, label in predictions.items():
            if index not in self.train_mask:
                self.train_new_mask = np.append(self.train_new_mask, index)
                self.y[index] = label
                mask = np.isin(self.test_new_mask, self.train_new_mask)
                self.test_new_mask = self.test_new_mask[~mask]

    def build_sparse_adjacency_matrix(self, edge_index, num_nodes):
        """
        Builds a sparse adjacency matrix from edge indices.

        Args:
            edge_index (np.ndarray): A 2xN array where each column represents an edge (source, target).
            num_nodes (int): The total number of nodes in the graph.

        Returns:
            csr_matrix: A sparse adjacency matrix in Compressed Sparse Row (CSR) format.
        """
        print('build sparse adjacency matrix')
        row = edge_index[0, :]  # Source nodes of edges
        col = edge_index[1, :]  # Target nodes of edges
        data = np.ones_like(row)  # Edge weights (default to 1 for unweighted graphs)
        adjacency_matrix = csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))  # Create CSR matrix
        print(adjacency_matrix.shape)  # Print the shape of the adjacency matrix
        return adjacency_matrix

    def add_edges_with_knn(self, adjacency_matrix, k=5):
        """
        Adds edges to the adjacency matrix based on K-Nearest Neighbors (KNN) of node features.

        Args:
            adjacency_matrix (csr_matrix): The original sparse adjacency matrix.
            k (int): The number of nearest neighbors to consider.

        Returns:
            csr_matrix: An enhanced adjacency matrix with additional edges from KNN.
        """
        print('add edges with knn')
        # Use approximate nearest neighbors for faster computation
        from annoy import AnnoyIndex
        n_nodes = self.data_x.shape[0]
        tree = AnnoyIndex(self.data_x.shape[1], metric='euclidean')
        for i in range(n_nodes):
            tree.add_item(i, self.data_x[i])
        tree.build(10)  # Build 10 trees for approximate search

        # Prepare row and column indices for new edges
        new_edges = []
        for i in range(n_nodes):
            neighbors = tree.get_nns_by_item(i, k + 1)[1:]  # Exclude self
            new_edges.extend([(i, j) for j in neighbors])

        # Create a new sparse matrix for additional edges
        row_indices = [edge[0] for edge in new_edges]
        col_indices = [edge[1] for edge in new_edges]
        new_edges_matrix = csr_matrix((np.ones_like(row_indices), (row_indices, col_indices)), shape=(n_nodes, n_nodes))

        # Add new edges to the original adjacency matrix
        enhanced_adj_matrix = adjacency_matrix + new_edges_matrix
        enhanced_adj_matrix[enhanced_adj_matrix > 1] = 1  # Ensure binary adjacency matrix

        return enhanced_adj_matrix

    def label_propagation_async(self, adjacency_matrix, max_iter=100, adaptive_threshold=True):
        """
        Performs asynchronous label propagation on the graph.

        Args:
            adjacency_matrix (csr_matrix): The sparse adjacency matrix of the graph.
            max_iter (int): Maximum number of iterations for label propagation.
            adaptive_threshold (bool): Whether to use an adaptive confidence threshold.

        Updates:
            self.y: Updates labels for high-confidence unknown nodes.
            self.train_new_mask: Updates the training mask to include high-confidence nodes.
            self.test_new_mask: Updates the testing mask to exclude high-confidence nodes.
        """
        print('label propagation async')
        labels = np.copy(self.y)  # Copy the original labels
        unknown_mask = labels == -100  # Mask for nodes with unknown labels
        known_mask = ~unknown_mask  # Mask for nodes with known labels
        num_classes = len(np.unique(labels[known_mask]))  # Number of classes (supports multi-class)

        # Initialize probability matrix for label propagation
        probabilities = np.zeros((len(labels), num_classes), dtype=np.float8)
        for c in range(num_classes):
            probabilities[known_mask, c] = (labels[known_mask] == c).astype(float)

        # Normalize adjacency matrix for balanced propagation
        adjacency_matrix = normalize(adjacency_matrix, norm='l1', axis=1).tocsr()

        for _ in range(max_iter):
            # Asynchronous update: Update nodes in random order
            update_order = np.where(unknown_mask)[0]
            np.random.shuffle(update_order)  # Randomize update order to avoid bias
            for node in update_order:
                neighbors = adjacency_matrix[node].indices
                if len(neighbors) == 0:
                    continue
                # Weighted average of neighbor probabilities
                neighbor_probs = probabilities[neighbors].mean(axis=0)
                probabilities[node] = neighbor_probs

            # Dynamic threshold adjustment
            if adaptive_threshold:
                threshold = np.percentile(probabilities[unknown_mask].max(axis=1), 90)  # 90th percentile as threshold
            else:
                threshold = 0.95

            # Early stopping if most unknown nodes are confidently labeled
            max_probs = probabilities[unknown_mask].max(axis=1)
            if (max_probs > threshold).mean() > 0.9:  # Stop if >90% of unknown nodes are confident
                break

        # Update labels for high-confidence unknown nodes
        high_conf_mask = probabilities.max(axis=1) > threshold
        self.y[unknown_mask & high_conf_mask] = probabilities[unknown_mask & high_conf_mask].argmax(axis=1)

        # Update training and testing masks
        new_train_nodes = np.where(unknown_mask & high_conf_mask)[0]
        self.train_new_mask = np.concatenate([self.train_mask, new_train_nodes])
        self.test_new_mask = np.setdiff1d(self.test_mask, new_train_nodes)

        print("Updated train_mask and test_mask after label propagation.")