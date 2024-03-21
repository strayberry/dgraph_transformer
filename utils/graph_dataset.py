import random
import numpy as np
import scipy.sparse as sp
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.neighbors import NearestNeighbors


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
        print('build sparse adjacency matrix')
        row = edge_index[0, :]
        col = edge_index[1, :]
        data = np.ones_like(row)
        adjacency_matrix = csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
        print(adjacency_matrix.shape)
        return adjacency_matrix

    def add_edges_with_knn(self, adjacency_matrix, k=5):
        print('add edges with knn')
        # Use node features to compute K-nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(self.data_x)
        distances, indices = nbrs.kneighbors(self.data_x)

        # Get the number of nodes
        num_nodes = self.data_x.shape[0]

        # Prepare the row and column indices for the new edges
        # We use array broadcasting to create a grid of indices
        row_indices = np.repeat(np.arange(num_nodes), k)
        col_indices = indices[:, 1:].flatten()  # Skip the first column which is the node itself

        # Create a new sparse matrix with additional edges
        # Use 1 to indicate the presence of an edge
        new_edges = csr_matrix((np.ones_like(row_indices), (row_indices, col_indices)), shape=(num_nodes, num_nodes))

        # Add the new edges to the original adjacency matrix
        # Since we're dealing with unweighted graphs, we can use binary addition
        enhanced_adj_matrix = adjacency_matrix + new_edges

        # Ensure the matrix is binary since KNN might introduce duplicate edges
        enhanced_adj_matrix[enhanced_adj_matrix > 1] = 1

        return enhanced_adj_matrix

    def label_propagation_async(self, adjacency_matrix, max_iter=100, threshold=0.95):
        print('label propagation async')
        num_classes = 2  # 已知的类别数，这里是0和1
        labels = np.copy(self.y)

        # 定义未知和已知掩码
        unknown_mask = labels == -100
        print(unknown_mask)
        known_mask = ~unknown_mask
        print(known_mask)

        # Initialize probabilities matrix
        probabilities = np.zeros((len(labels), num_classes), dtype=float)
        probabilities[known_mask, 0] = (labels[known_mask] == 0).astype(float)
        probabilities[known_mask, 1] = (labels[known_mask] == 1).astype(float)
        print(probabilities.shape)
        print(probabilities)

        # Ensure adjacency_matrix is in CSR format for efficiency
        adjacency_matrix = adjacency_matrix.tocsr()

        print('update neighbor_probs')
        for _ in range(max_iter):
            # Use matrix multiplication to spread probabilities
            neighbor_probs = adjacency_matrix.dot(probabilities)

            # Normalize probabilities for unknown nodes
            row_sums = neighbor_probs[unknown_mask].sum(axis=1).reshape(-1, 1)
            row_sums[row_sums == 0] = 1
            neighbor_probs[unknown_mask] /= row_sums

            # Update probabilities for unknown nodes
            probabilities[unknown_mask] = neighbor_probs[unknown_mask]
            #print(probabilities[unknown_mask])

        # 计算每个未知标签节点的最大概率及其对应的标签
        max_probs = probabilities[unknown_mask].max(axis=1)
        predicted_labels = probabilities[unknown_mask].argmax(axis=1)

        # 确定高置信度节点
        high_confidence_mask = max_probs > threshold
        high_confidence_indices = np.where(unknown_mask)[0][high_confidence_mask]

        # 更新高置信度未知节点的标签
        self.y[high_confidence_indices] = predicted_labels[high_confidence_mask]

        # 根据更新后的标签，重新定义train_new_mask和test_new_mask
        self.train_new_mask = np.concatenate([self.train_mask, high_confidence_indices])
        self.test_new_mask = np.setdiff1d(self.test_mask, high_confidence_indices)

        print("Updated train_mask and test_mask after label propagation.")