import os
import torch
import pandas as pd
import numpy as np

from utils import utils as u
from torch.utils.data.dataset import random_split
from torch_geometric.data import InMemoryDataset, Data
from typing import Optional, Callable, List


def raw_file_names():
    return [
            'elliptic_bitcoin_dataset_cont/elliptic_txs_features.csv',
            'elliptic_bitcoin_dataset_cont/elliptic_txs_classes.csv',
            'elliptic_bitcoin_dataset_cont/elliptic_txs_nodetime.csv',
            'elliptic_bitcoin_dataset_cont/elliptic_txs_edgelist_timed.csv',
        ]


def read_elliptic(folder):
    merged_df = preprocess_files(folder)
    sorted_merged_tensor = preprocess_merged_df(merged_df)
    edge_index_mapped = preprocess_edges(folder, sorted_merged_tensor)
    
    x = sorted_merged_tensor[:, 1:-1]
    x = (x - x.mean(0)) / x.std(0)
    y = sorted_merged_tensor[:, -1]
    edge_type = torch.ones(edge_index_mapped.shape[1])
    
    data = Data(x=x.float(), edge_index=edge_index_mapped, edge_attr=edge_type, y=y)
    print(data)
    return data


def preprocess_files(folder):
    nodes_features = pd.read_csv(os.path.join(folder, raw_file_names()[0]))
    nodes_features.rename(columns={nodes_features.columns[0]: 'txId'}, inplace=True)
    nodes_labels = pd.read_csv(os.path.join(folder, raw_file_names()[1]), dtype=int).apply(pd.to_numeric, errors='coerce').fillna(-1)
    nodes_times = pd.read_csv(os.path.join(folder, raw_file_names()[2])).apply(pd.to_numeric, errors='coerce').fillna(-1)
    # Merge the DataFrames on 'txId'
    merged_df = nodes_features.merge(nodes_times, on='txId').merge(nodes_labels, on='txId')
    return merged_df


def preprocess_merged_df(merged_df):
    tensors = [torch.tensor(merged_df[col].values).unsqueeze(1) for col in merged_df.columns]
    merged_tensor = torch.cat(tensors, dim=1)
    valid_rows = merged_tensor[:, -1] != -1
    merged_tensor = merged_tensor[valid_rows]
    sorted_indices = torch.argsort(merged_tensor[:, -2])
    return merged_tensor[sorted_indices].float()


def preprocess_edges(folder, sorted_merged_tensor):
    nodes_edges = pd.read_csv(os.path.join(folder, raw_file_names()[3]))
    nodes_edges = nodes_edges[['txId1', 'txId2']].drop_duplicates()

    # Convert to tensor
    edge_index = torch.tensor(nodes_edges.values, dtype=torch.int64).t().contiguous()

    # Extract the unique valid nodes from sorted_merged_tensor
    valid_nodes = set(sorted_merged_tensor[:, 0].numpy())

    # Filter edges
    valid_edges_mask = np.isin(edge_index[0].numpy(), list(valid_nodes)) & np.isin(edge_index[1].numpy(), list(valid_nodes))
    edge_index = edge_index[:, valid_edges_mask]

    # Create a Mapping:
    node_ids = sorted_merged_tensor[:, 0].long()
    mapping = {node_id.item(): i for i, node_id in enumerate(node_ids)}

    # Apply the Mapping:
    edge_index_mapped = edge_index.clone()
    for i in range(edge_index.shape[1]):
        edge_index_mapped[0, i], edge_index_mapped[1, i] = mapping[edge_index[0, i].item()], mapping[edge_index[1, i].item()]

    return edge_index_mapped


class EllipticTemporalDataset(InMemoryDataset):
    def __init__(self, root: str, name: str, 
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return [
            'elliptic_bitcoin_dataset_cont/elliptic_txs_features.csv',
            'elliptic_bitcoin_dataset_cont/elliptic_txs_classes.csv',
            'elliptic_bitcoin_dataset_cont/elliptic_txs_nodetime.csv',
            'elliptic_bitcoin_dataset_cont/elliptic_txs_edgelist_timed.csv',
        ]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download is not implemented as the data is not publicly available for direct downloading
        # Make sure you've manually downloaded the data and placed it in the `raw_dir`
        pass

    def process(self):
        data = read_elliptic(self.root)
        # If using the pre_transform function (e.g., for data augmentation), apply it
        #if self.pre_transform is not None:
        #    data = self.pre_transform(data)
        data_list = [data]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}()'


# Usage example:
if __name__ == "__main__":
    class Args:
        elliptic_args = {
            'folder': '/content/drive/MyDrive/EvolveGCN/data/elliptic_temporal/',
            'tar_file': 'elliptic_bitcoin_dataset_cont.tar.gz',
            'classes_file': 'elliptic_bitcoin_dataset_cont/elliptic_txs_classes.csv',
            'times_file': 'elliptic_bitcoin_dataset_cont/elliptic_txs_nodetime.csv',
            'edges_file': 'elliptic_bitcoin_dataset_cont/elliptic_txs_edgelist_timed.csv',
            'feats_file': 'elliptic_bitcoin_dataset_cont/elliptic_txs_features.csv'
        }

    args = Args()
    elliptic_dataset = EllipticTemporalDataset(root=args.elliptic_args['folder'])
    print(elliptic_dataset)
