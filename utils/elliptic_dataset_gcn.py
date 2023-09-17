import os
import torch
import tarfile
import numpy as np
from utils import utils as u
from torch.utils.data import Dataset
from torch_geometric.data import Data  # Import Data from torch_geometric


class EllipticTemporalDataset(Dataset):
    def __init__(self, args):
        self.args = args
        tar_file = os.path.join(args.elliptic_args['folder'], args.elliptic_args['tar_file'])
        tar_archive = tarfile.open(tar_file, 'r:gz')
        self.nodes = self.load_node_feats(args.elliptic_args, tar_archive)
        self.edges, self.edges_labels = self.load_transactions(args.elliptic_args, tar_archive)
        self.ids = self.nodes[:, 0]
        self.x = self.nodes[:, 1:-1]
        self.x = (self.x - self.x.mean(0)) / self.x.std(0)
        self.y = self.nodes[:, -1]
        self.num_classes = 2
        self.edge_indices = self.compute_edge_indices(self.edges)

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, idx):
        x = self.x[idx].unsqueeze(0)  # Add a batch dimension
        y = self.y[idx].unsqueeze(0)  # Add a batch dimension
        edge_index = self.edge_indices
        
        # Return a PyTorch Geometric Data object
        data = Data(x=x, y=y, edge_index=edge_index)
        return data

    def compute_edge_indices(self, edges):
        return edges[:, :2].t().contiguous()

    def load_transactions(self, elliptic_args, tar_archive):
        data = u.load_data_from_tar(elliptic_args['edges_file'], tar_archive, starting_line=1, tensor_const=torch.LongTensor)
        tcols = u.Namespace({'source': 0, 'target': 1, 'time': 2})

        data = torch.cat([data, data[:, [1, 0, 2]]])
        
        time_column = data[:, tcols.time]
        sorted_time, sorted_indices = torch.sort(time_column)
        sorted_edges = data[sorted_indices]

        self.max_time = data[:, tcols.time].max()
        self.min_time = data[:, tcols.time].min()

        return sorted_edges, torch.ones(data.size(0))

    def load_node_feats(self, elliptic_args, tar_archive):
        nodes_features = u.load_data_from_tar(elliptic_args['feats_file'], tar_archive, starting_line=0)
        nodes_labels = u.load_data_from_tar(elliptic_args['classes_file'], tar_archive, starting_line=1, replace_unknown=True).long()
        nodes_times = u.load_data_from_tar(elliptic_args['times_file'], tar_archive, starting_line=1, replace_unknown=True).long()
        ids = nodes_features[:, 0].unsqueeze(1)
        merged_tensor = torch.cat((ids, nodes_features[:, 1:], nodes_times[:, 1:], nodes_labels[:, 1:]), dim=1)
        

        # Filter out rows where the last value is -1
        valid_rows = merged_tensor[:, -1] != -1
        merged_tensor = merged_tensor[valid_rows]
        # sort with timestamp
        sorted_indices = torch.argsort(merged_tensor[:, -2])
        sorted_merged_tensor = merged_tensor[sorted_indices]
        
        return sorted_merged_tensor.float()
    

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
    elliptic_dataset = EllipticTemporalDataset(args)
    example_idx = 0  # Replace with the desired example index
    example_data = elliptic_dataset[example_idx]
    print(example_data)
