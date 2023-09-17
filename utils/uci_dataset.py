import os
import tarfile
import torch
import utils
import numpy as np
from torch.utils.data import Dataset

class UcIrvineMessageDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.uc_irc_args = args.uc_irc_args
        self.folder = self.uc_irc_args['folder']
        self.edges = self.load_edges()
        self.node_features = self.load_node_features()
        self.edge_types = self.load_edge_types()
        self.edge_timestamps = self.load_edge_timestamps()

    def load_node_features(self):
        node_features_file = os.path.join(self.folder, self.uc_irc_args['node_features_file'])
        node_features = np.loadtxt(node_features_file, skiprows=1)
        node_features = torch.FloatTensor(node_features)
        return node_features

    def load_edge_types(self):
        edge_types_file = os.path.join(self.folder, self.uc_irc_args['edge_types_file'])
        edge_types = np.loadtxt(edge_types_file, dtype=int, skiprows=1)
        edge_types = torch.LongTensor(edge_types)
        return edge_types

    def load_edge_timestamps(self):
        edge_timestamps_file = os.path.join(self.folder, self.uc_irc_args['edge_timestamps_file'])
        edge_timestamps = np.loadtxt(edge_timestamps_file, dtype=int, skiprows=1)
        edge_timestamps = torch.LongTensor(edge_timestamps)
        return edge_timestamps

    def load_edges(self):
        tar_file = os.path.join(self.folder, self.uc_irc_args['tar_file'])
        tar_archive = tarfile.open(tar_file, 'r:bz2')
        data = utils.load_data_from_tar(
            self.uc_irc_args['edges_file'], 
            tar_archive,
            starting_line=2,
            sep=' '
        )
        cols = utils.Namespace({'source': 0,
                                'target': 1,
                                'weight': 2,
                                'time': 3})
        
        data = data.long()
        self.num_nodes = int(data[:,[cols.source,cols.target]].max())

        #first id should be 0 (they are already contiguous)
        data[:,[cols.source,cols.target]] -= 1
        
        #add edges in the other direction (simmetric)
        data = torch.cat([data,
                          data[:,[cols.target,
                                  cols.source,
                                  cols.weight,
                                  cols.time]]],
                        dim=0)
        data[:,cols.time] = utils.aggregate_by_time(data[:,cols.time],
                                                    self.uc_irc_args['aggr_time'])
        
        ids = data[:,cols.source] * self.num_nodes + data[:,cols.target]
        self.num_non_existing = float(self.num_nodes**2 - ids.unique().size(0))
        idx = data[:,[cols.source,
                      cols.target,
                      cols.time]]
        self.max_time = data[:,cols.time].max()
        self.min_time = data[:,cols.time].min()

        return {'idx': idx, 'vals': torch.ones(idx.size(0))}
    
    def __len__(self):
        return len(self.edges)  # Update with your dataset's length

    def __getitem__(self, idx):
        edge_idx = self.edges['idx'][idx]
        source, target, weight, time = edge_idx

        # Assuming self.folder contains the path to your data files
        # Load the features for source and target nodes
        x_source = self.node_features[source]
        x_target = self.node_features[target]

        # Determine your edge_start_type and edge_end_type values based on edge_types
        edge_start_type = self.edge_types[idx]  # Assuming idx corresponds to edge index
        edge_end_type = self.edge_types[idx]  # Assuming idx corresponds to edge index

        # Get edge timestamps
        start_edge_timestamp = self.edge_timestamps[idx]  # Assuming idx corresponds to edge index
        end_edge_timestamp = self.edge_timestamps[idx]  # Assuming idx corresponds to edge index

        # Assuming you have the attention masks for back_x and front_x
        back_x_attention_mask = ...
        front_x_attention_mask = ...

        # Return the data as a dictionary
        return {
            'x': x_source.tolist(),
            'y': weight.tolist(),  # 'weight' seems to be your target value
            'start_edge_timestamp': start_edge_timestamp.tolist(),
            'end_edge_timestamp': end_edge_timestamp.tolist(),
            'edge_start_type': edge_start_type.tolist(),
            'edge_end_type': edge_end_type.tolist(),
            'back_x': x_target.tolist(),
            'front_x': x_source.tolist(),
            'back_x_attention_mask': back_x_attention_mask,
            'front_x_attention_mask': front_x_attention_mask
        }


# Usage example:
if __name__ == "__main__":
    class Args:
        uc_irc_args = {
            'folder': '/home/ubuntu/2022_finvcup_baseline/data/',
            'tar_file': 'opsahl-ucsocial.tar.bz2',
            'edges_file': 'opsahl-ucsocial/out.opsahl-ucsocial',
            'node_features_file': 'opsahl-ucsocial/node-attributes.txt',
            'edge_types_file': 'opsahl-ucsocial/edge-attributes.txt',
            'edge_timestamps_file': 'opsahl-ucsocial/time-attributes.txt',
            'aggr_time': 190080  # 216000 #172800, 86400 smaller numbers yields days with no edges
        }

    args = Args()

    uci_dataset = UcIrvineMessageDataset(args)
    example_idx = 0  # Replace with the desired example index
    example_data = uci_dataset[example_idx]
    print(example_data)  # Print the example data for demonstration
