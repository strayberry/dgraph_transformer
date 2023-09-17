import os
import torch
import tarfile
from datetime import datetime
from torch.utils.data import Dataset
from utils import utils as u

class AutonomousSystemsDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.data = self.load_data()

    def __len__(self):
        return len(self.data['idx'])     

    def __getitem__(self, idx):
        edge_idx = self.data['idx'][idx]
        edge_vals = self.data['vals'][idx]

        source, target, time = edge_idx
        label = edge_vals

        x = [source, target, time]  # Replace with appropriate processing
        y = label.tolist()  # Assuming 'label' is a scalar

        # Process the edge data as needed, similar to the GraphDataset example
        start_edge_timestamp = [time.tolist()]
        end_edge_timestamp = [time.tolist()]
        edge_start_type = [label.tolist()]
        edge_end_type = [label.tolist()]
        back_x = [x]  # Replace with appropriate processing
        front_x = [x]  # Replace with appropriate processing
        back_x_attention_mask = [1] * len(back_x)
        front_x_attention_mask = [1] * len(front_x)

        return {
            'x': x,
            'y': y,
            'start_edge_timestamp': start_edge_timestamp,
            'end_edge_timestamp': end_edge_timestamp,
            'edge_start_type': edge_start_type,
            'edge_end_type': edge_end_type,
            'back_x': back_x,
            'front_x': front_x,
            'back_x_attention_mask': back_x_attention_mask,
            'front_x_attention_mask': front_x_attention_mask
        }

    def load_data(self):
        folder = self.args.aut_sys_args['folder']
        tar_file = os.path.join(folder, self.args.aut_sys_args['tar_file'])
        tar_archive = tarfile.open(tar_file, 'r:gz')

        files = tar_archive.getnames()
        cont_files2times = self.times_from_names(files)

        edges = []
        cols = u.Namespace({'source': 0, 'target': 1, 'time': 2})

        for file in files:
            data = u.load_data_from_tar(file, tar_archive, starting_line=4, sep='\t', type_fn=int, tensor_const=torch.LongTensor)
            time_col = torch.zeros(data.size(0), 1, dtype=torch.long) + cont_files2times[file]
            data = torch.cat([data, time_col], dim=1)
            data = torch.cat([data, data[:, [cols.target, cols.source, cols.time]]])
            edges.append(data)

        edges = torch.cat(edges)
        _, edges[:, [cols.source, cols.target]] = edges[:, [cols.source, cols.target]].unique(return_inverse=True)

        # Use only the first X time steps
        indices = edges[:, cols.time] < self.args.aut_sys_args['steps_accounted']
        edges = edges[indices, :]

        # Time aggregation
        edges[:, cols.time] = u.aggregate_by_time(edges[:, cols.time], self.args.aut_sys_args['aggr_time'])

        num_nodes = int(edges[:, [cols.source, cols.target]].max() + 1)
        ids = edges[:, cols.source] * num_nodes + edges[:, cols.target]
        num_non_existing = float(num_nodes ** 2 - ids.unique().size(0))

        max_time = edges[:, cols.time].max()
        min_time = edges[:, cols.time].min()

        return {'idx': edges, 'vals': torch.ones(edges.size(0))}

    def times_from_names(self, files):
        files2times = {}
        times2files = {}

        base = datetime.strptime("19800101", '%Y%m%d')

        for file in files:
            delta = (datetime.strptime(file[2:-4], '%Y%m%d') - base).days
            files2times[file] = delta
            times2files[delta] = file

        cont_files2times = {}
        sorted_times = sorted(files2times.values())
        new_t = 0

        for t in sorted_times:
            file = times2files[t]
            cont_files2times[file] = new_t
            new_t += 1

        return cont_files2times


# Usage example:
if __name__ == "__main__":
    class Args:
        aut_sys_args = {
            'folder': '/home/ubuntu/2022_finvcup_baseline/data/',
            'tar_file': 'as-733.tar.gz',
            'steps_accounted': 100,  # Set the appropriate number of steps
            'aggr_time': 1  # Set the appropriate aggregation time
        }

    args = Args()
    aut_sys_dataset = AutonomousSystemsDataset(args)
    example_idx = 0  # Replace with the desired example index
    example_data = aut_sys_dataset[example_idx]
    print(example_data)
 