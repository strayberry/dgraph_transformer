import numpy as np
from torch.utils.data import Dataset


class GraphDataset(Dataset):
    def __init__(self, args):
        data = np.load(args.data_path)
        self.x = data['x']
        self.x = (self.x - self.x.mean(0)) / self.x.std(0)
        self.y = data['y']
        self.edge_index = data['edge_index']
        self.edge_type = data['edge_type']
        self.train_mask = data['train_mask']
        self.test_mask = data['test_mask']
        self.edge_timestamp = data['edge_timestamp']

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
        back_x_attention_mask = [1] * back_x.shape[0]
        front_x_attention_mask = [1] * front_x.shape[0]
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
