import torch
import os.path as osp
import numpy as np

from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from typing import Optional, Callable, List


def read_xygraphp1(folder):
    print('read_xygraphp1')
    names = ['phase1_gdata.npz']
    items = [np.load(folder+'/'+name) for name in names]
    
    x = items[0]['x']
    y = items[0]['y'].reshape(-1,1)
    edge_index = items[0]['edge_index']
    edge_type = items[0]['edge_type']
    np.random.seed(42)
    train_mask_t = items[0]['train_mask']
    np.random.shuffle(train_mask_t)
    train_mask = train_mask_t[:int(len(train_mask_t)/10*6)]
    valid_mask = train_mask_t[int(len(train_mask_t)/10*6):]
    test_mask = items[0]['test_mask']

    x = torch.tensor(x, dtype=torch.float).contiguous()
    y = torch.tensor(y, dtype=torch.int64)
    edge_index = torch.tensor(edge_index.transpose(), dtype=torch.int64).contiguous()
    edge_type = torch.tensor(edge_type, dtype=torch.float)
    train_mask = torch.tensor(train_mask, dtype=torch.int64)
    valid_mask = torch.tensor(valid_mask, dtype=torch.int64)
    test_mask = torch.tensor(test_mask, dtype=torch.int64)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_type, y=y)
    data.train_mask = train_mask
    data.valid_mask = valid_mask
    data.test_mask = test_mask

    return data

class XYGraphP1(InMemoryDataset):
    r"""
    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"xygraphp1"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = ''

    def __init__(self, root: str, name: str, 
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        
        self.name = name
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        names = ['phase1_gdata.npz']
        return names

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        pass
#         for name in self.raw_file_names:
#             download_url('{}/{}'.format(self.url, name), self.raw_dir)

    def process(self):
        data = read_xygraphp1(self.raw_dir)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}()'