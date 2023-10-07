from utils.elliptic_dataset_gcn import EllipticTemporalDataset
from utils.utils import prepare_folder
from models.mlp import MLP, MLPLinear
from models.gcn import GCN
from models.sage import SAGE

from config import parser_args
from logger import logger

import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.transforms as T
from torch.utils.data import random_split
from torch_geometric.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import pandas as pd
from sklearn.model_selection import train_test_split


gcn_parameters = {'lr':0.01
              , 'num_layers':2
              , 'hidden_channels':128
              , 'dropout':0.0
              , 'batchnorm': False
              , 'l2':5e-7
             }

sage_parameters = {'lr':0.01
              , 'num_layers':2
              , 'hidden_channels':128
              , 'dropout':0
              , 'batchnorm': False
              , 'l2':5e-7
             }


def train(model, data, train_idx, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    loss = F.cross_entropy(out, data.y[train_idx].long())
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test(model, data, split_idx):
    model.eval()

    # Obtain log probabilities (logits) from the model
    out = model(data.x, data.adj_t)  
    y_pred = out.exp()

    losses = {}
    accuracies = {}
    for key in ['train', 'valid', 'test']:
        node_id = split_idx[key]
        losses[key] = F.nll_loss(out[node_id], data.y[node_id].long()).item()

        pred = y_pred[node_id].max(1)[1]
        correct = float(pred.eq(data.y[node_id]).sum().item())
        acc = correct / len(node_id)
        accuracies[key] = acc

    return losses, accuracies, y_pred


def main():
    args = parser_args()
    args.elliptic_args = {'folder': '/home/ubuntu/2022_finvcup_baseline/data/elliptic_temporal','tar_file': 'elliptic_bitcoin_dataset_cont.tar.gz','classes_file': 'elliptic_bitcoin_dataset_cont/elliptic_txs_classes.csv','times_file': 'elliptic_bitcoin_dataset_cont/elliptic_txs_nodetime.csv','edges_file': 'elliptic_bitcoin_dataset_cont/elliptic_txs_edgelist_timed.csv','feats_file': 'elliptic_bitcoin_dataset_cont/elliptic_txs_features.csv'}
    args.dataset = 'Elliptic'
    args.model = 'sage'
    args.log_steps = 1
    logger.info(args)
    
    device = torch.device(f'{args.device}' if torch.cuda.is_available() else 'cpu')

    # Initialize your EllipticTemporalDataset here
    print(args.elliptic_args['folder'])
    dataset = EllipticTemporalDataset(root=args.elliptic_args['folder'], name='elliptic', transform=T.ToSparseTensor())
    print(f"dataset:{dataset}")
    data = dataset[0]
    print(f"data:{data}")
    data.adj_t = data.adj_t.to_symmetric()

    if args.dataset in ['Elliptic']:
        x = data.x
        x = (x-x.mean(0))/x.std(0)
        data.x = x
    if data.y.dim()==2:
        data.y = data.y.squeeze(1)      

    # Print the content of the data object
    print("Data object:", data)
    print("Number of nodes in data:", data.num_nodes)
    print("Number of features in data:", data.num_features)
    
    # Assuming the dataset is a single graph, we'll split node indices
    num_nodes = data.x.size(0)  # Assuming x is the node feature matrix
    node_indices = torch.arange(num_nodes)
    train_size = int(0.8 * num_nodes)
    valid_size = (num_nodes - train_size) // 2
    test_size = num_nodes - train_size - valid_size
    train_indices, valid_indices, test_indices = [list(subset.indices) for subset in random_split(node_indices, [train_size, valid_size, test_size])]
    split_idx = {'train':train_indices, 'valid':valid_indices, 'test':test_indices}
        
    data = data.to(device)
    train_idx = split_idx['train']

    model_dir = prepare_folder('elliptic', args.model)
    print('model_dir:', model_dir)
    
    in_channels = dataset.data.x.size(1)
    nlabels = 2
    if args.model == 'gcn':
        para_dict = gcn_parameters
        model_para = gcn_parameters.copy()
        model_para.pop('lr')
        model_para.pop('l2')
        model = GCN(in_channels = in_channels, out_channels = nlabels, **model_para).to(device)
    if args.model == 'sage':
        para_dict = sage_parameters
        model_para = sage_parameters.copy()
        model_para.pop('lr')
        model_para.pop('l2')
        model = SAGE(in_channels = in_channels, out_channels = nlabels, **model_para).to(device)

    print(f'Model {args.model} initialized')
    optimizer = torch.optim.Adam(model.parameters(), lr=para_dict['lr'], weight_decay=para_dict['l2'])

    print(sum(p.numel() for p in model.parameters()))

    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=para_dict['lr'], weight_decay=para_dict['l2'])
    min_valid_loss = 1e8

    for epoch in range(1, args.epoch + 1):
        train_loss = train(model, data, train_idx, optimizer)  
        print(f'Epoch: {epoch}/{args.epoch}, Train Loss: {train_loss:.4f}')

        losses, accuracies, y_pred = test(model, data, split_idx)  
        valid_loss = losses['valid']
        
        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            torch.save(model.state_dict(), model_dir + 'model.pt')

        if epoch % args.log_steps == 0:
            print(f'Epoch: {epoch:02d}, Valid Loss: {valid_loss:.4f}, Valid Acc: {accuracies["valid"]:.4f}')

    # After training is complete
    losses, accuracies, y_pred = test(model, data, split_idx)
    print(f'Test Loss: {losses["test"]:.4f}, Test Acc: {accuracies["test"]:.4f}')


if __name__ == "__main__":
    main()
