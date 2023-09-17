from utils.elliptic_dataset_gcn import EllipticTemporalDataset
from utils.utils import prepare_folder
from models.mlp import MLP, MLPLinear
from models.gcn import GCN
from models.sage import SAGE

from config import parser_args
from logger import logger

import torch
import torch.nn.functional as F
import torch.nn as nn
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


def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


@torch.no_grad()
def test(model, loader, device):
    model.eval()
    all_pred = []
    all_loss = []
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index)
        y_pred = out.argmax(dim=1)
        all_pred.append(y_pred)
        loss = F.nll_loss(out, data.y)
        all_loss.append(loss.item())
    average_loss = sum(all_loss) / len(loader)
    return average_loss, torch.cat(all_pred)


def main():
    args = parser_args()
    args.elliptic_args = {'folder': '/home/ubuntu/2022_finvcup_baseline/data/elliptic_temporal','tar_file': 'elliptic_bitcoin_dataset_cont.tar.gz','classes_file': 'elliptic_bitcoin_dataset_cont/elliptic_txs_classes.csv','times_file': 'elliptic_bitcoin_dataset_cont/elliptic_txs_nodetime.csv','edges_file': 'elliptic_bitcoin_dataset_cont/elliptic_txs_edgelist_timed.csv','feats_file': 'elliptic_bitcoin_dataset_cont/elliptic_txs_features.csv'}
    args.model = 'gcn'
    args.log_steps = 1
    logger.info(args)

    random_seed = 2023

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')

    # Initialize your EllipticTemporalDataset here
    dataset = EllipticTemporalDataset(args)

    # Splitting the dataset into train, validation, and test sets
    total_data_len = len(dataset)
    train_size = int(0.7 * total_data_len)
    valid_size = int(0.2 * total_data_len)
    test_size = total_data_len - train_size - valid_size

    train_indices = list(range(train_size))
    valid_indices = list(range(train_size, train_size + valid_size))
    test_indices = list(range(train_size + valid_size, total_data_len))

    # Create DataLoaders for train, validation, and test sets
    train_loader = DataLoader(dataset, batch_size=args.train_batch_size, sampler=SubsetRandomSampler(train_indices))
    valid_loader = DataLoader(dataset, batch_size=args.train_batch_size, sampler=SubsetRandomSampler(valid_indices))
    test_loader = DataLoader(dataset, batch_size=args.not_train_batch_size, sampler=SubsetRandomSampler(test_indices))

    
    nlabels = dataset.num_classes
    in_channels = dataset.x.size(1)
    out_channels = dataset.num_classes

    model_dir = prepare_folder('elliptic', args.model)
    print('model_dir:', model_dir)

    if args.model == 'gcn':
        para_dict = gcn_parameters
        model_para = gcn_parameters.copy()
        model_para.pop('lr')
        model_para.pop('l2')
        model = GCN(in_channels = in_channels, out_channels = out_channels, **model_para).to(device)
    if args.model == 'sage':
        para_dict = sage_parameters
        model_para = sage_parameters.copy()
        model_para.pop('lr')
        model_para.pop('l2')
        model = SAGE(in_channels = in_channels, out_channels = out_channels, **model_para).to(device)

    print(f'Model {args.model} initialized')

    print(sum(p.numel() for p in model.parameters()))

    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=para_dict['lr'], weight_decay=para_dict['l2'])
    min_valid_loss = 1e8

    for epoch in range(1, args.epoch + 1):
        train_loss = train(model, train_loader, optimizer, device)
        print(f'Epoch: {epoch}/{args.epoch}, Train Loss: {train_loss:.4f}')

        valid_loss, _ = test(model, valid_loader, device)  # Evaluate on the validation set
        
        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            torch.save(model.state_dict(), model_dir + 'model.pt')

        if epoch % args.log_steps == 0:
            print(f'Epoch: {epoch:02d}, Valid Loss: {valid_loss:.4f}')

    # After training is complete, you can evaluate the model on the test set
    test_loss, _ = test(model, test_loader, device)
    print(f'Test Loss: {test_loss:.4f}')


if __name__ == "__main__":
    main()
