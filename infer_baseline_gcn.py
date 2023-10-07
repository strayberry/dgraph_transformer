import os
import torch
import numpy as np
import torch.nn.functional as F
import torch_geometric.transforms as T

from models.mlp import MLP
from models.gcn import GCN
from models.sage import SAGE

from logger import logger
from torch.utils.data import random_split
from config import parser_args
from utils.utils import prepare_folder
from utils.evaluator import Evaluator
from utils.elliptic_dataset_gcn import EllipticTemporalDataset


mlp_parameters = {'lr':0.01
              , 'num_layers':2
              , 'hidden_channels':128
              , 'dropout':0.0
              , 'batchnorm': False
              , 'l2':5e-7
             }

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


@torch.no_grad()
def test(model, data, split_idx):
    model.eval()

    # Obtain log probabilities (logits) from the model
    out = model(data.x, data.adj_t)

    # Compute the probabilities by exponentiating the logits
    y_pred = F.softmax(out, dim=1)

    split_losses = {}
    split_accuracies = {}  
    for key in ['train', 'valid', 'test']:
        node_id = split_idx[key]
        split_losses[key] = F.cross_entropy(out[node_id], data.y[node_id].long()).item()

        _, pred_class = y_pred[node_id].max(dim=1)
        correct = (pred_class == data.y[node_id]).sum().item()
        accuracy = correct / len(node_id)
        split_accuracies[key] = accuracy

    return split_losses, split_accuracies, y_pred

def load_model(model_name, model_file, in_channels, nlabels, device):
    if model_name == 'mlp':
        para_dict = mlp_parameters
        model_para = mlp_parameters.copy()
        model_para.pop('lr')
        model_para.pop('l2')
        model = MLP(in_channels = 167, out_channels = nlabels, **model_para).to(device)
    if model_name == 'gcn':
        para_dict = gcn_parameters
        model_para = gcn_parameters.copy()
        model_para.pop('lr')
        model_para.pop('l2')
        model = GCN(in_channels = in_channels, out_channels = nlabels, **model_para).to(device)
    if model_name == 'sage':
        para_dict = sage_parameters
        model_para = sage_parameters.copy()
        model_para.pop('lr')
        model_para.pop('l2')
        model = SAGE(in_channels = in_channels, out_channels = nlabels, **model_para).to(device)
    #else:
    #    raise ValueError(f"Unknown model: {model_name}")
    
    model.load_state_dict(torch.load(model_file))
    return model

def main():    
    args = parser_args()
    args.elliptic_args = {'folder': '/home/ubuntu/2022_finvcup_baseline/data/elliptic_temporal','tar_file': 'elliptic_bitcoin_dataset_cont.tar.gz','classes_file': 'elliptic_bitcoin_dataset_cont/elliptic_txs_classes.csv','times_file': 'elliptic_bitcoin_dataset_cont/elliptic_txs_nodetime.csv','edges_file': 'elliptic_bitcoin_dataset_cont/elliptic_txs_edgelist_timed.csv','feats_file': 'elliptic_bitcoin_dataset_cont/elliptic_txs_features.csv'}
    args.dataset = 'Elliptic'
    args.model = 'gcn'
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

    model_file = './model_files/elliptic/{}/model.pt'.format(args.model)
    print('model_file:', model_file)
    model = load_model(args.model, model_file, data.x.size(-1), data.num_features, device)

    split_losses, split_accuracies, y_pred = test(model, data, split_idx)

    evaluator = Evaluator('auc')
    preds_train, preds_valid = y_pred[data.train_mask], y_pred[data.valid_mask]
    y_train, y_valid = data.y[data.train_mask], data.y[data.valid_mask]
    train_auc = evaluator.eval(y_train, preds_train)['auc']
    valid_auc = evaluator.eval(y_valid, preds_valid)['auc']
    
    print('Train AUC:', train_auc)
    print('Valid AUC:', valid_auc)
    
    preds = y_pred[data.test_mask].cpu().numpy()
    np.save('./submit/preds.npy', preds)

if __name__ == "__main__":
    main()
