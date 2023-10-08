import torch
import torch.nn.functional as F
import torch.nn as nn

from logger import logger
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split

from config import parser_args
from utils.elliptic_dataset import EllipticTemporalDataset
from utils.tools import AverageMeter, collate_fn
from utils.utils import prepare_folder
from models.mlp import MLP, MLPLinear


mlp_parameters = {'lr':0.01
              , 'num_layers':2
              , 'hidden_channels':128
              , 'dropout':0.0
              , 'batchnorm': False
              , 'l2':5e-7
             }


def train(model, train_loader, optimizer, device):
    model.train()

    total_loss = 0
    for data in train_loader:
        x = data['x'].to(device)  # Assuming 'x' is the input feature tensor in your dataset
        y = data['y'].to(device)  # Assuming 'y' is the label tensor in your dataset
        optimizer.zero_grad()

        out = model(x)  # Forward pass through the model for MLP

        loss = F.nll_loss(out, y)  # Compute the negative log likelihood loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


@torch.no_grad()
def test(model, loader, device):
    model.eval()

    all_pred = []
    all_loss = []
    all_accuracies = []

    for data in loader:
        x = data['x'].to(device)  # Assuming 'x' is the input feature tensor in your dataset
        y = data['y'].to(device)  # Assuming 'y' is the label tensor in your dataset

        out = model(x)  # Forward pass through the model for MLP

        # Predictions and loss
        y_pred = out.argmax(dim=1)  # Convert model's output probabilities to predicted classes
        all_pred.append(y_pred)
        loss = F.nll_loss(out, y)  # Compute the negative log likelihood loss
        all_loss.append(loss.item())

        # Accuracy calculation
        correct_predictions = y_pred.eq(y).sum().item()
        acc = correct_predictions / len(y)
        all_accuracies.append(acc)

    average_loss = sum(all_loss) / len(loader)
    average_accuracy = sum(all_accuracies) / len(loader)
    return average_loss, average_accuracy, torch.cat(all_pred)


def main():
    args = parser_args()
    args.elliptic_args = {'folder': '/home/ubuntu/2022_finvcup_baseline/data/elliptic_temporal','tar_file': 'elliptic_bitcoin_dataset_cont.tar.gz','classes_file': 'elliptic_bitcoin_dataset_cont/elliptic_txs_classes.csv','times_file': 'elliptic_bitcoin_dataset_cont/elliptic_txs_nodetime.csv','edges_file': 'elliptic_bitcoin_dataset_cont/elliptic_txs_edgelist_timed.csv','feats_file': 'elliptic_bitcoin_dataset_cont/elliptic_txs_features.csv'}
    args.dataset_name = 'Elliptic'
    args.model = 'mlp'
    args.log_steps = 1
    logger.info(args)
    
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
    train_loader = DataLoader(dataset, batch_size=args.train_batch_size, sampler=SubsetRandomSampler(train_indices), collate_fn=collate_fn)
    valid_loader = DataLoader(dataset, batch_size=args.train_batch_size, sampler=SubsetRandomSampler(valid_indices), collate_fn=collate_fn)
    test_loader = DataLoader(dataset, batch_size=args.not_train_batch_size, sampler=SubsetRandomSampler(test_indices), collate_fn=collate_fn)

    nlabels = dataset.num_classes

    model_dir = prepare_folder('elliptic', args.model)
    print('model_dir:', model_dir)

    if args.model == 'mlp':
        para_dict = mlp_parameters
        model_para = mlp_parameters.copy()
        model_para.pop('lr')
        model_para.pop('l2')
        model = MLP(in_channels = 167, out_channels = nlabels, **model_para).to(device)

    print(f'Model {args.model} initialized')

    print(sum(p.numel() for p in model.parameters()))

    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=para_dict['lr'], weight_decay=para_dict['l2'])
    min_valid_loss = 1e8

    for epoch in range(1, args.epoch + 1):
        train_loss = train(model, train_loader, optimizer, device)
        print(f'Epoch: {epoch}/{args.epoch}, Train Loss: {train_loss:.4f}')

        valid_loss, valid_accuracy, _ = test(model, valid_loader, device)  # Evaluate on the validation set

        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            torch.save(model.state_dict(), model_dir + 'model.pt')

        if epoch % args.log_steps == 0:
            print(f'Epoch: {epoch:02d}, Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.4f}')

    # After training is complete, you can evaluate the model on the test set
    test_loss, test_accuracy, _ = test(model, test_loader, device)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')


if __name__ == "__main__":
    main()
