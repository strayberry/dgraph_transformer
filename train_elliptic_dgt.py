import os
import torch
import numpy as np

from config import parser_args
from torch.utils.data import DataLoader

from tqdm import tqdm
from logger import logger
from torch.optim import AdamW
from torch.utils.data.sampler import SubsetRandomSampler
from transformers.optimization import get_linear_schedule_with_warmup
from sklearn.metrics import roc_auc_score

from utils.tools import AverageMeter, collate_fn
from utils.graph_dataset import GraphDataset
from utils.as_dataset import AutonomousSystemsDataset
from utils.elliptic_dataset import EllipticTemporalDataset
from utils.reddit_dataset import RedditDataset
from models.dgt import GraphTransformer


def train_all(args):
    validation_split = .2
    shuffle_dataset = True
    random_seed= 42

    # Creating data indices for training and validation splits:
    dataset_size = len(args.dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_dataloader = torch.utils.data.DataLoader(args.dataset, 
                                               batch_size=args.train_batch_size, 
                                               sampler=train_sampler,
                                               collate_fn=collate_fn)
    valid_dataloader = torch.utils.data.DataLoader(args.dataset, 
                                                    batch_size=args.train_batch_size,
                                                    sampler=valid_sampler,
                                                    collate_fn=collate_fn)
    

    model = GraphTransformer(args)
    model.to(args.device)

    train_steps = args.epoch * len(train_dataloader)
    optimizer = AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(args.warmup_ratio * train_steps),
        num_training_steps=train_steps - int(args.warmup_ratio * train_steps)
    )

    for epoch in range(1, args.epoch + 1):
        loss_avg = AverageMeter('loss')
        train_auc = AverageMeter('auc')
        with tqdm(train_dataloader, unit_scale=True, desc=f'epoch {epoch} training', colour='blue') as pbar:
            for batch in train_dataloader:
                try:
                    model.train()
                    inputs = {
                        'x': batch['x'].to(args.device),
                        'y': batch['y'].to(args.device),
                        'start_edge_timestamp': batch['start_edge_timestamp'].to(args.device),
                        'end_edge_timestamp': batch['end_edge_timestamp'].to(args.device),
                        'edge_start_type': batch['edge_start_type'].to(args.device),
                        'edge_end_type': batch['edge_end_type'].to(args.device),
                        'back_x': batch['back_x'].to(args.device),
                        'front_x': batch['front_x'].to(args.device)
                    }
                    outputs = model(**inputs)
                    loss = outputs['loss']
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    logits = outputs['logits']
                    try:
                        auc = roc_auc_score(batch['y'].tolist(), logits[:, 1].softmax(-1).tolist())
                        train_auc.update(auc)
                    except ValueError:
                        auc = auc

                    loss_avg.update(loss.item())
                    pbar.set_postfix({'loss': f'{loss_avg.avg:.5f}',
                                    'auc': f'{train_auc.avg:.5f}',
                                    'lr': f"{scheduler.get_last_lr()[0]:.7f}"})
                    pbar.update(1)
                except Exception as e:
                    print(e)
                    print(inputs)
                    pass
                continue

        valid_res = evaluate(model, valid_dataloader, args)
        logger.info(f'epoch {epoch} | loss {loss_avg.val:.5f} | '
                    f"valid loss {valid_res['loss']:.5f} | valid auc {valid_res['auc']:.5f}")

        torch.save(model.state_dict(), os.path.join(args.save_model_dir, f'epoch_{epoch}_model.bin'))
        torch.cuda.empty_cache()


@torch.no_grad()
def evaluate(model, valid_dataloader, args):
    model.eval()
    pred_list = []
    true_list = []
    with tqdm(valid_dataloader, unit_scale=True, desc=f'evaluating', colour='green') as pbar_eval:
        loss_avg = AverageMeter('loss')
        for batch in valid_dataloader:
            inputs = {
                'x': batch['x'].to(args.device),
                'y': batch['y'].to(args.device),
                'start_edge_timestamp': batch['start_edge_timestamp'].to(args.device),
                'end_edge_timestamp': batch['end_edge_timestamp'].to(args.device),
                'edge_start_type': batch['edge_start_type'].to(args.device),
                'edge_end_type': batch['edge_end_type'].to(args.device),
                'back_x': batch['back_x'].to(args.device),
                'front_x': batch['front_x'].to(args.device)
            }

            outputs = model(**inputs)
            loss = outputs['loss']
            logits = outputs['logits']
            loss_avg.update(loss.item())
            pred_list.extend(logits.softmax(-1)[:, 1].tolist())
            true_list.extend(batch['y'].tolist())
            pbar_eval.update(1)
    auc = roc_auc_score(true_list, pred_list)
    model.train()
    return {
        'auc': auc,
        'loss': loss_avg.avg
    }


if __name__ == '__main__':
    args = parser_args()
    args.elliptic_args = {'folder': '/home/ubuntu/2022_finvcup_baseline/data/elliptic_temporal','tar_file': 'elliptic_bitcoin_dataset_cont.tar.gz','classes_file': 'elliptic_bitcoin_dataset_cont/elliptic_txs_classes.csv','times_file': 'elliptic_bitcoin_dataset_cont/elliptic_txs_nodetime.csv','edges_file': 'elliptic_bitcoin_dataset_cont/elliptic_txs_edgelist_timed.csv','feats_file': 'elliptic_bitcoin_dataset_cont/elliptic_txs_features.csv'}
    args.dataset = EllipticTemporalDataset(args)
    args.dataset_name = 'Elliptic'
    #args.reddit_args = {'folder': '/home/ubuntu/2022_finvcup_baseline/data/reddit/','feats_file': 'web-redditEmbeddings-subreddits.csv','title_edges_file': 'soc-redditHyperlinks-title.tsv','body_edges_file': 'soc-redditHyperlinks-body.tsv','aggr_time': 1}
    #args.dataset = RedditDataset(args)
    #args.aut_sys_args = {'folder': '/home/ubuntu/2022_finvcup_baseline/data/','tar_file': 'as-733.tar.gz','steps_accounted': 100,'aggr_time': 1}
    #args.dataset = AutonomousSystemsDataset(args)
    logger.info(args)
    train_all(args)
