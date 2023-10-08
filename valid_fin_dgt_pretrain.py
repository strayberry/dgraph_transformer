import os
import torch
import random

from config import parser_args
from torch.utils.data import DataLoader
from tqdm import tqdm
from logger import logger
from sklearn.metrics import roc_auc_score

from utils.graph_dataset import GraphDataset
from utils.tools import AverageMeter, collate_fn
from models.dgt import GraphTransformer


def train(args):
    dataset = GraphDataset(args)
    train_mask = dataset.train_mask.tolist()
    all_train_index = dataset.train_index.tolist()
    valid_index = random.sample(train_mask, 10000)
    train_index = list(set(train_mask).difference(set(valid_index)))
    train_dataset = torch.utils.data.Subset(dataset=dataset, indices=train_index)
    val_dataset = torch.utils.data.Subset(dataset=dataset, indices=valid_index)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.train_batch_size,
                                  shuffle=True,
                                  num_workers=args.num_works,
                                  collate_fn=collate_fn)
    valid_dataloader = DataLoader(dataset=val_dataset,
                                  batch_size=args.not_train_batch_size,
                                  shuffle=False,
                                  num_workers=args.num_works,
                                  collate_fn=collate_fn)

    model = GraphTransformer(args)
    model.load_state_dict(torch.load(os.path.join(args.save_model_dir, args.pretrained_model)), strict=False)
    model.to(args.device)

    valid_res = evaluate(model, valid_dataloader, args)
    logger.info(f'valid pretrain model| '
                f"valid loss {valid_res['loss']:.5f} | valid auc {valid_res['auc']:.5f}")

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
    logger.info(args)
    train(args)
