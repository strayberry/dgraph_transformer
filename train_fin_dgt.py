import os
import torch
import random
import numpy as np

from tqdm import tqdm
from logger import logger
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from transformers.optimization import get_linear_schedule_with_warmup

from config import parser_args
from utils.graph_dataset import GraphDataset
from utils.tools import AverageMeter, collate_fn
from models.dgt import GraphTransformer


def train(args):
    dataset = GraphDataset(args, filter_labels=[])
    #train_mask = dataset.train_mask.tolist()
    #train_mask = random.sample(train_mask, 1000)
    #valid_index = random.sample(train_mask, 200)
    all_indices = list(range(len(dataset.x)))
    total_size = len(dataset.x)
    train_size = int(total_size * 4 / 6)
    valid_size = int(total_size * 1 / 6)
    test_size = total_size - train_size - valid_size
    random.shuffle(all_indices)
    train_index = all_indices[:train_size]
    valid_index = all_indices[train_size:train_size+valid_size]
    test_index = all_indices[train_size+valid_size:]

    train_dataset = torch.utils.data.Subset(dataset=dataset, indices=train_index)
    val_dataset = torch.utils.data.Subset(dataset=dataset, indices=valid_index)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.train_batch_size,
                                  shuffle=True,
                                  num_workers=args.num_works,
                                  collate_fn=collate_fn)
    valid_dataloader = DataLoader(dataset=val_dataset,
                                  batch_size=args.not_train_batch_size,
                                  shuffle=True,
                                  num_workers=args.num_works,
                                  collate_fn=collate_fn)

    model = GraphTransformer(args)
    model.load_state_dict(torch.load(os.path.join(args.save_model_dir, args.pretrained_model)), strict=False)
    model.to(args.device)

    train_steps = args.epoch * len(train_dataloader)
    optimizer = AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(args.warmup_ratio * train_steps),
        num_training_steps=train_steps - int(args.warmup_ratio * train_steps)
    )

    steps = 0
    auc = 0

    for epoch in range(1, args.epoch + 1):
        loss_avg = AverageMeter('loss')
        train_auc = AverageMeter('auc')
        with tqdm(train_dataloader, unit_scale=True, desc=f'epoch {epoch} training', colour='blue') as pbar:
            for batch in train_dataloader:
                try:
                    steps += 1
                    model.train()
                    inputs = {
                        'x': batch['x'].to(args.device),
                        'y': batch['y'].to(args.device),
                        'start_edge_timestamp': batch['start_edge_timestamp'].to(args.device),
                        'end_edge_timestamp': batch['end_edge_timestamp'].to(args.device),
                        'edge_start_type': batch['edge_start_type'].to(args.device),
                        'edge_end_type': batch['edge_end_type'].to(args.device),
                        'back_x': batch['back_x'].to(args.device),
                        'front_x': batch['front_x'].to(args.device),
                        'attention_mask': batch['attention_mask'].to(args.device)
                    }
                    outputs = model(**inputs)
                    loss = outputs['loss']
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        3
                    )

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
                    print(inputs)
                    print("Error in model forward pass:", e)
                    print("Input shapes:", {k: v.shape for k, v in inputs.items()})
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
            try:
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
                pred_list.append(logits.softmax(-1)[:, 1].cpu().detach().numpy())
                true_list.extend(batch['y'].cpu().numpy())
                pbar_eval.update(1)
            except Exception as e:
                print(inputs)
                print("Error in model forward pass:", e)
                print("Input shapes:", {k: v.shape for k, v in inputs.items()})
                pass
            continue
        
        pred_list = np.concatenate(pred_list, axis=0)
        true_list = np.array(true_list)
        true_list_binary = (true_list == 1).astype(int)

    auc = roc_auc_score(true_list_binary, pred_list)
    model.train()
    return {
        'auc': auc,
        'loss': loss_avg.avg
    }


if __name__ == '__main__':
    args = parser_args()
    logger.info(args)
    train(args)
