import os
import torch
import numpy as np

from tqdm import tqdm
from logger import logger
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from config import parser_args
from utils.graph_dataset import GraphDataset
from utils.tools import AverageMeter, collate_fn
from models.dgt import GraphTransformer


@torch.no_grad()
def interface(args):
    dataset = GraphDataset(args)
    #test_mask = dataset.test_mask.tolist()
    #test_index = random.sample(test_mask, 10000)
    #test_dataset = torch.utils.data.Subset(dataset=dataset, indices=test_index)
    test_dataset = torch.utils.data.Subset(dataset=dataset, indices=dataset.test_mask)
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=128,
                                 shuffle=False,
                                 num_workers=args.num_works,
                                 collate_fn=collate_fn)

    model = GraphTransformer(args)
    model.load_state_dict(torch.load(os.path.join(args.save_model_dir, args.trained_model)), strict=False)
    model.to(args.device)

    #pred_list = []
    #true_list = []
    save_list = []
    with tqdm(test_dataloader, unit_scale=True, desc=f'interface', colour='green') as pbar_eval:
        #loss_avg = AverageMeter('loss')
        for batch in test_dataloader:
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
            logits = outputs['logits']
            #loss = outputs['loss']
            #loss_avg.update(loss.item())
            #pred_list.extend(logits.softmax(-1)[:, 1].tolist())
            #true_list.extend(batch['y'].tolist())
            save_list.extend(logits.softmax(-1).tolist())
            pbar_eval.update(1)
    #auc = roc_auc_score(true_list, pred_list)
    #print({
    #    'auc': auc,
    #    'loss': loss_avg.avg
    #})
    np.save(args.submit_path, np.array(save_list))


if __name__ == '__main__':
    args = parser_args()
    logger.info(args)
    interface(args)
