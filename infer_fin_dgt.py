import os
import torch
import random
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
    #test_index = random.sample(test_mask, 100)
    #test_dataset = torch.utils.data.Subset(dataset=dataset, indices=test_index)
    test_dataset = torch.utils.data.Subset(dataset=dataset, indices=dataset.test_mask)
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=2,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=collate_fn)

    model = GraphTransformer(args)
    model.load_state_dict(torch.load(os.path.join(args.save_model_dir, args.trained_model)), strict=False)
    model.to(args.device)

    filtered_results = {}
    with tqdm(test_dataloader, unit_scale=True, desc=f'interface', colour='green') as pbar_eval:
        for i, batch in enumerate(test_dataloader):
            inputs = {k: v.to(args.device) for k, v in batch.items()}
            try:
                outputs = model(**inputs)
            except RuntimeError as e:
                #print(inputs)
                #print("Error in model forward pass:", e)
                #print("Input shapes:", {k: v.shape for k, v in inputs.items()})
                pass
            
            logits = outputs['logits']
            
            max_probs, predicted_indices = logits.softmax(-1).max(dim=1)
            for j, prob in enumerate(max_probs):
                if prob > 0.95:
                    index = dataset.test_mask[i * 2 + j].item()
                    predicted_label = predicted_indices[j].item()
                    filtered_results[index] = predicted_label

            pbar_eval.update(1)
    
    return filtered_results


if __name__ == '__main__':
    args = parser_args()
    logger.info(args)

    dataset = GraphDataset(args)
    filtered_results = interface(args)

    dataset.merge_predictions_to_train_set(filtered_results)

    np.savez(
        'updated_dataset.npz', 
        x=dataset.data_x, 
        y=dataset.y, 
        edge_index=dataset.edge_index,
        edge_type=dataset.edge_type,
        edge_timestamp=dataset.edge_timestamp,
        train_mask=dataset.train_new_mask,
        test_mask=dataset.test_new_mask
        )