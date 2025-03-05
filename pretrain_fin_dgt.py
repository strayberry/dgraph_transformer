import os
import torch
import random

from logger import logger
from typing import Tuple
from tqdm import tqdm
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertModel, BertConfig
from transformers.activations import GELUActivation
from transformers.models.bert.modeling_bert import BertEncoder
from transformers.optimization import get_linear_schedule_with_warmup
from sklearn.metrics import roc_auc_score

from config import parser_args
from utils.graph_dataset import GraphDataset
from utils.tools import AverageMeter, collate_fn


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear = nn.Linear(config.input_dim, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.act = GELUActivation()

    def forward(self,
                x=None):
        hidden_state = self.linear(x)
        hidden_state = self.LayerNorm(hidden_state)
        hidden_state = self.dropout(hidden_state)
        hidden_state = self.act(hidden_state)
        return hidden_state


class PretrainGraphTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        config = BertConfig.from_pretrained(args.torch_model_dir)
        config.use_relative_position = False
        config.input_dim = 17
        config.hidden_size = 128
        config.intermediate_size = 512
        config.num_attention_heads = 8
        config.num_hidden_layers = 2
        self.x_embedding = MLP(config)
        self.edge_type_embedding = nn.Embedding(12, config.hidden_size)
        self.timestamp_embedding = nn.Embedding(579, config.hidden_size)

        config.vocab_size = 4
        config.max_position_embeddings = 762
        self.node_transformer = BertEncoder(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classification = nn.Linear(config.hidden_size, 4)
        self.bilstm = nn.LSTM(input_size=config.hidden_size, hidden_size=config.hidden_size // 2, num_layers=1, bidirectional=True)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self,
                x=None,
                start_edge_timestamp=None,
                end_edge_timestamp=None,
                edge_start_type=None,
                edge_end_type=None,
                back_x=None,
                front_x=None,
                y=None,
                output_hidden_states=True,
                attention_mask=None):
        x_hidden_state = self.x_embedding(x)
        back_x_hidden_state = self.x_embedding(back_x)
        front_x_hidden_state = self.x_embedding(front_x)
        edge_start_hidden_state = self.edge_type_embedding(edge_start_type)
        edge_end_hidden_state = self.edge_type_embedding(edge_end_type)
        start_timestamp_hidden_state = self.timestamp_embedding(start_edge_timestamp)
        end_timestamp_hidden_state = self.timestamp_embedding(end_edge_timestamp)

        """
        Concatenate X to its preceding and following nodes.
        Add temporal information so that the output can contain both directionality and temporal information.
        The relationship between two points is mainly based on Transformer.

        """
        x_nodes_embedding = back_x_hidden_state + edge_start_hidden_state + start_timestamp_hidden_state
        nodes_x_embedding = front_x_hidden_state + edge_end_hidden_state + end_timestamp_hidden_state
        nodes_x_nodes_emb = torch.cat([nodes_x_embedding, x_hidden_state.unsqueeze(1), x_nodes_embedding], dim=1)
        
        input_shape = nodes_x_nodes_emb.size()[:-1]
        batch_size, seq_length = input_shape
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=nodes_x_nodes_emb.device)
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        hidden_state = self.node_transformer(
            nodes_x_nodes_emb,
            attention_mask=extended_attention_mask,
            output_hidden_states=output_hidden_states
        ).last_hidden_state.mean(1)

        outputs, _ = self.bilstm(hidden_state)  # Processing through BiLSTM
        outputs = self.dropout(outputs)  # Applying dropout on the BiLSTM output
        logits = self.classification(outputs)  # Classifying the processed output

        return_dict = {'logits': logits}
        if y is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, y)
            return_dict['loss'] = loss.mean()
        return return_dict

    def get_extended_attention_mask(
            self, attention_mask: Tensor, input_shape: Tuple[int]
    ) -> Tensor:

        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        extended_attention_mask = extended_attention_mask.to(dtype=torch.float)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask


def pretrain(args):
    dataset = GraphDataset(args)
    train_dataloader = DataLoader(dataset=dataset,
                                  batch_size=args.train_batch_size,
                                  shuffle=True,
                                  num_workers=args.num_works,
                                  collate_fn=collate_fn)
    train_mask = dataset.train_mask.tolist()
    valid_index = random.sample(train_mask, 10000)
    val_dataset = torch.utils.data.Subset(dataset=dataset, indices=valid_index)
    valid_dataloader = DataLoader(dataset=val_dataset,
                                  batch_size=args.not_train_batch_size,
                                  shuffle=False,
                                  num_workers=args.num_works,
                                  collate_fn=collate_fn)
    model = PretrainGraphTransformer(args)
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
        with tqdm(train_dataloader, unit_scale=True, desc=f'epoch {epoch} pretrain', colour='blue') as pbar:
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

                    torch.nn.utils.clip_grad_norm_(model.parameters(), 3)

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
        torch.save(model.state_dict(),
                   os.path.join(args.save_model_dir,
                                f"auc_{valid_res['auc']:.5f}_loss_{valid_res['loss']:.5f}_epoch_{epoch}_pretrain_model.bin"))
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
    pretrain(args)
