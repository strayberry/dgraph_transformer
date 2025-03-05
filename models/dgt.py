import torch
import numpy as np
from torch import Tensor, nn
from transformers.models.bert.modeling_bert import BertEncoder
from transformers.models.bert.configuration_bert import BertConfig
from transformers.activations import GELUActivation
from typing import Tuple


dgraph_fin_config = {
    'input_dim': 17,
    'hidden_size': 128,
    'intermediate_size': 512,
    'num_attention_heads': 8,
    'num_hidden_layers': 2,
    'edge_type_embedding': 12,
    'timestamp_embedding': 579,
    'vocab_size': 4,
    'max_position_embeddings': 762,
    'classification': 4
}

elliptic_config = {
    'input_dim': 167,
    'hidden_size': 128,
    'intermediate_size': 512,
    'num_attention_heads': 8,
    'num_hidden_layers': 2,
    'edge_type_embedding': 2,
    'timestamp_embedding': 49,
    'vocab_size': 4,
    'max_position_embeddings': 762,
    'classification': 2
}


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


class GraphTransformer(nn.Module):
    def __init__(self, args):
        self.args = args
        super().__init__()
        if self.args.dataset_name == 'DGraphFin':
            model_config = dgraph_fin_config.copy()
        elif self.args.dataset_name == 'Elliptic':
            model_config = elliptic_config.copy()
        # Ablation: Without using pre-trained weights
        if self.args.use_pretraining:
            config = BertConfig.from_pretrained(self.args.torch_model_dir)
        else:
            config = BertConfig()
        config.use_relative_position = False
        config.input_dim = model_config['input_dim']
        config.hidden_size = model_config['hidden_size']
        config.intermediate_size = model_config['intermediate_size']
        config.num_attention_heads = model_config['num_attention_heads']
        config.num_hidden_layers = model_config['num_hidden_layers']
        self.x_embedding = MLP(config)
        self.edge_type_embedding = nn.Embedding(model_config['edge_type_embedding'], config.hidden_size)
        self.timestamp_embedding = nn.Embedding(model_config['timestamp_embedding'], config.hidden_size)

        config.vocab_size = model_config['vocab_size']
        config.max_position_embeddings = model_config['max_position_embeddings']
        self.node_transformer = BertEncoder(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classification = nn.Linear(config.hidden_size, model_config['classification'])
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
        # Ablation: Without using temporal features
        if self.args.use_time_features:
            start_timestamp_hidden_state = self.timestamp_embedding(start_edge_timestamp)
            end_timestamp_hidden_state = self.timestamp_embedding(end_edge_timestamp)
        else:
            start_timestamp_hidden_state = torch.zeros_like(back_x_hidden_state)
            end_timestamp_hidden_state = torch.zeros_like(front_x_hidden_state)
        
        start_timestamp_hidden_state = self.timestamp_embedding(start_edge_timestamp)
        end_timestamp_hidden_state = self.timestamp_embedding(end_edge_timestamp)

        """
        Concatenate X to its preceding and following nodes.
        Add temporal information so that the output can contain both directionality and temporal information.
        The relationship between two points is mainly based on Transformer.
        
        """
        x_nodes_embedding = back_x_hidden_state + edge_start_hidden_state + start_timestamp_hidden_state
        nodes_x_embedding = front_x_hidden_state + edge_end_hidden_state + end_timestamp_hidden_state
        
        # Ablation: Without concatenating in the order of edge generation
        if self.args.use_time_ordering:
            nodes_x_nodes_emb = torch.cat([nodes_x_embedding, x_hidden_state.unsqueeze(1), x_nodes_embedding], dim=1)
        else:
            nodes_x_nodes_emb = torch.cat([x_hidden_state.unsqueeze(1), x_nodes_embedding, nodes_x_embedding], dim=1)

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
        
        # Ablation: Without using BiLSTM
        if self.args.use_bilstm:
            outputs, _ = self.bilstm(hidden_state)
        else:
            outputs = hidden_state
        outputs = self.dropout(outputs)
        logits = self.classification(outputs)

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
