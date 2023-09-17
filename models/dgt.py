import torch
from torch import Tensor, nn
from transformers.models.bert.modeling_bert import BertEncoder
from transformers.models.bert.configuration_bert import BertConfig
from transformers.activations import GELUActivation
from typing import Tuple


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
        super().__init__()
        config = BertConfig.from_pretrained(args.torch_model_dir)
        config.use_relative_position = False
        config.input_dim = 167 #dg 17
        config.hidden_size = 128 #dg 128
        config.intermediate_size = 512  #dg 512
        config.num_attention_heads = 8 #dg 8
        config.num_hidden_layers = 2   #dg 2
        self.x_embedding = MLP(config)
        self.edge_type_embedding = nn.Embedding(2, config.hidden_size) #dg 12
        self.timestamp_embedding = nn.Embedding(49, config.hidden_size)   #dg 579

        config.vocab_size = 4 #dg 4
        config.max_position_embeddings = 762 #dg 762
        self.node_transformer = BertEncoder(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classification1 = nn.Linear(config.hidden_size, 2)  #dg 2
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

        # 把x拼接在在x后面的节点的前面，把x拼接在在x前面的节点的后面，加上时间信息，使输出即能包含方向又能包含时间(方向，两点之间关系主要是transformer)
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

        outputs, _ = self.bilstm(hidden_state)  # Shape: (batch_size, seq_length, hidden_size)
        hidden_state = outputs.mean(1)  # Get the mean of the outputs over sequence length

        hidden_state = self.dropout(hidden_state)
        logits = self.classification1(hidden_state)

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
