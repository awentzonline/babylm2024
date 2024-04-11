import math
from typing import Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from .config import HFVanConfig


class VanAttention(nn.Module):
    def __init__(self, model_dims, heads):
        super().__init__()
        assert model_dims % heads == 0, 'Model dimensionality should be an integer multiple of heads'
        self.head_dims = model_dims // heads
        self.num_heads = heads
        self.model_dims = model_dims
        self.keys = nn.Linear(model_dims, model_dims, bias=False)
        self.values = nn.Linear(model_dims, model_dims, bias=False)
        self.queries = nn.Linear(model_dims, model_dims, bias=False)

    def forward(self, x_query, x_key, x_value, mask=None):
        q = self.split_heads(self.queries(x_query))
        k = self.split_heads(self.keys(x_key))
        v = self.split_heads(self.values(x_value))

        scores = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(x_query.shape[-1])
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        result = torch.matmul(weights, v)

        result = self.combine_heads(result)
        return result

    def split_heads(self, x):
        batch, seq, dims = x.shape
        return x.view(batch, seq, self.num_heads, self.head_dims).transpose(1, 2)

    def combine_heads(self, x):
        batch, _, seq = x.shape[:3]
        return x.transpose(1, 2).contiguous().view(batch, seq, self.model_dims)


class VanLayer(nn.Module):
    def __init__(self, model_dims, heads):
        super().__init__()
        self.norm_pre_attention = nn.LayerNorm(model_dims)
        self.attention = VanAttention(model_dims, heads)
        self.ff_net = nn.Sequential(
            nn.LayerNorm(model_dims),
            nn.Linear(model_dims, 4 * model_dims),
            nn.ReLU(),
            nn.Linear(4 * model_dims, model_dims),
        )

    def forward(self, x, labels=None, mask=None):
        normed_x = self.norm_pre_attention(x)
        attended = self.attention(normed_x, normed_x, normed_x, mask=mask)
        x = x + attended
        ffed = self.ff_net(x)
        return x + ffed


class VanDecoder(PreTrainedModel):
    config_class = HFVanConfig

    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList([
            VanLayer(config.model_dims, config.heads)
            for _ in range(config.num_hidden_layers)
        ])

    def forward(self, x, labels=None, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x


class HFVan(PreTrainedModel):
    config_class = HFVanConfig

    def __init__(self, config):
        super().__init__(config)
        self.decoder = VanDecoder(config)
        self.input_embedding = nn.Embedding(config.vocab_size, config.model_dims)
        self.predict_token = nn.Linear(config.model_dims, config.vocab_size)
        self.positional_encoding = PositionalEncoding(config.model_dims, config.max_seq_length)

    def forward(
        self,
        input_ids: Optional = None,
        attention_mask: Optional = None,
        token_type_ids: Optional = None,
        position_ids: Optional = None,
        head_mask: Optional = None,
        inputs_embeds: Optional = None,
        encoder_hidden_states: Optional = None,
        encoder_attention_mask: Optional = None,
        labels: Optional = None,
        past_key_values: Optional = None,
        use_cache: Optional = None,
        output_attentions: Optional = None,
        output_hidden_states: Optional = None,
        return_dict: Optional = None,
    ):
        tokens = self.input_embedding(input_ids)
        tokens = self.positional_encoding(tokens)
        attention_mask = self.make_causal_mask(tokens)
        feats = self.decoder(tokens, mask=attention_mask)
        logits = self.predict_token(feats)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.transpose(-1, -2), labels)

        if return_dict is not None and not return_dict:
            output = (logits, feats)
            output = (loss,) + output if loss else output
            return output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=None,  # transformer_outputs.past_key_values,
            hidden_states=feats,
            attentions=None,  # transformer_outputs.attentions,
            cross_attentions=None,  # transformer_outputs.cross_attentions,
        )

    def make_causal_mask(self, x):
        seq = x.shape[2]
        mask = torch.ones((seq, seq), device=x.device)
        mask = torch.tril(mask).bool()
        return mask

    def get_input_embeddings(self):
        return self.input_embedding

    def set_input_embeddings(self, new_embs):
        self.input_embedding = new_embs

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        """
        https://github.com/huggingface/transformers/blob/08a194fcd615dcf9406a7e319d637cc303097f46/src/transformers/models/gpt2/modeling_gpt2.py#L1227C5-L1272C28
        """
        token_type_ids = kwargs.get("token_type_ids", None)
        # Omit tokens covered by past_key_values
        if past_key_values:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -input_ids.shape[1] :]

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]
        else:
            position_ids = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
        )

        return model_inputs


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


AutoModel.register(HFVanConfig, VanDecoder)
AutoModelForCausalLM.register(HFVanConfig, HFVan)
