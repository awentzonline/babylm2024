from typing import Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, PreTrainedModel

from . import hrr
from .config import HFHoloConfig


class HoloLayer(nn.Module):
    def __init__(self, model_dims):
        super().__init__()

        self.keys = nn.Linear(model_dims, model_dims)
        self.values = nn.Linear(model_dims, model_dims)
        self.queries = nn.Linear(model_dims, model_dims)

    def forward(self, x, labels=None):
        xh = x #hrr.fft(x)

        xh_keys = hrr.fft(self.keys(xh))
        xh_values = hrr.fft(self.values(xh))
        xh_queries = hrr.fft(self.queries(xh))
        inv_xh_queries = hrr.inverse(xh_queries)

        kv = torch.multiply(xh_keys, xh_values).cumsum(dim=1)
        xh_queried = torch.multiply(kv, inv_xh_queries)
        # xh_queried = xh_queried / np.sqrt(x.shape[-1])
        xh_queried = xh_queried / (1e-9 + torch.norm(xh_queried, dim=-1, keepdim=True))
        values = torch.real(hrr.ifft(xh_queried)) / np.sqrt(x.shape[-1])
        return values


class HoloDecoder(PreTrainedModel):
    config_class = HFHoloConfig

    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList([
            HoloLayer(config.model_dims)
            for _ in range(config.num_hidden_layers)
        ])

    def forward(self, x, labels=None):
        for layer in self.layers:
            x = x + layer(x)
        return x


class HFHolo(PreTrainedModel):
    config_class = HFHoloConfig

    def __init__(self, config):
        super().__init__(config)
        self.decoder = HoloDecoder(config)
        self.input_embedding = nn.Embedding(config.vocab_size, config.model_dims)
        self.predict_token = nn.Linear(config.model_dims, config.vocab_size)

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
        feats = self.decoder(tokens)
        logits = self.predict_token(feats)
        if labels is not None:
            loss = F.cross_entropy(logits.transpose(-1, -2), labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}

    def get_input_embeddings(self):
        return self.input_embedding

    def set_input_embeddings(self, new_embs):
        self.input_embedding = new_embs


AutoModel.register(HFHoloConfig, HoloDecoder)
AutoModelForCausalLM.register(HFHoloConfig, HFHolo)
