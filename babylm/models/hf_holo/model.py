from typing import Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from . import hrr
from .config import HFHoloConfig


class SelfAttention(nn.Module):
    def __init__(self, model_dims):
        super().__init__()
        self.model_dims = model_dims
        self.queries = nn.Linear(model_dims, model_dims, bias=False)
        self.keys = nn.Linear(model_dims, model_dims, bias=False)
        self.values = nn.Linear(model_dims, model_dims, bias=False)

        self.init_weights()

    def forward(self, x, causal=True, mask=None):
        q = self.queries(x)
        k = self.keys(x)
        v = self.values(x)
        values_hat = hrr.key_value_query(k, v, q, causal=True, norm=True)
        return values_hat

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        https://www.cs.toronto.edu/~mvolkovs/ICML2020_tfixup.pdf
        """
        initializer_range = 1. / np.sqrt(self.model_dims)

        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_uniform_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class Rebind(nn.Module):
    def __init__(self, model_dims):
        super().__init__()
        self.rebind_a = nn.Linear(model_dims, model_dims, bias=False)
        self.rebind_b = nn.Linear(model_dims, model_dims, bias=False)

    def forward(self, x):
        rebind_a = self.rebind_a(x)
        rebind_b = self.rebind_b(x)
        x = hrr.rebind(x, rebind_a, rebind_b)
        # values_a = hrr.unbind(x, rebind_a)
        # x = x - hrr.bind(rebind_a, values_a) + hrr.bind(rebind_b, values_a)
        return x


class Transform(nn.Module):
    def __init__(self, model_dims):
        super().__init__()
        self.keys = nn.Linear(model_dims, model_dims, bias=False)
        self.ff_net = nn.Sequential(
            nn.Linear(2 * model_dims, 4 * model_dims),
            nn.ReLU(),
            nn.Linear(4 * model_dims, 2 * model_dims),
        )

    def forward(self, x):
        keys = self.keys(x)
        x = hrr.transform(x, keys, self.ff_net)
        # values = hrr.unbind(x, keys)
        # values_transformed = self.ff_net(values)
        # x = x - hrr.bind(keys, values) + hrr.bind(keys, values_transformed)
        return x


class HoloLayer(nn.Module):
    def __init__(self, model_dims, gain_init=1.):
        super().__init__()
        self.self_attention = SelfAttention(model_dims)
        self.rebind = Rebind(model_dims)
        self.transform = Transform(model_dims)

        self.gain = nn.Parameter(torch.full((1,), gain_init))

    def forward(self, x, mask=None, labels=None):
        values_hat = self.self_attention(x)
        x = x + values_hat
        # x = self.rebind(x)
        # x = self.transform(x)
        return x


class FullHoloLayer(nn.Module):
    def __init__(self, model_dims, gain_init=1., is_causal=True):
        super().__init__()
        self.queries = nn.Linear(model_dims, model_dims, bias=False)
        self.keys = nn.Linear(model_dims, model_dims, bias=False)
        self.values = nn.Linear(model_dims, model_dims, bias=False)

        self.transform_source = nn.Linear(model_dims, model_dims, bias=False)
        self.transform_target = nn.Linear(model_dims, model_dims, bias=False)
        self.ff_net = nn.Sequential(
            nn.Linear(2 * model_dims, 4 * model_dims),
            nn.ReLU(),
            nn.Linear(4 * model_dims, 2 * model_dims),
        )
        self.is_causal = is_causal

    def forward(self, x, mask=None, labels=None):
        x_hat = hrr.fft(x)
        # self attention
        q = self.queries(x)
        k = hrr.fft(self.keys(x))
        v = hrr.fft(self.values(x))
        attention_values = hrr.fft(hrr.inverse(q)) * k * v
        if self.is_causal:
            attention_values = attention_values.cumsum(-2)
        else:
            attention_values = attention_values.sum(-2, keepdim=True)

        x_hat = x_hat + attention_values
        # MLP transform
        source = self.transform_source(x)
        source_inv = hrr.fft(hrr.inverse(source))
        source = hrr.fft(source)
        target = hrr.fft(self.transform_target(x))
        t_args = source_inv * x_hat
        transformed = hrr.wrap_real_transform(self.ff_net, t_args)
        x_hat = x_hat + target * transformed - source * t_args

        return torch.real(hrr.ifft(x))


class HoloDecoder(PreTrainedModel):
    config_class = HFHoloConfig

    def __init__(self, config):
        super().__init__(config)
        layer_class = HoloLayer
        self.layers = nn.ModuleList([
            layer_class(config.model_dims)
            for _ in range(config.num_hidden_layers)
        ])

    def forward(self, x, mask=None, labels=None):
        loss = 0.

        for layer in self.layers:
            x = layer(x, mask=mask)

            if labels is not None:
                layer_loss = x.square().sum()
                loss = loss# + layer_loss
            print(list(map(float, (x.min(), x.mean(), x.std(), x.max()))))

        if labels is None:
            return x
        else:
            return x, loss


class HFHolo(PreTrainedModel):
    config_class = HFHoloConfig

    def __init__(self, config):
        super().__init__(config)
        self.decoder = HoloDecoder(config)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.model_dims)
        self.input_embedding = nn.Embedding(config.vocab_size, config.model_dims)
        self.input_embedding.weight.data.copy_(
            hrr.init(self.input_embedding.weight.shape),
        )
        self.position_embedding.weight.data.copy_(
            hrr.init(self.position_embedding.weight.shape),
        )
        self.predict_token = nn.Linear(config.model_dims, config.vocab_size, bias=False)
        self.predict_token.weight.data.copy_(
            hrr.init(self.predict_token.weight.shape),
        )
        self.register_buffer('result_vector', hrr.init((config.model_dims,)))
        self.cleanup_kv = CleanUpKV(config.model_dims)

        freeze_list = []
        if not config.learn_input_embs:
            freeze_list += [self.input_embedding, self.position_embedding]
        if not config.learn_output_embs:
            freeze_list += [self.predict_token]
        if freeze_list:
            list(map(lambda x: x.requires_grad_(False), freeze_list))

        # self.post_init()

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
        position_ids = torch.arange(tokens.shape[1]).long().to(tokens.device)
        position_ids = position_ids[None, :].repeat(tokens.shape[0], 1)
        positions = self.position_embedding(position_ids)
        inputs = hrr.bind(tokens, positions)
        #inputs = tokens + positions
        feats = self.decoder(inputs, labels=labels)
        if labels is not None:
            feats, decoder_loss = feats

        feats = hrr.unbind(feats, self.result_vector)
        feats = self.cleanup_kv(feats)
        feats = feats / (torch.linalg.norm(feats, dim=-1, keepdim=True) + 1e-9)
        logits = self.predict_token(feats)

        loss = 0.
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].transpose(-1, -2), labels[:, 1:])

            # logit_loss = logits.square().sum() * 0.01  # + logits.abs().sum() * 0.01
            # loss = loss + logit_loss  # + decoder_loss

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


class CleanupBlock(nn.Module):
    def __init__(self, model_dims, num_layers):
        super().__init__()
        self.modules = nn.ModuleList([
            CleanUpKV(model_dims) for _ in range(num_layers)
        ])

    def forward(self, x):
        for module in self.modules:
            x = module(x)


class CleanUpKV(nn.Module):
    def __init__(self, model_dims, num_support=256):
        super().__init__()
        # self.keys = nn.Parameter(hrr.init((num_support, model_dims)), requires_grad=True)
        # self.values = nn.Parameter(hrr.init((num_support, model_dims)), requires_grad=True)
        self.keys = nn.Parameter(torch.randn((num_support, model_dims)), requires_grad=True)
        self.values = nn.Parameter(torch.randn((num_support, model_dims)), requires_grad=True)

    def forward(self, x):
        x = x / (torch.linalg.norm(x, dim=-1, keepdim=True) + 1e-9)
        scores = torch.matmul(x, self.keys.transpose(-1, -2))
        weights = torch.softmax(scores, dim=-1)
        values = torch.matmul(weights, self.values)
        # dx = x - values
        # print('kv dx', list(map(float, [dx.min(), dx.mean(), dx.std(), dx.max()])))
        return values


AutoModel.register(HFHoloConfig, HoloDecoder)
AutoModelForCausalLM.register(HFHoloConfig, HFHolo)
