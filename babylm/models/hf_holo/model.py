from typing import Optional

import mup
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from babylm.models.hf_holo import hrr
from babylm.models.hf_holo.config import HFHoloConfig
from babylm.models.hf_vanilla.model import VanAttention


class Abs(nn.Module):
    def forward(self, x):
        return torch.abs(x)


class RRSelfAttention(nn.Module):
    def __init__(self, model_dims, **kwargs):
        super().__init__()
        self.model_dims = model_dims
        self.queries = nn.Linear(model_dims, model_dims, bias=False)
        self.keys = nn.Linear(model_dims, model_dims, bias=False)
        self.values = nn.Linear(model_dims, model_dims, bias=False)
        self.output = nn.Linear(model_dims, model_dims, bias=False)

        self.kvt = nn.Identity()
        self.vhat = nn.Identity()

    def forward(self, x, causal=True, mask=None):
        q = self.queries(x)
        k = self.keys(x)
        v = self.values(x)
        kvt = k * v
        if causal:
            # denom = 1 #/ x.shape[-1] #* torch.sqrt(torch.arange(1, x.shape[-2] + 1, device=x.device))[None, ..., None]
            denom = np.sqrt(x.shape[-1]) / x.shape[-2] #/ torch.arange(1, x.shape[-2] + 1, device=x.device)[None, ..., None]
            kvt = kvt.cumsum(-2) * denom
        else:
            kvt = kvt.sum(-2, keepdim=True) / x.shape[-1]
        kvt = self.kvt(kvt)
        values_hat = q * kvt
        values_hat = self.vhat(values_hat)
        # values_hat = values_hat / (torch.linalg.norm(values_hat, dim=-1, keepdim=True) + 1e-9)
        return self.output(values_hat)


class HRRSelfAttention(nn.Module):
    def __init__(self, model_dims, num_heads=8, **kwargs):
        super().__init__()
        self.model_dims = model_dims
        self.num_heads = num_heads
        assert model_dims % num_heads == 0, f'Num heads ({num_heads}) incompatible with model dims ({model_dims})'
        self.head_dims = model_dims // num_heads
        self.qkv = nn.Linear(self.model_dims, 3 * self.model_dims, bias=False)
        # self.queries = nn.Linear(self.head_dims, self.head_dims, bias=False)
        # self.keys = nn.Linear(self.head_dims, self.head_dims, bias=False)
        # self.values = nn.Linear(self.head_dims, self.head_dims, bias=False)
        self.output = nn.Linear(model_dims, model_dims, bias=False)
        self.dropout = nn.Dropout(0.1)
        self.precum = nn.Identity()
        self.postcum = nn.Identity()

    def forward(self, x, causal=True, mask=None):
        batch_size, seq_len = x.shape[:2]
        q, k, v = self.qkv(x).split(self.model_dims, dim=2)
        if self.num_heads > 1:
            q = q.view(batch_size, seq_len, self.num_heads, self.head_dims)
            k = k.view(batch_size, seq_len, self.num_heads, self.head_dims)
            v = v.view(batch_size, seq_len, self.num_heads, self.head_dims)
        # q = self.queries(x)
        # k = self.keys(x)
        # v = self.values(x)
        # k, v, inv_q = hrr.fft(k), hrr.fft(v), hrr.fft(q)
        # kv = hrr.bind(k, v)
        # kvt = self.precum(kv)
        # kvt = self.kvt(kv.cumsum(-2))
        # kvt = self.postcum(kvt)
        # values_hat = hrr.unbind(kvt, q)
        values_hat = hrr.key_value_query_lin(k, v, q, causal=causal, norm=False)
        if self.num_heads > 1:
            values_hat = values_hat.view(batch_size, seq_len, self.model_dims)
        # values_hat = hrr.unit_projection(values_hat)
        # values_hat = values_hat / (2 * v.shape[-1] ** 2)
        #return values_hat
        return self.dropout(self.output(values_hat))


class HRRSimpleSelfAttention(nn.Module):
    """
    Assumes the model can learn appropriate key/value bindings and the inverse queries.
    """
    def __init__(self, model_dims, num_heads=8):
        super().__init__()
        self.model_dims = model_dims
        self.num_heads = num_heads
        assert model_dims % num_heads == 0, f'Num heads ({num_heads}) incompatible with model dims ({model_dims})'
        self.head_dims = model_dims // num_heads
        self.queries = nn.Linear(self.model_dims, self.model_dims, bias=False)
        self.keyvalues = nn.Linear(self.model_dims, self.model_dims, bias=False)
        self.output = nn.Linear(model_dims, model_dims, bias=False)

    def forward(self, x, causal=True, mask=None):
        batch_size, seq_len = x.shape[:2]
        q = self.queries(x)
        kv = self.keyvalues(x)
        if self.num_heads > 1:
            q = q.view(batch_size, seq_len, self.num_heads, self.head_dims)
            kv = kv.view(batch_size, seq_len, self.num_heads, self.head_dims)
        q, kv = hrr.fft(q), hrr.fft(kv)
        q = q / torch.norm(q, dim=-1, keepdim=True)
        kv = kv / torch.norm(kv, dim=-1, keepdim=True)
        if causal:
            kv = torch.cumsum(kv, dim=1)
        else:
            kv = torch.sum(kv, dim=1, keepdim=True)
        qv = q * kv
        qv = qv
        values_hat = hrr.ifft(qv)
        if self.num_heads > 1:
            values_hat = values_hat.view(batch_size, seq_len, self.model_dims)
        values_hat = self.output(values_hat)
        return values_hat


class HRRBindingSelfAttention(nn.Module):
    """
    Assumes the model can learn appropriate key/value bindings and the inverse queries.
    The queries/keyvalues are hrr vectors instead of linear transformations.
    """
    def __init__(self, model_dims, num_heads=8):
        super().__init__()
        self.model_dims = model_dims
        self.num_heads = num_heads
        assert model_dims % num_heads == 0, f'Num heads ({num_heads}) incompatible with model dims ({model_dims})'
        self.head_dims = model_dims // num_heads
        # self.queries = nn.Parameter(
        #     hrr.init((1, 1, model_dims)), requires_grad=True
        # )
        # self.keyvalues = nn.Parameter(
        #     hrr.init((1, 1, model_dims)), requires_grad=True
        # )
        self.queries = nn.Parameter(
            torch.randn((1, 1, self.model_dims // 2 + 1)), requires_grad=True
        )
        self.keyvalues = nn.Parameter(
            torch.randn((1, 1, self.model_dims // 2 + 1)), requires_grad=True
        )
        self.output = nn.Linear(model_dims, model_dims, bias=False)

    def forward(self, x, causal=True, mask=None):
        batch_size, seq_len = x.shape[:2]
        #vq, vkv = hrr.fft(self.queries), hrr.fft(self.keyvalues)
        vq, vkv = self.queries, self.keyvalues
        if self.num_heads > 1:
            x = x.view(batch_size, seq_len, self.num_heads, self.head_dims)
            vq = vq.view(batch_size, seq_len, self.num_heads, self.head_dims)
            vkv = vkv.view(batch_size, seq_len, self.num_heads, self.head_dims)
        xhat = hrr.fft(x)
        q = xhat * vq
        kv = xhat * vkv    
        if causal:
            kv = torch.cumsum(kv, dim=1)
        else:
            kv = torch.sum(kv, dim=1, keepdim=True)
        qv = q * kv
        values_hat = hrr.ifft(qv)
        if self.num_heads > 1:
            values_hat = values_hat.view(batch_size, seq_len, self.model_dims)
        values_hat = self.output(values_hat)
        return values_hat


class RedundantHRRSelfAttention(nn.Module):
    """
    Use the mean of mutiple permuted copies of the data to reduce noise in the representation.
    Associative Long Short-Term Memory: https://arxiv.org/pdf/1602.03032v2
    """
    def __init__(self, model_dims, num_copies=10, perm_freq=True, **kwargs):
        super().__init__()
        self.model_dims = model_dims
        self.queries = nn.Linear(model_dims, model_dims, bias=False)
        self.keys = nn.Linear(model_dims, model_dims, bias=False)
        self.values = nn.Linear(model_dims, model_dims, bias=False)
        self.output = nn.Linear(model_dims, model_dims, bias=False)
        self.num_copies = num_copies
        self.perm_freq = perm_freq
        num_perms = model_dims // 2 + 1 if perm_freq else model_dims
        self.register_buffer('permutations', torch.randn(num_copies, num_perms).argsort(-1))

    def forward(self, x, causal=True, mask=None):
        q = self.queries(x)
        k = self.keys(x)
        v = self.values(x)
        if self.perm_freq:
            values_hat = hrr.perm_key_value_query(k, v, q, self.permutations, causal=causal)
        else:
            q = q[..., self.permutations].permute(2, 0, 1, 3)
            k = k[..., self.permutations].permute(2, 0, 1, 3)
            v = v[None, ...]
            values_hat = hrr.key_value_query(k, v, q, causal=causal)
            values_hat = values_hat.mean(0)
        values_hat = self.output(values_hat)
        return values_hat


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
            nn.Linear(2 * model_dims, 4 * model_dims, bias=False),
            nn.ReLU(),
            nn.Linear(4 * model_dims, 2 * model_dims, bias=False),
        )

    def forward(self, x):
        keys = self.keys(x)
        x = hrr.transform(x, keys, self.ff_net)
        # values = hrr.unbind(x, keys)
        # values_transformed = self.ff_net(values)
        # x = x - hrr.bind(keys, values) + hrr.bind(keys, values_transformed)
        return x


class MLP(nn.Module):
    def __init__(self, model_dims, ff_dims=None):
        super().__init__()
        ff_dims = ff_dims or model_dims * 4
        self.net = nn.Sequential(
            nn.Linear(model_dims, ff_dims, bias=False),
            # Abs(),
            nn.GELU(),
        )
        # move output to its own thing so we can target it for initialization
        self.dropout = nn.Dropout(0.1)
        self.output = nn.Linear(ff_dims, model_dims, bias=False)

    def forward(self, x):
        return self.dropout(self.output(self.net(x)))


class HoloLayer(nn.Module):
    def __init__(
        self, model_dims, rezero=False, attention_class=HRRSelfAttention,
        use_norm_bias=False, num_heads=8, num_layers=1,
    ):
        super().__init__()
        self.self_attention = attention_class(model_dims, num_heads=num_heads)
        #self.mlp = MLP(model_dims)
        self.num_layers = num_layers

        if rezero:
            self.norm_attn = nn.Identity()
            self.norm_mlp = nn.Identity()
            self.gain = nn.Parameter(torch.zeros(1))
        else:
            # self.norm_attn = nn.Identity()
            # self.norm_mlp = nn.Identity()
            self.norm_attn = HRRLayerNorm(model_dims)
            #self.norm_mlp = HRRLayerNorm(model_dims)
            # self.norm_attn = nn.LayerNorm(model_dims, bias=use_norm_bias)
            # self.norm_mlp = nn.LayerNorm(model_dims, bias=use_norm_bias)
            self.gain = 1.# / np.sqrt(num_layers)

    def forward(self, x, mask=None, labels=None):
        values_attn = self.self_attention(self.norm_attn(x), causal=True)
        x = x + self.gain * values_attn # torch.abs(values_attn)
        #x = x + 1. * values_attn # torch.abs(values_attn)
        #values_mlp = self.mlp(self.norm_mlp(x))
        #x = x + self.gain * values_mlp #torch.abs(values_mlp)
        return x


class HoloDecoder(PreTrainedModel):
    config_class = HFHoloConfig

    def __init__(self, config):
        super().__init__(config)
        layer_class = HoloLayer
        attention_class = dict(
            rr=RRSelfAttention,
            hrr=HRRSelfAttention,
            rhrr=RedundantHRRSelfAttention,
            bhrr=HRRBindingSelfAttention,
            shrr=HRRSimpleSelfAttention,
            van=VanAttention,
        )[config.attention_class]

        self.layers = nn.ModuleList([
            layer_class(
                config.model_dims, attention_class=attention_class,
                rezero=config.rezero, use_norm_bias=config.use_norm_bias,
                num_heads=config.num_heads, num_layers=config.num_hidden_layers,
            )
            for _ in range(config.num_hidden_layers)
        ])

    def forward(self, x, mask=None, labels=None):
        loss = torch.tensor(0., device=x.device)

        for layer in self.layers:
            x = layer(x, mask=mask)

            if labels is not None:
                layer_loss = x.square().sum()
                loss = loss# + layer_loss
            # print(list(map(float, (x.min(), x.mean(), x.std(), x.max()))))

        if labels is None:
            return x
        else:
            return x, loss


class HFHolo(PreTrainedModel):
    config_class = HFHoloConfig

    def __init__(self, config):
        super().__init__(config)
        #self.decoder = HoloDecoder(config)
        if config.position_embedding == 'learn':
            self.position_embedding = nn.Embedding(config.max_seq_len, config.model_dims)
        elif config.position_embedding == 'sin':
            self.position_embedding = PositionalEncoding(config.model_dims, config.max_seq_len)
        self.input_embedding = nn.Embedding(config.vocab_size, config.model_dims)
        # if config.attention_class == 'hrr':
        #     self.input_embedding.weight.data.copy_(
        #         hrr.init(self.input_embedding.weight.shape),
        #     )
        #     self.position_embedding.weight.data.copy_(
        #         hrr.init(self.position_embedding.weight.shape),
        #     )
        # self.predict_token = nn.Linear(config.model_dims, config.vocab_size, bias=False)
        # self.predict_token.weight.data.copy_(
        #     hrr.init(self.predict_token.weight.shape),
        # )
        if config.rezero:
            self.norm = nn.Identity()
        else:
            #self.norm = HRRLayerNorm(config.model_dims)
            self.norm = nn.LayerNorm(config.model_dims, bias=config.use_norm_bias)

        self.dropout = nn.Dropout(0.1)
        self.predict_token = mup.MuReadout(config.model_dims, config.vocab_size, bias=False)
        # self.result_vector = nn.Parameter(
        #     torch.randn((1, 1, config.model_dims // 2 + 1)), requires_grad=True
        # )
        # self.result_vector = nn.Parameter(
        #     hrr.init((1, 1, config.model_dims)), requires_grad=True
        # )
        # self.register_buffer('result_vector', hrr.init((config.model_dims,)).contiguous())
        # self.cleanup_kv = CleanUpKV(config.model_dims)
        # self.fuck = nn.Linear(config.model_dims, config.model_dims, bias=False)

        freeze_list = []
        if not config.learn_input_embs:
            freeze_list += [self.input_embedding, self.position_embedding]
        if not config.learn_output_embs:
            freeze_list += [self.predict_token]
        if freeze_list:
            list(map(lambda x: x.requires_grad_(False), freeze_list))

        self.loss_fn = {
            'xent': F.cross_entropy,
            'focal': focal_loss,
            'asl': ASLSingleLabel(),
        }[config.loss]

        self.post_init()

    def _init_weights(self, module):
        module.apply(self._init_module)

    def _init_module(self, module, readout_zero_init=False, query_zero_init=False):
        if isinstance(module, mup.MuReadout) and readout_zero_init:
            module.weight.data.zero_()
        elif isinstance(module, (nn.Linear, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617

            if hasattr(module.weight, 'infshape'):
                mup.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            else:
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            if self.config.hrr_embedding:
                module.weight.data.copy_(
                    hrr.init(module.weight.shape),
                )
            else:
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        hrr_std = 1 / self.config.model_dims
        # depth_std = self.config.initializer_range / np.sqrt(2 * self.config.num_hidden_layers)
        # for name, module in module.named_modules():
        #     for target_name in ('qkv',):
        #         if target_name in name and query_zero_init:
        #             if hasattr(module.weight, 'infshape'):
        #                 mup.init.normal_(module.weight, mean=0.0, std=hrr_std)
        #             else:
        #                 module.weight.data.normal_(mean=0.0, std=hrr_std)
        #             if module.bias is not None:
        #                 module.bias.data.zero_()

            # if "output" in name:
            #     if hasattr(module.weight, 'infshape'):
            #         mup.init.normal_(module.weight, mean=0.0, std=depth_std)
            #     else:
            #         module.weight.data.normal_(mean=0.0, std=depth_std)

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
        if self.config.hrr_embedding:
            inputs = hrr.bind(tokens, positions)
        else:
            inputs = tokens + positions
        #feats = self.decoder(inputs, labels=None)
        #feats = feats / (1e-8 + torch.norm(feats, dim=-1, keepdim=True))
        feats = inputs
        #feats = self.norm(feats)
        # feats = self.decoder(inputs, labels=labels)
        # feats = F.relu(feats)
        # feats = tokens #F.relu(tokens) #+ positions
        # feats = self.fuck(feats)
        # if labels is not None:
        #     feats, decoder_loss = feats

        #feats = hrr.bind(feats, self.result_vector)
        # feats = hrr.fft(feats)
        # feats = feats * self.result_vector
        # feats = hrr.ifft(feats)
        # feats = hrr.unbind(feats, self.result_vector)
        # feats = self.cleanup_kv(feats)
        # feats = feats / (torch.linalg.norm(feats, dim=-1, keepdim=True) + 1e-9)
        feats = self.dropout(feats)
        logits = self.predict_token(feats)
        # logits = logits / (1e-8 + torch.norm(logits, dim=-1, keepdim=True)) * self.config.vocab_size
        
        loss = 0.
        if labels is not None:
            preds = logits[:, :-1].contiguous().view(-1, logits.shape[-1])
            targets = labels[:, 1:].contiguous().view(-1)
            loss = self.loss_fn(preds, targets)

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

    @classmethod
    def mup_base_shapes(cls, filename=None, base_kwargs=None, delta_kwargs=None):
        if not hasattr(cls, '_mup_base_shapes'):
            print('getting muP base shapes')
            base_kwargs = base_kwargs or {}
            delta_kwargs = delta_kwargs or {}
            if not 'model_dims' in base_kwargs:
                base_kwargs['model_dims'] = 128
            if not 'model_dims' in delta_kwargs:
                delta_kwargs['model_dims'] = 256
            base_config = HFHoloConfig(
                **base_kwargs,
            )
            delta_config = HFHoloConfig(
                **delta_kwargs
            )
            base_model = HFHolo(config=base_config)
            delta_model = HFHolo(config=delta_config)
            base_shapes = mup.make_base_shapes(base_model, delta_model, savefile=filename)
            cls._mup_base_shapes = base_shapes
            del base_model
            del delta_model
            base_model = delta_model = None
        return cls._mup_base_shapes


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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, :x.size(1)]


def focal_loss(logits, targets, gamma=2.):
    """Adapted from https://github.com/artemmavrin/focal-loss/blob/master/src/focal_loss/_categorical_focal_loss.py"""
    probs = torch.softmax(logits, dim=-1)
    probs = probs[torch.arange(logits.shape[0]), targets]
    focal_modulation = (1 - probs) ** gamma
    xent_loss = F.cross_entropy(logits, targets, reduction='none')
    return torch.mean(focal_modulation * xent_loss)


class ASLSingleLabel(nn.Module):
    '''
    This loss is intended for single-label classification problems

    From: https://github.com/lnsmith54/CFL/blob/main/OLTR/loss/asl_focal_loss.py
    '''
    def __init__(self, gamma_pos=0, gamma_neg=4, eps: float = 0.1, reduction='mean'):
        super(ASLSingleLabel, self).__init__()

        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.targets_classes = []
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction
#        print("ASLSingleLabel: gamma_pos=", gamma_pos," gamma_neg=",gamma_neg, " eps=",eps)

    def forward(self, inputs, target):
        '''
        "input" dimensions: - (batch_size,number_classes)
        "target" dimensions: - (batch_size)
        '''
        num_classes = inputs.size()[-1]
        log_preds = self.logsoftmax(inputs)
        if len(list(target.size()))>1:
            target = torch.argmax(target, 1)
        self.targets_classes = torch.zeros_like(inputs).scatter_(1, target.long().unsqueeze(1), 1)

        # ASL weights
        targets = self.targets_classes
        anti_targets = 1 - targets
        xs_pos = torch.exp(log_preds)
        xs_neg = 1 - xs_pos
        xs_pos = xs_pos * targets
        xs_neg = xs_neg * anti_targets
        asymmetric_w = torch.pow(1 - xs_pos - xs_neg,
                                 self.gamma_pos * targets + self.gamma_neg * anti_targets)
        log_preds = log_preds * asymmetric_w

        if self.eps > 0:  # label smoothing
            self.targets_classes = self.targets_classes.mul(1 - self.eps).add(self.eps / num_classes)

        # loss calculation
        loss = - self.targets_classes.mul(log_preds)

        loss = loss.sum(dim=-1)
        if self.reduction == 'mean':
            loss = loss.mean()

        return loss


class HRRLayerNorm(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
        self.norm = nn.LayerNorm(dims)
    
    def forward(self, x):
        return self.norm(x)# / self.dims


HFHolo.register_for_auto_class("AutoModel")
HFHolo.register_for_auto_class("AutoModelForCausalLM")
AutoModel.register(HFHoloConfig, HoloDecoder)
AutoModelForCausalLM.register(HFHoloConfig, HFHolo)
