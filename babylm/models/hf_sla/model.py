from typing import Optional

import mup
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from babylm.models.hf_sla.config import HFSLAConfig


class Abs(nn.Module):
    def forward(self, x):
        return torch.abs(x)


class SLASelfAttention(nn.Module):
    def __init__(self, f_attn, model_dims, num_heads=8, **kwargs):
        super().__init__()
        self.f_attn = f_attn
        self.model_dims = model_dims
        self.num_heads = num_heads
        assert model_dims % num_heads == 0, f'Num heads ({num_heads}) incompatible with model dims ({model_dims})'
        self.head_dims = model_dims // num_heads
        self.qkv = nn.Linear(self.model_dims, 3 * self.model_dims, bias=False)
        self.output = nn.Linear(model_dims, model_dims, bias=False)

    def forward(self, x, causal=True, mask=None):
        batch_size, seq_len = x.shape[:2]
        q, k, v = self.qkv(x).split(self.model_dims, dim=-1)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dims)
        q = q.permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dims)
        k = k.permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dims)
        v = v.permute(0, 2, 1, 3)
        values_hat = self.f_attn(k, v, q)
        values_hat = values_hat.permute(0, 2, 1, 3).contiguous()
        values_hat = values_hat.view(batch_size, seq_len, self.model_dims)
        return self.output(values_hat)


class MLP(nn.Module):
    def __init__(self, model_dims, ff_dims=None):
        super().__init__()
        ff_dims = ff_dims or model_dims * 4
        self.net = nn.Sequential(
            nn.Linear(model_dims, ff_dims, bias=False),
            nn.GELU(),
        )
        # move output to its own thing so we can target it for initialization
        self.output = nn.Linear(ff_dims, model_dims, bias=False)

    def forward(self, x):
        return self.output(self.net(x))


class SLALayer(nn.Module):
    def __init__(
        self, f_attn, model_dims, rezero=False, attention_class=SLASelfAttention,
        use_norm_bias=False, num_heads=8, num_layers=1,
    ):
        super().__init__()
        self.self_attention = attention_class(f_attn, model_dims, num_heads=num_heads)
        self.mlp = MLP(model_dims)
        self.num_layers = num_layers

        if rezero:
            self.norm_attn = nn.Identity()
            self.norm_mlp = nn.Identity()
            self.gain = nn.Parameter(torch.zeros(1))
        else:
            self.norm_attn = nn.LayerNorm(model_dims, bias=use_norm_bias)
            self.norm_mlp = nn.LayerNorm(model_dims, bias=use_norm_bias)
            self.gain = 1. #/ np.sqrt(2 * num_layers)

    def forward(self, x, mask=None, labels=None):
        values_attn = self.self_attention(self.norm_attn(x))
        x = x + self.gain * values_attn  #torch.abs(values_attn)
        values_mlp = self.mlp(self.norm_mlp(x))
        x = x + self.gain * values_mlp  #torch.tanh(values_mlp)#torch.abs(values_mlp)
        return x


class SLADecoder(PreTrainedModel):
    config_class = HFSLAConfig

    def __init__(self, config):
        super().__init__(config)
        layer_class = SLALayer
        attention_class = SLASelfAttention
        self.layers = nn.ModuleList([
            layer_class(
                config.f_attn,
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


class HFSLA(PreTrainedModel):
    config_class = HFSLAConfig

    def __init__(self, config):
        super().__init__(config)
        self.decoder = SLADecoder(config)
        if config.position_embedding == 'learn':
            self.position_embedding = nn.Embedding(config.max_seq_len, config.model_dims)
        elif config.position_embedding == 'sin':
            self.position_embedding = PositionalEncoding(config.model_dims, config.max_seq_len)
        self.input_embedding = nn.Embedding(config.vocab_size, config.model_dims)
        if config.rezero:
            self.norm = nn.Identity()
        else:
            self.norm = nn.LayerNorm(config.model_dims, bias=config.use_norm_bias)

        self.predict_token = mup.MuReadout(config.model_dims, config.vocab_size, bias=False)

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
        depth_std = self.config.initializer_range / np.sqrt(2 * self.config.num_hidden_layers)
        for name, module in module.named_modules():
            for target_name in ('qkv',):
                if target_name in name and query_zero_init:
                    if hasattr(module.weight, 'infshape'):
                        mup.init.normal_(module.weight, mean=0.0, std=hrr_std)
                    else:
                        module.weight.data.normal_(mean=0.0, std=hrr_std)
                    if module.bias is not None:
                        module.bias.data.zero_()

            if "output" in name:
                if hasattr(module.weight, 'infshape'):
                    mup.init.normal_(module.weight, mean=0.0, std=depth_std)
                else:
                    module.weight.data.normal_(mean=0.0, std=depth_std)

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
        feats = self.decoder(inputs, labels=None)

        feats = self.norm(feats)
        logits = self.predict_token(feats)

        loss = 0.
        if labels is not None:
            preds = logits[:, :-1].contiguous().view(-1, logits.shape[-1])
            targets = labels[:, 1:].contiguous().view(-1)
            loss = self.loss_fn(preds, targets)

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
            base_config = HFSLAConfig(
                model_dims=128,
                **base_kwargs,
            )
            delta_config = HFSLAConfig(
                model_dims=256,
                **delta_kwargs
            )
            base_model = HFSLA(config=base_config)
            delta_model = HFSLA(config=delta_config)
            base_shapes = mup.make_base_shapes(base_model, delta_model, savefile=filename)
            cls._mup_base_shapes = base_shapes
            del base_model
            del delta_model
            base_model = delta_model = None
        return cls._mup_base_shapes


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


HFSLA.register_for_auto_class("AutoModel")
HFSLA.register_for_auto_class("AutoModelForCausalLM")
AutoModel.register(HFSLAConfig, SLADecoder)
AutoModelForCausalLM.register(HFSLAConfig, HFSLA)
