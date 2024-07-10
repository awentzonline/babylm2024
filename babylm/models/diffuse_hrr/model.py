from dataclasses import asdict
from typing import Optional

from diffusers import DDIMScheduler, UNet1DModel
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
import mup
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from babylm.models.diffuse_hrr.config import HRRDiffusionConfig
from babylm.models.hf_holo import hrr


class Normalize(nn.Module):
    def __init__(self, dim=-1, p=2., eps=1e-12):
        super().__init__()
        self.dim = dim
        self.p = p
        self.eps = eps

    def forward(self, x):
        return F.normalize(x, dim=self.dim, p=self.p, eps=self.eps)


class AdaptiveLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, elementwise_affine=False, eps=eps)

    def forward(self, x, scale, shift):
        x = self.norm(x)
        return x * (1. + scale) + shift


class HRRSelfAttention(nn.Module):
    def __init__(self, model_dims, num_heads=8, **kwargs):
        super().__init__()
        self.model_dims = model_dims
        self.num_heads = num_heads
        assert model_dims % num_heads == 0, f'Num heads ({num_heads}) incompatible with model dims ({model_dims})'
        self.head_dims = model_dims // num_heads
        self.qkv = nn.Linear(self.model_dims, 3 * self.model_dims, bias=False)
        self.output = nn.Linear(model_dims, model_dims, bias=False)
        self.norm = Normalize()

    def forward(self, x, causal=True, mask=None):
        batch_size, seq_len = x.shape[:2]
        q, k, v = self.qkv(x).split(self.model_dims, dim=2)
        if self.num_heads > 1:
            q = q.view(batch_size, seq_len, self.num_heads, self.head_dims)
            k = k.view(batch_size, seq_len, self.num_heads, self.head_dims)
            v = v.view(batch_size, seq_len, self.num_heads, self.head_dims)
        # q, k, v = map(self.norm, (q, k, v))
        #values_hat = hrr.key_value_query_lin(k, v, q, causal=causal, norm=False)
        values_hat = hrr.key_value_query(k, v, q, causal=causal, norm=False)
        if self.num_heads > 1:
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


class HoloLayer(nn.Module):
    def __init__(
        self, model_dims, rezero=False, attention_class=HRRSelfAttention,
        use_norm_bias=False, num_heads=8, **_
    ):
        super().__init__()
        self.self_attention = attention_class(model_dims, num_heads=num_heads)
        self.mlp = MLP(model_dims)

        self.norm_attn = AdaptiveLayerNorm(model_dims)
        self.norm_mlp = AdaptiveLayerNorm(model_dims)
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_dims, 6 * model_dims),
        )

    def forward(self, x, conditions, mask=None, labels=None):
        scale_attn, shift_attn, gate_attn, scale_mlp, shift_mlp, gate_mlp = \
            self.adaln_modulation(conditions).chunk(6, dim=-1)

        x_attn = self.norm_attn(x, scale_attn, shift_attn)
        values_attn = self.self_attention(x_attn, causal=True)
        x = x + gate_attn * values_attn

        x_mlp = self.norm_mlp(x, scale_mlp, shift_mlp)
        values_mlp = self.mlp(x_mlp)
        x = x + gate_mlp * values_mlp
        return x


class HoloDecoder(PreTrainedModel):
    config_class = HRRDiffusionConfig

    def __init__(self, config):
        super().__init__(config)
        layer_class = HoloLayer
        attention_class = dict(
            hrr=HRRSelfAttention,
        )[config.attention_class]

        self.layers = nn.ModuleList([
            layer_class(
                config.model_dims, attention_class=attention_class,
                rezero=config.rezero, use_norm_bias=config.use_norm_bias,
                num_heads=config.num_heads, num_layers=config.num_hidden_layers,
            )
            for _ in range(config.num_hidden_layers)
        ])

    def forward(self, x, condtions, mask=None, labels=None):
        loss = torch.tensor(0., device=x.device)

        for layer in self.layers:
            x = layer(x, condtions, mask=mask)

            if labels is not None:
                layer_loss = x.square().sum()
                loss = loss# + layer_loss
            # print(list(map(float, (x.min(), x.mean(), x.std(), x.max()))))

        if labels is None:
            return x
        else:
            return x, loss


class HRRDiffuser(PreTrainedModel):
    config_class = HRRDiffusionConfig

    def __init__(self, config):
        super().__init__(config)
        self.noise_scheduler = DDIMScheduler(**config.noise_scheduler_config)
        self.decoder = HoloDecoder(config)
        if config.position_embedding == 'learn':
            self.position_embedding = nn.Embedding(config.max_seq_len, config.model_dims)
        elif config.position_embedding == 'sin':
            self.position_embedding = PositionalEncoding(config.model_dims, config.max_seq_len)
        self.input_embedding = nn.Embedding(config.vocab_size, config.model_dims)
        # diffusion timestep
        self.timestep_embedding = nn.Embedding(self.noise_scheduler.config.num_train_timesteps, config.model_dims)

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
        }[config.loss]

        self.post_init()

    def _init_weights(self, module):
        module.apply(self._init_module)

    def _init_module(self, module, readout_zero_init=True, query_zero_init=False):
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
            if module.weight is not None:
                module.weight.data.fill_(1.0)

        for layer in self.decoder.layers:
            nn.init.constant_(layer.adaln_modulation[-1].weight, 0.)
            nn.init.constant_(layer.adaln_modulation[-1].bias, 0.)

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
        num_inference_steps: int = 50,
        eta: float = 0.,
    ):
        input_embs = self.input_embedding(input_ids)
        position_ids = torch.arange(input_embs.shape[1]).long().to(input_embs.device)
        position_ids = position_ids[None, :].repeat(input_embs.shape[0], 1)
        position_embs = self.position_embedding(position_ids)
        x = input_embs# + position_embs
        bsz = x.shape[0]
        noise = randn_tensor(x.shape, generator=None, device=self.device, dtype=self.dtype)

        loss = None
        # Training is very different from generating so `forward`` has to do a lot to fit in with the other frameworks
        if labels is None:
            # generate
            time_embs = self.timestep_embedding(torch.LongTensor(0).to(self.device))
            x = x + position_embs
            x_tp1_noise = randn_tensor((bsz, 1, x.shape[-1]), generator=None, device=self.device, dtype=self.dtype)
            x_tp1_position_embs = self.position_embedding(torch.LongTensor([x.shape[1]]).to(self.device))[None, ...]
            self.noise_scheduler.set_timesteps(num_inference_steps)
            for t in self.noise_scheduler.timesteps:
                timesteps = torch.LongTensor([t]).to(self.device)
                time_embs = self.timestep_embedding(timesteps).unsqueeze(1)
                conditions = time_embs
                x_tp1_noise = x_tp1_noise + x_tp1_position_embs
                x_inputs = torch.concatenate([x, x_tp1_noise], dim=1)
                pred_x_tp1_noise = self.decoder(x_inputs, conditions, labels=None)[:, -1]
                x_tp1_noise = self.noise_scheduler.step(
                    pred_x_tp1_noise, t, x_tp1_noise, eta=eta, use_clipped_model_output=False, generator=None,
                ).prev_sample
            x_inputs = torch.concatenate([x, x_tp1_noise], dim=1)
            logits = self.predict_token(x_inputs)
        else:
            # Sample a random timestep for each sequence
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=self.device, dtype=torch.long
            )
            time_embs = self.timestep_embedding(timesteps).unsqueeze(1)
            conditions = time_embs
            x_t = x + position_embs
            # Add noise to the model input according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_x = self.noise_scheduler.add_noise(x_t, noise, timesteps)
            pred_x = self.decoder(noisy_x, conditions, labels=None)

            if self.noise_scheduler.config.prediction_type == "epsilon":
                x_target = noise
            elif self.noise_scheduler.config.prediction_type == "v_prediction":
                x_target = self.noise_scheduler.get_velocity(x_t, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

            recon_noise_loss = F.mse_loss(pred_x, x_target)

            # train a model to decode tokens from the input embeddings
            # logits = self.predict_token(F.normalize(input_embs, dim=-1))
            # input_embs = self.noise_scheduler.add_noise(x_tp1, x_tp1_noise, timesteps)
            logits = self.predict_token(input_embs)
            decode_tokens_loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                input_ids.view(-1)  # [:, 1:].contiguous().view(-1)
            )

            loss = recon_noise_loss + decode_tokens_loss

        if return_dict is not None and not return_dict:
            output = (logits,)# feats)
            output = (loss,) + output if loss else output
            return output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=None,  # transformer_outputs.past_key_values,
            # hidden_states=feats,
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
            if not 'model_dims' in delta_kwargs:
                delta_kwargs['model_dims'] = 256
            base_config = cls.config_class(
                model_dims=128,
                **base_kwargs,
            )
            delta_config = cls.config_class(
                **delta_kwargs
            )
            base_model = cls(config=base_config)
            delta_model = cls(config=delta_config)
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


HRRDiffuser.register_for_auto_class("AutoModel")
HRRDiffuser.register_for_auto_class("AutoModelForCausalLM")
AutoModel.register(HRRDiffusionConfig, HoloDecoder)
AutoModelForCausalLM.register(HRRDiffusionConfig, HRRDiffuser)
