from dataclasses import asdict, dataclass
from typing import Callable, List, Optional, Union

import numpy as np


from transformers import AutoConfig, PretrainedConfig


class HRRDiffusionConfig(PretrainedConfig):
    model_type = 'hrrdiff'

    def __init__(
        self,
        vocab_size: int = 50_257,  # 50_304,  # gpt2 tokens padded (50257) to nearest multiple of 64
        model_dims: int = 768, #128,
        num_hidden_layers: int = 4,
        max_seq_len: int = 1024,
        learn_input_embs: bool = True,
        learn_output_embs: bool = True,
        attention_class: str = 'hrr',
        initializer_range: float = 0.2,
        rezero: bool = False,
        loss: str = 'xent',
        use_norm_bias: bool = True,
        num_heads: int = 2,
        hrr_embedding: bool = False,
        position_embedding: str = 'learn',
        f_attn: Callable = None,
        noise_scheduler_config: dict = None,
        tie_word_embeddings: bool = False,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.model_dims = model_dims
        self.num_hidden_layers = num_hidden_layers
        self.max_seq_len = max_seq_len
        self.learn_input_embs = learn_input_embs
        self.learn_output_embs = learn_output_embs
        self.attention_class = attention_class
        self.initializer_range = initializer_range
        self.rezero = rezero
        self.loss = loss
        self.use_norm_bias = use_norm_bias
        self.num_heads = num_heads
        self.hrr_embedding = hrr_embedding
        self.position_embedding = position_embedding
        self.f_attn = f_attn
        self.tie_word_embeddings = tie_word_embeddings

        self.noise_scheduler_config = dict(
            num_train_timesteps=1000,
            beta_start= 0.0001,
            beta_end= 0.02,
            beta_schedule="linear",
            trained_betas=None,
            clip_sample = True,
            set_alpha_to_one = True,
            steps_offset = 0,
            prediction_type = "epsilon",
            thresholding = False,
            dynamic_thresholding_ratio = 0.995,
            clip_sample_range = 1.0,
            sample_max_value = 1.0,
            timestep_spacing = "leading",
            rescale_betas_zero_snr = False,
        )
        if noise_scheduler_config is not None:
            self.noise_scheduler_config.update(noise_scheduler_config)
        super().__init__(**kwargs)


AutoConfig.register(HRRDiffusionConfig.model_type, HRRDiffusionConfig)
HRRDiffusionConfig.register_for_auto_class()
