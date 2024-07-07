from dataclasses import asdict, dataclass
from typing import Callable, List, Optional, Union

import numpy as np


from transformers import AutoConfig, PretrainedConfig


class VectorDiffusionConfig(PretrainedConfig):
    model_type = 'vecdiff'

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


@dataclass
class DDIMSchedulerConfig:
    num_train_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "linear"
    trained_betas: Optional[Union[np.ndarray, List[float]]] = None
    clip_sample: bool = True
    set_alpha_to_one: bool = True
    steps_offset: int = 0
    prediction_type: str = "epsilon"
    thresholding: bool = False
    dynamic_thresholding_ratio: float = 0.995
    clip_sample_range: float = 1.0
    sample_max_value: float = 1.0
    timestep_spacing: str = "leading"
    rescale_betas_zero_snr: bool = False


AutoConfig.register(VectorDiffusionConfig.model_type, VectorDiffusionConfig)
VectorDiffusionConfig.register_for_auto_class()
