from transformers import AutoConfig, PretrainedConfig


class HFHoloConfig(PretrainedConfig):
    model_type = 'holo'

    def __init__(
        self,
        vocab_size: int = 50_257,  # 50_304,  # gpt2 tokens padded (50257) to nearest multiple of 64
        model_dims: int = 128,
        num_hidden_layers: int = 3,
        max_seq_len: int = 1024,
        learn_input_embs: bool = True,
        learn_output_embs: bool = True,
        attention_class: str = 'hrr',
        initializer_range: float = 0.02,
        rezero: bool = False,
        loss: str = 'xent',
        use_norm_bias: bool = True,
        num_heads: int = 8,
        hrr_embedding: bool = False,
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
        super().__init__(**kwargs)


# AutoConfig.register(HFHoloConfig.model_type, HFHoloConfig)
HFHoloConfig.register_for_auto_class()
