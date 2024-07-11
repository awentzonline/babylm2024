from transformers import AutoConfig, PretrainedConfig


class HFVanConfig(PretrainedConfig):
    model_type = 'vanilla'

    def __init__(
        self,
        vocab_size: int = 50_257,
        model_dims: int = 1024,
        num_hidden_layers: int = 2,
        heads: int = 1,  #, 8,
        max_seq_length: int = 1024,
        use_norm: bool = False,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.model_dims = model_dims
        self.num_hidden_layers = num_hidden_layers
        self.heads = heads
        self.max_seq_length = max_seq_length
        self.use_norm = use_norm
        super().__init__(**kwargs)


AutoConfig.register(HFVanConfig.model_type, HFVanConfig)
HFVanConfig.register_for_auto_class()