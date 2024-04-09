from transformers import AutoConfig, PretrainedConfig


class HFHoloConfig(PretrainedConfig):
    model_type = 'holo'

    def __init__(
        self,
        vocab_size: int = 50_257,
        model_dims: int = 128,
        num_hidden_layers: int = 2,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.model_dims = model_dims
        self.num_hidden_layers = num_hidden_layers
        super().__init__(**kwargs)


AutoConfig.register(HFHoloConfig.model_type, HFHoloConfig)
