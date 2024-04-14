from transformers import AutoConfig, PretrainedConfig


class HFHoloConfig(PretrainedConfig):
    model_type = 'holo'

    def __init__(
        self,
        vocab_size: int = 50_257,
        model_dims: int = 128,
        num_hidden_layers: int = 2,
        max_seq_len: int = 1024,
        learn_input_embs: bool = True,
        learn_output_embs: bool = True,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.model_dims = model_dims
        self.num_hidden_layers = num_hidden_layers
        self.max_seq_len = max_seq_len
        self.learn_input_embs = learn_input_embs
        self.learn_output_embs = learn_output_embs
        super().__init__(**kwargs)


AutoConfig.register(HFHoloConfig.model_type, HFHoloConfig)
