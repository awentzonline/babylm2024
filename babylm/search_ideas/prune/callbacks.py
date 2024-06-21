from transformers import (
    TrainerCallback, TrainerControl, TrainerState, TrainingArguments,
)

class PruneCallback(TrainerCallback):
    def __init__(self, prune_func):
        self.prune_func = prune_func

    def on_optimizer_step(
        self, args: TrainingArguments, state: TrainerState,
        control: TrainerControl, **kwargs
    ):
        model = kwargs['model']
        self.prune_func(model, state, control)