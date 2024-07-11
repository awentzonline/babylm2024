from transformers import TrainerCallback


class UpdateEMACallback(TrainerCallback):
    def on_end_step(self, *args, model=None, **kwargs):
        model.update_ema_model(model.config.ema_weight)