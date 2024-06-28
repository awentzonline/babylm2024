from typing import Any, Dict, List, Optional, Union

from datasets import Dataset
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Sampler
from transformers import Trainer, TrainerCallback, TrainingArguments
from transformers.trainer import seed_worker

from babylm.hf_mup_trainer import MuPTrainer


class DynamicBatchSampler(Sampler):
    def __init__(self, dataset: Dataset, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size
        self.difficulties = np.ones(len(dataset))

    def __iter__(self):
        while True:
            probabilities = self.difficulties / self.difficulties.sum()
            batch = np.random.choice(
                len(self.dataset),
                size=self.batch_size,
                replace=False,
                p=probabilities
            )
            yield batch.tolist()

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def update_difficulties(self, indices: List[int], losses: List[float]):
        for idx, loss in zip(indices, losses):
            self.difficulties[idx] = loss


class DynamicBatchCallback(TrainerCallback):
    def __init__(self, sampler: DynamicBatchSampler):
        self.sampler = sampler

    def on_step_end(self, args: TrainingArguments, state, control, **kwargs):
        print(kwargs)
        model = kwargs.get("model")
        inputs = kwargs.get("inputs")

        # Calculate per-example losses
        with torch.no_grad():
            outputs = model(**inputs)
            shift_logits = outputs.logits[..., :-1].contiguous()
            shift_labels = inputs["input_ids"][..., 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            losses = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='none'
            )

            # Reshape losses to (batch_size, sequence_length-1) and take mean over sequence dimension
            per_example_losses = losses.view(shift_labels.size(0), -1).mean(dim=1)

        #indices = inputs.get("index", None)
        indices = inputs["index"]
        if indices is not None:
            self.sampler.update_difficulties(indices.tolist(), per_example_losses)


class DynamicBatchTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.sampler = kwargs.pop("sampler")
        super().__init__(*args, **kwargs)

    # def _get_train_sampler(self) -> Optional[Sampler]:
    #     return self.sampler
        # if self.train_dataset is None or not has_length(self.train_dataset):
        #     return None

        # # Build the sampler.
        # if self.args.group_by_length:
        #     if is_datasets_available() and isinstance(self.train_dataset, datasets.Dataset):
        #         lengths = (
        #             self.train_dataset[self.args.length_column_name]
        #             if self.args.length_column_name in self.train_dataset.column_names
        #             else None
        #         )
        #     else:
        #         lengths = None
        #     model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
        #     return LengthGroupedSampler(
        #         self.args.train_batch_size * self.args.gradient_accumulation_steps,
        #         dataset=self.train_dataset,
        #         lengths=lengths,
        #         model_input_name=model_input_name,
        #     )

        # else:
        #     return RandomSampler(self.train_dataset)

    def get_train_dataloader(self):
        return self.accelerator.prepare(torch.utils.data.DataLoader(
            self.train_dataset,
            #batch_size=self.args.per_device_train_batch_size,
            batch_sampler=self.sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_persistent_workers,
            # worker_init_fn=seed_worker,
            # prefetch_factor=self.args.dataloader_prefetch_factor,
        ))

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)



        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def on_update_sampler(self, inputs, outputs):
        with torch.no_grad():
            shift_logits = outputs.logits[..., :-1].contiguous()
            shift_labels = inputs["input_ids"][..., 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            losses = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='none'
            )
            # Reshape losses to (batch_size, sequence_length-1) and take mean over sequence dimension
            per_example_losses = losses.view(shift_labels.size(0), -1).mean(dim=1)

        indices = inputs["index"]
        if indices is not None:
            self.sampler.update_difficulties(indices.tolist(), per_example_losses)