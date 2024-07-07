# Adapted from huggingface/diffusers DDIMScheduler example
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from diffusers.utils import BaseOutput
from diffusers.schedulers import DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
import numpy as np
import torch
from transformers import Trainer
from transformers.trainer import _is_peft_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES


@dataclass
class VectorPipelineOutput(BaseOutput):
    """
    Output class for 1D vector pipelines.

    Args:
        vectors (`np.ndarray`)
            List of denoised vectors of length `batch_size` NumPy array of shape `(batch_size, num_channels)`.
    """

    vectors: np.ndarray


class VectorDiffusionPipeline(DiffusionPipeline):
    r"""
    Pipeline for sequence generation.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        unet ([`UNet1DModel`]):
            A `UNet1DModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    model_cpu_offload_seq = "unet"

    def __init__(self, unet, scheduler):
        super().__init__()

        # make sure scheduler can always be converted to DDIM
        scheduler = DDIMScheduler.from_config(scheduler.config)

        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: float = 0.0,
        num_inference_steps: int = 50,
        use_clipped_model_output: Optional[bool] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[VectorPipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers. A value of `0` corresponds to
                DDIM and `1` corresponds to DDPM.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            use_clipped_model_output (`bool`, *optional*, defaults to `None`):
                If `True` or `False`, see documentation for [`DDIMScheduler.step`]. If `None`, nothing is passed
                downstream to the scheduler (use `None` for schedulers which don't support this argument).
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Example:

        ```py
        >>> from diffusers import DDIMPipeline
        >>> import PIL.Image
        >>> import numpy as np

        >>> # load model and scheduler
        >>> pipe = DDIMPipeline.from_pretrained("fusing/ddim-lsun-bedroom")

        >>> # run pipeline in inference (sample random noise and denoise)
        >>> image = pipe(eta=0.0, num_inference_steps=50)

        >>> # process image to PIL
        >>> image_processed = image.cpu().permute(0, 2, 3, 1)
        >>> image_processed = (image_processed + 1.0) * 127.5
        >>> image_processed = image_processed.numpy().astype(np.uint8)
        >>> image_pil = PIL.Image.fromarray(image_processed[0])

        >>> # save image
        >>> image_pil.save("test.png")
        ```

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """

        # Sample gaussian noise to begin loop
        if isinstance(self.unet.config.sample_size, int):
            input_shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
            )
        else:
            input_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        vectors = randn_tensor(input_shape, generator=generator, device=self._execution_device, dtype=self.unet.dtype)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            model_output = self.unet(vectors, t).sample

            # 2. predict previous mean of image x_t-1 and add variance depending on eta
            # eta corresponds to η in paper and should be between [0, 1]
            # do x_t -> x_t-1
            vectors = self.scheduler.step(
                model_output, t, vectors, eta=eta, use_clipped_model_output=use_clipped_model_output, generator=generator
            ).prev_sample

        if not return_dict:
            return (vectors,)

        return VectorPipelineOutput(vectors=vectors)


# TODO: Add schedule to the model instead?
# class VectorDiffuserTrainer(Trainer):
#     def __init__(self, *args, scheduler=None, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.scheduler = scheduler

#     def compute_loss(self, model, inputs, return_outputs=False):
#         """
#         How the loss is computed by Trainer. By default, all models return the loss in the first element.

#         Subclassed to pass the scheduler to the model.
#         """
#         if self.label_smoother is not None and "labels" in inputs:
#             labels = inputs.pop("labels")
#         else:
#             labels = None
#         outputs = model(scheduler=self.scheduler, **inputs)
#         # Save past state if it exists
#         # TODO: this needs to be fixed and made cleaner later.
#         if self.args.past_index >= 0:
#             self._past = outputs[self.args.past_index]

#         if labels is not None:
#             unwrapped_model = self.accelerator.unwrap_model(model)
#             if _is_peft_model(unwrapped_model):
#                 model_name = unwrapped_model.base_model.model._get_name()
#             else:
#                 model_name = unwrapped_model._get_name()
#             if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
#                 loss = self.label_smoother(outputs, labels, shift_labels=True)
#             else:
#                 loss = self.label_smoother(outputs, labels)
#         else:
#             if isinstance(outputs, dict) and "loss" not in outputs:
#                 raise ValueError(
#                     "The model did not return a loss from the inputs, only the following keys: "
#                     f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
#                 )
#             # We don't use .loss here since the model may return tuples instead of ModelOutput.
#             loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

#         return (loss, outputs) if return_outputs else loss
