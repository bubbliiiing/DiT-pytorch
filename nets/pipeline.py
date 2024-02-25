# Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
# William Peebles and Saining Xie
#
# Copyright (c) 2021 OpenAI
# MIT License
#
# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Optional, Tuple, Union

import torch

from diffusers.models import AutoencoderKL
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import randn_tensor
from diffusers.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers import logging
from .transformer_2d import Transformer2DModel

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
    layout: Optional["torch.layout"] = None,
):
    """A helper function to create random tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators, you can seed each batch size individually. If CPU generators are passed, the tensor
    is always created on the CPU.
    """
    # device on which tensor is created defaults to device
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
            if device != "mps":
                logger.info(
                    f"The passed generator was created on 'cpu' even though a tensor on {device} was expected."
                    f" Tensors will be created on 'cpu' and then moved to {device}. Note that one can probably"
                    f" slighly speed up this function by passing a generator that was created on the {device} device."
                )
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return latents

class DiTPipeline(DiffusionPipeline):
    r"""
    Pipeline for image generation based on a Transformer backbone instead of a UNet.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        transformer ([`Transformer2DModel`]):
            A class conditioned `Transformer2DModel` to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        scheduler ([`DDIMScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
    """

    def __init__(
        self,
        transformer: Transformer2DModel,
        vae: AutoencoderKL,
        scheduler: KarrasDiffusionSchedulers,
        id2label: Optional[Dict[int, str]] = None,
    ):
        super().__init__()
        self.register_modules(transformer=transformer, vae=vae, scheduler=scheduler)

        # create a imagenet -> id dictionary for easier use
        self.labels = {}
        if id2label is not None:
            for key, value in id2label.items():
                for label in value.split(","):
                    self.labels[label.lstrip().rstrip()] = int(key)
            self.labels = dict(sorted(self.labels.items()))

    def get_label_ids(self, label: Union[str, List[str]]) -> List[int]:
        r"""

        Map label strings from ImageNet to corresponding class ids.

        Parameters:
            label (`str` or `dict` of `str`):
                Label strings to be mapped to class ids.

        Returns:
            `list` of `int`:
                Class ids to be processed by pipeline.
        """

        if not isinstance(label, list):
            label = list(label)

        for l in label:
            if l not in self.labels:
                raise ValueError(
                    f"{l} does not exist. Please make sure to select one of the following labels: \n {self.labels}."
                )

        return [self.labels[l] for l in label]

    @torch.no_grad()
    def __call__(
        self,
        class_labels: List[int],
        guidance_scale: float = 4.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 50,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        # 生成的批次大小、隐含层的大小与隐含层通道
        batch_size = len(class_labels)
        latent_size = self.transformer.config.sample_size
        latent_channels = self.transformer.config.in_channels

        # --------------------------------- #
        #   前处理
        # --------------------------------- #
        # 生成latent
        latents = randn_tensor(
            shape=(batch_size, latent_channels, latent_size, latent_size),
            generator=generator,
            device=self.device,
            dtype=self.transformer.dtype,
        )
        latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1 else latents

        # 将输入的label 与 null label进行concat，null label是负向提示类。
        class_labels = torch.tensor(class_labels, device=self.device).reshape(-1)
        class_null = torch.tensor([1000] * batch_size, device=self.device)
        class_labels_input = torch.cat([class_labels, class_null], 0) if guidance_scale > 1 else class_labels

        # 设置生成的步数
        self.scheduler.set_timesteps(num_inference_steps)

        # --------------------------------- #
        #   扩散生成
        # --------------------------------- #
        # 开始N步扩散的循环
        for t in self.progress_bar(self.scheduler.timesteps):
            if guidance_scale > 1:
                half = latent_model_input[: len(latent_model_input) // 2]
                latent_model_input = torch.cat([half, half], dim=0)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            # 处理timesteps
            timesteps = t
            if not torch.is_tensor(timesteps):
                is_mps = latent_model_input.device.type == "mps"
                if isinstance(timesteps, float):
                    dtype = torch.float32 if is_mps else torch.float64
                else:
                    dtype = torch.int32 if is_mps else torch.int64
                timesteps = torch.tensor([timesteps], dtype=dtype, device=latent_model_input.device)
            elif len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(latent_model_input.device)
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timesteps = timesteps.expand(latent_model_input.shape[0])

            # 将隐含层特征、时间步和种类输入传入到transformers中
            noise_pred = self.transformer(
                latent_model_input, timestep=timesteps, class_labels=class_labels_input
            ).sample

            # perform guidance
            if guidance_scale > 1:
                # 在通道上做分割，取出生图部分的通道
                eps, rest = noise_pred[:, :latent_channels], noise_pred[:, latent_channels:]
                cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)

                half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
                eps = torch.cat([half_eps, half_eps], dim=0)

                noise_pred = torch.cat([eps, rest], dim=1)

            # 对结果进行分割，取出生图部分的通道
            if self.transformer.config.out_channels // 2 == latent_channels:
                model_output, _ = torch.split(noise_pred, latent_channels, dim=1)
            else:
                model_output = noise_pred

            # 通过采样器将这一步噪声施加到隐含层
            latent_model_input = self.scheduler.step(model_output, t, latent_model_input).prev_sample

        if guidance_scale > 1:
            latents, _ = latent_model_input.chunk(2, dim=0)
        else:
            latents = latent_model_input

        # --------------------------------- #
        #   后处理
        # --------------------------------- #
        # 通过vae进行解码
        latents = 1 / self.vae.config.scaling_factor * latents
        samples = self.vae.decode(latents).sample

        samples = (samples / 2 + 0.5).clamp(0, 1)

        # 转化为float32类别
        samples = samples.cpu().permute(0, 2, 3, 1).float().numpy()

        if output_type == "pil":
            samples = self.numpy_to_pil(samples)

        if not return_dict:
            return (samples,)

        return ImagePipelineOutput(images=samples)
