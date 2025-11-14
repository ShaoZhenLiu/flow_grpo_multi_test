# Copied from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux_kontext.py

import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import PIL
import numpy as np
import torch
import torch.nn.functional as F

# from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
# from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
# from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.utils import logging

from omnigen2.models.transformers.repo import OmniGen2RotaryPosEmbed
# from .sd3_sde_with_logprob import sde_step_with_logprob
from .omni_gen2_sde_with_logprob import sde_step_with_logprob


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Type hinting would cause circular import, self should be `OmniGen2Pipeline`
def cache_init(self, num_steps: int):
    '''
    Initialization for cache.
    '''
    cache_dic = {}
    cache = {}
    cache_index = {}
    cache[-1]={}
    cache_index[-1]={}
    cache_index['layer_index']={}
    cache[-1]['layers_stream']={}
    cache_dic['cache_counter'] = 0

    for j in range(len(self.transformer.layers)):
        cache[-1]['layers_stream'][j] = {}
        cache_index[-1][j] = {}

    cache_dic['Delta-DiT'] = False
    cache_dic['cache_type'] = 'random'
    cache_dic['cache_index'] = cache_index
    cache_dic['cache'] = cache
    cache_dic['fresh_ratio_schedule'] = 'ToCa' 
    cache_dic['fresh_ratio'] = 0.0
    cache_dic['fresh_threshold'] = 3
    cache_dic['soft_fresh_weight'] = 0.0
    cache_dic['taylor_cache'] = True
    cache_dic['max_order'] = 4
    cache_dic['first_enhance'] = 5

    current = {}
    current['activated_steps'] = [0]
    current['step'] = 0
    current['num_steps'] = num_steps

    return cache_dic, current


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# @torch.no_grad()
# def pipeline_with_logprob(
#     self,
#     image: Optional[PipelineImageInput] = None,
#     prompt: Union[str, List[str]] = None,
#     prompt_2: Optional[Union[str, List[str]]] = None,
#     negative_prompt: Union[str, List[str]] = None,
#     negative_prompt_2: Optional[Union[str, List[str]]] = None,
#     height: Optional[int] = None,
#     width: Optional[int] = None,
#     num_inference_steps: int = 28,
#     sigmas: Optional[List[float]] = None,
#     guidance_scale: float = 3.5,
#     num_images_per_prompt: Optional[int] = 1,
#     generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
#     latents: Optional[torch.FloatTensor] = None,
#     prompt_embeds: Optional[torch.FloatTensor] = None,
#     pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
#     negative_prompt_embeds: Optional[torch.FloatTensor] = None,
#     negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
#     output_type: Optional[str] = "pil",
#     joint_attention_kwargs: Optional[Dict[str, Any]] = None,
#     callback_on_step_end_tensor_inputs: List[str] = ["latents"],
#     max_sequence_length: int = 512,
#     max_area: int = 1024**2,
#     _auto_resize: bool = True,
#     noise_level: float = 0.7,
# ):

@torch.no_grad()
def pipeline_with_logprob(
    self,
    prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    prompt_attention_mask: Optional[torch.LongTensor] = None,
    negative_prompt_attention_mask: Optional[torch.LongTensor] = None,
    max_sequence_length: Optional[int] = None,
    input_images: Optional[List[PIL.Image.Image]] = None,
    num_images_per_prompt: int = 1,
    height: Optional[int] = None,
    width: Optional[int] = None,
    max_pixels: int = 1024 * 1024,
    max_input_image_side_length: int = 1024,
    align_res: bool = True,
    num_inference_steps: int = 28,
    text_guidance_scale: float = 4.0,
    image_guidance_scale: float = 1.0,
    cfg_range: Tuple[float, float] = (0.0, 1.0),
    attention_kwargs: Optional[Dict[str, Any]] = None,
    timesteps: List[int] = None,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    verbose: bool = False,
    step_func=None,
    noise_level: float = 0.7,
):
    
    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor

    self._text_guidance_scale = text_guidance_scale
    self._image_guidance_scale = image_guidance_scale
    self._cfg_range = cfg_range
    self._attention_kwargs = attention_kwargs

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    # 3. Encode input prompt
    (
        prompt_embeds,
        prompt_attention_mask,
        negative_prompt_embeds,
        negative_prompt_attention_mask,
    ) = self.encode_prompt(
        prompt,
        self.text_guidance_scale > 1.0,
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        prompt_attention_mask=prompt_attention_mask,
        negative_prompt_attention_mask=negative_prompt_attention_mask,
        max_sequence_length=max_sequence_length,
    )

    dtype = self.vae.dtype
    # 3. Prepare control image
    ref_latents = self.prepare_image(
        images=input_images,
        batch_size=batch_size,
        num_images_per_prompt=num_images_per_prompt,
        max_pixels=max_pixels,
        max_side_length=max_input_image_side_length,
        device=device,
        dtype=dtype,
    )

    if input_images is None:
        input_images = []
    
    if len(input_images) == 1 and align_res:
        width, height = ref_latents[0][0].shape[-1] * self.vae_scale_factor, ref_latents[0][0].shape[-2] * self.vae_scale_factor
        ori_width, ori_height = width, height
    else:
        ori_width, ori_height = width, height

        cur_pixels = height * width
        ratio = (max_pixels / cur_pixels) ** 0.5
        ratio = min(ratio, 1.0)

        height, width = int(height * ratio) // 16 * 16, int(width * ratio) // 16 * 16
    
    if len(input_images) == 0:
        self._image_guidance_scale = 1

    # 4. Prepare latents.
    latent_channels = self.transformer.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        latent_channels,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    freqs_cis = OmniGen2RotaryPosEmbed.get_freqs_cis(
        self.transformer.config.axes_dim_rope,
        self.transformer.config.axes_lens,
        theta=10000,
    )
    
    # mu = np.sqrt(latents.shape[-2] * latents.shape[-1]) / 40
    # image_seq_len = latents.shape[1]
    # mu = calculate_shift(
    #     image_seq_len,
    #     self.scheduler.config.get("base_image_seq_len", 256),
    #     self.scheduler.config.get("max_image_seq_len", 4096),
    #     self.scheduler.config.get("base_shift", 0.5),
    #     self.scheduler.config.get("max_shift", 1.15),
    # )
    # sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    
    # timesteps, num_inference_steps = retrieve_timesteps(
    #     self.scheduler,
    #     num_inference_steps,
    #     device,
    #     # sigmas=sigmas,
    #     mu=mu,
    # )
    
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        timesteps,
        num_tokens=latents.shape[-2] * latents.shape[-1]
    )
    
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
    self._num_timesteps = len(timesteps)

    enable_taylorseer = getattr(self, "enable_taylorseer", False)
    if enable_taylorseer:
        model_pred_cache_dic, model_pred_current = cache_init(self, num_inference_steps)
        model_pred_ref_cache_dic, model_pred_ref_current = cache_init(self, num_inference_steps)
        model_pred_uncond_cache_dic, model_pred_uncond_current = cache_init(self, num_inference_steps)
        self.transformer.enable_taylorseer = True
    # elif self.transformer.enable_teacache:
    #     # Use different TeaCacheParams for different conditions
    #     teacache_params = TeaCacheParams()
    #     teacache_params_uncond = TeaCacheParams()
    #     teacache_params_ref = TeaCacheParams()

    all_latents = [latents]
    all_log_probs = []

    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if enable_taylorseer:
                self.transformer.cache_dic = model_pred_cache_dic
                self.transformer.current = model_pred_current
            # elif self.transformer.enable_teacache:
            #     teacache_params.is_first_or_last_step = i == 0 or i == len(timesteps) - 1
            #     self.transformer.teacache_params = teacache_params

            model_pred = self.predict(
                t=t,
                latents=latents,
                prompt_embeds=prompt_embeds,
                freqs_cis=freqs_cis,
                prompt_attention_mask=prompt_attention_mask,
                ref_image_hidden_states=ref_latents,
            )
            text_guidance_scale = self.text_guidance_scale if self.cfg_range[0] <= i / len(timesteps) <= self.cfg_range[1] else 1.0
            image_guidance_scale = self.image_guidance_scale if self.cfg_range[0] <= i / len(timesteps) <= self.cfg_range[1] else 1.0
            
            if text_guidance_scale > 1.0 and image_guidance_scale > 1.0:
                if enable_taylorseer:
                    self.transformer.cache_dic = model_pred_ref_cache_dic
                    self.transformer.current = model_pred_ref_current
                # elif self.transformer.enable_teacache:
                #     teacache_params_ref.is_first_or_last_step = i == 0 or i == len(timesteps) - 1
                #     self.transformer.teacache_params = teacache_params_ref

                model_pred_ref = self.predict(
                    t=t,
                    latents=latents,
                    prompt_embeds=negative_prompt_embeds,
                    freqs_cis=freqs_cis,
                    prompt_attention_mask=negative_prompt_attention_mask,
                    ref_image_hidden_states=ref_latents,
                )

                if enable_taylorseer:
                    self.transformer.cache_dic = model_pred_uncond_cache_dic
                    self.transformer.current = model_pred_uncond_current
                # elif self.transformer.enable_teacache:
                #     teacache_params_uncond.is_first_or_last_step = i == 0 or i == len(timesteps) - 1
                #     self.transformer.teacache_params = teacache_params_uncond

                model_pred_uncond = self.predict(
                    t=t,
                    latents=latents,
                    prompt_embeds=negative_prompt_embeds,
                    freqs_cis=freqs_cis,
                    prompt_attention_mask=negative_prompt_attention_mask,
                    ref_image_hidden_states=None,
                )

                model_pred = model_pred_uncond + image_guidance_scale * (model_pred_ref - model_pred_uncond) + \
                    text_guidance_scale * (model_pred - model_pred_ref)
            elif text_guidance_scale > 1.0:
                if enable_taylorseer:
                    self.transformer.cache_dic = model_pred_uncond_cache_dic
                    self.transformer.current = model_pred_uncond_current
                # elif self.transformer.enable_teacache:
                #     teacache_params_uncond.is_first_or_last_step = i == 0 or i == len(timesteps) - 1
                #     self.transformer.teacache_params = teacache_params_uncond

                model_pred_uncond = self.predict(
                    t=t,
                    latents=latents,
                    prompt_embeds=negative_prompt_embeds,
                    freqs_cis=freqs_cis,
                    prompt_attention_mask=negative_prompt_attention_mask,
                    ref_image_hidden_states=None,
                )
                model_pred = model_pred_uncond + text_guidance_scale * (model_pred - model_pred_uncond)

            # latents = self.scheduler.step(model_pred, t, latents, return_dict=False)[0]
            # log_prob = torch.randn_like(latents)
            
            latents, log_prob, prev_latents_mean, std_dev_t = sde_step_with_logprob(
                self=self.scheduler, 
                model_output=model_pred.float(), 
                # t.unsqueeze(0).repeat(latents.shape[0]), 
                timestep=t, 
                sample=latents.float(),
                noise_level=noise_level,
            )

            latents = latents.to(dtype=dtype)
            
            all_latents.append(latents)
            all_log_probs.append(log_prob)

            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()
            
            if step_func is not None:
                step_func(i, self._num_timesteps)

    if enable_taylorseer:
        del model_pred_cache_dic, model_pred_ref_cache_dic, model_pred_uncond_cache_dic
        del model_pred_current, model_pred_ref_current, model_pred_uncond_current

    latents = latents.to(dtype=dtype)
    if self.vae.config.scaling_factor is not None:
        latents = latents / self.vae.config.scaling_factor
    if self.vae.config.shift_factor is not None:
        latents = latents + self.vae.config.shift_factor
    image = self.vae.decode(latents, return_dict=False)[0]

    image = F.interpolate(image, size=(ori_height, ori_width), mode='bilinear')

    image = self.image_processor.postprocess(image, output_type=output_type)
    
    # Offload all models
    self.maybe_free_model_hooks()

    # 图片，所有的潜空间，所有的生图概率，频率，参考潜空间（是收入图片的）
    data = {
        "image": image,
        "all_latents": all_latents,
        "all_log_probs": all_log_probs,
        "ref_latents": ref_latents,
        
        # 加上这些，就不需要 encode prompt 了
        "prompt_embeds": prompt_embeds,
        "prompt_attention_mask": prompt_attention_mask,
        "negative_prompt_embeds": negative_prompt_embeds,
        "negative_prompt_attention_mask": negative_prompt_attention_mask,
    }
    
    return data

