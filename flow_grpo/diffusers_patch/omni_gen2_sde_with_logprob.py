# Copied from https://github.com/kvablack/ddpo-pytorch/blob/main/ddpo_pytorch/diffusers_patch/ddim_with_logprob.py
# We adapt it from flow to flow matching.

import math
from typing import Optional, Union
import torch

from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from omnigen2.schedulers.scheduling_flow_match_euler_maruyama_discrete import FlowMatchEulerMaruyamaDiscreteScheduler


def expand_as(tensor, other):
    """
    Expands a tensor to match the dimensions of another tensor.
    
    If tensor has shape [b] and other has shape [b, c, h, w],
    this function will reshape tensor to [b, 1, 1, 1] to enable broadcasting.
    
    Args:
        tensor (`torch.FloatTensor`): The tensor to expand
        other (`torch.FloatTensor`): The tensor whose shape will be matched
        
    Returns:
        `torch.FloatTensor`: The expanded tensor
    """
    for _ in range(other.ndim - tensor.ndim):
        tensor = tensor.unsqueeze(-1)
    return tensor


def sde_step_with_logprob(
    self: Union[FlowMatchEulerDiscreteScheduler, FlowMatchEulerMaruyamaDiscreteScheduler],
    model_output: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    sample: torch.FloatTensor,
    prev_sample: Optional[torch.FloatTensor] = None,
    generator: Optional[torch.Generator] = None,
    noise_level: float = 0.7,
    sde_type: Optional[str] = 'sde',
):
    """
    Predict the sample from the previous timestep by reversing the SDE. This function propagates the flow
    process from the learned model outputs (most often the predicted velocity).

    Args:
        model_output (`torch.FloatTensor`):
            The direct output from learned flow model.
        timestep (`float`):
            The current discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`):
            A current instance of a sample created by the diffusion process.
        generator (`torch.Generator`, *optional*):
            A random number generator.
    """
    # bf16 can overflow here when compute prev_sample_mean, we must convert all variable to fp32
    model_output = model_output.to(dtype=torch.float32)
    sample = sample.to(dtype=torch.float32)
    if prev_sample is not None:
        prev_sample = prev_sample.to(dtype=torch.float32)

    # if self.step_index is None:
    #     self._init_step_index(timestep)
    # print("[DEBUG] self.step_index: ", self.step_index)
    
    # Upcast to avoid precision issues when computing prev_sample
    # sample = sample.to(torch.float32)
    # step_index = [self.index_for_timestep(t) for t in timestep]
    # prev_step_index = [step + 1 for step in step_index]
    
    step_index = self.index_for_timestep(timestep)
    # print("[DEBUG] step_index: ", step_index)
    prev_step_index = step_index + 1
    t = self._timesteps[step_index].view(-1, *([1] * (len(sample.shape) - 1)))
    t_next = self._timesteps[prev_step_index].view(-1, *([1] * (len(sample.shape) - 1)))

    sigma_t = self.get_sigma_t(t, t_next if step_index == 0 else None).view(-1, *([1] * (len(sample.shape) - 1)))

    # sigma_t = expand_as(sigma_t, sample)
    # t = expand_as(t, sample)
    # t_next = expand_as(t_next, sample)

    dt = t_next - t

    sigma_t = sigma_t.to(dtype=torch.float32)
    t = t.to(dtype=torch.float32)
    t_next = t_next.to(dtype=torch.float32)
    dt = dt.to(dtype=torch.float32)

    # 这里的 sigma_t 是sd3那边的 std_dev_t
    # 这里的 1-t 是sd3那边的 sigma
    prev_sample_mean = (
        sample * (1 - sigma_t**2 / (2 * (1 - t)) * dt)
        + model_output * (1 + sigma_t**2 * t / (2 * (1 - t))) * dt
    )
    variance_noise = randn_tensor(
        model_output.shape,
        generator=generator,
        device=sample.device,
        dtype=sample.dtype,
    )
    prev_sample = (
        prev_sample_mean + sigma_t * torch.sqrt(dt) * variance_noise
    )

    # if img_mask is not None:
    #     img_mask = expand_as(img_mask, sample).expand(sample.shape)
    #     prev_sample = prev_sample * img_mask

    # if return_log_prob: TODO 这个跟之前的还不一样了，是不是需要重新计算一下？
    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2)
        / (2 * ((sigma_t ** 2) * dt))
        - torch.log(sigma_t * torch.sqrt(dt))
        - 0.5 * torch.log(2 * torch.as_tensor(math.pi, device=sample.device))
    )

    # log_prob = (log_prob * img_mask).sum(
    #     dim=tuple(range(-log_prob.ndim + 1, 0)), dtype=torch.float32
    # ) / img_mask.sum(
    #     dim=tuple(range(-log_prob.ndim + 1, 0)), dtype=torch.float32
    # )
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    
    # self._step_index += 1
    
    # prev_sample, prev_sample_mean, log_prob = self.step(model_output, timestep, sample, return_log_prob=True, return_dict=False)[0]
    
    if torch.isnan(prev_sample_mean).any():
        print("[WARNING] NaN detected in prev_sample_mean!!")
    
    return prev_sample, log_prob, prev_sample_mean, sigma_t


# def forward_logprob(
#     latents: List[torch.Tensor],
#     latents_next: List[torch.Tensor],
#     t: torch.Tensor,
#     t_next: torch.Tensor,
#     step_index: int,
#     img_mask,
#     model,
#     model_kwargs: Dict[str, Any],
#     model_pred_kwargs: Dict[str, Any],
#     # model_pred_ref_kwargs: Dict[str, Any],
#     # model_pred_uncond_kwargs: Dict[str, Any],
#     scheduler,
#     apply_cfg: bool = True,
#     text_guidance_scale: float = 1.0,
#     image_guidance_scale: float = 1.0,
# ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     # cfg
    
#     model_pred = model(**model_kwargs, **model_pred_kwargs)
#     if apply_cfg:
#         if text_guidance_scale > 1.0 and image_guidance_scale > 1.0:
#             model_pred, model_pred_ref, model_pred_uncond = model_pred.chunk(3)
#             model_pred = (
#                 model_pred_uncond
#                 + image_guidance_scale * (model_pred_ref - model_pred_uncond)
#                 + text_guidance_scale * (model_pred - model_pred_ref)
#             )
#         elif text_guidance_scale > 1.0:
#             model_pred, model_pred_uncond = model_pred.chunk(2)
#             model_pred = model_pred_uncond + text_guidance_scale * (model_pred - model_pred_uncond)


#     sigma_t = scheduler.get_sigma_t(t, t_next if step_index == 0 else None)  # [batch_size]
#     sigma_t = expand_as(sigma_t.unsqueeze(1), latents)  # [batch_size, max_img_len, dim]
#     t = expand_as(t.unsqueeze(1), latents)  # [batch_size, max_img_len, dim]
#     t_next = expand_as(t_next.unsqueeze(1), latents)  # [batch_size, max_img_len, dim]
#     dt = t_next - t
    
#     sigma_t = sigma_t.to(dtype=torch.float32)
#     t = t.to(dtype=torch.float32)
#     t_next = t_next.to(dtype=torch.float32)
#     dt = dt.to(dtype=torch.float32)

#     prev_sample_mean = (
#         latents.to(dtype=torch.float32) * (1 - sigma_t**2 / (2 * (1 - t)) * dt)
#         + model_pred * (1 + sigma_t**2 * t / (2 * (1 - t))) * dt
#     )

#     log_prob = (
#         -((latents_next.to(dtype=torch.float32).detach() - prev_sample_mean) ** 2)
#         / (2 * (sigma_t**2 * dt))  # Fix: denominator is 2 * σ² * dt
#         - torch.log(sigma_t * torch.sqrt(dt))  # Fix: log(σ * √dt)
#         - 0.5
#         * torch.log(
#             2 * torch.as_tensor(math.pi, device=latents.device)
#         )  # Fix: 0.5 coefficient
#     )

#     img_mask = expand_as(img_mask, latents).expand(latents.shape)
#     log_prob = (log_prob * img_mask.detach()).sum(
#         dim=tuple(range(-log_prob.ndim + 1, 0)), dtype=torch.float32
#     ) / img_mask.detach().sum(
#         dim=tuple(range(-log_prob.ndim + 1, 0)), dtype=torch.float32
#     )
   
#     return log_prob, prev_sample_mean, sigma_t**2