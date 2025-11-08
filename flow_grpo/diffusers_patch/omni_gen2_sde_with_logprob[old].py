# Copied from https://github.com/kvablack/ddpo-pytorch/blob/main/ddpo_pytorch/diffusers_patch/ddim_with_logprob.py
# We adapt it from flow to flow matching.

import math
from typing import Optional, Union
import torch

from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

def sde_step_with_logprob(
    self: FlowMatchEulerDiscreteScheduler,
    model_output: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    sample: torch.FloatTensor,
    noise_level: float = 0.7,
    prev_sample: Optional[torch.FloatTensor] = None,
    generator: Optional[torch.Generator] = None,
    sde_type: Optional[str] = 'sde',
    epsilon: Optional[float] = 1e-8,
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
    model_output=model_output.float()
    sample=sample.float()
    if prev_sample is not None:
        prev_sample=prev_sample.float()

        # 使用与ODE相同的索引逻辑
    if self.step_index is None:
        self._init_step_index(timestep)
    
    sigma = self._sigmas[self.step_index]
    sigma_next = self._sigmas[self.step_index + 1]
    dt = sigma_next - sigma  # dt为正（正向过程）
    
    # 调整形状以匹配sample
    sigma = sigma.view(-1, *([1] * (len(sample.shape) - 1)))
    dt = dt.view(-1, *([1] * (len(sample.shape) - 1)))
    
    # if sde_type == 'sde':
    # 数值稳定性保护
    safe_sigma = torch.clamp(sigma, epsilon, 1.0 - epsilon)
    
    # 简化的正向SDE：dx = v * dt + noise_level * sqrt(sigma) * sqrt(dt) * dW
    drift = model_output * dt  # 确定性漂移项（与ODE相同）
    
    # 随机扩散项（针对正向过程设计）
    diffusion_std = noise_level * torch.sqrt(safe_sigma * dt + epsilon)
    noise = torch.randn_like(sample) * diffusion_std
    
    prev_sample = sample + drift + noise
    prev_sample_mean = sample + drift  # 确定性部分
    
    # 计算log概率
    log_prob = -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * diffusion_std**2 + epsilon)
    log_prob = log_prob - 0.5 * torch.log(2 * torch.tensor(math.pi) * (diffusion_std**2 + epsilon))
    
    std_dev_t = noise_level * torch.sqrt(safe_sigma)  # 用于返回
        
    # 更新步索引
    self._step_index += 1
    
    return prev_sample, log_prob.mean(dim=tuple(range(1, log_prob.ndim))), prev_sample_mean, std_dev_t
