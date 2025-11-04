from typing import Any, Callable, Dict, List, Optional, Union
import torch
import torch.distributed as dist
import numpy as np
import random
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
# from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit import calculate_shift, calculate_dimensions

from diffusers.image_processor import PipelineImageInput
from .sd3_sde_with_logprob import sde_step_with_logprob

# Copied from diffusers.pipelines.qwenimage.pipeline_qwenimage.calculate_shift
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

def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio

    width = round(width / 32) * 32
    height = round(height / 32) * 32

    return width, height, None


@torch.no_grad()  # 禁用梯度计算，用于推理阶段
def pipeline_with_logprob(
    self,
    prompt: Union[str, List[str]],
    input_images: Union[PipelineImageInput, List[PipelineImageInput]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    max_input_image_size: int = 1024,
    timesteps: List[int] = None,
    guidance_scale: float = 2.5,
    img_guidance_scale: float = 1.6,
    use_input_image_size_as_output: bool = False,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.Tensor] = None,
    output_type: Optional[str] = "pil",
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    noise_level: float = 0.7,  # 噪声水平
    process_index: int = 0,  # 进程索引
    sde_window_size: int = 0,  # SDE窗口大小
    sde_window_range: tuple[int, int] = (0, 5),  # SDE窗口范围
):
    """
    基于图像生成管道，支持对数概率计算和SDE（随机微分方程）训练

    这个函数是Stable Diffusion 3管道的扩展版本，主要特点包括：
    1. 支持在去噪过程中计算每一步的对数概率
    2. 提供SDE窗口机制，可在指定时间步范围内添加随机噪声
    3. 返回生成过程中的中间结果，便于模型训练和分析

    主要应用场景：
    - 扩散模型的强化学习训练（如DPO、PPO等）
    - 概率模型的可视化和分析
    - 需要获取生成过程详细信息的实验研究

    使用注意事项：
    - 函数使用@torch.no_grad()装饰器，不计算梯度，适合推理阶段
    - 返回的log_prob可用于计算损失函数，但需要相应的训练逻辑
    - SDE窗口机制有助于在特定时间步范围内进行训练采样
    """
    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor
    num_cfg = 2 if input_images is not None else 1
    use_img_cfg = True if input_images is not None else False
    if isinstance(prompt, str):
        prompt = [prompt]
        input_images = [input_images]

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        input_images,
        height,
        width,
        use_input_image_size_as_output,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
    )

    self._guidance_scale = guidance_scale
    self._interrupt = False

    # 2. Define call parameters
    batch_size = len(prompt)
    device = self._execution_device

    # 3. process multi-modal instructions
    if max_input_image_size != self.multimodal_processor.max_image_size:
        self.multimodal_processor.reset_max_image_size(max_image_size=max_input_image_size)
    processed_data = self.multimodal_processor(
        prompt,
        input_images,
        height=height,
        width=width,
        use_img_cfg=use_img_cfg,
        use_input_image_size_as_output=use_input_image_size_as_output,
        num_images_per_prompt=num_images_per_prompt,
    )
    # processed_data的内部是这样的： {
    #     "input_ids": all_padded_input_ids,
    #     "attention_mask": all_attention_mask,
    #     "position_ids": all_position_ids,
    #     "input_pixel_values": all_pixel_values,
    #     "input_image_sizes": all_image_sizes,
    # }
    processed_data["input_ids"] = processed_data["input_ids"].to(device)
    processed_data["attention_mask"] = processed_data["attention_mask"].to(device)
    processed_data["position_ids"] = processed_data["position_ids"].to(device)

    # 4. Encode input images
    input_img_latents = self.encode_input_images(processed_data["input_pixel_values"], device=device)

    # 5. Prepare timesteps
    sigmas = np.linspace(1, 0, num_inference_steps + 1)[:num_inference_steps]
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler, num_inference_steps, device, timesteps, sigmas=sigmas
    )
    self._num_timesteps = len(timesteps)

    # 6. Prepare latents
    transformer_dtype = self.transformer.dtype
    if use_input_image_size_as_output:
        height, width = processed_data["input_pixel_values"][0].shape[-2:]
    latent_channels = self.transformer.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        latent_channels,
        height,
        width,
        torch.float32,
        device,
        generator,
        latents,
    )
    
    # 设置随机种子
    random.seed(process_index)
    
    # 计算SDE窗口
    # 假设一共denoising10步，sde_window_size=2, sde_window_range=(0, 10), sde_window可能是(0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 8), (7, 9), (8, 10)
    if sde_window_size > 0:
        start = random.randint(sde_window_range[0], sde_window_range[1] - sde_window_size)
        end = start + sde_window_size
        sde_window = (start, end)
    else:
        # 最后一步接近图片，高斯分布很尖，概率很大，容易精度溢出，不参与训练
        sde_window = (0, len(timesteps)-1)
    # print(f'sde_window: {sde_window}')
    
    # 初始化存储列表
    all_latents = []
    all_log_probs = []
    all_timesteps = []

    # 8. Denoising loop
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            # 确定当前步骤的噪声水平
            if i < sde_window[0]:
                cur_noise_level = 0
            elif i == sde_window[0]:
                cur_noise_level= noise_level
                all_latents.append(latents)  # 记录初始潜变量
            elif i > sde_window[0] and i < sde_window[1]:
                cur_noise_level = noise_level
            else:
                cur_noise_level= 0
            
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * (num_cfg + 1))
            latent_model_input = latent_model_input.to(transformer_dtype)

            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latent_model_input.shape[0])
            
            print(f"pipeline中 latents shape: {latents.shape}")
            print(f"pipeline中 latent_model_input shape: {latent_model_input.shape}")
            print(f"pipeline中 input_ids shape: {processed_data['input_ids'].shape}")
            print(f"pipeline中 input_img_latents shape: {[img_latent.shape for img_latent in input_img_latents]}")

            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                input_ids=processed_data["input_ids"],
                input_img_latents=input_img_latents,
                input_image_sizes=processed_data["input_image_sizes"],
                attention_mask=processed_data["attention_mask"],
                position_ids=processed_data["position_ids"],
                return_dict=False,
            )[0]

            if num_cfg == 2:
                cond, uncond, img_cond = torch.split(noise_pred, len(noise_pred) // 3, dim=0)
                noise_pred = uncond + img_guidance_scale * (img_cond - uncond) + guidance_scale * (cond - img_cond)
            else:
                cond, uncond = torch.split(noise_pred, len(noise_pred) // 2, dim=0)
                noise_pred = uncond + guidance_scale * (cond - uncond)

            # compute the previous noisy sample x_t -> x_t-1
            # 这一步是 ODE ，我需要将其转换为 SDE
            # latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            
            latents, log_prob, prev_latents_mean, std_dev_t = sde_step_with_logprob(
                self.scheduler, 
                noise_pred.float(), 
                t.unsqueeze(0).repeat(latents.shape[0]), 
                latents.float(),
                noise_level=cur_noise_level,
            )
                
            # 在SDE窗口内记录潜变量和对数概率
            if i >= sde_window[0] and i < sde_window[1]:
                all_latents.append(latents)
                all_log_probs.append(log_prob)
                all_timesteps.append(t)

            progress_bar.update()

    if not output_type == "latent":
        latents = latents.to(self.vae.dtype)
        latents = latents / self.vae.config.scaling_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)
    else:
        image = latents

    # Offload all models
    self.maybe_free_model_hooks()
    
    # 返回结果字典
    ret = {
        "images": image,
        "all_latents": all_latents,  # 所有潜变量
        "all_log_probs": all_log_probs,  # 所有对数概率
        "all_timesteps": all_timesteps,  # 所有时间步
        # "prompt_embeds": prompt_embeds,  # 提示嵌入
        # "negative_prompt_embeds": negative_prompt_embeds if has_neg_prompt else None,  # 负向提示嵌入
        # "prompt_embeds_mask": prompt_embeds_mask,  # 提示嵌入掩码
        # "negative_prompt_embeds_mask": negative_prompt_embeds_mask if has_neg_prompt else None,  # 负向提示嵌入掩码
        "image_latents": input_img_latents,  # 图像潜变量
        "multimodal_data": processed_data,  # 【新加的】处理后的数据
    }
    return ret
