from typing import Any, Dict, List, Optional, Union
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
    image: Optional[Union[PipelineImageInput, List[PipelineImageInput]]] = None,  # 输入图像（多图）
    prompt: Union[str, List[str]] = None,  # 文本提示
    negative_prompt: Union[str, List[str]] = None,  # 负向提示
    true_cfg_scale: float = 4.0,  # 真实的分类器自由引导尺度
    height: Optional[int] = None,  # 输出图像高度
    width: Optional[int] = None,  # 输出图像宽度
    num_inference_steps: int = 50,  # 推理步数
    sigmas: Optional[List[float]] = None,  # 噪声调度参数
    guidance_scale: Optional[float] = None,  # 引导尺度
    num_images_per_prompt: int = 1,  # 每个提示生成的图像数量
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,  # 随机数生成器
    latents: Optional[torch.Tensor] = None,  # 初始潜变量
    # prompt_embeds: Optional[torch.Tensor] = None,  # 预计算的提示嵌入
    # prompt_embeds_mask: Optional[torch.Tensor] = None,  # 提示嵌入掩码
    # negative_prompt_embeds: Optional[torch.Tensor] = None,  # 负向提示嵌入
    # negative_prompt_embeds_mask: Optional[torch.Tensor] = None,  # 负向提示嵌入掩码
    output_type: Optional[str] = "pil",  # 输出类型
    attention_kwargs: Optional[Dict[str, Any]] = None,  # 注意力机制参数
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],  # 步结束回调的输入张量
    # max_sequence_length: int = 512,  # 最大序列长度
    noise_level: float = 0.7,  # 噪声水平
    process_index: int = 0,  # 进程索引
    sde_window_size: int = 0,  # SDE窗口大小
    sde_window_range: tuple[int, int] = (0, 5),  # SDE窗口范围
):
    """
    基于Stable Diffusion 3的图像生成管道，支持对数概率计算和SDE（随机微分方程）训练

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
    # 计算输出图像尺寸
    # image_size = image[0].size if isinstance(image, list) else image.size
    # calculated_width, calculated_height, _ = calculate_dimensions(1024 * 1024, image_size[0] / image_size[1])
    # height = height or calculated_height
    # width = width or calculated_width
    height = height
    width = width
    # 调整尺寸为VAE缩放因子和2的倍数
    multiple_of = self.vae_scale_factor * 2
    width = width // multiple_of * multiple_of
    height = height // multiple_of * multiple_of
    
    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        image,
        height,
        width,
        # negative_prompt=negative_prompt,
        # prompt_embeds=prompt_embeds,
        # negative_prompt_embeds=negative_prompt_embeds,
        # prompt_embeds_mask=prompt_embeds_mask,
        # negative_prompt_embeds_mask=negative_prompt_embeds_mask,
        use_input_image_size_as_output=False,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        # max_sequence_length=max_sequence_length,
    )

    # 设置模型参数
    self._guidance_scale = guidance_scale
    self._attention_kwargs = attention_kwargs
    self._current_timestep = None
    self._interrupt = False

    # 2. Define call parameters
    # 根据输入确定批次大小
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
        prompt = [prompt]  # 转换为列表以支持batch处理
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    # elif prompt_embeds is not None:
    #     batch_size = prompt_embeds.shape[0]
    else:
        raise ValueError("必须提供prompt")

    device = self._execution_device  # 获取执行设备
    
    # 3. Preprocess image
    # 图像预处理：调整尺寸和预处理
    # 修改图像预处理逻辑，处理图像列表
        # 3. 多图像预处理 - 支持batch处理
    if image is not None:
        if isinstance(image, list):
            # 情况1: image是图像列表，每个元素可能是一张或多张图像
            processed_batches = []
            
            for img_item in image:
                if isinstance(img_item, list):
                    # 子列表：一个样本包含多张图像
                    processed_images = []
                    for img in img_item:
                        if not (isinstance(img, torch.Tensor) and img.size(1) == self.latent_channels):
                            img_resized = self.image_processor.resize([img], height, width)
                            img_processed = self.image_processor.preprocess(img_resized, height, width)
                            processed_images.append(img_processed)
                    
                    if processed_images:
                        # 堆叠多张图像 [num_images, C, H, W]
                        img_tensor = torch.stack(processed_images)
                        processed_batches.append(img_tensor)
                    else:
                        processed_batches.append(None)
                else:
                    # 单张图像
                    if not (isinstance(img_item, torch.Tensor) and img_item.size(1) == self.latent_channels):
                        img_resized = self.image_processor.resize([img_item], height, width)
                        img_processed = self.image_processor.preprocess(img_resized, height, width)
                        processed_batches.append(img_processed.unsqueeze(0))  # 添加batch维度
                    else:
                        processed_batches.append(img_item.unsqueeze(0))
            
            # 检查批次一致性
            if batch_size != len(processed_batches):
                raise ValueError(f"prompt数量({batch_size})与image数量({len(processed_batches)})不匹配")
            
            # 合并所有batch
            try:
                # 尝试沿batch维度拼接
                prompt_image = torch.cat(processed_batches, dim=0)
                image = prompt_image.unsqueeze(2) if prompt_image.dim() == 4 else prompt_image
            except:
                # 如果形状不一致，使用列表处理
                prompt_image = processed_batches
                image = prompt_image
        else:
            # 单张图像输入，扩展到batch大小
            if not (isinstance(image, torch.Tensor) and image.size(1) == self.latent_channels):
                image = self.image_processor.resize(image, height, width)
                prompt_image = self.image_processor.preprocess(image, height, width)
                # 扩展到batch大小
                if batch_size > 1:
                    prompt_image = prompt_image.repeat(batch_size, 1, 1, 1)
                image = prompt_image.unsqueeze(2)
            else:
                prompt_image = image
    else:
        prompt_image = None
        image = None
    
    # 检查是否存在负向提示
    has_neg_prompt = negative_prompt is not None
    
    if negative_prompt is not None and isinstance(negative_prompt, str):
        negative_prompt = [negative_prompt] * batch_size
    
    # # 编码提示文本为嵌入向量
    # prompt_embeds, prompt_embeds_mask = self.encode_prompt(
    #     image=torch.cat([prompt_image, prompt_image], dim=0),  # 拼接正负提示的图像
    #     prompt=prompt+negative_prompt,  # 合并正负提示
    #     prompt_embeds=prompt_embeds,
    #     prompt_embeds_mask=prompt_embeds_mask,
    #     device=device,
    #     num_images_per_prompt=num_images_per_prompt,
    #     max_sequence_length=max_sequence_length,
    # )
    
    # 编码提示文本为嵌入向量
    # 5. 文本编码 - 支持batch处理
    if prompt_image is not None:
        if isinstance(prompt_image, torch.Tensor):
            # 使用相同的图像用于所有提示（或根据batch扩展）
            if has_neg_prompt:
                batch_images = torch.cat([prompt_image, prompt_image], dim=0)
            else:
                batch_images = prompt_image
        else:
            # 图像列表，需要特殊处理
            batch_images = prompt_image
    else:
        batch_images = None

    prompt_embeds, prompt_embeds_mask = self.encode_prompt(
        image=batch_images,
        prompt=prompt + (negative_prompt if negative_prompt else []),
        # prompt_embeds=prompt_embeds,
        # prompt_embeds_mask=prompt_embeds_mask,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        # max_sequence_length=max_sequence_length,
    )
    
    # 分割正负提示嵌入
    if has_neg_prompt:
        prompt_embeds, negative_prompt_embeds = prompt_embeds.chunk(2, dim=0)
        prompt_embeds_mask, negative_prompt_embeds_mask = prompt_embeds_mask.chunk(2, dim=0)
    
    # 4. Prepare latent variables
    # 准备潜变量
    num_channels_latents = self.transformer.config.in_channels // 4
    # 计算总样本数（考虑num_images_per_prompt）
    total_batch_size = batch_size * num_images_per_prompt
    latents, image_latents = self.prepare_latents(
        image,
        total_batch_size,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )
    
    # # 定义图像形状信息
    # img_shapes = [
    #     [
    #         (1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2),
    #         (1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2),  # 统一使用1024
    #         # (1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2),
    #         # (1, calculated_height // self.vae_scale_factor // 2, calculated_width // self.vae_scale_factor // 2),
    #     ]
    # ] * batch_size
    
    
    # 7. 定义图像形状信息 - 扩展到batch大小
    base_img_shape = [
        (1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2),
        (1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2),
    ]
    img_shapes = [base_img_shape] * batch_size
    
    # 5. Prepare timesteps
    # 准备时间步
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
    image_seq_len = latents.shape[1]
    
    # 计算shift参数用于时间步调整
    mu = calculate_shift(
        image_seq_len,
        self.scheduler.config.get("base_image_seq_len", 256),
        self.scheduler.config.get("max_image_seq_len", 4096),
        self.scheduler.config.get("base_shift", 0.5),
        self.scheduler.config.get("max_shift", 1.15),
    )
    
    # 获取时间步
    # 8. 准备时间步 - 扩展到batch大小
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        mu=mu,
    )
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
    self._num_timesteps = len(timesteps)

    # 9. 处理引导参数 - 扩展到batch大小
    if self.transformer.config.guidance_embeds and guidance_scale is None:
        raise ValueError("guidance_scale is required for guidance-distilled model.")
    elif self.transformer.config.guidance_embeds:
        guidance = torch.full([batch_size], guidance_scale, device=device, dtype=torch.float32)
        guidance = guidance.expand(total_batch_size)
        # guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        # guidance = guidance.expand(latents.shape[0])
    elif not self.transformer.config.guidance_embeds and guidance_scale is not None:
        logger.warning(
            f"guidance_scale is passed as {guidance_scale}, but ignored since the model is not guidance-distilled."
        )
        guidance = None
    elif not self.transformer.config.guidance_embeds and guidance_scale is None:
        guidance = None
        
    if self.attention_kwargs is None:
        self._attention_kwargs = {}

    # 计算文本序列长度
    txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist() if prompt_embeds_mask is not None else None
    negative_txt_seq_lens = (
        negative_prompt_embeds_mask.sum(dim=1).tolist() if negative_prompt_embeds_mask is not None else None
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
    print(f'sde_window: {sde_window}')
    
    # 6. Prepare image embeddings
    # 初始化存储列表
    all_latents = []
    all_log_probs = []
    all_timesteps = []

    # 7. Denoising loop
    # 去噪循环
    self.scheduler.set_begin_index(0)
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
                
            self._current_timestep = t
            timestep = t.expand(latents.shape[0]).to(latents.dtype)
            latent_model_input = latents
            
            # 拼接潜变量和图像潜变量
            if image_latents is not None:
                latent_model_input = torch.cat([latents, image_latents], dim=1)
                
            # 前向传播获取噪声预测 - 支持batch
            # 注意：这里需要根据batch_size调整拼接逻辑
            if has_neg_prompt:
                # 拼接正负样本
                noise_pred = self.transformer(
                    hidden_states=torch.cat([latent_model_input, latent_model_input], dim=0),
                    timestep=torch.cat([timestep, timestep], dim=0) / 1000,
                    guidance=guidance,
                    encoder_hidden_states_mask=torch.cat([prompt_embeds_mask, negative_prompt_embeds_mask], dim=0),
                    encoder_hidden_states=torch.cat([prompt_embeds, negative_prompt_embeds], dim=0),
                    img_shapes=img_shapes * 2,
                    txt_seq_lens=txt_seq_lens + negative_txt_seq_lens,
                )[0]
                
                # 分割正负噪声预测
                noise_pred, neg_noise_pred = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred[:, :latents.size(1)]
                neg_noise_pred = neg_noise_pred[:, :latents.size(1)]
                
                # 应用分类器自由引导
                comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)
            else:
                # 没有负向提示的情况
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    encoder_hidden_states=prompt_embeds,
                    img_shapes=img_shapes,
                    txt_seq_lens=txt_seq_lens,
                )[0]
                noise_pred = noise_pred[:, :latents.size(1)]
                comb_pred = noise_pred
            
            # 归一化处理
            if has_neg_prompt:
                cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                noise_pred = comb_pred * (cond_norm / noise_norm)
            else:
                noise_pred = comb_pred
            
            # # 归一化处理
            # cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
            # noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
            # noise_pred = comb_pred * (cond_norm / noise_norm)
            
            # 执行SDE步并计算对数概率
            latents_dtype = latents.dtype
            latents, log_prob, prev_latents_mean, std_dev_t = sde_step_with_logprob(
                self.scheduler, 
                noise_pred.float(), 
                t.unsqueeze(0).repeat(latents.shape[0]), 
                latents.float(),
                noise_level=cur_noise_level,
            )
            
            # 保持数据类型一致
            if latents.dtype != latents_dtype:
                latents = latents.to(latents_dtype)
                
            # 在SDE窗口内记录潜变量和对数概率
            if i >= sde_window[0] and i < sde_window[1]:
                all_latents.append(latents)
                all_log_probs.append(log_prob)
                all_timesteps.append(t)
                
            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()

    # # 后处理：解包潜变量并通过VAE解码生成图像
    # latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
    # latents = latents.to(self.vae.dtype)
    
    # # 应用VAE的均值和标准差归一化
    # latents_mean = (
    #     torch.tensor(self.vae.config.latents_mean)
    #     .view(1, self.vae.config.z_dim, 1, 1, 1)
    #     .to(latents.device, latents.dtype)
    # )
    # latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
    #     latents.device, latents.dtype
    # )
    # latents = latents / latents_std + latents_mean
    
    # # VAE解码生成图像
    # image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
    # image = self.image_processor.postprocess(image, output_type=output_type)
    
    # 14. 后处理 - 支持batch
    # 注意：这里需要处理batch维度的解包
    processed_images = []
    for i in range(latents.shape[0]):
        # 对每个样本单独处理
        sample_latents = latents[i:i+1]  # 保持batch维度
        sample_latents = self._unpack_latents(sample_latents, height, width, self.vae_scale_factor)
        sample_latents = sample_latents.to(self.vae.dtype)
        
        # 应用VAE的均值和标准差归一化
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(sample_latents.device, sample_latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            sample_latents.device, sample_latents.dtype
        )
        sample_latents = sample_latents / latents_std + latents_mean
        
        # VAE解码生成图像
        sample_image = self.vae.decode(sample_latents, return_dict=False)[0][:, :, 0]
        processed_images.append(sample_image)
    
    # 合并所有batch的图像
    image_batch = torch.cat(processed_images, dim=0)
    image = self.image_processor.postprocess(image_batch, output_type=output_type)
    
    # Offload all models
    self.maybe_free_model_hooks()
    
    # 返回结果字典
    ret = {
        "images": image,
        "all_latents": all_latents,  # 所有潜变量
        "all_log_probs": all_log_probs,  # 所有对数概率
        "all_timesteps": all_timesteps,  # 所有时间步
        "prompt_embeds": prompt_embeds,  # 提示嵌入
        # "negative_prompt_embeds": negative_prompt_embeds if has_neg_prompt else None,  # 负向提示嵌入
        "prompt_embeds_mask": prompt_embeds_mask,  # 提示嵌入掩码
        # "negative_prompt_embeds_mask": negative_prompt_embeds_mask if has_neg_prompt else None,  # 负向提示嵌入掩码
        "image_latents": image_latents,  # 图像潜变量
    }
    return ret
