from collections import defaultdict
import contextlib
import os
import sys
import datetime
from concurrent import futures
import time
import json
import hashlib
from absl import app, flags
from ml_collections import config_flags
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data.distributed import DistributedSampler
import logging
import itertools

# Setup logger
logger = logging.getLogger(__name__)
# from diffusers import QwenImageEditPipeline, QwenImageTransformer2DModel
from diffusers import OmniGenPipeline, OmniGenTransformer2DModel
from diffusers.utils.torch_utils import is_compiled_module
# from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit import calculate_shift, calculate_dimensions

from flow_grpo.fsdp_utils import FSDPConfig, fsdp_wrapper, init_distributed, save_fsdp_checkpoint, OptimizerOffload
import numpy as np
import flow_grpo.prompts
import flow_grpo.rewards
from flow_grpo.stat_tracking import PerPromptStatTracker
# from flow_grpo.diffusers_patch.qwenimage_edit_pipeline_with_logprob import pipeline_with_logprob
from flow_grpo.diffusers_patch.omni_gen_pipline_with_logprob import pipeline_with_logprob
from flow_grpo.diffusers_patch.sd3_sde_with_logprob import sde_step_with_logprob
import torch
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, PeftModel
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from flow_grpo.ema import EMAModuleWrapper

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

def gather_tensor(tensor, world_size):
    if world_size == 1:
        return tensor
    
    gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gather_list, tensor)
    return torch.cat(gather_list)

def set_seed(seed, device_specific=True):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device_specific and torch.cuda.is_available():
        # For device-specific seeding
        torch.cuda.manual_seed_all(seed + dist.get_rank() if dist.is_initialized() else seed)

class GenevalPromptImageDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.dataset = dataset
        self.file_path = os.path.join(dataset, f'multihuman_{split}_metadata.jsonl')
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item['omnigen_prompt'] for item in self.metadatas]
            self.eval_prompts = [item['prompt'] for item in self.metadatas]
            self.vlm_questions = [item['vlm_questions'] for item in self.metadatas]
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        item = {
            "prompt": self.prompts[idx],
            "eval_prompt": self.eval_prompts[idx],
            "vlm_questions": self.vlm_questions[idx],
            "metadata": self.metadatas[idx]
        }
        # Assuming 'image' in metadata contains a path to the image file
        image_path_ls = self.metadatas[idx]['people']
        
        # item["image_path"] = image_path_ls
        
        item["prompt_with_image_path"] = f"{self.prompts[idx]}_{'_'.join(image_path_ls)}"
        # image = Image.open(os.path.join(self.dataset, image_path)).convert('RGB')
        # item["image"] = [Image.open(image_path).convert('RGB') for image_path in image_path_ls]
        item["image"] = image_path_ls
        return item

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        eval_prompts = [example["eval_prompt"] for example in examples]
        vlm_questions = [example["vlm_questions"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        images = [example["image"] for example in examples]
        prompt_with_image_paths = [example["prompt_with_image_path"] for example in examples]
        return prompts, eval_prompts, vlm_questions, metadatas, images, prompt_with_image_paths


class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size  # Batch size per replica
        self.k = k                    # Number of repetitions per sample
        self.num_replicas = num_replicas  # Total number of replicas
        self.rank = rank              # Current replica rank
        self.seed = seed              # Random seed for synchronization
        
        # Compute the number of unique samples needed per iteration
        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0, f"k can not divide n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k  # Number of unique samples
        self.epoch = 0

    def __iter__(self):
        while True:
            # Generate a deterministic random sequence to ensure all replicas are synchronized
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            
            # Randomly select m unique samples
            indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()
            
            # Repeat each sample k times to generate n*b total samples
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            
            # Shuffle to ensure uniform distribution
            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]
            
            # Split samples to each replica
            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            
            # Return current replica's sample indices
            yield per_card_samples[self.rank]
    
    def set_epoch(self, epoch):
        self.epoch = epoch  # Used to synchronize random state across epochs


def calculate_zero_std_ratio(prompts, gathered_rewards):
    """
    Calculate the proportion of unique prompts whose reward standard deviation is zero.
    
    Args:
        prompts: List of prompts.
        gathered_rewards: Dictionary containing rewards, must include the key 'ori_avg'.
        
    Returns:
        zero_std_ratio: Proportion of prompts with zero standard deviation.
        prompt_std_devs: Mean standard deviation across all unique prompts.
    """
    # Convert prompt list to NumPy array
    prompt_array = np.array(prompts)
    
    # Get unique prompts and their group information
    unique_prompts, inverse_indices, counts = np.unique(
        prompt_array, 
        return_inverse=True,
        return_counts=True
    )
    
    # Group rewards for each prompt
    grouped_rewards = gathered_rewards['ori_avg'][np.argsort(inverse_indices)]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    
    # Calculate standard deviation for each group
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    
    # Calculate the ratio of zero standard deviation
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)
    
    return zero_std_ratio, prompt_std_devs.mean()

def create_generator(prompts, base_seed):
    generators = []
    for prompt in prompts:
        # Use a stable hash (SHA256), then convert it to an integer seed
        hash_digest = hashlib.sha256(prompt.encode()).digest()
        prompt_hash_int = int.from_bytes(hash_digest[:4], 'big')  # Take the first 4 bytes as part of the seed
        seed = (base_seed + prompt_hash_int) % (2**31) # Ensure the number is within a valid range
        gen = torch.Generator().manual_seed(seed)
        generators.append(gen)
    return generators


def compute_log_prob(transformer, pipeline, sample, j, config, rank, samples_prosessed_tmp_data):
    """
    再次基于之前存储的中间结果，计算当前时间步的 log prob
    transformer: phi3 的 DiT
    pipeline: 原始的 pipeline，用于调用 scheduler
    sample: 包含所有中间结果的字典
    j: 当前时间步
    config: 一些默认的参数
    rank: 当前的机器，似乎在这没用
    """
    
    num_cfg = 2
    
    
    # print(f"sample['latents'] shape: {sample['latents'].shape}")
    # print(f"j: {j}")
    # print(f"sample['latents'][:, j] shape: {sample['latents'][:, j].shape}")
    
    latent_model_input = torch.cat([sample["latents"][:, j]] * (num_cfg + 1))
    # print(f"latent_model_input shape: {latent_model_input.shape}")
    latent_model_input = latent_model_input.to(transformer.dtype)
    
    timestep = sample["timesteps"][:, j].expand(latent_model_input.shape[0])
    # print(f"timestep shape: {timestep.shape}")
    
    
    multimodal_data = samples_prosessed_tmp_data["multimodal_data"]
    # print(f"input_ids shape: {multimodal_data['input_ids'].shape}")
    # print(f"attention_mask shape: {multimodal_data['attention_mask'].shape}")
    # print(f"position_ids shape: {multimodal_data['position_ids'].shape}")
    # print(f"input_image_sizes: {multimodal_data['input_image_sizes']}")
    
    # 检查图像潜变量
    # print(f"image_latents shape: {[img_latent.shape for img_latent in samples_prosessed_tmp_data['image_latents']]}")
    
    # 扩展多模态数据以匹配批次尺寸
    # batch_size = latent_model_input.shape[0]
    # orig_batch_size = multimodal_data["input_ids"].shape[0]
    
    # print(f"batch_size: {batch_size}, orig_batch_size: {orig_batch_size}")
    
    noise_pred = transformer(
        hidden_states=latent_model_input,
        timestep=timestep,
        # input_ids=processed_data["input_ids"],
        # input_img_latents=input_img_latents,
        # input_image_sizes=processed_data["input_image_sizes"],
        # attention_mask=processed_data["attention_mask"],
        # position_ids=processed_data["position_ids"],
        input_ids=samples_prosessed_tmp_data["multimodal_data"]["input_ids"],
        input_img_latents=samples_prosessed_tmp_data["image_latents"],
        input_image_sizes=samples_prosessed_tmp_data["multimodal_data"]["input_image_sizes"],
        attention_mask=samples_prosessed_tmp_data["multimodal_data"]["attention_mask"],
        position_ids=samples_prosessed_tmp_data["multimodal_data"]["position_ids"],
        return_dict=False,
    )[0]

    img_guidance_scale, guidance_scale = 1.6, 2.5
    cond, uncond, img_cond = torch.split(noise_pred, len(noise_pred) // 3, dim=0)
    noise_pred = uncond + img_guidance_scale * (img_cond - uncond) + guidance_scale * (cond - img_cond)

    # compute the previous noisy sample x_t -> x_t-1
    # 这一步是 ODE ，我需要将其转换为 SDE
    # latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
    
    prev_sample, log_prob, prev_sample_mean, std_dev_t = sde_step_with_logprob(
        pipeline.scheduler, 
        noise_pred.float(), 
        # t.unsqueeze(0).repeat(latents.shape[0]), 
        # latents.float(),
        # noise_level=cur_noise_level,
        sample["timesteps"][:, j],
        sample["latents"][:, j].float(),
        prev_sample=sample["next_latents"][:, j].float(),
        noise_level=config.sample.noise_level,
    )
    
    return prev_sample, log_prob, prev_sample_mean, std_dev_t

def eval(pipeline, test_dataloader, config, rank, local_rank, world_size, device, global_step, reward_fn, executor, autocast, ema, transformer_trainable_parameters):
    if config.train.ema:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
    all_rewards = defaultdict(list)
    
    # 创建保存图片的目录
    save_dir = "results/omni_gen"
    os.makedirs(save_dir, exist_ok=True)
    
    for test_batch in tqdm(
            test_dataloader,
            desc="Eval: ",
            disable=local_rank != 0,
            position=0,
        ):
        prompts, eval_prompts, vlm_questions, prompt_metadata, ref_images, _ = test_batch
        # ref_images = [ref_image.resize((config.resolution, config.resolution)) for ref_image in ref_images]
        ref_images = [
            [ref_img for ref_img in ref_image]
            for ref_image in ref_images
        ]
        # with autocast():
        with torch.no_grad():
            # 通过pipeline生成图像并计算log概率
            collected_data = pipeline_with_logprob(
                pipeline,
                prompt=prompts,
                input_images=ref_images,
                height=config.resolution,
                width=config.resolution,
                guidance_scale=2.5,
                img_guidance_scale=2.0,
                num_inference_steps=config.sample.num_steps,
                # guidance_scale=config.sample.guidance_scale,  # 这里只设计了text的guidance_scale，没有设计image的
                output_type="pt",
                noise_level=0,
                # generator=generator,
                sde_window_size=0,
                # sde_window_range=config.sample.sde_window_range,
                # process_index=rank
            )
        images = collected_data["images"]
        rewards = executor.submit(reward_fn, images, prompts, prompt_metadata, ref_images, eval_prompts, vlm_questions, only_strict=False)
        # yield to to make sure reward computation starts
        time.sleep(0)
        rewards, reward_metadata = rewards.result()

        for key, value in rewards.items():
            rewards_gather = gather_tensor(torch.as_tensor(value, device=device).contiguous(), world_size).cpu().float().numpy()
            all_rewards[key].append(rewards_gather)
    
    last_batch_images_gather = gather_tensor(torch.as_tensor(images, device=device), world_size).cpu().float().numpy()
    last_batch_prompt_ids = pipeline.tokenizer(
        prompts,
        padding="max_length",
        max_length=256,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)
    # last_batch_prompt_ids_gather = gather_tensor(last_batch_prompt_ids, world_size).cpu().float().numpy()
    last_batch_prompt_ids_gather = gather_tensor(last_batch_prompt_ids, world_size).cpu().long().numpy()
    last_batch_prompts_gather = pipeline.tokenizer.batch_decode(
        last_batch_prompt_ids_gather, skip_special_tokens=True
    )
    last_batch_rewards_gather = {}
    for key, value in rewards.items():
        last_batch_rewards_gather[key] = gather_tensor(torch.as_tensor(value, device=device).contiguous(), world_size).cpu().float().numpy()

    all_rewards = {key: np.concatenate(value) for key, value in all_rewards.items()}
    if rank == 0:
        
        num_samples = min(15, len(last_batch_images_gather))
        sample_indices = range(num_samples)
        for idx, index in enumerate(sample_indices):
            image = last_batch_images_gather[index]
            pil = Image.fromarray(
                (image.transpose(1, 2, 0) * 255).astype(np.uint8)
            )
            pil = pil.resize((config.resolution, config.resolution))
            
            # 保存到results/omni_gen目录
            save_path = os.path.join(save_dir, f"step_{global_step}_sample_{idx}.jpg")
            pil.save(save_path)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # num_samples = min(15, len(last_batch_images_gather))
            # sample_indices = range(num_samples)
            for idx, index in enumerate(sample_indices):
                image = last_batch_images_gather[index]
                pil = Image.fromarray(
                    (image.transpose(1, 2, 0) * 255).astype(np.uint8)
                )
                pil = pil.resize((config.resolution, config.resolution))
                pil.save(os.path.join(tmpdir, f"{idx}.jpg"))
            sampled_prompts = [last_batch_prompts_gather[index] for index in sample_indices]
            sampled_rewards = [{k: last_batch_rewards_gather[k][index] for k in last_batch_rewards_gather} for index in sample_indices]
            for key, value in all_rewards.items():
                print(key, value.shape)
            wandb.log(
                {
                    "eval_images": [
                        wandb.Image(
                            os.path.join(tmpdir, f"{idx}.jpg"),
                            caption=f"{prompt:.1000} | " + " | ".join(f"{k}: {v:.2f}" for k, v in reward.items() if v != -10),
                        )
                        for idx, (prompt, reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                    ],
                    **{f"eval_reward_{key}": np.mean(value[value != -10]) for key, value in all_rewards.items()},
                },
                step=global_step,
            )
    if config.train.ema:
        ema.copy_temp_to(transformer_trainable_parameters)


def get_transformer_layer_cls():
    from diffusers.models.transformers.transformer_omnigen import OmniGenBlock
    from diffusers.models.attention_processor import Attention
    
    # from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformerBlock
    
    # from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionBlock, Qwen2_5_VLDecoderLayer
    # return {
    #     QwenImageTransformerBlock,
    #     # QwenImageResidualBlock,
    #     # QwenImageResample,
    #     # QwenImageResidualBlock,
    #     # QwenImageMidBlock,
    #     # QwenImageAttentionBlock,
    #     Qwen2_5_VLVisionBlock,
    #     Qwen2_5_VLDecoderLayer
    #     }
    return {
        OmniGenBlock,
        Attention,
    }
    

def main(_):
    # basic Accelerate and logging setup
    config = FLAGS.config

    # Initialize distributed training
    is_distributed, rank, world_size, local_rank = init_distributed()
    device = torch.device(f'cuda:{local_rank}') if torch.cuda.is_available() else torch.device('cpu')
    
    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    # number of timesteps within each trajectory to train on
    if config.sample.sde_window_size > 0:
        num_train_timesteps = config.sample.sde_window_size
    else:
        num_train_timesteps = config.sample.num_steps - 1

    # Create project directory
    project_dir = os.path.join(config.logdir, config.run_name)
    os.makedirs(project_dir, exist_ok=True)
    if rank == 0:
        wandb.init(
            project="flow_grpo_omni_gen",
            # mode="disabled"
        )
    logger.info(f"\n{config}")

    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if config.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif config.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    from diffusers import AutoencoderKL
    from diffusers.models.transformers.transformer_omnigen import OmniGenTransformer2DModel
    # vae = AutoencoderKL.from_pretrained(
    #     config.pretrained.model, 
    #     subfolder="vae",
    #     torch_dtype=torch.float32  # 强制使用FP32
    # )
    
    omni_transformer = OmniGenTransformer2DModel.from_pretrained(
        config.pretrained.model, 
        subfolder="transformer",
        torch_dtype=torch.float32  # 强制使用FP32
    )

    # load scheduler, tokenizer and models.
    pipeline = OmniGenPipeline.from_pretrained(
        config.pretrained.model, 
        transformer=omni_transformer,
        torch_dtype=inference_dtype
    )
    
    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    # pipeline.text_encoder.requires_grad_(False)  # omni_gen没有text_encoder，就不需要这个
    pipeline.transformer.requires_grad_(not config.use_lora)
    # disable safety checker
    pipeline.safety_checker = None
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        disable=local_rank != 0,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    # Move vae and text_encoder to device and cast to inference_dtype
    pipeline.vae.to(device, dtype=torch.float32)
    # pipeline.text_encoder.to(device, dtype=inference_dtype)
    pipeline.transformer.to(device)
    
    # # 运行这个来查看实际模块名
    # model = pipeline.transformer  # 你的模型实例

    # print("模型中的前50个模块名:")
    # modules_list = []
    # for name, module in model.named_modules():
    #     if name:  # 跳过空字符串（根模块）
    #         modules_list.append(name)
    #         if len(modules_list) <= 50:
    #             print(name)

    # # 特别关注注意力相关模块
    # print("\n注意力相关模块:")
    # for name, module in model.named_modules():
    #     if any(keyword in name.lower() for keyword in ['attn', 'attention', 'q', 'k', 'v']):
    #         print(f"{name} - {type(module).__name__}")
    
    if config.use_lora:
        # Set correct lora layers
        target_modules = [
            # 注意力机制中的Q/K/V/Out线性层
            "attn.to_k",
            "attn.to_q",
            "attn.to_v",
            "attn.to_out.0",  # 注意：to_out是一个ModuleList，to_out.0是线性层
            
            # MLP层（前馈网络）
            "mlp.gate_up_proj",  # 门控和上投影组合
            "mlp.down_proj",     # 下投影
            
            # "attn.add_k_proj",
            # "attn.add_q_proj",
            # "attn.add_v_proj",
            # "attn.to_add_out",
            # "img_mlp.net.0.proj",
            # "img_mlp.net.2",
            # "txt_mlp.net.0.proj",
            # "txt_mlp.net.2",
        ]
        # print("[lora] target_modules:", target_modules)
        transformer_lora_config = LoraConfig(
            r=64,
            lora_alpha=128,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        if config.train.lora_path:
            pipeline.transformer = PeftModel.from_pretrained(pipeline.transformer, config.train.lora_path)
            # After loading with PeftModel.from_pretrained, all parameters have requires_grad set to False. You need to call set_adapter to enable gradients for the adapter parameters.
            pipeline.transformer.set_adapter("default")
        else:
            pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config)
    
    transformer = pipeline.transformer

    # Setup FSDP configuration
    fsdp_config = FSDPConfig(
        sharding_strategy="FULL_SHARD",
        backward_prefetch="BACKWARD_PRE",
        cpu_offload=False,  # Set to True if memory is limited
        num_replicate=1,
        num_shard=world_size,
        mixed_precision_dtype=inference_dtype,
        # mixed_precision_dtype=torch.float32,
        use_activation_checkpointing=config.activation_checkpointing,
        use_device_mesh=False, 
    )
    # Wrap language model with FSDP
    transformer.cpu().to(dtype=torch.float32)
    transformer = fsdp_wrapper(transformer, fsdp_config, get_transformer_layer_cls)
    pipeline.transformer = transformer

    if config.train.beta > 0:
        transformer_ref = OmniGenTransformer2DModel.from_pretrained(
            config.pretrained.model,
            subfolder="transformer",
            torch_dtype=inference_dtype
            # torch_dtype=torch.float32
        )
        transformer_ref.eval()
        transformer_ref.requires_grad_(False)
        transformer_ref.cpu().to(dtype=torch.float32)
        transformer_ref = fsdp_wrapper(transformer_ref, fsdp_config, get_transformer_layer_cls)

    # pipeline.text_encoder.cpu().to(dtype=torch.float32)
    # pipeline.text_encoder = fsdp_wrapper(pipeline.text_encoder, fsdp_config, get_transformer_layer_cls)

    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    # This ema setting affects the previous 20 × 8 = 160 steps on average.
    ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=8, device=device)
    # ema = None
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        transformer_trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    if config.fsdp_optimizer_offload:
        optimizer = OptimizerOffload(optimizer)
    
    train_dataset = GenevalPromptImageDataset(config.dataset, 'train')
    test_dataset = GenevalPromptImageDataset(config.dataset, 'test')

    train_sampler = DistributedKRepeatSampler( 
        dataset=train_dataset,
        batch_size=config.sample.train_batch_size,
        k=config.sample.num_image_per_prompt,
        num_replicas=world_size,
        rank=rank,
        seed=42
    )

    # Create a DataLoader; note that shuffling is not needed here because it’s controlled by the Sampler.
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=1,
        collate_fn=GenevalPromptImageDataset.collate_fn,
        # persistent_workers=True
    )

    test_dataloader = DataLoader(
        test_dataset,
        sampler=DistributedSampler(test_dataset, shuffle=False),
        batch_size=config.sample.test_batch_size,
        collate_fn=GenevalPromptImageDataset.collate_fn,
        shuffle=False,
        num_workers=8,
    )

    if config.sample.num_image_per_prompt == 1:
        config.per_prompt_stat_tracking = False
    # initialize stat tracker
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(config.sample.global_std)

    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    if config.mixed_precision == "fp16":
        autocast = lambda: torch.amp.autocast("cuda", dtype=torch.float16)
    elif config.mixed_precision == "bf16":
        autocast = lambda: torch.amp.autocast("cuda", dtype=torch.bfloat16)
    else:
        autocast = contextlib.nullcontext

    # FSDP doesn't need deepspeed configuration
    # prepare prompt and reward fn
    reward_fn = getattr(flow_grpo.rewards, 'multi_score')(device, config.reward_fn)
    eval_reward_fn = getattr(flow_grpo.rewards, 'multi_score')(device, config.reward_fn)
    
    # FSDP setup completed above
    # executor to perform callbacks asynchronously. this is beneficial for the llava callbacks which makes a request to a
    # remote server running llava inference.
    executor = futures.ThreadPoolExecutor(max_workers=8)

    # Train!
    samples_per_epoch = (
        config.sample.train_batch_size
        * world_size
        * config.sample.num_batches_per_epoch
    )
    total_train_batch_size = (
        config.train.batch_size
        * world_size
        * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}"
    )
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
    )
    logger.info(
        f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}"
    )
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")
    # 初始化训练计数器
    epoch = 0
    global_step = 0
    # 创建训练数据加载器的迭代器
    train_iter = iter(train_dataloader)

    # 主训练循环
    while True:
        #################### 评估阶段 ####################
        # 设置transformer为评估模式
        pipeline.transformer.eval()
        
        # 定期保存模型检查点
        if epoch % config.save_freq == 0 and epoch > 0:
            save_fsdp_checkpoint(config.save_dir, transformer, global_step, rank)
        
        # 定期进行评估
        if epoch % config.eval_freq == 0:
            eval(pipeline, test_dataloader, config, rank, local_rank, world_size, device, global_step, eval_reward_fn, executor, autocast, ema, transformer_trainable_parameters)

        
        #################### 采样阶段 ####################
        # 设置transformer为评估模式
        pipeline.transformer.eval()
        samples = []  # 存储采样结果
        samples_prosessed_tmp_data = []  # 存储处理前的采样结果
        
        # 对每个采样批次进行循环
        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=local_rank != 0,  # 只在主进程显示进度条
            position=0,
        ):
            # 设置采样器的epoch，确保数据shuffle的一致性
            train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)
            
            # 从数据加载器获取下一个批次的数据
            prompts, eval_prompts, vlm_questions, prompt_metadata, ref_images, prompt_with_image_paths = next(train_iter)
            
            # 调整参考图像尺寸
            ref_images = [
                [ref_img for ref_img in ref_image]
                for ref_image in ref_images
            ]
            
            # 对提示文本进行tokenize
            prompt_ids = pipeline.tokenizer(
                prompt_with_image_paths,
                padding="max_length",
                max_length=256,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)

            # 设置随机数生成器（如果需要固定的潜在表示）
            if config.sample.same_latent:
                generator = create_generator(prompts, base_seed=epoch*10000+i)
            else:
                generator = None
                
            # 使用自动混合精度进行采样
            # with autocast():
            with torch.no_grad():  # 采样阶段不需要梯度
                # 通过pipeline生成图像并计算log概率
                collected_data = pipeline_with_logprob(
                    pipeline,
                    prompt=prompts,
                    input_images=ref_images,
                    height=config.resolution,
                    width=config.resolution,
                    num_inference_steps=config.sample.num_steps,
                    # guidance_scale=config.sample.guidance_scale,  # 这里只设计了text的guidance_scale，没有设计image的
                    output_type="pt",
                    noise_level=config.sample.noise_level,
                    generator=generator,
                    sde_window_size=config.sample.sde_window_size,
                    sde_window_range=config.sample.sde_window_range,
                    process_index=rank
                )

            # 整理采样数据
            # print(f"[DEBUG] collected_data['all_latents'] shape: {[latent.shape for latent in collected_data['all_latents']]}")
            # print(f"[DEBUG] collected_data['all_log_probs'] shape: {[log_prob.shape for log_prob in collected_data['all_log_probs']]}")
            
            latents = torch.stack(collected_data["all_latents"], dim=1)  # 所有时间步的潜在表示
            log_probs = torch.stack(collected_data["all_log_probs"], dim=1)  # 所有时间步的log概率
            timesteps = torch.stack(collected_data["all_timesteps"]).unsqueeze(0)  # .repeat(config.sample.train_batch_size, 1)
            images = collected_data["images"]  # 生成的图像
            
            # print(f"[DEBUG] latents shape: {latents.shape}")  # TODO 这里面都是nan，需要查看下具体是什么问题
            
            # 异步计算奖励（使用线程池执行器）
            rewards = executor.submit(reward_fn, images, prompts, prompt_metadata, ref_images, eval_prompts, vlm_questions, only_strict=True)
            
            # 短暂暂停以确保奖励计算开始
            time.sleep(0)

            # 存储采样数据
            samples.append(
                {
                    "prompt_ids": prompt_ids,  # 提示文本的token IDs
                    # "prompt_embeds": collected_data["prompt_embeds"],  # 提示文本的嵌入
                    # "prompt_embeds_mask": collected_data["prompt_embeds_mask"],  # 提示嵌入的掩码
                    # "negative_prompt_embeds": collected_data["negative_prompt_embeds"],  # 负提示嵌入
                    # "negative_prompt_embeds_mask": collected_data["negative_prompt_embeds_mask"],  # 负提示嵌入掩码
                    # "multimodal_data": collected_data["multimodal_data"],  # 【新加的】处理后的数据
                    # "input_ids": collected_data["multimodal_data"]["input_ids"],  # 【新加的】处理后的数据
                    # "attention_mask": collected_data["multimodal_data"]["attention_mask"],  # 【新加的】处理后的数据
                    # "position_ids": collected_data["multimodal_data"]["position_ids"],  # 【新加的】处理后的数据
                    # # "input_pixel_values": collected_data["multimodal_data"]["input_pixel_values"],  # 【新加的】处理后的数据
                    # "input_image_sizes": collected_data["multimodal_data"]["input_image_sizes"],  # 这是一个列表 {0: [[50, 306], [316, 572]], 1: [[50, 306], [316, 572]], 4: [[58, 314], [317, 573]], 5: [[58, 314], [317, 573]]}
                    
                    # "image_latents": collected_data["image_latents"],  # 图像潜在表示，但是是一个列表，包含多个图像
                    "timesteps": timesteps,  # 时间步
                    "latents": latents[:, :-1],  # 每个时间步t之前的潜在表示
                    "next_latents": latents[:, 1:],  # 每个时间步t之后的潜在表示
                    "log_probs": log_probs,  # 模型预测的log概率
                    "rewards": rewards,  # 奖励（Future对象）
                }
            )
            samples_prosessed_tmp_data.append(
                {
                    "multimodal_data": collected_data["multimodal_data"],
                    "image_latents": collected_data["image_latents"],  # 图像潜在表示，但是是一个列表，包含多个图像
                }
            )
        
        # 计算最大的提示嵌入长度（用于填充）
        # max_prompt_embeds_len = max([sample["prompt_embeds_mask"].shape[1] for sample in samples])
        
        # 等待所有奖励计算完成
        for sample in tqdm(
            samples,
            desc="Waiting for rewards",
            disable=local_rank!=0,
            position=0,
        ):
            # # 对提示嵌入进行填充，使所有样本长度一致
            # seq_pad_len = max_prompt_embeds_len - sample["prompt_embeds"].shape[1]
            # sample["prompt_embeds"] = torch.nn.functional.pad(
            #     sample["prompt_embeds"],  # [B, L, D]
            #     (0, 0, 0, seq_pad_len),   # 在序列维度(L)进行填充
            #     value=0,
            # )
            # sample["prompt_embeds_mask"] = torch.nn.functional.pad(
            #     sample["prompt_embeds_mask"],  # [B, L]
            #     (0, seq_pad_len),              # 在序列维度(L)进行填充
            #     value=0,
            # )
            # # 对负提示嵌入进行同样的填充
            # sample["negative_prompt_embeds"] = torch.nn.functional.pad(
            #     sample["negative_prompt_embeds"],  # [B, L, D]
            #     (0, 0, 0, seq_pad_len),            # 在序列维度(L)进行填充
            #     value=0,
            # )
            # sample["negative_prompt_embeds_mask"] = torch.nn.functional.pad(
            #     sample["negative_prompt_embeds_mask"],  # [B, L]
            #     (0, seq_pad_len),                       # 在序列维度(L)进行填充
            #     value=0,
            # )

            # 获取奖励计算结果
            rewards, reward_metadata = sample["rewards"].result()
            # {
            #     "multi_human": 0.25,
            #     "avg": 0.5
            # }
            # # 将奖励转换为tensor
            sample["rewards"] = {
                key: torch.as_tensor(value, device=device).float()
                for key, value in rewards.items()
            }

        # print("=== 调试信息：每个样本的字段尺寸 ===")
        # for i, sample in enumerate(samples):
        #     print(f"样本 {i}:")
        #     for key in sample.keys():
        #         print(f"  {key}: {sample[key].shape}")
        #     print()

        # 将所有样本数据拼接成一个大的tensor
        # new_samples = samples.copy()
        # new_samples = {
        #     k: torch.cat([s[k] for s in new_samples], dim=0)
        #     if not isinstance(new_samples[0][k], dict) else  # 如果不是字典，直接拼接
        #     {
        #         sub_key: torch.cat([s[k][sub_key] for s in new_samples], dim=0)  # 如果是字典，对字典中的每个子键进行拼接
        #         for sub_key in new_samples[0][k]
        #     }
        #     for k in new_samples[0].keys()  # 遍历所有样本的键；注意，这里用的是samples[0]，这是因为我们假设所有样本的键都是相同的
        # }

        # 定期记录生成的图像到wandb
        if epoch % 10 == 0 and rank == 0:
            # 使用临时目录保存图像
            with tempfile.TemporaryDirectory() as tmpdir:
                num_samples = min(15, len(images))
                sample_indices = random.sample(range(len(images)), num_samples)

                # 保存选中的样本图像
                for idx, i in enumerate(sample_indices):
                    image = images[i]
                    pil = Image.fromarray(
                        (image.cpu().float().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    )
                    pil = pil.resize((config.resolution, config.resolution))
                    pil.save(os.path.join(tmpdir, f"{idx}.jpg"))

                # 获取对应的提示和奖励
                sampled_prompts = [prompts[i] for i in sample_indices]
                sampled_rewards = [rewards['avg'][i] for i in sample_indices]

                # 记录到wandb
                wandb.log(
                    {
                        "images": [
                            wandb.Image(
                                os.path.join(tmpdir, f"{idx}.jpg"),
                                caption=f"{prompt:.100} | avg: {avg_reward:.2f}",
                            )
                            for idx, (prompt, avg_reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                        ],
                    },
                    step=global_step,
                )
        
        # 准备奖励数据用于训练
        # 首先收集所有批次的奖励数据用于全局计算
        all_rewards_avg = []
        all_prompt_ids = []

        # 遍历每个批次，收集奖励和提示ID
        for sample in samples:
            # 保存原始平均值
            sample["rewards"]["ori_avg"] = sample["rewards"]["avg"].clone()
            
            # 将奖励沿着时间步维度重复
            sample["rewards"]["avg"] = sample["rewards"]["avg"].unsqueeze(1).repeat(1, num_train_timesteps)
            
            # 收集奖励和提示ID用于全局计算
            all_rewards_avg.append(sample["rewards"]["avg"])
            all_prompt_ids.append(sample["prompt_ids"])

        # 将所有批次的奖励和提示ID拼接起来
        all_rewards_avg_cat = torch.cat(all_rewards_avg, dim=0)
        all_prompt_ids_cat = torch.cat(all_prompt_ids, dim=0)

        # 在所有进程间收集奖励数据
        gathered_rewards_avg = gather_tensor(all_rewards_avg_cat, world_size)
        gathered_prompt_ids = gather_tensor(all_prompt_ids_cat, world_size)

        # 转换为numpy用于统计计算
        gathered_rewards_avg_np = gathered_rewards_avg.cpu().float().numpy()
        gathered_prompt_ids_np = gathered_prompt_ids.cpu().long().numpy()

        # 记录奖励统计信息
        if rank == 0:
            wandb.log(
                {
                    "epoch": epoch,
                    "reward_avg": gathered_rewards_avg_np.mean(),
                },
                step=global_step,
            )

        # 每个提示的统计跟踪（如果启用）
        if config.per_prompt_stat_tracking:
            # 解码提示文本
            prompts = pipeline.tokenizer.batch_decode(gathered_prompt_ids_np, skip_special_tokens=True)
            
            # 更新统计跟踪器并计算优势
            advantages_global = stat_tracker.update(prompts, gathered_rewards_avg_np)
            
            if local_rank == 0:
                print("len(prompts)", len(prompts))
                print("len unique prompts", len(set(prompts)))

            # 获取统计信息
            group_size, trained_prompt_num = stat_tracker.get_stats()
            zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(prompts, {'ori_avg': gathered_rewards_avg_np})

            # 记录统计信息
            if rank == 0:
                wandb.log(
                    {
                        "group_size": group_size,
                        "trained_prompt_num": trained_prompt_num,
                        "zero_std_ratio": zero_std_ratio,
                        "reward_std_mean": reward_std_mean,
                    },
                    step=global_step,
                )
            stat_tracker.clear()  # 清空统计器
        else:
            # 标准化优势（如果没有启用每个提示的统计跟踪）
            advantages_global = (gathered_rewards_avg_np - gathered_rewards_avg_np.mean()) / (gathered_rewards_avg_np.std() + 1e-4)

        # 将全局优势分布回各个进程和批次
        advantages_global_tensor = torch.as_tensor(advantages_global, device=device)
        # 将全局优势按照进程和批次分割
        advantages_per_rank = advantages_global_tensor.reshape(world_size, -1, advantages_global_tensor.shape[-1])[rank]

        # 将优势分配回各个批次
        start_idx = 0
        for i, sample in enumerate(samples):
            batch_size = sample["rewards"]["avg"].shape[0]
            end_idx = start_idx + batch_size
            
            # 分配当前批次的优势
            sample["advantages"] = advantages_per_rank[start_idx:end_idx]
            start_idx = end_idx
            
            # 清理不需要的数据
            del sample["rewards"]
            del sample["prompt_ids"]

        # 验证分配是否正确
        total_assigned = sum(sample["advantages"].shape[0] for sample in samples)
        if local_rank == 0:
            print(f"优势分配验证: 总分配样本数={total_assigned}, 全局优势样本数={advantages_per_rank.shape[0]}")
            print(f"平均优势绝对值: {advantages_per_rank.abs().mean()}")

        # 获取训练数据的维度信息
        total_batch_size = sum(sample["timesteps"].shape[0] for sample in samples)
        gradient_accumulation_steps = config.train.gradient_accumulation_steps * num_train_timesteps
        
        # # 准备奖励数据用于训练
        # new_samples["rewards"]["ori_avg"] = new_samples["rewards"]["avg"]  # 保存原始平均值
        # # 将奖励沿着时间步维度重复，便于后续引入时间步相关的优势计算
        # new_samples["rewards"]["avg"] = new_samples["rewards"]["avg"].unsqueeze(1).repeat(1, num_train_timesteps)
        
        # # 在所有进程间收集奖励数据
        # gathered_rewards = {key: gather_tensor(value, world_size) for key, value in new_samples["rewards"].items()}
        # gathered_rewards = {key: value.cpu().float().numpy() for key, value in gathered_rewards.items()}
        
        # # 记录奖励统计信息
        # if rank == 0:
        #     wandb.log(
        #         {
        #             "epoch": epoch,
        #             **{f"reward_{key}": value.mean() for key, value in gathered_rewards.items() 
        #             if '_strict_accuracy' not in key and '_accuracy' not in key},
        #         },
        #         step=global_step,
        #     )

        # # 每个提示的统计跟踪（如果启用）
        # if config.per_prompt_stat_tracking:
        #     # 收集所有进程的提示
        #     # prompt_ids = gather_tensor(samples["prompt_ids"], world_size).cpu().float().numpy()
        #     prompt_ids = gather_tensor(new_samples["prompt_ids"], world_size).cpu().long().numpy()
        #     prompts = pipeline.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
            
        #     # 更新统计跟踪器并计算优势
        #     advantages = stat_tracker.update(prompts, gathered_rewards['avg'])
            
        #     if local_rank == 0:
        #         print("len(prompts)", len(prompts))
        #         print("len unique prompts", len(set(prompts)))

        #     # 获取统计信息
        #     group_size, trained_prompt_num = stat_tracker.get_stats()
        #     zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(prompts, gathered_rewards)

        #     # 记录统计信息
        #     if rank == 0:
        #         wandb.log(
        #             {
        #                 "group_size": group_size,
        #                 "trained_prompt_num": trained_prompt_num,
        #                 "zero_std_ratio": zero_std_ratio,
        #                 "reward_std_mean": reward_std_mean,
        #             },
        #             step=global_step,
        #         )
        #     stat_tracker.clear()  # 清空统计器
        # else:
        #     # 标准化优势（如果没有启用每个提示的统计跟踪）
        #     advantages = (gathered_rewards['avg'] - gathered_rewards['avg'].mean()) / (gathered_rewards['avg'].std() + 1e-4)

        # # 将优势分布回各个进程
        # advantages = torch.as_tensor(advantages)
        # new_samples["advantages"] = (
        #     advantages.reshape(world_size, -1, advantages.shape[-1])[rank]
        #     .to(device)
        # )
        
        # if local_rank == 0:
        #     print("advantages: ", samples["advantages"].abs().mean())

        # # 清理不需要的数据
        # del samples["rewards"]
        # del samples["prompt_ids"]

        # # 获取训练数据的维度信息
        # # total_batch_size, num_timesteps = new_samples["timesteps"].shape
        # # print("total_batch_size", total_batch_size)  # 8
        # # print("num_timesteps", num_timesteps)  # 2
        # gradient_accumulation_steps = config.train.gradient_accumulation_steps * num_train_timesteps

        #################### 训练阶段 ####################
        # 内部epoch循环（多次使用同一样本数据进行训练）
        # print("[DEBUG] mini_batch", total_batch_size//config.sample.num_batches_per_epoch)
        for inner_epoch in range(config.train.num_inner_epochs):
            # 重新批次化训练数据
            """
            samples_batched = {
                k: v.reshape(-1, total_batch_size//config.sample.num_batches_per_epoch, *v.shape[1:])
                for k, v in samples.items()
            }

            # 将字典转换为列表，便于迭代
            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]
            """
            samples_batched = samples

            # 设置模型为训练模式
            pipeline.transformer.train()
            info = defaultdict(list)  # 存储训练信息
            
            # 对每个批次进行训练
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=local_rank != 0,
            ):
                # 所有需要训练的时间步
                train_timesteps = [step_index for step_index in range(num_train_timesteps)]
                
                # 对每个时间步进行训练
                for j in tqdm(
                    train_timesteps,
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=local_rank != 0,
                ):
                    # 手动梯度累积（用于FSDP）
                    if (i * num_train_timesteps + j + 1) % gradient_accumulation_steps == 0:
                        should_sync = True  # 需要同步梯度
                    else:
                        should_sync = False
                    
                    # 计算当前时间步的log概率
                    # with autocast():
                    # prossesed_data = collected_data["multimodal_data"]
                    prev_sample, log_prob, prev_sample_mean, std_dev_t = compute_log_prob(transformer, pipeline, sample, j, config, rank, samples_prosessed_tmp_data[i])
                    
                    # 如果使用KL散度正则化，计算参考模型的输出
                    if config.train.beta > 0:
                        with torch.no_grad():
                            _, _, prev_sample_mean_ref, _ = compute_log_prob(transformer_ref, pipeline, sample, j, config, rank, samples_prosessed_tmp_data[i])
                    
                    # GRPO（Guided Reward Policy Optimization）逻辑
                    advantages = torch.clamp(
                        sample["advantages"][:, j],
                        -config.train.adv_clip_max,
                        config.train.adv_clip_max,
                    )
                    
                    # 计算重要性采样比率
                    ratio = torch.exp(log_prob - sample["log_probs"][:, j])
                    print("ratio", ratio)
                    
                    # PPO裁剪损失
                    unclipped_loss = -advantages * ratio
                    clipped_loss = -advantages * torch.clamp(
                        ratio,
                        1.0 - config.train.clip_range,
                        1.0 + config.train.clip_range,
                    )
                    policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                    policy_loss = policy_loss / gradient_accumulation_steps  # 梯度累积归一化
                    
                    # 如果使用KL散度正则化
                    if config.train.beta > 0:
                        kl_loss = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean(dim=(1,2), keepdim=True) / (2 * std_dev_t ** 2)
                        kl_loss = torch.mean(kl_loss)
                        kl_loss = kl_loss / gradient_accumulation_steps
                        loss = policy_loss + config.train.beta * kl_loss
                    else:
                        loss = policy_loss

                    # 收集训练统计信息
                    info["approx_kl"].append(0.5 * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2))
                    info["clipfrac"].append(torch.mean((torch.abs(ratio - 1.0) > config.train.clip_range).float()))
                    info["clipfrac_gt_one"].append(torch.mean((ratio - 1.0 > config.train.clip_range).float()))
                    info["clipfrac_lt_one"].append(torch.mean((1.0 - ratio > config.train.clip_range).float()))
                    info["policy_loss"].append(policy_loss)
                    if config.train.beta > 0:
                        info["kl_loss"].append(kl_loss)
                    info["loss"].append(loss)

                    # 反向传播
                    loss.backward()
                    
                    # 如果需要同步梯度（梯度累积步骤完成）
                    if should_sync:
                        # 梯度裁剪
                        torch.nn.utils.clip_grad_norm_(transformer.parameters(), config.train.max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad()

                    # 记录训练信息
                    if should_sync:
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        
                        # 在多进程环境下汇总统计信息
                        if is_distributed:
                            for k, v in info.items():
                                dist.all_reduce(v, op=dist.ReduceOp.SUM)
                                info[k] = v / world_size
                        
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        
                        # 记录到wandb
                        if rank == 0:
                            wandb.log(info, step=global_step)
                        
                        global_step += 1
                        info = defaultdict(list)  # 重置信息字典
            
            # 如果使用EMA（指数移动平均），更新EMA参数
            if config.train.ema:
                ema.step(transformer_trainable_parameters, global_step)
        
        # 增加epoch计数器
        epoch += 1


if __name__ == "__main__":
    app.run(main)

