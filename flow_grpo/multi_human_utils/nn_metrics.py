# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import os
# import time
from typing import Union
import facer
import hpsv2
import huggingface_hub
import torch
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
from hpsv2.utils import hps_version_map
from huggingface_hub import snapshot_download
from onnx2torch import convert
from PIL import Image
import numpy as np


def get_model_root():
    """
    获取模型根目录，支持环境变量和默认路径
    """
    # 1. 首先检查环境变量
    model_root = os.environ.get('MULTIHUMAN_MODEL_ROOT')
    if model_root and os.path.exists(model_root):
        return model_root
    
    # 2. 检查常见的模型存储路径
    possible_paths = [
        "/data4/shaozhen.liu/code/MultiHuman-Testbench/models",  # 您当前的路径
        "./models",  # 当前目录下的models文件夹
        "~/multihuman_models",  # 用户主目录下的模型文件夹
        "/opt/multihuman_models",  # 系统级模型路径
    ]
    
    for path in possible_paths:
        expanded_path = os.path.expanduser(path)
        if os.path.exists(expanded_path):
            return expanded_path
    
    # 3. 如果都找不到，使用当前工作目录下的models文件夹
    default_path = "./models"
    os.makedirs(default_path, exist_ok=True)
    return default_path

############# Initialize HPS model
def initialize_hps_model(device, model_root=None):
    """
    Initializes the HPSv2 model with pretrained weights and tokenizer.

    Args:
        device (torch.device): The device to load the model onto.
        model_root (str, optional): 模型根目录，如果为None则自动检测

    Returns:
        tuple: (hps_model, tokenizer, preprocess_val) for inference.
    """
    if model_root is None:
        model_root = get_model_root()
    
    hps_model_dict = {}
    if not hps_model_dict:
        model, preprocess_train, preprocess_val = create_model_and_transforms(
            "ViT-H-14",
            "laion2B-s32B-b79K",
            precision="amp",
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False,
            cache_dir=model_root,  # 使用统一的模型根目录
        )
        hps_model_dict["model"] = model
        hps_model_dict["preprocess_val"] = preprocess_val

    hps_model = hps_model_dict["model"]
    preprocess_val = hps_model_dict["preprocess_val"]
    hps_version = "v2.1"
    
    # 检查模型文件是否已存在
    # local_model_path = os.path.join(model_root, f"HPSv2/{hps_version_map[hps_version]}")
    local_model_path = os.path.join(model_root, f"{hps_version_map[hps_version]}")
    if os.path.exists(local_model_path):
        cp = local_model_path
    else:
        # 如果不存在，下载到统一模型目录
        cp = huggingface_hub.hf_hub_download(
            "xswu/HPSv2", 
            hps_version_map[hps_version], 
            local_dir=os.path.join(model_root, "HPSv2")
        )
    
    hps_checkpoint = torch.load(cp, map_location=device)
    hps_model.load_state_dict(hps_checkpoint["state_dict"])
    tokenizer = get_tokenizer("ViT-H-14")
    hps_model = hps_model.to(device)
    hps_model.eval()
    return hps_model, tokenizer, preprocess_val

########### Init peripherals
def init_peripherals(device, model_root=None):
    """
    Initializes all required peripherals for multi-human evaluation, including
    face detection, identity embedding, and HPS scoring.

    Args:
        device (torch.device): The device to load models onto.
        model_root (str, optional): 模型根目录，如果为None则自动检测

    Returns:
        tuple: Initialized (face_detector, ant_model, hps_model, cosine_sim, tokenizer, preprocess_val).
    """
    if model_root is None:
        model_root = get_model_root()
    
    face_detector = facer.face_detector("retinaface/mobilenet", device=device)
    
    # Antelopev2 模型路径处理
    antelope_path = os.path.join(model_root, "antelopev2")
    if not os.path.exists(antelope_path):
        # 如果模型不存在，下载到统一目录
        snapshot_download("DIAMONIK7777/antelopev2", local_dir=antelope_path)
    else:
        print(f"使用已存在的 Antelopev2 模型: {antelope_path}")
    
    ant_model = convert(os.path.join(antelope_path, "glintr100.onnx")).eval().to(device)
    for param in ant_model.parameters():
        param.requires_grad_(False)
    
    hps_model, tokenizer, preprocess_val = initialize_hps_model(device, model_root)
    cosine_sim = torch.nn.CosineSimilarity(dim=0)
    return face_detector, ant_model, hps_model, cosine_sim, tokenizer, preprocess_val

########### Detect Faces
def face_detect(face_detector, image_input, device):
    """
    Detects faces in an image using the specified face detector.
    
    Args:
        face_detector: The face detection model.
        image_input (str or PIL.Image): Path to the image file or PIL.Image object.
        device (torch.device): The device for inference.
    
    Returns:
        tuple: (faces, face_image) where faces is the detection result and face_image is the tensor.
    """
    # 统一处理图像输入
    if isinstance(image_input, str):
        # 文件路径
        image_tensor = facer.read_hwc(image_input)
    elif isinstance(image_input, Image.Image):
        # PIL.Image对象
        np_image = np.array(image_input.convert('RGB'))
        image_tensor = torch.from_numpy(np_image)
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")
    
    # 转换为BCHW格式并移动到设备
    face_image = facer.hwc2bchw(image_tensor).to(device=device)
    
    # 图像: 1 x 3 x h x w
    with torch.inference_mode():
        faces = face_detector(face_image)
    return faces, face_image


########### HPS Score
def hps_score_function(
    img_path: Union[list, str, Image.Image],
    prompt: str,
    tokenizer,
    preprocess_val,
    device,
    hps_model,
) -> list:
    """
    Computes the Human Preference Score (HPS) between an image and a prompt.

    Args:
        img_path (Union[list, str, Image.Image]): Input image(s) for evaluation.
        prompt (str): Text prompt describing the image.
        tokenizer: Tokenizer for processing the prompt.
        preprocess_val: Preprocessing function for the image.
        device (torch.device): Device for model inference.
        hps_model: The HPS model.

    Returns:
        list: List of HPS scores for each image.
    """
    if isinstance(img_path, list):
        result = []
        for one_img_path in img_path:
            # Load your image and prompt
            with torch.no_grad():
                # Process the image
                image = (
                    preprocess_val(one_img_path)
                    .unsqueeze(0)
                    .to(device=device, non_blocking=True)
                )
                # Process the prompt
                text = tokenizer([prompt]).to(device=device, non_blocking=True)
                # Calculate the HPS
                with torch.amp.autocast("cuda"):
                    outputs = hps_model(image, text)
                    image_features, text_features = (
                        outputs["image_features"],
                        outputs["text_features"],
                    )
                    logits_per_image = image_features @ text_features.T
                    hps_score = torch.diagonal(logits_per_image).cpu().numpy()
            result.append(hps_score[0])
        return result
