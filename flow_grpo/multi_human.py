# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import argparse
import json
import os
import pathlib
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


from .multi_human_utils import (crop_face, face_detect, gather_and_print_scores,
                   hps_score_function, hungarian_algorithm, init_peripherals,
                   )


class MultiHumanScorer:
    def __init__(
        self, 
        use_gpu: bool = False, 
        enable_mllm: bool = False, 
        model_root: str = "/data4/shaozhen.liu/code/MultiHuman-Testbench/models"
    ):
        """
        多人生成图像评估器
        
        Args:
            use_gpu: 是否使用GPU加速
            enable_mllm: 是否启用MLLM评估
            model_root: 模型根目录，如果为None则自动检测
        """
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.enable_mllm = enable_mllm
        self.MLLM_AWAKE = enable_mllm
        self.model_root = model_root
        
        # 初始化模型和工具
        self.face_detector, self.ant_model, self.hps_model, self.cosine_sim, self.tokenizer, self.preprocess_val = (
            self._init_peripherals(self.device, self.model_root)
        )
        
        # 初始化统计变量
        self.reset_stats()
    
    def _init_peripherals(self, device, model_root=None):
        """初始化所有必要的模型和工具"""
        return init_peripherals(device, model_root)
    
    def reset_stats(self):
        """重置统计计数器"""
        self.total = 0
        self.face_sim = 0
        self.accuracy = 0
        self.hps = 0
        
        # 按人数分组的统计
        self.split_metrics = {1: [0, 0, 0], 2: [0, 0, 0], 3: [0, 0, 0], 4: [0, 0, 0], 5: [0, 0, 0]}
        self.total_people = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        
        # MLLM统计
        if self.MLLM_AWAKE:
            from .multi_human_utils import mllm_vqa
            self._mllm_fn = mllm_vqa
            self.lmm_score = 0
            self.num_lmm = 0
    
    @torch.no_grad()
    def __call__(self, 
                images: Union[List[Image.Image], List[str], List[np.ndarray]],  # 应该是numpy
                prompts: List[str],
                reference_peoples: Union[List[List[str]], List[List[Image.Image]]] = None,
                vlm_questions: List[List[str]] = None) -> dict:
        """
        对单张图像进行评估打分
        
        Args:
            image: 输入图像（PIL图像、文件路径或numpy数组）
            prompt: 对应的文本提示
            reference_people: 参考人物图像路径列表
            vlm_questions: VLM问题列表
            
        Returns:
            包含各项评分的字典
        """
        # 处理图像输入
        pil_images = []
        for img in images:
            if isinstance(img, str):
                if not os.path.exists(img):
                    raise FileNotFoundError(f"Image file {img} does not exist.")
                pil_images.append(Image.open(img))
            elif isinstance(img, np.ndarray):
                pil_images.append(Image.fromarray(img))
            else:
                pil_images.append(img)
        
        # 初始化结果列表
        all_results = []

        # 对每张图像进行循环处理
        for i, pil_image in enumerate(pil_images):
            # 检测人脸
            faces, face_image = self._face_detect(self.face_detector, pil_image, self.device)
            
            # 获取当前图像对应的提示和VLM问题
            current_prompt = prompts[i] if isinstance(prompts, list) and i < len(prompts) else prompts
            # current_eval_prompt = eval_prompts[i] if isinstance(eval_prompts, list) and i < len(eval_prompts) else eval_prompts
            current_vlm_question = vlm_questions[i] if isinstance(vlm_questions, list) and i < len(vlm_questions) else vlm_questions
            
            # 计算HPS分数
            hps_score = self._hps_score_function(
                [pil_image], current_prompt, self.tokenizer, self.preprocess_val, self.device, self.hps_model
            )[0]
            
            # MLLM评估
            mllm_answers = None
            if self.MLLM_AWAKE and self.enable_mllm and current_vlm_question:
                mllm_answers = self._mllm_vqa(current_vlm_question, pil_image)
            
            # 初始化当前图像的结果
            result = {
                "hps": float(hps_score),  # 这个需要
                "num_detected_faces": 0,
                "accuracy": 0,  # 这个需要
                "id_match": 0.0,  # 这个也需要
                "all_ids": [],
                "mllm_answers": mllm_answers
            }
            
            # 人脸匹配（如果有参考人物）
            if reference_peoples and len(reference_peoples) > 0:
                # 获取当前图像对应的参考人物
                current_reference_people = reference_peoples[i] if isinstance(reference_peoples, list) and i < len(reference_peoples) else reference_peoples
                
                num_people = len(current_reference_people)
                
                # 过滤没有检测到人脸的情况
                no_gen = len(faces.keys()) == 0 or faces["rects"].nelement() == 0
                if not no_gen:
                    num_genned_faces = len(faces["rects"])
                    result["num_detected_faces"] = num_genned_faces
                    
                    # 提取生成的人脸特征
                    gen_embeds = []
                    for g_face in range(num_genned_faces):
                        gen_embeds.append(self.ant_model(self._crop_face(faces, face_image, g_face)))
                    
                    # 提取参考人脸特征
                    person_embeds = []
                    for person in current_reference_people:
                        person_face, person_image = self._face_detect(self.face_detector, person, self.device)
                        person_embeds.append(self.ant_model(self._crop_face(person_face, person_image, 0)))
                    
                    # 构建代价矩阵并进行匈牙利匹配
                    if len(person_embeds) > 0 and len(gen_embeds) > 0:
                        cost_matrix = self._build_cost_matrix(person_embeds, gen_embeds)
                        _, dict_assignments = self._hungarian_algorithm(cost_matrix)
                        
                        # 计算匹配相似度
                        matched_similarity, all_ids = self._compute_matching_scores(
                            cost_matrix, dict_assignments, num_people, num_genned_faces
                        )
                        
                        result["id_match"] = float(matched_similarity / num_people)
                        result["all_ids"] = all_ids
                        result["accuracy"] = int(num_people == num_genned_faces)
                    
                    # 更新统计
                    self._update_stats(num_people, hps_score, result["id_match"], result["accuracy"])
            else:
                print(f"[DEBUG] Warning: No reference people provided for image {i}, skip face matching.")
            
            # 将当前图像的结果添加到结果列表
            # 先测试一下 1：1：1
            reward = result["hps"] + result["accuracy"] + result["id_match"]
            all_results.append(reward)

        # 返回所有图像的结果
        return all_results


    def _build_cost_matrix(self, person_embeds, gen_embeds):
        """构建代价矩阵"""
        cost_matrix = np.zeros((len(person_embeds), len(gen_embeds)))
        for i_p, p_ in enumerate(person_embeds):
            for i_g, g_ in enumerate(gen_embeds):
                cost_matrix[i_p, i_g] = (
                    self.cosine_sim(p_.flatten(), g_.flatten()).cpu().numpy()
                )
        return cost_matrix
    
    def _compute_matching_scores(self, cost_matrix, dict_assignments, num_people, num_genned_faces):
        """计算匹配分数"""
        matched_similarity = 0.0
        all_ids = []
        for person_id in range(len(cost_matrix)):
            assignment_id = dict_assignments[person_id]
            if num_genned_faces > assignment_id:
                matched_similarity += cost_matrix[person_id, assignment_id]
                all_ids.append(float(cost_matrix[person_id, assignment_id]))
            else:
                all_ids.append(0.0)
        return matched_similarity, all_ids
    
    def _update_stats(self, num_people, hps_score, id_match, accuracy):
        """更新统计信息"""
        self.total += 1
        self.total_people[num_people] += 1
        self.hps += hps_score
        self.face_sim += id_match
        self.accuracy += accuracy
        
        # 更新分组统计
        self.split_metrics[num_people][0] += hps_score
        self.split_metrics[num_people][1] += id_match
        self.split_metrics[num_people][2] += accuracy
        
        # 更新MLLM统计（如果需要）
        if self.MLLM_AWAKE and self.enable_mllm:
            # 这里需要根据MLLM结果更新统计
            pass
    
    def get_stats(self):
        """获取当前统计信息"""
        if self.total == 0:
            return {
                "total": 0,
                "accuracy": 0,
                "face_similarity": 0,
                "hps": 0,
                "per_group": {}
            }
        
        stats = {
            "total": self.total,
            "accuracy": self.accuracy / self.total,
            "face_similarity": self.face_sim / self.total,
            "hps": self.hps / self.total,
            "per_group": {}
        }
        
        for num_people in range(1, 6):
            if self.total_people[num_people] > 0:
                stats["per_group"][num_people] = {
                    "count": self.total_people[num_people],
                    "accuracy": self.split_metrics[num_people][2] / self.total_people[num_people],
                    "face_similarity": self.split_metrics[num_people][1] / self.total_people[num_people],
                    "hps": self.split_metrics[num_people][0] / self.total_people[num_people]
                }
        
        return stats
    
    def print_stats(self):
        """打印统计信息（兼容原gather_and_print_scores功能）"""
        stats = self.get_stats()
        
        if stats["total"] == 0:
            print("No samples evaluated yet.")
            return
        
        print(f"Total Images: {stats['total']}")
        print(f"Overall Accuracy: {stats['accuracy']:.4f}, "
              f"Face Similarity: {stats['face_similarity']:.4f}, "
              f"HPS: {stats['hps']:.4f}")
        
        for num_people, group_stats in stats["per_group"].items():
            print(f"People={num_people}: "
                  f"Accuracy: {group_stats['accuracy']:.4f}, "
                  f"Face Similarity: {group_stats['face_similarity']:.4f}, "
                  f"HPS: {group_stats['hps']:.4f} "
                  f"(Count: {group_stats['count']})")
    
    # 以下方法需要您提供原始实现
    def _face_detect(self, face_detector, image, device):
        """人脸检测 - 需要原始face_detect函数"""
        return face_detect(face_detector, image, device)
    
    def _hps_score_function(self, images, prompt, tokenizer, preprocess_val, device, hps_model):
        """HPS评分 - 需要原始hps_score_function函数"""
        return hps_score_function(images, prompt, tokenizer, preprocess_val, device, hps_model)
    
    def _mllm_vqa(self, questions, image):
        """MLLM VQA - 需要原始mllm_vqa函数"""
        return self._mllm_fn(questions, image)
    
    def _crop_face(self, faces, face_image, index):
        """裁剪人脸 - 需要原始crop_face函数"""
        return crop_face(faces, face_image, index)
    
    def _hungarian_algorithm(self, cost_matrix):
        """匈牙利算法 - 需要原始hungarian_algorithm函数"""
        return hungarian_algorithm(cost_matrix)
