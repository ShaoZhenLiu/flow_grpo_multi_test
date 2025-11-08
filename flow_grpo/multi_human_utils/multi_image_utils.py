import torch
import torch.nn.functional as F

def batch_list_to_tensor(batch_list):
    """
    将列表套列表转换为padded tensor和mask
    
    Args:
        batch_list: List[List[Tensor]] - 外层是batch，内层是图片列表
    
    Returns:
        padded_tensor: Tensor [batch_size, max_num_images, c, h, w]
        mask: Tensor [batch_size, max_num_images] (1表示有效，0表示padding)
    """
    batch_size = len(batch_list)
    
    # 获取每个样本的图片数量和最大数量
    num_images_per_sample = [len(sample) for sample in batch_list]
    max_num_images = max(num_images_per_sample)
    
    # 获取单个图片的形状
    c, h, w = batch_list[0][0].shape
    
    # 初始化padded tensor和mask
    padded_tensor = torch.zeros(batch_size, max_num_images, c, h, w)
    mask = torch.zeros(batch_size, max_num_images, dtype=torch.bool)
    
    # 填充数据
    for i, sample in enumerate(batch_list):
        num_images = len(sample)
        # 将图片堆叠
        images_tensor = torch.stack(sample, dim=0)  # [num_images, c, h, w]
        padded_tensor[i, :num_images] = images_tensor
        mask[i, :num_images] = 1  # 有效位置标记为1
    
    return padded_tensor, mask, num_images_per_sample


def merge_samples(samples_list, masks_list=None):
    """
    合并多个采样，统一padding到最大维度
    
    Args:
        samples_list: List[Tensor] - 每个tensor形状为 [batch_size, num_images, c, h, w]
        masks_list: List[Tensor] - 对应的mask列表，形状为 [batch_size, num_images]
    
    Returns:
        merged_tensor: 合并后的tensor
        merged_mask: 合并后的mask
        original_shapes: 原始形状信息，用于恢复
    """
    # 找出所有采样中最大的num_images
    max_num_images = max(sample.shape[1] for sample in samples_list)
    c, h, w = samples_list[0].shape[2:]
    
    padded_samples = []
    padded_masks = []
    original_shapes = []
    
    for i, sample in enumerate(samples_list):
        batch_size, num_images = sample.shape[:2]
        
        # 记录原始形状
        original_shapes.append((batch_size, num_images))
        
        # 如果需要padding
        if num_images < max_num_images:
            pad_size = max_num_images - num_images
            
            # 对tensor进行padding
            padded_sample = F.pad(sample, (0, 0, 0, 0, 0, 0, 0, pad_size))
            
            # 对mask进行相应的padding
            if masks_list is not None and i < len(masks_list):
                sample_mask = masks_list[i]
                # 在mask的num_images维度添加0（表示padding）
                padded_mask = F.pad(sample_mask.unsqueeze(-1), (0, 0, 0, pad_size)).squeeze(-1)
            else:
                # 如果没有提供mask，创建一个新的
                padded_mask = torch.cat([
                    torch.ones(batch_size, num_images, dtype=torch.bool),
                    torch.zeros(batch_size, pad_size, dtype=torch.bool)
                ], dim=1)
        else:
            padded_sample = sample
            if masks_list is not None and i < len(masks_list):
                padded_mask = masks_list[i]
            else:
                padded_mask = torch.ones(batch_size, num_images, dtype=torch.bool)
        
        padded_samples.append(padded_sample)
        padded_masks.append(padded_mask)
    
    # 沿着batch维度合并
    merged_tensor = torch.cat(padded_samples, dim=0)
    merged_mask = torch.cat(padded_masks, dim=0)
    
    return merged_tensor, merged_mask, original_shapes


def recover_from_padded(padded_tensor, mask, original_shapes=None):
    """
    从padded tensor恢复为原始列表套列表格式
    
    Args:
        padded_tensor: 填充后的tensor [total_batch_size, max_num_images, c, h, w]
        mask: 对应的mask [total_batch_size, max_num_images]
        original_shapes: 可选的原始形状信息列表
    
    Returns:
        recovered_data: 恢复后的列表套列表格式
    """
    total_batch_size = padded_tensor.shape[0]
    recovered_data = []
    
    start_idx = 0
    if original_shapes:
        # 如果有原始形状信息，按原始形状恢复
        for batch_size, num_images in original_shapes:
            batch_samples = []
            for i in range(start_idx, start_idx + batch_size):
                valid_indices = mask[i].nonzero(as_tuple=True)[0]
                sample_images = []
                for j in valid_indices:
                    sample_images.append(padded_tensor[i, j])
                batch_samples.append(sample_images)
            recovered_data.extend(batch_samples)
            start_idx += batch_size
    else:
        # 如果没有原始形状信息，根据mask恢复
        for i in range(total_batch_size):
            valid_indices = mask[i].nonzero(as_tuple=True)[0]
            sample_images = []
            for j in valid_indices:
                sample_images.append(padded_tensor[i, j])
            recovered_data.append(sample_images)
    
    return recovered_data

# 步骤1: 将多个批次数据转换为tensor和mask
def process_multiple_batches(batch_data_list):
    """
    处理多个批次数据
    """
    all_tensors = []
    all_masks = []
    all_num_images = []
    
    for batch_data in batch_data_list:
        tensor, mask, num_images = batch_list_to_tensor(batch_data)
        all_tensors.append(tensor)
        all_masks.append(mask)
        all_num_images.append(num_images)
    
    return all_tensors, all_masks, all_num_images

# 步骤2: 合并所有批次
def merge_all_batches(tensors, masks):
    """
    合并所有批次的tensor和mask
    """
    return merge_samples(tensors, masks)

# # 完整示例
# # 模拟多个批次数据
# batch1_data = [
#     [torch.randn(16, 32, 32) for _ in range(3)],  # 3张图
#     [torch.randn(16, 32, 32) for _ in range(2)],  # 2张图
# ]

# batch2_data = [
#     [torch.randn(16, 32, 32) for _ in range(5)],  # 5张图
#     [torch.randn(16, 32, 32) for _ in range(4)],  # 4张图
#     [torch.randn(16, 32, 32) for _ in range(3)],  # 3张图
# ]

# # 处理每个批次
# tensors, masks, num_images_list = process_multiple_batches([batch1_data, batch2_data])

# # 合并所有批次
# merged_tensor, merged_mask, original_shapes = merge_all_batches(tensors, masks)

# print(f"Batch1 tensor shape: {tensors[0].shape}")  # [2, 3, 16, 32, 32]
# print(f"Batch2 tensor shape: {tensors[1].shape}")  # [3, 5, 16, 32, 32]
# print(f"Merged tensor shape: {merged_tensor.shape}")  # [5, 5, 16, 32, 32]
# print(f"Merged mask shape: {merged_mask.shape}")  # [5, 5]

# # 验证mask是否正确
# print("\nMask验证:")
# print("Batch1原始mask:")
# print(masks[0])
# print("Batch2原始mask:") 
# print(masks[1])
# print("合并后mask:")
# print(merged_mask)

if __name__ == "__main__":
    # 使用示例
    batch_data = [
        [torch.randn(16, 32, 32) for _ in range(3)],  # 样本1有3张图
        [torch.randn(16, 32, 32) for _ in range(5)],  # 样本2有5张图
        [torch.randn(16, 32, 32) for _ in range(2)],  # 样本3有2张图
    ]

    padded_tensor, mask, num_images = batch_list_to_tensor(batch_data)
    print(f"Padded tensor shape: {padded_tensor.shape}")  # [3, 5, 16, 32, 32]
    print(f"Mask shape: {mask.shape}")  # [3, 5]
    print(f"Num images per sample: {num_images}")  # [3, 5, 2]
    
    
    # 使用示例
    samples = [
        torch.randn(2, 3, 16, 32, 32),  # 采样1: batch_size=2, num_images=3
        torch.randn(3, 5, 16, 32, 32),  # 采样2: batch_size=3, num_images=5
        torch.randn(1, 4, 16, 32, 32),  # 采样3: batch_size=1, num_images=4
    ]

    merged_tensor, merged_mask, original_shapes = merge_samples(samples, mask)
    print(f"Merged tensor shape: {merged_tensor.shape}")  # [6, 5, 16, 32, 32]


    # 恢复示例
    recovered_batch = recover_from_padded(merged_tensor, merged_mask, original_shapes)
    print(f"Recovered batch length: {len(recovered_batch)}")  # 6
    print(f"First sample images: {len(recovered_batch[0])}")  # 根据原始形状恢复
    
    
    # # 完整使用示例
    # processor = ImageSequenceProcessor()

    # # 模拟多个训练批次
    # batch1 = [
    #     [torch.randn(16, 32, 32) for _ in range(i)] for i in [3, 2, 4]
    # ]
    # batch2 = [
    #     [torch.randn(16, 32, 32) for _ in range(i)] for i in [1, 5, 2]
    # ]

    # # 准备批次
    # prepared1 = processor.prepare_training_batch(batch1)
    # prepared2 = processor.prepare_training_batch(batch2)

    # # 合并批次
    # merged_tensor, merged_mask = processor.merge_multiple_batches([prepared1, prepared2])

    # print(f"Final merged shape: {merged_tensor.shape}")