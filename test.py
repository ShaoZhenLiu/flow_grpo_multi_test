import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
from PIL import Image

# 使用本地模型路径
pipe = FluxKontextPipeline.from_pretrained("/data4/shaozhen.liu/model/FLUX.1-Kontext-dev/", torch_dtype=torch.bfloat16)
pipe.to("cuda:1")

input_image = load_image("assests/ym.jpg")
input_image_2 = load_image("assests/zjm.jpg")
input_image_3 = load_image("assests/gyy.jpg")

# 调整两张图片到相同尺寸
input_image = input_image.resize((512, 512))
input_image_2 = input_image_2.resize((512, 512))
input_image_3 = input_image_3.resize((512, 512))

# 方法1：水平拼接两张图片 (PIL方式)
combined_width = input_image.width + input_image_2.width + input_image_3.width
combined_height = max(input_image.height, input_image_2.height, input_image_3.height)

# 创建新的画布
combined_image = Image.new('RGB', (combined_width, combined_height))

# 粘贴两张图片
combined_image.paste(input_image, (0, 0))
combined_image.paste(input_image_2, (input_image.width, 0))
combined_image.paste(input_image_3, (input_image.width + input_image_2.width, 0))
#保存
combined_image.save("reference_combined.png")

# 保存拼接后的参考图
combined_image.save("reference_combined.png")

# 使用拼接后的图像进行编辑
result = pipe(
    image=combined_image,  # 使用拼接后的图像作为输入
  prompt="Three people, matching the identity of their respective reference images, are walking together naturally in the school. The left, middle, and right individuals should preserve the facial features and gender of the left, middle, and right reference images, respectively. Ensure natural posture, clothing, and a coherent background.don't only copy paste the reference image, ensure the identity of the person,and make it look like a real scene",
    guidance_scale=2.5
).images[0]

result.save("three.png")