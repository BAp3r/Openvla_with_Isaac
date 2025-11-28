from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch
import numpy as np

# 加载离线模型和处理器
model_path = '/home/wuyou/code1/openvla/openvla-7b'		# 离线模型文件夹路径

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    model_path,
    attn_implementation="flash_attention_2",
    # attn_implementation="sdpa",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to("cuda:0")

print('-' * 50)
print(f"Model is using attention implementation: {vla.config._attn_implementation}")
print('-' * 50)

# 直接使用噪声图像作为模型的输入
noise = torch.randn((3, 224, 224), dtype=torch.float32)  # 模型输入尺寸为 224x224
noise_image = Image.fromarray((noise.numpy().transpose(1, 2, 0) * 255).astype(np.uint8))

# 格式化提示语
prompt = "In: What action should the robot take to {<INSTRUCTION>}?\nOut:"

# 预测动作
inputs = processor(prompt, noise_image).to("cuda:0", dtype=torch.bfloat16)
action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

# 打印输出
print("Predicted Action:", action)
