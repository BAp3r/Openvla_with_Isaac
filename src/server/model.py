"""
OpenVLA 模型加载与管理
"""

import os
import time
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor


def get_best_attn_implementation():
    """自动检测最佳的 Attention 实现"""
    # 优先级: flash_attention_2 > sdpa > eager
    try:
        import flash_attn
        print("✓ Flash Attention 2 可用，启用 flash_attention_2")
        return "flash_attention_2"
    except ImportError:
        pass
    
    # 检查 PyTorch 版本是否支持 SDPA
    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        print("✓ PyTorch SDPA 可用，启用 sdpa")
        return "sdpa"
    
    print("⚠ 使用默认 eager attention")
    return "eager"


class OpenVLAModel:
    """OpenVLA 模型封装类"""
    
    def __init__(self, model_path: str, device: str = "cuda:0", enable_compile: bool = False):
        self.model_path = model_path
        self.device = device
        self.attn_impl = get_best_attn_implementation()
        
        self._print_config()
        self._load_model(enable_compile)
    
    def _print_config(self):
        """打印配置信息"""
        print(f"=" * 60)
        print(f"OpenVLA 模型配置:")
        print(f"  模型路径: {self.model_path}")
        print(f"  设备: {self.device}")
        print(f"  Attention 实现: {self.attn_impl}")
        print(f"  PyTorch 版本: {torch.__version__}")
        print(f"  CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"=" * 60)
    
    def _load_model(self, enable_compile: bool):
        """加载模型"""
        print("正在加载模型...")
        load_start = time.time()
        
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            attn_implementation=self.attn_impl,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto"
        )
        
        # 设置为推理模式
        self.model.eval()
        
        # 可选：启用 torch.compile 加速
        if enable_compile and hasattr(torch, 'compile'):
            print("正在编译模型 (torch.compile)...")
            self.model = torch.compile(self.model, mode="reduce-overhead")
            print("✓ torch.compile 完成")
        
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        
        load_time = time.time() - load_start
        print(f"✓ 模型加载完成，耗时: {load_time:.2f}s")
    
    def predict_action(self, image, instruction: str, unnorm_key: str = "bridge_orig"):
        """预测动作"""
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        
        with torch.inference_mode():
            inputs = self.processor(prompt, image).to(self.model.device, dtype=torch.bfloat16)
            action = self.model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
        
        return action
    
    @property
    def model_device(self):
        """获取模型设备"""
        return self.model.device if hasattr(self.model, 'device') else self.device
