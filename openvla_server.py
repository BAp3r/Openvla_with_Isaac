# Terminal 1: conda activate openvla
# python openvla_server.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import base64
import numpy as np
from PIL import Image
import io
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
import os
import time

app = FastAPI()

# ==================== 推理优化配置 ====================
MODEL_PATH = os.getenv("OPENVLA_MODEL_PATH", "/home/wuyou/code1/openvla/openvla-7b")
DEVICE = os.getenv("OPENVLA_DEVICE", "cuda:0")

# 检测可用的 Attention 实现
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

attn_impl = get_best_attn_implementation()

print(f"=" * 60)
print(f"OpenVLA Server 启动配置:")
print(f"  模型路径: {MODEL_PATH}")
print(f"  设备: {DEVICE}")
print(f"  Attention 实现: {attn_impl}")
print(f"  PyTorch 版本: {torch.__version__}")
print(f"  CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"=" * 60)

# 加载 OpenVLA 模型（带优化）
print("正在加载模型...")
load_start = time.time()

model = AutoModelForVision2Seq.from_pretrained(
    MODEL_PATH,
    attn_implementation=attn_impl,  # 使用检测到的最佳实现
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    device_map="auto"
)

# 设置为推理模式
model.eval()

# 可选：启用 torch.compile 加速（PyTorch 2.0+）
ENABLE_TORCH_COMPILE = os.getenv("ENABLE_TORCH_COMPILE", "false").lower() == "true"
if ENABLE_TORCH_COMPILE and hasattr(torch, 'compile'):
    print("正在编译模型 (torch.compile)...")
    model = torch.compile(model, mode="reduce-overhead")
    print("✓ torch.compile 完成")

processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

load_time = time.time() - load_start
print(f"✓ 模型加载完成，耗时: {load_time:.2f}s")

class Message(BaseModel):
    role: str
    content: str  # 可以是文本或包含图像的列表

class ChatCompletionRequest(BaseModel):
    model: str = "openvla-7b"
    messages: List[dict]
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.0

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    model: str
    choices: List[dict]

def decode_image(base64_string: str) -> Image.Image:
    """解码 base64 图像"""
    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data)).convert("RGB")

# 推理计数器和统计
inference_count = 0
total_inference_time = 0.0

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    global inference_count, total_inference_time
    
    try:
        # 解析消息，提取图像和指令
        image = None
        instruction = ""
        
        for msg in request.messages:
            if msg["role"] == "user":
                content = msg["content"]
                if isinstance(content, list):
                    for item in content:
                        if item.get("type") == "image_url":
                            # 解析 base64 图像
                            url = item["image_url"]["url"]
                            if url.startswith("data:image"):
                                base64_data = url.split(",")[1]
                                image = decode_image(base64_data)
                        elif item.get("type") == "text":
                            instruction = item["text"]
                else:
                    instruction = content
        
        if image is None:
            raise HTTPException(status_code=400, detail="需要提供图像")
        
        # 使用 OpenVLA 推理（带优化）
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        
        inference_start = time.time()
        
        # 禁用梯度计算加速推理
        with torch.inference_mode():
            inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
            action = model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
        
        inference_time = time.time() - inference_start
        inference_count += 1
        total_inference_time += inference_time
        
        # 返回 OpenAI 格式的响应
        action_list = action.tolist()
        action_str = ",".join([f"{a:.6f}" for a in action_list])
        
        return {
            "id": f"chatcmpl-openvla-{inference_count}",
            "object": "chat.completion",
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": action_str  # 7-DoF: [x, y, z, rx, ry, rz, gripper]
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "inference_time_ms": round(inference_time * 1000, 2),
                "total_inferences": inference_count,
                "avg_inference_time_ms": round((total_inference_time / inference_count) * 1000, 2)
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": "openvla-7b", "object": "model"}]
    }

@app.get("/")
async def root():
    """根路径欢迎页面"""
    return {
        "message": "OpenVLA API Server",
        "version": "1.0.0",
        "endpoints": {
            "chat_completions": "POST /v1/chat/completions",
            "models": "GET /v1/models",
            "health": "GET /health",
            "stats": "GET /stats",
            "docs": "GET /docs"
        },
        "status": "running",
        "attn_implementation": attn_impl
    }

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(model.device) if hasattr(model, 'device') else DEVICE,
        "attn_implementation": attn_impl,
        "cuda_available": torch.cuda.is_available(),
        "gpu_memory_used_gb": torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0,
        "gpu_memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
    }

@app.get("/stats")
async def get_stats():
    """获取推理统计信息"""
    return {
        "total_inferences": inference_count,
        "total_inference_time_s": round(total_inference_time, 2),
        "avg_inference_time_ms": round((total_inference_time / inference_count) * 1000, 2) if inference_count > 0 else 0,
        "model_path": MODEL_PATH,
        "attn_implementation": attn_impl
    }

@app.post("/v1/actions")
async def predict_action_simple(image_base64: str, instruction: str):
    """简化的动作预测接口（非 OpenAI 格式）"""
    try:
        image = decode_image(image_base64)
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        
        with torch.inference_mode():
            inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
            action = model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
        
        return {
            "action": action.tolist(),
            "action_names": ["x", "y", "z", "rx", "ry", "rz", "gripper"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("启动 OpenVLA API Server...")
    print("API 文档: http://0.0.0.0:8000/docs")
    print("健康检查: http://0.0.0.0:8000/health")
    print("推理统计: http://0.0.0.0:8000/stats")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)