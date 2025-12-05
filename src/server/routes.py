"""
FastAPI 路由定义
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import base64
import numpy as np
from PIL import Image
import io
import time

router = APIRouter()

# 全局模型引用（由 app.py 注入）
_model = None
_inference_count = 0
_total_inference_time = 0.0


def set_model(model):
    """设置模型实例"""
    global _model
    _model = model


class ChatCompletionRequest(BaseModel):
    model: str = "openvla-7b"
    messages: List[dict]
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.0


def decode_image(base64_string: str) -> Image.Image:
    """解码 base64 图像"""
    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data)).convert("RGB")


@router.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """OpenAI 兼容的 Chat Completion 接口"""
    global _inference_count, _total_inference_time
    
    if _model is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
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
        
        # 推理
        inference_start = time.time()
        action = _model.predict_action(image, instruction)
        inference_time = time.time() - inference_start
        
        _inference_count += 1
        _total_inference_time += inference_time
        
        # 返回响应
        action_list = action.tolist()
        action_str = ",".join([f"{a:.6f}" for a in action_list])
        
        return {
            "id": f"chatcmpl-openvla-{_inference_count}",
            "object": "chat.completion",
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": action_str
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "inference_time_ms": round(inference_time * 1000, 2),
                "total_inferences": _inference_count,
                "avg_inference_time_ms": round((_total_inference_time / _inference_count) * 1000, 2)
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v1/models")
async def list_models():
    """列出可用模型"""
    return {
        "object": "list",
        "data": [{"id": "openvla-7b", "object": "model"}]
    }


@router.get("/")
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
        "attn_implementation": _model.attn_impl if _model else "unknown"
    }


@router.get("/health")
async def health_check():
    """健康检查端点"""
    import torch
    return {
        "status": "healthy",
        "model_loaded": _model is not None,
        "device": str(_model.model_device) if _model else "unknown",
        "attn_implementation": _model.attn_impl if _model else "unknown",
        "cuda_available": torch.cuda.is_available(),
        "gpu_memory_used_gb": torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0,
        "gpu_memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
    }


@router.get("/stats")
async def get_stats():
    """获取推理统计信息"""
    return {
        "total_inferences": _inference_count,
        "total_inference_time_s": round(_total_inference_time, 2),
        "avg_inference_time_ms": round((_total_inference_time / _inference_count) * 1000, 2) if _inference_count > 0 else 0,
        "model_path": _model.model_path if _model else "unknown",
        "attn_implementation": _model.attn_impl if _model else "unknown"
    }


@router.post("/v1/actions")
async def predict_action_simple(image_base64: str, instruction: str):
    """简化的动作预测接口"""
    if _model is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    try:
        image = decode_image(image_base64)
        action = _model.predict_action(image, instruction)
        
        return {
            "action": action.tolist(),
            "action_names": ["x", "y", "z", "rx", "ry", "rz", "gripper"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
