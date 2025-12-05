"""
OpenVLA Server 主应用
"""

import os
from fastapi import FastAPI

from .model import OpenVLAModel
from .routes import router, set_model


def create_app() -> FastAPI:
    """创建 FastAPI 应用"""
    app = FastAPI(
        title="OpenVLA API Server",
        description="OpenVLA 视觉语言动作模型 API 服务",
        version="1.0.0"
    )
    
    # 加载模型
    model_path = os.getenv("OPENVLA_MODEL_PATH", "/home/wuyou/code1/openvla/openvla-7b")
    device = os.getenv("OPENVLA_DEVICE", "cuda:0")
    enable_compile = os.getenv("ENABLE_TORCH_COMPILE", "false").lower() == "true"
    
    model = OpenVLAModel(model_path, device, enable_compile)
    set_model(model)
    
    # 注册路由
    app.include_router(router)
    
    return app


# 创建应用实例
app = create_app()


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """运行服务器"""
    import uvicorn
    
    print("\n" + "=" * 60)
    print("启动 OpenVLA API Server...")
    print(f"API 文档: http://{host}:{port}/docs")
    print(f"健康检查: http://{host}:{port}/health")
    print(f"推理统计: http://{host}:{port}/stats")
    print("=" * 60 + "\n")
    
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
