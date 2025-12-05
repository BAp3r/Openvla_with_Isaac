"""
OpenVLA API 客户端
"""

import requests
import base64
import numpy as np
from PIL import Image
import io


class OpenVLAClient:
    """OpenVLA API 客户端"""
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        self.base_url = base_url
        self.timeout = timeout
        self.connected = False
        self._check_connection()
    
    def _check_connection(self):
        """检查服务器连接"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                self.connected = True
                info = response.json()
                print("✓ OpenVLA 服务器已连接")
                print(f"  设备: {info.get('device', 'unknown')}")
                print(f"  Attention: {info.get('attn_implementation', 'unknown')}")
            else:
                print("✗ OpenVLA 服务器连接失败")
        except Exception as e:
            print(f"✗ 无法连接 OpenVLA 服务器: {e}")
            print("  请确保已运行: python -m src.server.app")
    
    def predict_action(self, image: np.ndarray, instruction: str) -> np.ndarray:
        """调用 OpenVLA API 获取动作"""
        # 将图像转换为 base64
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        pil_image = Image.fromarray(image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        base64_image = base64.b64encode(buffer.getvalue()).decode()
        
        # 构造 OpenAI 格式请求
        payload = {
            "model": "openvla-7b",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                    {"type": "text", "text": instruction}
                ]
            }]
        }
        
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        
        result = response.json()
        action_str = result["choices"][0]["message"]["content"]
        action = np.array([float(x) for x in action_str.split(",")])
        
        # 打印推理时间
        if "usage" in result:
            print(f"  推理耗时: {result['usage'].get('inference_time_ms', 'N/A')} ms")
        
        return action
    
    def get_stats(self) -> dict:
        """获取服务器统计信息"""
        response = requests.get(f"{self.base_url}/stats", timeout=5)
        response.raise_for_status()
        return response.json()
    
    def health_check(self) -> dict:
        """健康检查"""
        response = requests.get(f"{self.base_url}/health", timeout=5)
        response.raise_for_status()
        return response.json()
