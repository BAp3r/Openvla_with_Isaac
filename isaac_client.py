# Terminal 2: 在 Isaac Sim 环境中运行
# ./python.sh isaac_sim_client.py

import requests
import base64
import numpy as np
from PIL import Image
import io

# Isaac Sim imports
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.franka import Franka
from omni.isaac.sensor import Camera

class OpenVLAClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def predict_action(self, image: np.ndarray, instruction: str) -> np.ndarray:
        """调用 OpenVLA API 获取动作"""
        # 将图像转换为 base64
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
            timeout=30
        )
        response.raise_for_status()
        
        # 解析动作
        action_str = response.json()["choices"][0]["message"]["content"]
        action = np.array([float(x) for x in action_str.split(",")])
        return action

# 初始化仿真
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

# 添加 Franka Panda
franka = world.scene.add(
    Franka(prim_path="/World/Franka", name="franka")
)

# 添加相机
camera = Camera(
    prim_path="/World/Camera",
    position=np.array([1.0, 0.0, 1.0]),
    frequency=30,
    resolution=(256, 256)
)

# 初始化 OpenVLA 客户端
vla_client = OpenVLAClient("http://localhost:8000")

world.reset()

# 主循环
instruction = "pick up the red cube"
while simulation_app.is_running():
    world.step(render=True)
    
    # 获取相机图像
    image = camera.get_rgb()
    
    # 每隔一定步数调用 VLA
    if world.current_time_step_index % 10 == 0:
        try:
            action = vla_client.predict_action(image, instruction)
            # action: [x, y, z, rx, ry, rz, gripper]
            
            # 应用动作到 Franka
            current_pos = franka.get_world_pose()[0]
            target_pos = current_pos + action[:3] * 0.01  # 缩放位置增量
            
            # 这里需要根据你的控制器实现动作执行
            # franka.apply_action(...)
            
        except Exception as e:
            print(f"VLA 调用失败: {e}")

simulation_app.close()