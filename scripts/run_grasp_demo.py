#!/usr/bin/env python
"""
启动 Isaac Sim 抓取演示

使用方法:
    cd ~/IsaacSim && ./python.sh ~/code1/openvla_isaac/scripts/run_grasp_demo.py
    
注意: 必须使用 Isaac Sim 的 Python 环境运行
"""

import sys
import os

# 设置资产路径
os.environ.setdefault("ISAACSIM_ASSETS_PATH", "/home/wuyou/Data1/isaac_sim_assets/Assets/Isaac/5.1")

# ===================== Isaac Sim 初始化 (必须最先执行) =====================
from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": False,
    "width": 1280,
    "height": 720,
    "window_width": 1920,
    "window_height": 1080,
})

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 导入模块 (必须在 SimulationApp 之后)
import numpy as np
from PIL import Image

from isaacsim.core.api.world import World

from src.isaac.scene import create_grasp_scene
from src.isaac.camera import capture_viewport_image, set_viewport_camera
from src.isaac.controllers import GraspController
from src.clients.vla_client import OpenVLAClient
from src.utils.input_handler import InputHandler


def main():
    print("\n正在初始化 Isaac Sim...")
    
    # 创建仿真世界
    world = World(stage_units_in_meters=1.0)
    
    # 创建场景
    scene_objects = create_grasp_scene(world)
    franka = scene_objects["franka"]
    camera_path = scene_objects["camera_path"]
    
    # 初始化 OpenVLA 客户端
    print("\n正在连接 OpenVLA 服务器...")
    vla_client = OpenVLAClient("http://localhost:8000")
    
    # 创建抓取控制器
    grasp_controller = GraspController(franka, vla_client, debug_dir=project_root)
    
    # 启动输入处理
    input_handler = InputHandler()
    
    # 重置仿真
    world.reset()
    
    # 设置 Viewport 相机
    print("\n设置 Viewport 相机...")
    set_viewport_camera(camera_path)
    
    # 等待场景稳定
    print("\n等待场景稳定...")
    for i in range(100):
        world.step(render=True)
        if i % 25 == 0:
            print(f"  预热进度: {i}/100")
    
    # 测试截图
    print("\n测试相机截图...")
    test_img = capture_viewport_image()
    if test_img is not None:
        print(f"  ✓ 截图成功 - 尺寸: {test_img.shape}, 值范围: [{test_img.min()}, {test_img.max()}]")
        Image.fromarray(test_img).save(f"{project_root}/debug_camera_test.png")
        print("  ✓ 测试图像已保存: debug_camera_test.png")
    else:
        print("  ⚠ 截图失败")
    
    print("\n✓ 仿真已启动，等待指令...")
    
    # 主循环
    frame_count = 0
    while simulation_app.is_running():
        # 检查新指令
        cmd = input_handler.get_command()
        if cmd:
            translated = input_handler.translate_command(cmd)
            if translated == "QUIT":
                print("\n退出程序...")
                break
            elif translated == "STOP":
                grasp_controller.stop()
            else:
                grasp_controller.set_instruction(translated)
        
        # 仿真步进
        world.step(render=True)
        frame_count += 1
        
        # 每帧都执行底层控制
        grasp_controller.control_step()
        
        # 获取相机图像并更新目标（低频）
        if frame_count % 5 == 0:
            camera_image = capture_viewport_image()
            if camera_image is not None and camera_image.max() > 0:
                grasp_controller.update_target(camera_image)
            elif frame_count % 50 == 0:
                print("  ⚠ 相机图像无效，跳过 VLA 调用")
    
    # 清理
    input_handler.stop()
    simulation_app.close()
    print("程序已退出")


if __name__ == "__main__":
    main()
