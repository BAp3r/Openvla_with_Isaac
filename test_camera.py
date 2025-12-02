# 测试 Isaac Sim 相机 API
import numpy as np
from PIL import Image
import os

os.environ.setdefault("ISAACSIM_ASSETS_PATH", "/home/wuyou/Data1/isaac_sim_assets/Assets/Isaac/5.1")

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False, "width": 1280, "height": 720})

from isaacsim.core.api.world import World
from isaacsim.core.api.objects import FixedCuboid, DynamicCuboid
from isaacsim.sensors.camera import Camera

# 创建世界
world = World(stage_units_in_meters=1.0)

# 添加地面
world.scene.add(
    FixedCuboid(
        prim_path="/World/Ground",
        name="ground",
        position=np.array([0.0, 0.0, -0.01]),
        scale=np.array([4.0, 4.0, 0.02]),
        color=np.array([0.5, 0.5, 0.5])
    )
)

# 添加一个红色方块
world.scene.add(
    DynamicCuboid(
        prim_path="/World/RedCube",
        name="red_cube",
        position=np.array([0.5, 0.0, 0.1]),
        scale=np.array([0.1, 0.1, 0.1]),
        color=np.array([1.0, 0.0, 0.0]),
        mass=0.1
    )
)

# 添加相机
print("创建相机...")
camera = Camera(
    prim_path="/World/Camera",
    position=np.array([1.0, 0.0, 0.5]),
    frequency=30,
    resolution=(256, 256),
    orientation=np.array([0.653, -0.271, 0.271, -0.653])  # 朝向原点
)

# 重置世界
print("重置世界...")
world.reset()

# 初始化相机 (在 world.reset 之后)
print("初始化相机...")
camera.initialize()

# 预热
print("预热渲染...")
for i in range(100):
    world.step(render=True)
    if i % 20 == 0:
        print(f"  Step {i}/100")

# 尝试获取图像
print("\n测试相机 API:")

# 方法 1: get_rgba
print("\n1. 测试 camera.get_rgba():")
try:
    img = camera.get_rgba()
    if img is not None:
        print(f"   形状: {img.shape}")
        print(f"   类型: {img.dtype}")
        print(f"   值范围: [{img.min()}, {img.max()}]")
        if img.max() > 0:
            pil_img = Image.fromarray(img[:, :, :3])
            pil_img.save("test_camera_rgba.png")
            print("   ✓ 保存为 test_camera_rgba.png")
        else:
            print("   ✗ 图像全黑!")
    else:
        print("   ✗ 返回 None")
except Exception as e:
    print(f"   ✗ 错误: {e}")

# 方法 2: get_rgb
print("\n2. 测试 camera.get_rgb():")
try:
    img = camera.get_rgb()
    if img is not None:
        print(f"   形状: {img.shape}")
        print(f"   类型: {img.dtype}")
        print(f"   值范围: [{img.min()}, {img.max()}]")
        if img.max() > 0:
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            pil_img = Image.fromarray(img)
            pil_img.save("test_camera_rgb.png")
            print("   ✓ 保存为 test_camera_rgb.png")
        else:
            print("   ✗ 图像全黑!")
    else:
        print("   ✗ 返回 None")
except Exception as e:
    print(f"   ✗ 错误: {e}")

# 方法 3: get_current_frame
print("\n3. 测试 camera.get_current_frame():")
try:
    frame = camera.get_current_frame()
    if frame is not None:
        print(f"   类型: {type(frame)}")
        if isinstance(frame, dict):
            print(f"   Keys: {frame.keys()}")
            for key, value in frame.items():
                if hasattr(value, 'shape'):
                    print(f"   {key}: shape={value.shape}, dtype={value.dtype}, range=[{value.min()}, {value.max()}]")
                    if 'rgb' in key.lower() or 'rgba' in key.lower():
                        if value.max() > 0:
                            if value.dtype != np.uint8:
                                value = (value * 255).astype(np.uint8)
                            if len(value.shape) == 3:
                                pil_img = Image.fromarray(value[:, :, :3] if value.shape[2] >= 3 else value)
                                pil_img.save(f"test_camera_{key}.png")
                                print(f"   ✓ 保存为 test_camera_{key}.png")
    else:
        print("   ✗ 返回 None")
except Exception as e:
    print(f"   ✗ 错误: {e}")

# 列出相机的所有方法
print("\n4. 相机可用方法:")
methods = [m for m in dir(camera) if not m.startswith('_') and callable(getattr(camera, m, None))]
for m in sorted(methods):
    if 'get' in m.lower() or 'data' in m.lower() or 'image' in m.lower() or 'frame' in m.lower():
        print(f"   - {m}")

# 多运行几帧再试
print("\n5. 再等待 50 帧后重试 get_rgba:")
for _ in range(50):
    world.step(render=True)

try:
    img = camera.get_rgba()
    if img is not None and img.max() > 0:
        pil_img = Image.fromarray(img[:, :, :3])
        pil_img.save("test_camera_rgba_final.png")
        print(f"   ✓ 成功! 保存为 test_camera_rgba_final.png")
        print(f"   形状: {img.shape}, 值范围: [{img.min()}, {img.max()}]")
    else:
        print(f"   ✗ 图像仍然全黑或为 None")
except Exception as e:
    print(f"   ✗ 错误: {e}")

print("\n测试完成，按 Ctrl+C 退出...")

# 保持运行以便查看窗口
try:
    while simulation_app.is_running():
        world.step(render=True)
except KeyboardInterrupt:
    pass

simulation_app.close()
