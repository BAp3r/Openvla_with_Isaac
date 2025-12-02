# 测试 Isaac Sim 5.x 相机 - 使用 Replicator/RenderProduct API
import numpy as np
from PIL import Image
import os

os.environ.setdefault("ISAACSIM_ASSETS_PATH", "/home/wuyou/Data1/isaac_sim_assets/Assets/Isaac/5.1")

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False, "width": 1280, "height": 720})

import omni.replicator.core as rep
from pxr import UsdGeom, Gf, UsdLux
import omni.usd

from isaacsim.core.api.world import World
from isaacsim.core.api.objects import FixedCuboid, DynamicCuboid

# 创建世界
world = World(stage_units_in_meters=1.0)

# ========== 添加光源 ==========
print("添加光源...")
stage = omni.usd.get_context().get_stage()

# 1. 添加 Dome Light (环境光/天空光)
dome_light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
dome_light.GetIntensityAttr().Set(1000)
print("  ✓ 添加 DomeLight (环境光)")

# 2. 添加 Distant Light (平行光/太阳光)
distant_light = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
distant_light.GetIntensityAttr().Set(3000)
distant_light.GetAngleAttr().Set(0.53)  # 太阳角度
# 设置光源方向
xform = UsdGeom.Xformable(distant_light.GetPrim())
xform.AddRotateXYZOp().Set(Gf.Vec3d(-45, 30, 0))
print("  ✓ 添加 DistantLight (平行光)")

# 3. 添加一个 Sphere Light (点光源) 作为补充
sphere_light = UsdLux.SphereLight.Define(stage, "/World/SphereLight")
sphere_light.GetIntensityAttr().Set(5000)
sphere_light.GetRadiusAttr().Set(0.1)
# 设置位置
xform = UsdGeom.Xformable(sphere_light.GetPrim())
xform.AddTranslateOp().Set(Gf.Vec3d(0.5, 0.5, 1.0))
print("  ✓ 添加 SphereLight (点光源)")

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

# 添加黄色方块
world.scene.add(
    DynamicCuboid(
        prim_path="/World/YellowCube",
        name="yellow_cube",
        position=np.array([0.3, 0.2, 0.1]),
        scale=np.array([0.08, 0.08, 0.08]),
        color=np.array([1.0, 1.0, 0.0]),
        mass=0.1
    )
)

# ========== 方法 1: 使用 USD 直接创建相机 ==========
print("\n创建 USD 相机...")
stage = omni.usd.get_context().get_stage()

# 创建相机 prim
camera_path = "/World/MyCamera"
camera_prim = stage.DefinePrim(camera_path, "Camera")
camera = UsdGeom.Camera(camera_prim)

# 设置相机属性
camera.GetFocalLengthAttr().Set(24.0)
camera.GetHorizontalApertureAttr().Set(20.955)

# 设置相机位置和朝向
xform = UsdGeom.Xformable(camera_prim)
xform.ClearXformOpOrder()

# 位置
translate_op = xform.AddTranslateOp()
translate_op.Set(Gf.Vec3d(1.0, 0.0, 0.6))

# 旋转 - 让相机朝向原点
rotate_op = xform.AddRotateXYZOp()
rotate_op.Set(Gf.Vec3d(-30, 0, 0))  # 俯视角度

print(f"  相机路径: {camera_path}")

# 重置世界
print("\n重置世界...")
world.reset()

# 预热
print("预热渲染...")
for i in range(50):
    world.step(render=True)
    if i % 10 == 0:
        print(f"  Step {i}/50")

# ========== 方法 2: 使用 Replicator 创建 RenderProduct ==========
print("\n创建 Replicator RenderProduct...")

# 创建渲染产品
render_product = rep.create.render_product(camera_path, (256, 256))

# 创建 annotator 获取 RGB 数据
rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
rgb_annotator.attach([render_product])

print("  RenderProduct 已创建")

# 等待渲染
print("\n等待渲染初始化...")
for i in range(100):
    world.step(render=True)
    if i % 20 == 0:
        print(f"  Step {i}/100")

# 获取图像
print("\n尝试获取图像...")

# 方法 A: 使用 annotator
print("\n方法 A - Replicator Annotator:")
try:
    rep.orchestrator.step()  # 触发渲染
    data = rgb_annotator.get_data()
    if data is not None:
        print(f"   类型: {type(data)}")
        if hasattr(data, 'shape'):
            print(f"   形状: {data.shape}")
            print(f"   dtype: {data.dtype}")
            print(f"   值范围: [{data.min()}, {data.max()}]")
            print(f"   均值: {data.mean()}")
            print(f"   唯一值数量: {len(np.unique(data))}")
            # 打印一些像素值
            print(f"   左上角像素: {data[0, 0, :]}")
            print(f"   中心像素: {data[128, 128, :]}")
            print(f"   右下角像素: {data[-1, -1, :]}")
            
            # 保存原始数据
            np.save("test_replicator_raw.npy", data)
            print("   ✓ 原始数据保存为 test_replicator_raw.npy")
            
            if data.max() > 0:
                img_data = data.astype(np.float32)
                img_data = img_data - img_data.min()
                if img_data.max() > 0:
                    img_data = img_data / img_data.max()
                img_data = np.clip(img_data, 0.0, 1.0)
                img_data = (img_data * 255).astype(np.uint8)
                # 取前3通道
                if len(img_data.shape) == 3 and img_data.shape[2] >= 3:
                    img_data = img_data[:, :, :3]
                pil_img = Image.fromarray(img_data)
                pil_img.save("test_replicator_rgb.png")
                print("   ✓ 保存为 test_replicator_rgb.png")
            else:
                print("   ✗ 图像全黑!")
        elif isinstance(data, dict):
            print(f"   字典 keys: {data.keys()}")
    else:
        print("   ✗ 返回 None")
except Exception as e:
    print(f"   ✗ 错误: {e}")
    import traceback
    traceback.print_exc()

# ========== 方法 3: 使用 omni.syntheticdata ==========
print("\n方法 B - SyntheticData API:")
try:
    import omni.syntheticdata as sd
    
    # 获取传感器
    sensors = sd.SyntheticData.Get()
    print(f"   SyntheticData 实例: {sensors}")
    
    # 尝试获取 viewport
    import omni.kit.viewport.utility as vp_util
    viewport = vp_util.get_active_viewport()
    if viewport:
        print(f"   Viewport: {viewport}")
        
        # 截取 viewport
        import omni.renderer_capture
        capture = omni.renderer_capture.capture_next_frame()
        print(f"   Capture: {capture}")
except Exception as e:
    print(f"   ✗ 错误: {e}")

# ========== 方法 4: 直接从 Viewport 截图 ==========
print("\n方法 C - Viewport 截图:")
try:
    import omni.kit.viewport.utility as vp_util
    from omni.kit.viewport.utility import capture_viewport_to_file
    
    viewport_api = vp_util.get_active_viewport()
    if viewport_api:
        # 保存截图
        output_path = "/home/wuyou/code1/openvla_isaac/test_viewport_capture.png"
        capture_viewport_to_file(viewport_api, output_path)
        print(f"   ✓ Viewport 截图保存为: {output_path}")
    else:
        print("   ✗ 无法获取 viewport")
except Exception as e:
    print(f"   ✗ 错误: {e}")
    import traceback
    traceback.print_exc()

# 多等几帧再试 annotator
print("\n再等待 50 帧...")
for _ in range(50):
    world.step(render=True)
    rep.orchestrator.step()

print("\n重试 Replicator Annotator:")
try:
    data = rgb_annotator.get_data()
    if data is not None and hasattr(data, 'max') and data.max() > 0:
        img_data = data.astype(np.float32)
        img_data = img_data - img_data.min()
        if img_data.max() > 0:
            img_data = img_data / img_data.max()
        img_data = np.clip(img_data, 0.0, 1.0)
        img_data = (img_data * 255).astype(np.uint8)
        if len(img_data.shape) == 3 and img_data.shape[2] >= 3:
            img_data = img_data[:, :, :3]
        pil_img = Image.fromarray(img_data)
        pil_img.save("test_replicator_rgb_final.png")
        print(f"   ✓ 成功! 保存为 test_replicator_rgb_final.png")
        print(f"   形状: {data.shape}, 值范围: [{data.min()}, {data.max()}]")
    else:
        print(f"   ✗ 仍然失败")
except Exception as e:
    print(f"   ✗ 错误: {e}")

print("\n测试完成，10秒后自动退出...")
print("(你可以在 Isaac Sim 窗口中查看场景)")

# 保持运行一会儿
import time
start = time.time()
while simulation_app.is_running() and (time.time() - start) < 10:
    world.step(render=True)

simulation_app.close()
print("程序退出")
