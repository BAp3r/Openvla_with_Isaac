# =============================================================================
# OpenVLA + Isaac Sim 抓取演示
# 
# 运行方式:
#   Terminal 1: conda activate openvla && python openvla_server.py
#   Terminal 2: cd ~/IsaacSim && ./python.sh ~/code1/openvla_isaac/demo_grasp.py
# =============================================================================

import requests
import base64
import numpy as np
from PIL import Image
import io
import threading
import time
import os

# ===================== Isaac Sim 初始化 (必须最先执行) =====================
from isaacsim import SimulationApp

# 显式指定资产根路径，避免找不到默认 Assets 目录
# 注意：根据你的目录结构，Assets 根目录应该是包含 Isaac 文件夹的那一层
os.environ.setdefault("ISAACSIM_ASSETS_PATH", "/home/wuyou/Data1/isaac_sim_assets/Assets/Isaac/5.1")

simulation_app = SimulationApp({
    "headless": False,
    "width": 1280,
    "height": 720,
    "window_width": 1920,
    "window_height": 1080,
})

# Isaac Sim imports (必须在 SimulationApp 之后)
import omni
from isaacsim.core.api.world import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.core.prims import XFormPrim
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.robot.manipulators.examples.franka.controllers import PickPlaceController
from isaacsim.robot.manipulators.examples.franka.controllers.rmpflow_controller import RMPFlowController
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.rotations import quat_to_rot_matrix
import scipy.spatial.transform as tf
from pxr import UsdGeom, Gf, UsdPhysics, UsdLux

# ===================== OpenVLA 客户端 =====================
class OpenVLAClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.connected = False
        self._check_connection()
    
    def _check_connection(self):
        """检查服务器连接"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                self.connected = True
                print("✓ OpenVLA 服务器已连接")
                print(f"  设备: {response.json().get('device', 'unknown')}")
            else:
                print("✗ OpenVLA 服务器连接失败")
        except Exception as e:
            print(f"✗ 无法连接 OpenVLA 服务器: {e}")
            print("  请确保已运行: python openvla_server.py")
    
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
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        action_str = result["choices"][0]["message"]["content"]
        action = np.array([float(x) for x in action_str.split(",")])
        
        # 打印推理时间
        if "usage" in result:
            print(f"  推理耗时: {result['usage'].get('inference_time_ms', 'N/A')} ms")
        
        return action


# ===================== 场景构建 =====================
def create_scene(world: World):
    """创建演示场景: 桌子 + 香蕉 + 苹果"""
    
    # ===== 添加光源 (必须！否则相机图像全黑) =====
    import omni.usd
    from pxr import UsdLux, UsdGeom, Gf
    
    stage = omni.usd.get_context().get_stage()
    
    # 1. Dome Light (环境光)
    dome_light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome_light.GetIntensityAttr().Set(1000)
    
    # 2. Distant Light (平行光/太阳光)
    distant_light = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
    distant_light.GetIntensityAttr().Set(3000)
    xform = UsdGeom.Xformable(distant_light.GetPrim())
    xform.AddRotateXYZOp().Set(Gf.Vec3d(-45, 30, 0))
    
    # 3. Sphere Light (补充光源)
    sphere_light = UsdLux.SphereLight.Define(stage, "/World/SphereLight")
    sphere_light.GetIntensityAttr().Set(5000)
    sphere_light.GetRadiusAttr().Set(0.1)
    xform = UsdGeom.Xformable(sphere_light.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(0.5, 0.5, 1.0))
    
    print("  ✓ 光源已添加 (DomeLight + DistantLight + SphereLight)")
    
    # 添加地面（使用固定立方体，避免依赖官方 GroundPlane 资产）
    world.scene.add(
        FixedCuboid(
            prim_path="/World/Ground",
            name="ground",
            position=np.array([0.0, 0.0, -0.01]),
            scale=np.array([4.0, 4.0, 0.02]),
            color=np.array([0.5, 0.5, 0.5])
        )
    )
    
    # ===== 桌子 (用几个方块组合) =====
    table_height = 0.4
    table_size = [0.6, 0.8, 0.02]  # 桌面
    
    # 桌面
    table_top = world.scene.add(
        FixedCuboid(
            prim_path="/World/Table/Top",
            name="table_top",
            position=np.array([0.5, 0.0, table_height]),
            scale=np.array(table_size),
            color=np.array([0.6, 0.4, 0.2])  # 棕色
        )
    )
    
    # 桌腿
    leg_size = [0.04, 0.04, table_height]
    leg_positions = [
        [0.5 - 0.25, -0.35, table_height / 2],
        [0.5 - 0.25, 0.35, table_height / 2],
        [0.5 + 0.25, -0.35, table_height / 2],
        [0.5 + 0.25, 0.35, table_height / 2],
    ]
    for i, pos in enumerate(leg_positions):
        world.scene.add(
            FixedCuboid(
                prim_path=f"/World/Table/Leg{i}",
                name=f"table_leg_{i}",
                position=np.array(pos),
                scale=np.array(leg_size),
                color=np.array([0.5, 0.3, 0.1])
            )
        )
    
    # ===== 香蕉 (用黄色长方体表示) =====
    banana = world.scene.add(
        DynamicCuboid(
            prim_path="/World/Banana",
            name="banana",
            position=np.array([0.4, -0.1, table_height + 0.03]),
            scale=np.array([0.12, 0.03, 0.03]),
            color=np.array([1.0, 0.9, 0.0]),  # 黄色
            mass=0.1
        )
    )
    
    # ===== 苹果 (用红色立方体表示) =====
    apple = world.scene.add(
        DynamicCuboid(
            prim_path="/World/Apple",
            name="apple",
            position=np.array([0.5, 0.15, table_height + 0.03]),
            scale=np.array([0.05, 0.05, 0.05]),
            color=np.array([0.9, 0.1, 0.1]),  # 红色
            mass=0.15
        )
    )
    
    # ===== Franka Panda 机械臂 =====
    # 注意：Franka 类内部也会尝试加载资产，如果找不到默认路径会报错
    # 这里我们显式指定 usd_path 为 None，让它使用默认逻辑，或者手动指定一个本地 USD 路径
    # 如果本地有 Franka USD，最好指定 usd_path="/path/to/franka.usd"
    try:
        franka = world.scene.add(
            Franka(
                prim_path="/World/Franka",
                name="franka",
                position=np.array([0.0, 0.0, 0.0])
            )
        )
    except Exception as e:
        print(f"⚠ 无法加载默认 Franka 资产: {e}")
        print("尝试使用备用 USD 路径...")
        # 尝试从 Isaac Sim 资产目录加载
        assets_root = os.environ.get("ISAACSIM_ASSETS_PATH")
        # 更新为 Isaac Sim 5.1 的新路径结构
        franka_usd_path = f"{assets_root}/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
        
        franka = world.scene.add(
            Franka(
                prim_path="/World/Franka",
                name="franka",
                position=np.array([0.0, 0.0, 0.0]),
                usd_path=franka_usd_path
            )
        )
    
    # ===== 相机 (第三人称视角) =====
    # 使用 USD 创建相机并设置为 Viewport 视角
    import omni.usd
    from pxr import UsdGeom
    
    stage = omni.usd.get_context().get_stage()
    camera_path = "/World/VLACamera"
    camera_prim = stage.DefinePrim(camera_path, "Camera")
    usd_camera = UsdGeom.Camera(camera_prim)
    usd_camera.GetFocalLengthAttr().Set(18.0)  # 广角
    usd_camera.GetHorizontalApertureAttr().Set(20.955)
    
    # 设置相机位置
    xform = UsdGeom.Xformable(camera_prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(1.2, 0.0, 0.8))
    xform.AddRotateXYZOp().Set(Gf.Vec3d(-40, 0, 0))  # 俯视角度
    
    print("\n✓ 场景创建完成:")
    print("  - 桌子: 棕色木桌")
    print("  - 香蕉: 黄色 (位于桌面左侧)")
    print("  - 苹果: 红色 (位于桌面右侧)")
    print("  - Franka Panda: 7-DoF 机械臂")
    print("  - 相机: 第三人称视角 (VLACamera)")
    
    return {
        "franka": franka,
        "banana": banana,
        "apple": apple,
        "camera_path": camera_path,  # 返回相机路径而不是 Camera 对象
        "table_height": table_height
    }


# ===================== 抓取控制器 =====================
class GraspController:
    """基于 OpenVLA 的抓取控制器"""
    
    def __init__(self, franka, vla_client: OpenVLAClient):
        self.franka = franka
        self.vla_client = vla_client
        
        # 使用 RMPFlow 控制器
        self.controller = RMPFlowController(
            name="target_follower_controller", 
            robot_articulation=franka
        )
        
        # 状态
        self.current_instruction = None
        self.is_executing = False
        self.step_count = 0
        
        # 目标位置（初始化为当前位置）
        self.target_pos, self.target_rot = self.franka.gripper.get_world_pose()
        
        # 控制参数
        self.action_scale = 0.5  # 大幅增加动作缩放因子
        self.gripper_threshold = 0.5
        self.gripper_open = True
        
    def set_instruction(self, instruction: str):
        """设置新的指令"""
        self.current_instruction = instruction
        self.is_executing = True
        self.step_count = 0
        
        # 重置控制器状态
        self.controller.reset()
        
        # 重置目标为当前位置
        self.target_pos, self.target_rot = self.franka.gripper.get_world_pose()
        print(f"\n>>> 开始执行指令: {instruction}")
        print(f"    初始位置: {self.target_pos}")
    
    def stop(self):
        """停止执行"""
        self.is_executing = False
        self.current_instruction = None
        print(">>> 停止执行")
    
    def update_target(self, camera_image: np.ndarray):
        """调用 VLA 更新目标位置"""
        if not self.is_executing or self.current_instruction is None:
            return
        
        self.step_count += 1
        
        # 保存调试图像
        if self.step_count % 10 == 0:
            try:
                debug_img = camera_image
                if debug_img.dtype != np.uint8:
                    debug_img = np.clip(debug_img, 0.0, None)
                    if debug_img.max() > 0:
                        debug_img = debug_img / debug_img.max()
                    debug_img = (debug_img * 255).astype(np.uint8)
                img = Image.fromarray(debug_img)
                img.save(f"debug_step_{self.step_count}.png")
            except Exception as e:
                print(f"  调试图像保存失败: {e}")

        try:
            print(f"\n[Step {self.step_count}] 调用 OpenVLA...")
            action = self.vla_client.predict_action(camera_image, self.current_instruction)
            
            # 解析动作: [x, y, z, rx, ry, rz, gripper]
            # OpenVLA 输出通常是相对末端执行器坐标系的
            # 使用 self.action_scale (0.5) 而不是硬编码的 0.05
            delta_pos_local = action[:3] * self.action_scale
            delta_rpy_local = action[3:6] * self.action_scale
            
            # 获取当前 EE 姿态 (Isaac Sim 返回 [w, x, y, z])
            ee_pos, ee_quat = self.franka.gripper.get_world_pose()
            
            # --- 位置更新 ---
            # 将四元数转为旋转矩阵
            rot_matrix = quat_to_rot_matrix(ee_quat)
            
            # 将局部位移转换到世界坐标系
            delta_pos_world = rot_matrix @ delta_pos_local
            
            # 更新目标位置 (基于当前真实位置累加)
            self.target_pos = ee_pos + delta_pos_world
            
            # 限制工作空间
            self.target_pos = np.clip(self.target_pos, 
                                 [0.2, -0.5, 0.05],
                                 [0.8, 0.5, 0.6])

            # --- 旋转更新 ---
            # Scipy Rotation 使用 [x, y, z, w] 格式
            # Isaac Sim 使用 [w, x, y, z] 格式
            current_quat_scipy = np.array([ee_quat[1], ee_quat[2], ee_quat[3], ee_quat[0]])
            current_rot = tf.Rotation.from_quat(current_quat_scipy)
            
            # 计算相对旋转 (假设 VLA 输出为 XYZ 欧拉角)
            delta_rot = tf.Rotation.from_euler('xyz', delta_rpy_local)
            
            # 应用旋转: New = Current * Delta (局部坐标系下的旋转)
            new_rot = current_rot * delta_rot
            
            # 转回 Isaac Sim 格式 [w, x, y, z]
            new_quat_scipy = new_rot.as_quat() # [x, y, z, w]
            self.target_rot = np.array([new_quat_scipy[3], new_quat_scipy[0], new_quat_scipy[1], new_quat_scipy[2]])

            # --- 夹爪控制 ---
            gripper_action = action[6]
            
            print(f"  Raw Action: {action[:3]}")
            print(f"  动作(Local): {delta_pos_local}")
            print(f"  动作(World): {delta_pos_world}")
            print(f"  Gripper: {gripper_action:.2f}")
            
            # 更新夹爪状态
            if gripper_action > self.gripper_threshold:
                self.gripper_open = True
                print("  夹爪: 打开")
            elif gripper_action < -self.gripper_threshold:
                self.gripper_open = False
                print("  夹爪: 关闭")
                
            print(f"  新目标位置: {self.target_pos}")
            
        except Exception as e:
            print(f"  ✗ VLA 调用失败: {e}")

    def control_step(self):
        """执行底层控制（每帧调用）"""
        if not self.is_executing:
            return

        # 使用 RMPFlow 计算关节动作
        actions = self.controller.forward(
            target_end_effector_position=self.target_pos,
            target_end_effector_orientation=self.target_rot
        )
        
        # 应用关节动作
        self.franka.apply_action(actions)
        
        # 应用夹爪动作
        if self.gripper_open:
            self.franka.gripper.open()
        else:
            self.franka.gripper.close()


# ===================== 终端输入线程 =====================
class InputHandler:
    """处理终端输入"""
    
    def __init__(self):
        self.current_command = None
        self.running = True
        self._start_input_thread()
    
    def _start_input_thread(self):
        """启动输入监听线程"""
        def input_loop():
            print("\n" + "=" * 60)
            print("OpenVLA + Isaac Sim 抓取演示")
            print("=" * 60)
            print("可用指令:")
            print("  - 抓取香蕉 / pick up the banana")
            print("  - 抓取苹果 / pick up the apple")
            print("  - 把香蕉放到苹果旁边 / place the banana next to the apple")
            print("  - stop: 停止当前动作")
            print("  - quit/exit: 退出程序")
            print("=" * 60)
            
            while self.running:
                try:
                    cmd = input("\n请输入指令 >>> ").strip()
                    if cmd:
                        self.current_command = cmd
                except EOFError:
                    break
                except Exception as e:
                    print(f"输入错误: {e}")
        
        thread = threading.Thread(target=input_loop, daemon=True)
        thread.start()
    
    def get_command(self) -> str:
        """获取并清除当前命令"""
        cmd = self.current_command
        self.current_command = None
        return cmd
    
    def stop(self):
        self.running = False


# ===================== 主函数 =====================
def main():
    print("\n正在初始化 Isaac Sim...")
    
    # 创建仿真世界
    world = World(stage_units_in_meters=1.0)
    
    # 创建场景
    scene_objects = create_scene(world)
    franka = scene_objects["franka"]
    camera_path = scene_objects["camera_path"]
    
    # 初始化 OpenVLA 客户端
    print("\n正在连接 OpenVLA 服务器...")
    vla_client = OpenVLAClient("http://localhost:8000")
    
    # 创建抓取控制器
    grasp_controller = GraspController(franka, vla_client)
    
    # 启动输入处理
    input_handler = InputHandler()
    
    # 重置仿真
    world.reset()
    
    # ===== 设置 Viewport 使用我们的相机 =====
    print("\n设置 Viewport 相机...")
    import omni.kit.viewport.utility as vp_util
    
    viewport_api = vp_util.get_active_viewport()
    if viewport_api:
        viewport_api.set_active_camera(camera_path)
        print(f"  ✓ Viewport 已切换到相机: {camera_path}")
    else:
        print("  ⚠ 无法获取 Viewport")
    
    # 等待场景稳定
    print("\n等待场景稳定...")
    for i in range(100):
        world.step(render=True)
        if i % 25 == 0:
            print(f"  预热进度: {i}/100")
    
    # ===== 创建用于截图的函数 =====
    import tempfile
    import omni.kit.viewport.utility as vp_util
    
    def capture_viewport_image(size=(256, 256)):
        """从 Viewport 截取图像"""
        try:
            viewport_api = vp_util.get_active_viewport()
            if viewport_api is None:
                return None
            
            # 使用临时文件
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                temp_path = f.name
            
            # 截图到文件
            from omni.kit.viewport.utility import capture_viewport_to_file
            capture_viewport_to_file(viewport_api, temp_path)
            
            # 读取并调整大小
            img = Image.open(temp_path)
            img = img.resize(size, Image.LANCZOS)
            img_array = np.array(img)[:, :, :3]  # 只取 RGB
            
            # 删除临时文件
            os.unlink(temp_path)
            
            return img_array
        except Exception as e:
            print(f"  截图错误: {e}")
            return None
    
    # 测试截图
    print("\n测试相机截图...")
    test_img = capture_viewport_image()
    if test_img is not None:
        print(f"  ✓ 截图成功 - 尺寸: {test_img.shape}, 值范围: [{test_img.min()}, {test_img.max()}]")
        Image.fromarray(test_img).save("debug_camera_test.png")
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
            cmd_lower = cmd.lower()
            if cmd_lower in ["quit", "exit", "q"]:
                print("\n退出程序...")
                break
            elif cmd_lower == "stop":
                grasp_controller.stop()
            else:
                # 翻译常见中文指令
                if "香蕉" in cmd:
                    instruction = "pick up the banana"
                elif "苹果" in cmd:
                    instruction = "pick up the apple"
                else:
                    instruction = cmd
                grasp_controller.set_instruction(instruction)
        
        # 仿真步进
        world.step(render=True)
        frame_count += 1
        
        # 每帧都执行底层控制
        grasp_controller.control_step()
        
        # 获取相机图像并更新目标（低频）
        if frame_count % 5 == 0:  # 降低相机采集频率
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
