"""
Isaac Sim 场景构建
"""

import numpy as np


def add_lights(stage):
    """添加场景光源"""
    from pxr import UsdLux, UsdGeom, Gf
    
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


def add_ground(world):
    """添加地面"""
    from isaacsim.core.api.objects import FixedCuboid
    
    world.scene.add(
        FixedCuboid(
            prim_path="/World/Ground",
            name="ground",
            position=np.array([0.0, 0.0, -0.01]),
            scale=np.array([4.0, 4.0, 0.02]),
            color=np.array([0.5, 0.5, 0.5])
        )
    )


def add_table(world, table_height: float = 0.4):
    """添加桌子"""
    from isaacsim.core.api.objects import FixedCuboid
    
    table_size = [0.6, 0.8, 0.02]
    
    # 桌面
    world.scene.add(
        FixedCuboid(
            prim_path="/World/Table/Top",
            name="table_top",
            position=np.array([0.5, 0.0, table_height]),
            scale=np.array(table_size),
            color=np.array([0.6, 0.4, 0.2])
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


def add_objects(world, table_height: float = 0.4):
    """添加桌面物体"""
    from isaacsim.core.api.objects import DynamicCuboid
    
    # 香蕉 (黄色长方体)
    banana = world.scene.add(
        DynamicCuboid(
            prim_path="/World/Banana",
            name="banana",
            position=np.array([0.4, -0.1, table_height + 0.03]),
            scale=np.array([0.12, 0.03, 0.03]),
            color=np.array([1.0, 0.9, 0.0]),
            mass=0.1
        )
    )
    
    # 苹果 (红色立方体)
    apple = world.scene.add(
        DynamicCuboid(
            prim_path="/World/Apple",
            name="apple",
            position=np.array([0.5, 0.15, table_height + 0.03]),
            scale=np.array([0.05, 0.05, 0.05]),
            color=np.array([0.9, 0.1, 0.1]),
            mass=0.15
        )
    )
    
    return {"banana": banana, "apple": apple}


def add_franka(world, assets_path: str = None):
    """添加 Franka 机械臂"""
    import os
    from isaacsim.robot.manipulators.examples.franka import Franka
    
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
        
        if assets_path is None:
            assets_path = os.environ.get("ISAACSIM_ASSETS_PATH", "")
        
        franka_usd_path = f"{assets_path}/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
        
        franka = world.scene.add(
            Franka(
                prim_path="/World/Franka",
                name="franka",
                position=np.array([0.0, 0.0, 0.0]),
                usd_path=franka_usd_path
            )
        )
    
    return franka


def add_camera(stage, camera_path: str = "/World/VLACamera", 
               position: tuple = (1.2, 0.0, 0.8), 
               rotation: tuple = (-40, 0, 0)):
    """添加相机"""
    from pxr import UsdGeom, Gf
    
    camera_prim = stage.DefinePrim(camera_path, "Camera")
    usd_camera = UsdGeom.Camera(camera_prim)
    usd_camera.GetFocalLengthAttr().Set(18.0)
    usd_camera.GetHorizontalApertureAttr().Set(20.955)
    
    xform = UsdGeom.Xformable(camera_prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*position))
    xform.AddRotateXYZOp().Set(Gf.Vec3d(*rotation))
    
    return camera_path


def create_grasp_scene(world, assets_path: str = None):
    """创建完整的抓取演示场景"""
    import omni.usd
    
    stage = omni.usd.get_context().get_stage()
    table_height = 0.4
    
    # 添加光源
    add_lights(stage)
    
    # 添加地面
    add_ground(world)
    
    # 添加桌子
    add_table(world, table_height)
    
    # 添加物体
    objects = add_objects(world, table_height)
    
    # 添加机械臂
    franka = add_franka(world, assets_path)
    
    # 添加相机
    camera_path = add_camera(stage)
    
    print("\n✓ 场景创建完成:")
    print("  - 桌子: 棕色木桌")
    print("  - 香蕉: 黄色 (位于桌面左侧)")
    print("  - 苹果: 红色 (位于桌面右侧)")
    print("  - Franka Panda: 7-DoF 机械臂")
    print("  - 相机: 第三人称视角 (VLACamera)")
    
    return {
        "franka": franka,
        "banana": objects["banana"],
        "apple": objects["apple"],
        "camera_path": camera_path,
        "table_height": table_height
    }
