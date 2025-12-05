"""
机器人控制器
"""

import numpy as np
from PIL import Image
import scipy.spatial.transform as tf


class GraspController:
    """基于 OpenVLA 的抓取控制器"""
    
    def __init__(self, franka, vla_client, action_scale: float = 0.5, 
                 gripper_threshold: float = 0.5, debug_dir: str = "."):
        from isaacsim.robot.manipulators.examples.franka.controllers.rmpflow_controller import RMPFlowController
        from isaacsim.core.utils.rotations import quat_to_rot_matrix
        
        self.franka = franka
        self.vla_client = vla_client
        self.quat_to_rot_matrix = quat_to_rot_matrix
        self.debug_dir = debug_dir
        
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
        self.action_scale = action_scale
        self.gripper_threshold = gripper_threshold
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
            self._save_debug_image(camera_image)

        try:
            print(f"\n[Step {self.step_count}] 调用 OpenVLA...")
            action = self.vla_client.predict_action(camera_image, self.current_instruction)
            
            # 解析动作: [x, y, z, rx, ry, rz, gripper]
            delta_pos_local = action[:3] * self.action_scale
            delta_rpy_local = action[3:6] * self.action_scale
            
            # 获取当前 EE 姿态
            ee_pos, ee_quat = self.franka.gripper.get_world_pose()
            
            # 位置更新
            rot_matrix = self.quat_to_rot_matrix(ee_quat)
            delta_pos_world = rot_matrix @ delta_pos_local
            self.target_pos = ee_pos + delta_pos_world
            
            # 限制工作空间
            self.target_pos = np.clip(self.target_pos, 
                                      [0.2, -0.5, 0.05],
                                      [0.8, 0.5, 0.6])

            # 旋转更新
            current_quat_scipy = np.array([ee_quat[1], ee_quat[2], ee_quat[3], ee_quat[0]])
            current_rot = tf.Rotation.from_quat(current_quat_scipy)
            delta_rot = tf.Rotation.from_euler('xyz', delta_rpy_local)
            new_rot = current_rot * delta_rot
            new_quat_scipy = new_rot.as_quat()
            self.target_rot = np.array([new_quat_scipy[3], new_quat_scipy[0], 
                                        new_quat_scipy[1], new_quat_scipy[2]])

            # 夹爪控制
            gripper_action = action[6]
            
            print(f"  Raw Action: {action[:3]}")
            print(f"  动作(Local): {delta_pos_local}")
            print(f"  动作(World): {delta_pos_world}")
            print(f"  Gripper: {gripper_action:.2f}")
            
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
    
    def _save_debug_image(self, camera_image: np.ndarray):
        """保存调试图像"""
        try:
            debug_img = camera_image
            if debug_img.dtype != np.uint8:
                debug_img = np.clip(debug_img, 0.0, None)
                if debug_img.max() > 0:
                    debug_img = debug_img / debug_img.max()
                debug_img = (debug_img * 255).astype(np.uint8)
            
            img = Image.fromarray(debug_img)
            img.save(f"{self.debug_dir}/debug_step_{self.step_count}.png")
        except Exception as e:
            print(f"  调试图像保存失败: {e}")
