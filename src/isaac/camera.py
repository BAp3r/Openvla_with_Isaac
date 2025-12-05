"""
相机工具函数
"""

import os
import tempfile
import numpy as np
from PIL import Image


def capture_viewport_image(size=(256, 256)):
    """从 Viewport 截取图像"""
    import omni.kit.viewport.utility as vp_util
    from omni.kit.viewport.utility import capture_viewport_to_file
    
    try:
        viewport_api = vp_util.get_active_viewport()
        if viewport_api is None:
            return None
        
        # 使用临时文件
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
        
        # 截图到文件
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


def set_viewport_camera(camera_path: str):
    """设置 Viewport 使用指定相机"""
    import omni.kit.viewport.utility as vp_util
    
    viewport_api = vp_util.get_active_viewport()
    if viewport_api:
        viewport_api.set_active_camera(camera_path)
        print(f"  ✓ Viewport 已切换到相机: {camera_path}")
        return True
    else:
        print("  ⚠ 无法获取 Viewport")
        return False
