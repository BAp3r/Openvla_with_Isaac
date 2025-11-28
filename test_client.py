"""
OpenVLA API 测试客户端
用于测试 openvla_server.py 是否正常工作
"""

import requests
import base64
import numpy as np
from PIL import Image
import io
import time
import json

# API 服务器地址
API_BASE_URL = "http://localhost:8000"


def image_to_base64(image: Image.Image) -> str:
    """将 PIL 图像转换为 base64 字符串"""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def create_test_image(width=256, height=256, color="red") -> Image.Image:
    """创建测试图像"""
    color_map = {
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
        "random": None
    }
    
    if color == "random":
        # 随机噪声图像
        arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        return Image.fromarray(arr)
    else:
        return Image.new("RGB", (width, height), color_map.get(color, (128, 128, 128)))


def test_health():
    """测试健康检查端点"""
    print("=" * 50)
    print("测试 /health 端点...")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        print(f"状态码: {response.status_code}")
        print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False


def test_root():
    """测试根路径端点"""
    print("=" * 50)
    print("测试 / 端点...")
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=10)
        print(f"状态码: {response.status_code}")
        print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False


def test_models():
    """测试模型列表端点"""
    print("=" * 50)
    print("测试 /v1/models 端点...")
    try:
        response = requests.get(f"{API_BASE_URL}/v1/models", timeout=10)
        print(f"状态码: {response.status_code}")
        print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False


def test_chat_completion(instruction: str = "pick up the red cube", image_color: str = "random"):
    """测试 OpenAI 格式的聊天补全端点"""
    print("=" * 50)
    print(f"测试 /v1/chat/completions 端点...")
    print(f"  指令: {instruction}")
    print(f"  图像: {image_color} 色图像")
    
    try:
        # 创建测试图像
        image = create_test_image(256, 256, image_color)
        base64_img = image_to_base64(image)
        
        # 构造 OpenAI 格式的请求
        payload = {
            "model": "openvla-7b",
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_img}"}
                    },
                    {
                        "type": "text",
                        "text": instruction
                    }
                ]
            }]
        }
        
        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}/v1/chat/completions",
            json=payload,
            timeout=60
        )
        elapsed_time = time.time() - start_time
        
        print(f"状态码: {response.status_code}")
        print(f"请求耗时: {elapsed_time:.2f}s")
        
        if response.status_code == 200:
            result = response.json()
            action_str = result["choices"][0]["message"]["content"]
            action = [float(x) for x in action_str.split(",")]
            
            print(f"\n预测动作 (7-DoF):")
            print(f"  x:       {action[0]:.6f}")
            print(f"  y:       {action[1]:.6f}")
            print(f"  z:       {action[2]:.6f}")
            print(f"  rx:      {action[3]:.6f}")
            print(f"  ry:      {action[4]:.6f}")
            print(f"  rz:      {action[5]:.6f}")
            print(f"  gripper: {action[6]:.6f}")
            
            if "usage" in result:
                print(f"\n推理统计:")
                print(f"  推理耗时: {result['usage']['inference_time_ms']:.2f} ms")
                print(f"  总推理次数: {result['usage']['total_inferences']}")
                print(f"  平均耗时: {result['usage']['avg_inference_time_ms']:.2f} ms")
            
            return True
        else:
            print(f"❌ 错误响应: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False


def test_stats():
    """测试统计端点"""
    print("=" * 50)
    print("测试 /stats 端点...")
    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=10)
        print(f"状态码: {response.status_code}")
        print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 50)
    print("OpenVLA API 测试客户端")
    print("=" * 50)
    print(f"服务器地址: {API_BASE_URL}")
    
    results = []
    
    # 测试基本端点
    results.append(("根路径", test_root()))
    results.append(("健康检查", test_health()))
    results.append(("模型列表", test_models()))
    
    # 测试推理（多次）
    results.append(("推理测试 1 - pick up", test_chat_completion("pick up the red cube", "red")))
    results.append(("推理测试 2 - move", test_chat_completion("move to the left", "random")))
    
    # 测试统计
    results.append(("推理统计", test_stats()))
    
    # 汇总结果
    print("\n" + "=" * 50)
    print("测试结果汇总:")
    print("=" * 50)
    
    passed = 0
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{len(results)} 测试通过")
    print("=" * 50)


if __name__ == "__main__":
    run_all_tests()
