#!/usr/bin/env python
"""
启动 OpenVLA 服务器

使用方法:
    conda activate openvla
    python scripts/run_server.py
    
或者:
    python -m src.server.app
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.server.app import run_server

if __name__ == "__main__":
    run_server()
