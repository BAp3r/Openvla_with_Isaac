# OpenVLA + Isaac Sim Integration

OpenVLA 视觉语言动作模型与 Isaac Sim 仿真平台的集成项目。

## 项目结构

```
openvla_isaac/
├── configs/                    # 配置文件
│   ├── server_config.yaml      # OpenVLA 服务器配置
│   └── scene_config.yaml       # Isaac Sim 场景配置
├── scripts/                    # 启动脚本
│   ├── run_server.py           # 启动 OpenVLA 服务器
│   └── run_grasp_demo.py       # 启动抓取演示
├── src/                        # 源代码
│   ├── server/                 # OpenVLA 服务器模块
│   │   ├── __init__.py
│   │   ├── app.py              # FastAPI 应用
│   │   ├── model.py            # 模型加载与管理
│   │   └── routes.py           # API 路由
│   ├── isaac/                  # Isaac Sim 仿真模块
│   │   ├── __init__.py
│   │   ├── scene.py            # 场景构建
│   │   ├── camera.py           # 相机工具
│   │   └── controllers.py      # 机器人控制器
│   ├── clients/                # 客户端模块
│   │   ├── __init__.py
│   │   └── vla_client.py       # OpenVLA API 客户端
│   └── utils/                  # 工具模块
│       ├── __init__.py
│       └── input_handler.py    # 终端输入处理
├── debug_notes.md              # 调试踩坑记录
├── readme.md                   # 本文档
└── LICENSE
```

## 快速开始

### 1. 启动 OpenVLA 服务器

```bash
# 激活 OpenVLA 环境
conda activate openvla

# 启动服务器
cd ~/code1/openvla_isaac
python scripts/run_server.py

# 或者使用模块方式
python -m src.server.app
```

### 2. 启动 Isaac Sim 抓取演示

```bash
# 使用 Isaac Sim 的 Python 环境运行
cd ~/IsaacSim
./python.sh ~/code1/openvla_isaac/scripts/run_grasp_demo.py
```

### 3. 交互指令

在终端输入以下指令：
- `抓取香蕉` 或 `pick up the banana`
- `抓取苹果` 或 `pick up the apple`
- `stop` - 停止当前动作
- `quit` 或 `exit` - 退出程序

## API 端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/` | GET | 服务器信息 |
| `/health` | GET | 健康检查 |
| `/stats` | GET | 推理统计 |
| `/v1/models` | GET | 模型列表 |
| `/v1/chat/completions` | POST | OpenAI 兼容推理 |
| `/v1/actions` | POST | 简化动作预测 |
| `/docs` | GET | API 文档 |

## 配置说明

### 服务器配置 (`configs/server_config.yaml`)

```yaml
model:
  path: "/path/to/openvla-7b"
  device: "cuda:0"
server:
  host: "0.0.0.0"
  port: 8000
```

### 场景配置 (`configs/scene_config.yaml`)

```yaml
simulation:
  headless: false
  assets_path: "/path/to/isaac_sim_assets"
camera:
  position: [1.2, 0.0, 0.8]
  rotation: [-40, 0, 0]
```

## 常见问题

详见 [debug_notes.md](debug_notes.md)

## License

MIT License
