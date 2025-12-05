# OpenVLA IsaacSim 调试踩坑记录

## 项目重构 (2025-12-05)

### 新目录结构
```
openvla_isaac/
├── configs/           # 配置文件
├── scripts/           # 启动脚本  
├── src/
│   ├── server/        # OpenVLA API 服务
│   ├── isaac/         # Isaac Sim 仿真模块
│   ├── clients/       # API 客户端
│   └── utils/         # 工具函数
```

---

## 1. 相机图像全黑/全白/全灰
- **原因**：
  - 没有添加光源，场景无照明导致全黑。
  - HDR/线性渲染数据未做归一化，直接转 uint8 导致全白或全灰。
  - Replicator/Camera API 在 Isaac Sim 5.x 下部分方法失效。
- **解决方法**：
  - 场景初始化时添加 DomeLight、DistantLight、SphereLight。
  - 保存/处理图像前先归一化到 0-1，再乘 255 转 uint8。
  - 使用 Viewport 截图方式获取图像，保证和窗口一致。

## 2. 相机 API 兼容性问题
- **表现**：`get_rgba`、Replicator Annotator 得到的图片全白或全灰。
- **解决**：
  - 直接用 Viewport 截图 (`capture_viewport_to_file`)，再读取 PNG。
  - 创建 USD Camera，并设置为 Viewport 活动相机。

## 3. 磁盘空间不足
- **表现**：报错 `No space left on device`，仿真无法正常运行。
- **解决**：清理 pip/conda 缓存、IsaacSim 缓存、apt autoremove。

## 4. 光源缺失
- **表现**：场景物体不可见，图像全黑。
- **解决**：添加多种光源，推荐 DomeLight + DistantLight + SphereLight。

## 5. 图像调试
- **建议**：
  - 保存调试图像时务必归一化，避免全白。
  - 打印 min/max/mean 统计值辅助排查。

---

# 一周内 TODO

1. **完善仿真场景**
   - 增加更多物体和交互任务
   - 优化相机视角和分辨率
2. **OpenVLA 推理链路测试**
   - 用真实场景图片测试动作输出
   - 检查动作分布合理性
3. **自动化调试脚本**
   - 封装一键清理缓存/调试环境脚本
   - 自动保存和归档调试图片
4. **文档完善**
   - 补充环境搭建、常见问题 FAQ
   - 记录所有已知坑和解决方法
5. **Git 版本管理**
   - 提交所有修复和文档到仓库
   - 标记本周调试分支

---

# Git 操作建议

```bash
cd ~/code1/openvla_isaac
# 添加并提交所有更改
 git add .
 git commit -m "修复相机图像问题，添加光源，更新调试文档"
 git push
```
