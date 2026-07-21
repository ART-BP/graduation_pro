# Go2W 学习型局部地形规划器

本仓库用于训练 Unitree Go2W 的高层局部规划策略。策略不控制关节，只输出底层控制器能够执行的速度指令：

```text
[linear.x, angular.z]
```

训练采用 PPO 和基于实际执行结果的奖励，不使用 DWA、TEB、MPPI 生成专家标签，也不训练固定高程阈值分类器。

## 固定版本

第一版严格固定以下组合：

- NVIDIA Isaac Lab `2.3.2`
- NVIDIA Isaac Sim `5.1.0`
- RSL-RL `3.1.2`（由 Isaac Lab 2.3.2 提供）
- 基础镜像 `nvcr.io/nvidia/isaac-lab:2.3.2`
- 项目镜像 `go2w-terrain-planner:0.1.0`

Dockerfile 不使用 `latest`。如果基础镜像需要 NGC 权限，请先按照 NVIDIA NGC 的要求执行 `docker login nvcr.io`。

## 当前阶段和边界

当前版本是阶段 1 的完整训练闭环：

- 用 Isaac Lab `DirectRLEnv` 运行并行环境；
- 用运动学刚体作为 Go2W 外形占位对象；
- 用独立的高层速度执行模型模拟速度滞后、加速度限制、摩擦、有限地形阻力、进入角度、跟踪噪声和卡住；
- 用单级突变、障碍高度、坑深和足迹支撑变化分别构造单调的碰撞与失稳结果；
- 用解析式地形快速生成与实机接口等价的四通道局部地图；
- 仿真先生成 `200 × 200` 原始地图，再使用共享预处理逻辑保守下采样到 `100 × 100`；
- 用 CNN + GRU Actor 和特权 Critic 训练 PPO；
- 支持无显示器训练、回放、评估和 ONNX 导出。

当前版本没有声称完成以下两项：

- Go2W 完整关节、轮腿接触和原厂底层控制器的物理模型；
- XT16 十六线 Ray Caster 到点云、再到 Grid Map 的完整仿真链路。

这两项属于阶段 2。替换时保持四通道地图和 `[v_cmd,w_cmd]` 接口不变即可。资产边界定义在 `robots/go2w_cfg.py`，地图替换边界是 `SimulatedLocalMap.generate()`。

## 数据与接口

实机局部环境模块继续由以下脚本启动：

```bash
local_map_ws/start_local_environment.sh
```

规划器对应的 ROS 输入契约为：

- `/local_environment/grid_map`：`grid_map_msgs/GridMap`
- `/Odometry`：`nav_msgs/Odometry`，提供位姿和当前实测 `[v,w]`
- `/local_goal`：局部目标
- 历史速度指令
- `/local_environment/ground_reference`：与地图同时间戳的 `geometry_msgs/PointStamped`，`point.z` 为 odom 系绝对脚下地面参考高度
- 可选 IMU 历史

Grid Map 原始尺寸为 `200 × 200`、分辨率为 `0.05 m`、范围为 `10 m × 10 m`。共享预处理器将三个原始图层转换为：

```text
relative_ground_height = ground_height
height_range
observed_mask
height_valid_mask = isfinite(ground_height) AND isfinite(height_range)
```

这里的 `ground_height` 已由真实地图节点或仿真地图生成器转换为相对于机器人脚下参考地面的高度，平地约为 0；共享预处理器不会再减 FAST-LIO 的 IMU 高度。

时序缓存保存每帧相对地图所使用的绝对 `ground_reference_z`。历史地图完成 SE(2) 对齐后，先用
`source_reference_z - current_reference_z` 统一垂直零点，再送入网络。该标量不是 Actor 地图通道，由
`local_map_ws` 与 Grid Map 同时间戳发布。

所有无效高度都被固定值替换，进入网络的张量不允许包含 `NaN` 或 `Inf`。第一版将地图下采样到 `100 × 100`。

实机 Grid Map 的栅格轴与 `odom` 对齐，送入 Actor 前还要调用 `world_aligned_map_to_robot_frame()` 按当前 yaw 转成机器人坐标约定；随后再进入 `TemporalGridBuffer`。这样仿真端和实机端的“地图前方”方向一致。

Actor 输入为：

```text
map_seq       [B, 5, 4, 100, 100]
goal          [B, 3]       # normalized_distance, sin(angle), cos(angle)
current_speed [B, 2]       # 当前实测 v, w
cmd_history   [B, 4, 2]
motion_history[B, 4, 3]    # 相邻里程计的 dx, dy, dyaw
```

默认展平维度为 `200025`。Critic 额外使用仿真真值，但这些特权信息不会进入 Actor 或 ONNX 模型。

PPO 使用带雅可比修正的 Tanh-squashed Gaussian，采样动作和概率计算都严格位于 `[-1,1]`。环境依据
`configs/action.yaml` 映射到物理速度。导出的 ONNX 已包含这一步映射，直接输出物理单位的
`[v_cmd,w_cmd]`。

## 代码结构

```text
go2w_terrain_planner/
├── Dockerfile
├── docker-compose.yaml
├── configs/
│   ├── observation.yaml
│   ├── action.yaml
│   ├── reward.yaml
│   ├── terrain.yaml
│   ├── sensor.yaml
│   └── training.yaml
├── docker/entrypoint.sh
├── scripts/
│   ├── rsl_rl/train.py
│   ├── rsl_rl/play.py
│   ├── evaluate.py
│   ├── export_onnx.py
│   └── verify_onnx.py
├── source/go2w_terrain_planner/go2w_terrain_planner/
│   ├── mapping/
│   ├── models/
│   ├── robots/
│   ├── tasks/direct/terrain_navigation/
│   └── utils/
└── tests/
    ├── unit/
    └── smoke/
```

核心执行链路为：

```text
解析式地形真值 → 执行、碰撞和失稳结果
解析式地形真值 → 独立观测扰动 → 四通道局部地图
  → 五帧 SE(2) 与垂直参考对齐缓存
  → 共享 CNN
  → GRU
  → 融合局部目标、当前实测速度、指令历史和运动历史
  → Actor 归一化动作
  → 配置化速度映射和执行模型
  → 位姿、碰撞、失稳、卡住和目标进展
  → PPO 奖励与终止
```

## 配置

六份 YAML 是运行时配置，不是示例文件：

- `observation.yaml`：地图范围、原始/网络尺寸、通道和历史长度；
- `action.yaml`：物理速度范围、加速度、跟踪滞后和卡住模型；
- `reward.yaml`：全部奖励权重和终止阈值；
- `terrain.yaml`：地形类型、几何随机范围、摩擦和课程参数；
- `sensor.yaml`：高程噪声、漏测、仅射线观测、位姿和时间同步扰动；
- `training.yaml`：环境数、随机种子、PPO/runner 参数、训练迭代和网络尺寸。

自定义配置目录必须同时包含这六个同名文件：

```bash
train --project-config-dir /workspace/data/my_configs
```

命令行的 `--num_envs`、`--seed` 和 `--max_iterations` 优先于 YAML。

## 本地构建与单元测试

本地不需要启动 Isaac Sim。建议先构建测试阶段：

```bash
cd /home/zezhao/graduation_project/go2w_terrain_planner

docker build \
  --target unit-test \
  -t go2w-terrain-planner:test \
  .
```

正式镜像：

```bash
docker build \
  --target runtime \
  --build-arg GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)" \
  -t go2w-terrain-planner:0.1.0 \
  .
```

如果本机已有 Python 3.10+、NumPy、PyTorch、PyYAML 和 pytest，也可执行：

```bash
PYTHONPATH=source/go2w_terrain_planner \
GO2W_SKIP_TASK_REGISTRATION=1 \
python -m pytest -q tests/unit
```

`GO2W_SKIP_TASK_REGISTRATION=1` 会阻止单元测试注册和加载 Isaac 任务，不会启动 SimulationApp。

## GPU 服务器运行

服务器需要 NVIDIA 驱动、NVIDIA Container Toolkit 和可工作的：

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

新容器与现有 YOLO 容器完全独立，只共享宿主机目录。不要修改 YOLO 容器、镜像或 Python 环境。

首次 GPU 冒烟测试只启动一个环境：

```bash
docker run --rm \
  --gpus all \
  --network host \
  --ipc host \
  -e ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  -v /data/go2w_training:/workspace/data \
  go2w-terrain-planner:0.1.0 \
  smoke --num_envs 1 --steps 100
```

冒烟测试会初始化 Isaac Sim、创建任务、生成有限观测、随机执行、显式 reset，并把结果写到：

```text
/data/go2w_training/smoke/smoke_test.json
```

正式训练：

```bash
docker run --rm \
  --gpus all \
  --network host \
  --ipc host \
  -e ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  -v /data/go2w_training:/workspace/data \
  go2w-terrain-planner:0.1.0 \
  train --num_envs 32 --seed 42 --max_iterations 1500
```

默认先使用 `--num_envs 32`。确认显存和训练吞吐稳定后再增加到 `64` 或 `128`；五帧 `100 × 100`
地图会被 PPO rollout 缓存，环境数对显存近似线性增长。

从某个 checkpoint 恢复：

```bash
docker run --rm --gpus all --network host --ipc host \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -v /data/go2w_training:/workspace/data \
  go2w-terrain-planner:0.1.0 \
  train --resume --checkpoint /workspace/data/runs/rsl_rl/go2w_terrain_navigation/RUN/model_500.pt
```

回放并录制无显示器视频：

```bash
docker run --rm --gpus all --network host --ipc host \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -v /data/go2w_training:/workspace/data \
  go2w-terrain-planner:0.1.0 \
  play --num_envs 4 \
  --checkpoint /workspace/data/runs/rsl_rl/go2w_terrain_navigation/RUN/model_1500.pt \
  --video --video_length 500
```

固定步数评估并保存指标：

```bash
docker run --rm --gpus all --network host --ipc host \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -v /data/go2w_training:/workspace/data \
  go2w-terrain-planner:0.1.0 \
  evaluate --num_envs 16 --steps 2000 \
  --checkpoint /workspace/data/runs/rsl_rl/go2w_terrain_navigation/RUN/model_1500.pt
```

也可以使用 Compose：

```bash
GO2W_DATA_DIR=/data/go2w_training docker compose build planner
GO2W_DATA_DIR=/data/go2w_training docker compose run --rm planner smoke
GO2W_DATA_DIR=/data/go2w_training docker compose run --rm planner train --num_envs 32
```

## ONNX 导出与验证

导出模型的固定张量名为：

```text
input:  policy_observation [B, 200025]
output: velocity_command   [B, 2]
```

执行：

```bash
docker run --rm \
  -v /data/go2w_training:/workspace/data \
  go2w-terrain-planner:0.1.0 \
  export \
  --checkpoint /workspace/data/runs/rsl_rl/go2w_terrain_navigation/RUN/model_1500.pt \
  --output /workspace/data/export/planner.onnx

docker run --rm \
  -v /data/go2w_training:/workspace/data \
  go2w-terrain-planner:0.1.0 \
  verify-onnx --model /workspace/data/export/planner.onnx
```

验证脚本检查输入/输出名称、动态 batch、有限值以及两个物理速度范围。训练和导出必须使用同一套配置，否则网络尺寸或 checkpoint 参数会不一致。

## 数据持久化

所有训练和评估结果都位于挂载的 `/workspace/data`，典型结构是：

```text
/data/go2w_training/
├── runs/rsl_rl/go2w_terrain_navigation/<timestamp>_phase1_proxy/
│   ├── configs/          # 六份配置快照
│   ├── params/           # Isaac Lab/Hydra 实际配置
│   ├── model_*.pt        # RSL-RL checkpoint
│   ├── checkpoints/model_latest.pt
│   ├── videos/
│   ├── metrics/
│   └── metadata.json     # seed、Git、镜像、Isaac、RSL-RL 版本
├── evaluations/<timestamp>/metrics.json
├── smoke/smoke_test.json
└── export/planner.onnx
```

容器删除后这些文件仍保留在宿主机。不要在没有 `-v /data/go2w_training:/workspace/data` 的临时容器中正式训练。

## 镜像迁移

私有仓库方式：

```bash
docker tag go2w-terrain-planner:0.1.0 REGISTRY/go2w-terrain-planner:0.1.0
docker push REGISTRY/go2w-terrain-planner:0.1.0

# 服务器
docker pull REGISTRY/go2w-terrain-planner:0.1.0
```

离线 tar 包方式：

```bash
docker save go2w-terrain-planner:0.1.0 | gzip > go2w-terrain-planner-0.1.0.tar.gz

# 服务器
gzip -dc go2w-terrain-planner-0.1.0.tar.gz | docker load
```

## 下一阶段替换顺序

1. 获取合法的 Go2W USD/URDF 和原厂速度控制接口，在 `robots/go2w_cfg.py` 配置资产。
2. 用完整 Go2W articulation 和底层速度跟踪器替换 `RigidObject` 与 `VelocityExecutionModel`。
3. 保持四通道接口，增加 XT16 Ray Caster、点云投影和五次有效观测融合。
4. 采集实机的指令、实际里程计增量、地形图和通过结果，标定速度滞后、衰减、转向误差和卡住概率。
5. 对比解析地图与 Ray Caster 地图的通道直方图、空洞率、空间相关性和时序漂移，再进行域随机化调整。
6. 增加 ROS 推理节点，复用本仓库的预处理、坐标变换、时序缓存和 ONNX 物理速度输出。

## 常见问题

### `could not select device driver ... gpu`

服务器未安装或未配置 NVIDIA Container Toolkit。先确保 `nvidia-smi` 在宿主机正常，再验证 CUDA 测试容器。

### `permission denied /var/run/docker.sock`

当前用户没有 Docker daemon 权限。使用服务器规定的 Docker 用户组或由管理员执行；不要通过修改现有 YOLO 容器绕过权限。

### 基础镜像拉取失败

确认镜像名和 `2.3.2` 标签，并完成 NGC 登录和 EULA/隐私环境变量设置。不要临时改成 `latest`。

### `CUDA out of memory`

降低 `--num_envs`，其次降低 `configs/observation.yaml` 的 `output_size`。修改地图尺寸后必须重新训练，旧 checkpoint 不能直接加载。

### checkpoint 参数尺寸不匹配

训练、回放、评估、导出使用了不同配置。使用训练目录中的 `configs/` 快照作为 `--project-config-dir`。

### ONNX 输出不是 `[-1,1]`

这是预期行为。ONNX 输出是 `configs/action.yaml` 定义的物理 `[linear.x,angular.z]`，归一化动作只存在于 PPO 环境内部。

### 单元测试加载了 Isaac Sim

确认使用 `unit-test` 镜像阶段，或在本机设置 `GO2W_SKIP_TASK_REGISTRATION=1`。
