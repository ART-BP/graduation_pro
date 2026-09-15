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
- 用可配置的 XT16 式 16 线首回波射线生成稀疏点云，再投影为与实机接口等价的四通道局部地图；
- 保留解析式稠密观测作为快速消融基线，可由配置切换；
- 仿真先生成 `200 × 200` 原始地图，再使用共享预处理逻辑保守下采样到 `100 × 100`；
- 用 CNN + GRU Actor 和特权 Critic 训练 PPO；
- 支持无显示器训练、回放、评估和 ONNX 导出。

第二版训练配置在不改变 Actor 四通道观测和网络接口的前提下，增加了以下机制：

- 课程从`step(2)`启动，采用50%当前前沿、45%低等级回放和5%未来一级挑战；每级先用至少100个前沿回合完成难度爬升，再要求至少30个难度不低于0.9的完整难度回合且专项EMA成功率达到0.85，才允许单调解锁下一级；
- 地形难度与目标距离采用两个进度状态：当前前沿负责选择正在挑战的地形，已掌握等级负责目标距离上限，因此进入新地形时不会同时增加目标距离，且目标上限不会回退；
- 每张网络地图快照融合最近 12 次经过 SE(2) 和高度参考对齐的观测，使十次更新以前的有效栅格仍可保留；Actor 仍只接收最近 5 张滚动地图快照；
- 地图 CNN 通道扩大为 `24/48/96/128`，末端空间结构从 `4×4` 提高到 `6×6`，CNN 与 GRU 特征提高到 192 维，融合层和 Critic 提高到 256 维；
- 局部目标距离接口覆盖 `0～10 m`，episode初始目标仍至少为`1.5 m`；采样上限依据已掌握等级从`4 m`线性扩展到`10 m`，比当前地形前沿落后一级，避免同时增加地形与距离难度；
- `pit(5)`采用浅窄坑到完整坑洼的等级内连续课程，`wall(6)`也从较窄绕行宽度逐步扩展；容易几何回合不计入完整难度掌握统计，目标保证位于坑远端安全余量之外，侧向绕坑不再受到错误的进入角失稳惩罚；
- 等级6墙体回合的进展奖励使用有限墙端点构造的最短绕行势能，而最终成功仍由真实欧氏距离判定；Critic在不增加维度的情况下接收该剩余路径势能和归一化墙宽，避免直线距离奖励诱导策略撞墙；
- 进展奖励只在刷新本回合历史最近目标距离时结算，无法通过前后振荡重复领取；距离增大仍采用轻惩罚，以允许绕障所需的横移和短时退让；
- Tanh 高斯策略将预变换均值限制为 `±1.5`，标准差通过 sigmoid 平滑约束在 `0.08～0.60`；重参数采样使 squashed entropy 的雅可比项能够正确反传，避免探索噪声在硬截断上限处锁死；
- 训练入口检测非有限损失和异常增大的 surrogate loss，触发时立即中止，避免继续保存受污染的 checkpoint；
- PPO 使用固定 `5e-5` 学习率，避免固定版本 RSL-RL 的自适应调度在低 KL 时将学习率放大到不适合本任务的量级；
- PPO每轮使用48步环境数据、3次学习epoch和更低的Critic损失权重，以降低扩大模型后的梯度方差；固定学习率模式下不依赖实际不会生效的`desired_kl`；
- `model_best.pt`优先比较“地形等级+等级内难度”的连续课程进度，同进度下再比较前沿成功率EMA，避免低等级高成功率模型覆盖已进入高难地形的模型；
- 墙体、混合障碍和多路径障碍随机化有限横向宽度，确保地图内存在可学习的绕行通道；
- 评估结果除总体成功率外，输出每类地形的完成回合数与成功率。

当前版本没有声称完成以下两项：

- Go2W 完整关节、轮腿接触和原厂底层控制器的物理模型；
- 基于 RTX 材质、强度和厂商精确标定参数的高保真 XT16 仿真；当前版本采用适配动态解析地形的 GPU 批量首回波射线模型，垂直角可由实测标定表替换。

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

仿真端默认使用 `sensor.observation_source: raycast`。射线传感器输出保存在
`SimulatedLocalMap.lidar_sensor.last_pointcloud`，形状为 `[B,N,3]`，坐标系为当前机器人局部坐标系，
无回波射线填充为 `NaN`；`last_ranges` 和 `last_hit_mask` 分别提供量程与有效回波掩码。地形执行、碰撞和
失稳判定仍使用独立的解析真值，Actor 只能看到由射线穿越与命中点投影得到的地图，避免特权信息泄漏。

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
其中目标距离以 `10 m` 归一化并截断到 `[0,1]`。机器人接近目标时能够自然观测到
`0～1.5 m` 区间；`reset_minimum_distance_m=1.5` 只约束episode初始采样，不限制运行时接口。

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
解析式地形真值 → XT16式首回波射线 → 稀疏点云
  → 射线穿越/命中栅格投影 → 四通道局部地图
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
- `sensor.yaml`：`raycast/analytic`观测源、16线扫描角、量程、角分辨率、扫描频率，以及回波漏测、位姿和时间同步扰动；
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
  train --num_envs 32 --seed 42 --max_iterations 10000
```

第四版长融合大模型基线必须从头建立新run。地图编码器、GRU、融合层和Critic的参数形状已经改变，第三版及更早checkpoint不能通过`--resume`或`--finetune`加载；训练入口会在启动时明确拒绝不兼容网络。第五版只修改课程、代理地形、奖励与PPO采样配置，没有改变网络形状。第六版接入稀疏激光雷达观测后网络形状仍兼容，但输入分布已显著变化，建议从头训练；如需迁移第五版权重，应使用`--finetune`建立新run，不建议恢复旧优化器和课程状态。

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

`--resume`只应用于观测源和网络结构均相同的checkpoint；严格续训会同时恢复模型、优化器、迭代计数以及每个并行环境的课程等级/成功率状态。`--finetune`只复用兼容且有限值正常的模型权重，并以新优化器和新课程开始独立微调，适合从第五版解析观测迁移到第六版射线观测：

```bash
docker run --rm --gpus all --network host --ipc host \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -v /data/go2w_training:/workspace/data \
  go2w-terrain-planner:0.1.0 \
  train --finetune --checkpoint /workspace/data/runs/rsl_rl/go2w_terrain_navigation/RUN/model_500.pt
```

训练入口会在载入和保存时检查模型张量；严格续训还会检查优化器状态与动作标准差参数化版本。包含 NaN/Inf 的 checkpoint 会被拒绝。训练过程中还会维护 `model_best.pt`，但部署前仍应通过独立的确定性评估比较该模型与周期 checkpoint。

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

评估和回放默认使用 `--terrain-min-level 2 --terrain-max-level 9`，因此跳过
`flat(0)` 和 `ramp(1)`，直接从台阶及更高难度地形中均匀采样。可使用
`--terrain-min-level 6` 只测试墙体、立柱、混合地形和多路径地形。评估结果会记录每类地形的实际步数、完成 episode 数量和成功率。

短时间训练压力测试也可以绕过课程采样：

```bash
docker run --rm --gpus all --network host --ipc host \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -v /data/go2w_training:/workspace/data \
  go2w-terrain-planner:0.1.0 \
  train --num_envs 32 --max_iterations 10 \
  --terrain-min-level 2 --terrain-max-level 9
```

不提供这两个参数时，正式训练使用`configs/terrain.yaml`中的前沿—回放—挑战三级课程。挑战样本从当前等级的下一级采样，并以最易几何提前适应；当前前沿的几何难度随学习状态连续增长，已掌握回放使用完整几何范围。地形索引依次为：
`flat=0, ramp=1, step=2, stairs=3, rough=4, pit=5, wall=6, pillar=7, mixed=8, multi_route=9`。

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
├── runs/rsl_rl/go2w_terrain_navigation/<timestamp>_phase6_xt16_raycast_observation/
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
3. 用实测 XT16 标定角与噪声统计校准当前 Ray Caster，并对射线点云投影和多次有效观测融合开展仿真—实机一致性评估。
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
