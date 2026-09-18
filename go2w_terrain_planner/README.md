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
- 用可配置的 XT16 式 16 线首回波射线生成每帧32000槽点云，再按实机低分位地面与稳健高度跨度规则投影为四通道局部地图；
- 保留解析式稠密观测作为快速消融基线，可由配置切换；
- 仿真直接生成并向网络提供 `200 × 200`、`0.05 m/格` 的局部地图，不再下采样小台阶和窄障碍；
- 用目标引导的紧凑地图特征金字塔、运动历史GRU和特权 Critic 训练 PPO；
- 支持无显示器训练、回放、评估和 ONNX 导出。

当前训练配置包含以下机制：

- 课程等级与地形类型已解耦，从阶段1的平地短距离趋近开始，依次训练随机方向转向、缓坡与起伏、小台阶/沟坎/矮障碍、稀疏大障碍绕行、混合局部规划、双向楼梯、长距离复杂混合场景和Sim2Real域随机化；
- 所有并行环境共享唯一课程前沿，每次采样60%当前阶段、35%已掌握阶段回放和5%未来一阶段预适应；每阶段至少累计3200个前沿回合和960个完整难度回合，且两项EMA成功率均达到0.85才单调解锁下一阶段；
- 当前地图由最近12次经过SE(2)与高度参考对齐的观测稳健融合，因此十次更新以前的有效栅格仍可保留；这些内部观测只用于生成当前融合结果，不再作为五张完整地图写入PPO rollout；
- Actor地图由当前四通道融合地图、最近变化、栅格年龄和观测置信度组成，共7通道；特征金字塔加入机器人坐标编码、多尺度空洞卷积和局部目标引导的空间注意力，在`200×200`输入下保留`50×50`细节层和`25×25`上下文层，输出384维地图特征；
- 最近4步运动增量与速度指令组成`[dx,dy,dyaw,v_cmd,w_cmd]`序列，由128维GRU编码；GRU只处理低维运动历史，不处理高分辨率地图；
- 局部目标接口仍覆盖`0～10 m`，但episode采样范围由每个能力阶段显式定义：从阶段1的`1.0～2.5 m`扩展到阶段9的`5.0～10.0 m`，不再用地形枚举值线性推导目标距离；
- 所有几何原语都在阶段内从易到难连续增长；楼梯回合以50%概率生成上楼或下楼方向，第9阶段才随等级内难度逐步打开摩擦、位姿、测距、丢点、遮挡和执行跟踪噪声；
- 深坑、墙体、立柱、混合障碍和多路径障碍的进展与航向奖励使用安全余量扩张后的矩形障碍及两侧绕行点构造无碰撞引导；最终成功仍仅由真实目标欧氏距离判定，Actor也不会接收该解析真值；44维特权 Critic 同时接收导航势能、地形几何、风险、未知比例、速度跟踪误差、指令历史和运动历史；
- 进展奖励只在刷新本回合历史最近目标距离时结算，无法通过前后振荡重复领取；距离增大仍采用轻惩罚，以允许绕障所需的横移和短时退让；
- Tanh 高斯策略将预变换均值限制为 `±1.5`，标准差通过 sigmoid 平滑约束在 `0.08～0.60`；重参数采样使 squashed entropy 的雅可比项能够正确反传，避免探索噪声在硬截断上限处锁死；
- 归一化动作采用分段零中心映射，动作`0`严格对应物理速度`[0,0]`；动作、目标或地图异常时安全门控直接输出零速度；
- 训练入口检测非有限损失和异常增大的 surrogate loss，触发时立即中止，避免继续保存受污染的 checkpoint；
- PPO 使用固定 `5e-5` 学习率，避免固定版本 RSL-RL 的自适应调度在低 KL 时将学习率放大到不适合本任务的量级；
- PPO每轮每环境收集96步，32环境时形成3072条样本，并使用8个mini-batch和3次学习epoch，以增加包含障碍交互的单轮轨迹长度并控制扩大模型后的梯度方差；固定学习率模式下不依赖实际不会生效的`desired_kl`；
- `model_best.pt`优先比较“能力阶段+阶段内难度”的连续课程进度，同进度下再比较前沿成功率EMA；
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
`SimulatedLocalMap.lidar_sensor.last_pointcloud`，默认形状为 `[B,32000,3]`，坐标系为当前机器人局部坐标系，
无回波射线填充为 `NaN`；`last_hit_mask` 提供有效回波掩码。地形执行、碰撞和
失稳判定仍使用独立的解析真值，Actor 只能看到由射线穿越与命中点投影得到的地图，避免特权信息泄漏。
为保持32个并行环境的吞吐，地形求交先在2度水平锚点上进行，只在相邻量程连续的扇区内插值到每线2000
个水平采样槽，几何突变边界采用最近有效锚点。独立的纯PyTorch投影器随后执行12 m输入裁剪、0.05 m栅格分组、10%
地面分位数和5%～95%高度跨度估计。它不链接ROS、PCL、Grid Map或`local_map_ws`，但保持主要观测语义一致。

内部融合缓存保存每帧相对地图所使用的绝对 `ground_reference_z`。观测完成 SE(2) 对齐后，先用
`source_reference_z - current_reference_z` 统一垂直零点，再送入网络。该标量不是 Actor 地图通道，由
`local_map_ws` 与 Grid Map 同时间戳发布。

所有无效高度都被固定值替换，进入网络的张量不允许包含 `NaN` 或 `Inf`。地面高程按`±1 m`归一化，高度差保持`0～3 m`覆盖范围；网络直接使用`200 × 200`地图。

实机部署适配器应直接输出机器人中心坐标系下的四通道地图，再进入 `TemporalGridBuffer`；仿真端和实机端都约定地图第一维为机器人前向、第二维为机器人左向。

Actor 输入为：

```text
compact_map   [B, 7, 200, 200]
  channel 0-3 当前融合地图：ground, height_range, observed, height_valid
  channel 4   recent_change：相邻融合地图的几何与掩码变化强度
  channel 5   cell_age：最近观测距当前帧的归一化年龄，0最新、1最旧/未知
  channel 6   confidence：12次融合窗口内按时间加权的观测置信度
goal          [B, 3]       # normalized_distance, sin(angle), cos(angle)
current_speed [B, 2]       # 当前实测 v,w除以物理速度尺度
cmd_history   [B, 4, 2]    # 历史指令除以相同物理速度尺度
motion_history[B, 4, 3]    # dx,dy除以5 m，dyaw除以pi
```

Actor辅助输入在拼接前使用固定物理尺度归一化，不依赖训练数据的运行均值，因此训练、评估和实机部署具有相同语义。默认 Actor 展平维度由 `800025` 降为 `280025`，PPO中每步地图观测存储量减少约65%。12帧融合历史只存在于环境表示模块内部，不随96步rollout重复存储。Critic 使用独立的44维特权状态，这些仿真真值不会进入 Actor 或 ONNX 模型。
其中目标距离以 `10 m` 归一化并截断到 `[0,1]`。机器人接近目标时能够自然观测到
`0～1.5 m` 区间；`reset_minimum_distance_m=1.5` 只约束episode初始采样，不限制运行时接口。

PPO 使用带雅可比修正的 Tanh-squashed Gaussian，采样动作和概率计算都严格位于 `[-1,1]`。环境依据
`configs/action.yaml` 进行分段零中心映射：负半轴映射到最小速度，正半轴映射到最大速度，因此归一化动作
`[0,0]`严格对应物理速度`[0,0]`。导出的 ONNX 已包含这一步映射，直接输出物理单位的`[v_cmd,w_cmd]`。

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
解析式地形真值 → XT16式首回波锚点 → 32000槽点云
  → 12 m裁剪与分位数高程投影 + 射线穿越掩码 → 四通道局部地图
  → 12次观测的SE(2)与垂直参考对齐融合
  → 当前融合地图 + 最近变化 + 栅格年龄/置信度
  → 残差多尺度二维特征金字塔
  → 局部目标引导的空间注意力
  → GRU编码四步运动/指令历史
  → 融合地图、局部目标、当前实测速度和GRU特征
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
- `sensor.yaml`：`raycast/analytic`观测源、每帧点数、16线扫描角、量程、点云裁剪、分位数投影，以及回波漏测、位姿和时间同步扰动；
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

当前训练run为 `phase12_nine_stage_curriculum`。策略观测和网络仍是版本6，但课程checkpoint状态已升级为九阶段能力语义，不能对phase11及以前运行使用`--resume`；正式第二版训练建议从头建立新run。

默认先使用 `--num_envs 32`。确认显存和训练吞吐稳定后再增加到 `64` 或 `128`；PPO只缓存单张7通道`200 × 200`紧凑地图，环境数对显存仍近似线性增长。

从某个 checkpoint 恢复：

```bash
docker run --rm --gpus all --network host --ipc host \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -v /data/go2w_training:/workspace/data \
  go2w-terrain-planner:0.1.0 \
  train --resume --checkpoint /workspace/data/runs/rsl_rl/go2w_terrain_navigation/RUN/model_500.pt
```

`--resume`只应用于本版本同配置产生的checkpoint；严格续训会同时恢复模型、优化器、迭代计数以及全局能力阶段、阶段内难度、成功率EMA和计数。`--finetune`只恢复策略权重：

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

评估和回放默认均匀覆盖 `--curriculum-min-stage 1 --curriculum-max-stage 9`。例如使用`--curriculum-min-stage 5 --curriculum-max-stage 5`可只评估稀疏大障碍绕行阶段。评估结果同时记录每个能力阶段和每类地形的步数、完成回合数与成功率。

短时间训练压力测试也可以绕过课程采样：

```bash
docker run --rm --gpus all --network host --ipc host \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -v /data/go2w_training:/workspace/data \
  go2w-terrain-planner:0.1.0 \
  train --num_envs 32 --max_iterations 10 \
  --curriculum-min-stage 5 --curriculum-max-stage 5
```

不提供这两个参数时，正式训练使用`configs/terrain.yaml`中的前沿—回放—挑战课程。阶段定义、地形混合权重、目标距离与朝向范围均可在`curriculum.stages`中直接查看和修改。

也可以使用 Compose：

```bash
GO2W_DATA_DIR=/data/go2w_training docker compose build planner
GO2W_DATA_DIR=/data/go2w_training docker compose run --rm planner smoke
GO2W_DATA_DIR=/data/go2w_training docker compose run --rm planner train --num_envs 32
```

## ONNX 导出与验证

导出模型的固定张量名为：

```text
input:  policy_observation [B, 280025]
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
├── runs/rsl_rl/go2w_terrain_navigation/<timestamp>_phase12_nine_stage_curriculum/
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
