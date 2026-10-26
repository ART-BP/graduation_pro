# Go2W C++ 策略部署包

该 ROS1/catkin 包把训练好的 Go2W 高层地形策略部署为纯 C++ 节点。运行时链路为：

```text
普通 RSL-RL .pt checkpoint
  └─（部署前一次性转换）→ planner.onnx
      └─ C++ ONNX Runtime → geometry_msgs/Twist [v_cmd, w_cmd]
```

训练生成的 `.pt` 是 Python checkpoint（包含 `state_dict` 和训练元数据），不是 TorchScript，不能由 C++ 直接加载。ONNX 转换仍使用仓库已有的模型类恢复权重；机器狗端不运行 Python、PyTorch 或 Isaac Lab。

## 功能

- 订阅 `/local_environment/grid_map`、`/Odometry` 和同时间戳的 `/local_environment/ground_reference`；
- 把 odom 轴对齐的滚动地图重采样到机器人前向/左向坐标；
- 复现训练端 12 帧 SE(2) 与高度参考对齐、稳健融合和 7 通道紧凑地图；
- 复现 8 步速度指令/运动历史及固定的 `280045` 维 Actor 输入顺序；
- 通过 ONNX Runtime C++ 执行 `policy_observation -> velocity_command`；
- 检查模型张量名、输入维度、地图覆盖率、输出范围和输入超时；
- 默认不上使能，并在禁用、无目标、地图异常、输入超时、推理异常或到达目标时立即发布零速度。

该节点输出高层 `/cmd_vel`，不直接发送关节或轮电机命令。实机必须把此话题接到已经验证过的 Unitree 高层速度控制桥，并保留急停和原厂安全控制器。

## 1. 从 checkpoint 导出

先构建训练镜像 `go2w-terrain-planner:0.1.0`，然后执行：

```bash
cd /home/zezhao/graduation_project/local_map_ws/src/go2w_policy_deployment
./scripts/export_policy.sh /absolute/path/to/model_best.pt
```

默认生成并验证 `models/planner.onnx`。也可以把第二个参数设为其他输出路径。模型文件属于构建产物，已由 `.gitignore` 排除。

若 checkpoint 对应的是某次运行保存的配置快照，显式传入该目录：

```bash
GO2W_PROJECT_CONFIG_DIR=/absolute/path/to/run/configs \
  ./scripts/export_policy.sh /absolute/path/to/model_best.pt
```

训练与导出必须使用 checkpoint 对应的六份配置。当前包配置固定对应：`7 × 200 × 200` 地图、12 帧地图融合、8 步历史、输入 `policy_observation [B,280045]`、输出 `velocity_command [B,2]`。若训练配置发生变化，必须同步更新 `config/policy.yaml` 并重新导出。

## 2. 安装 ONNX Runtime C++ SDK

在机器狗的目标架构上准备 ONNX Runtime C/C++ SDK，目录至少应包含：

```text
/opt/onnxruntime/
├── include/onnxruntime_cxx_api.h
└── lib/libonnxruntime.so
```

Jetson/ARM64 必须使用与 JetPack、CUDA 和目标架构匹配的构建，不能复制 x86_64 动态库。CPU 版无需 CUDA provider，也能运行本节点。

## 3. 编译和测试

```bash
cd /home/zezhao/graduation_project/local_map_ws
./build.sh -DONNXRUNTIME_ROOT=/opt/onnxruntime
catkin_make run_tests_go2w_policy_deployment
catkin_test_results
```

如果构建机尚未安装 ONNX Runtime，核心观测库及单元测试仍会编译，但 CMake 会明确提示并跳过 `go2w_policy_node`。部署到机器人前必须确认构建日志中已生成该可执行文件。运行时若动态链接器找不到库，可将 `/opt/onnxruntime/lib` 加入系统 linker 配置或 `LD_LIBRARY_PATH`。

## 4. 启动

只启动策略节点：

```bash
source /home/zezhao/graduation_project/local_map_ws/devel/setup.bash
roslaunch go2w_policy_deployment policy.launch \
  model:=/absolute/path/to/planner.onnx
```

同时启动 FAST-LIO、局部地图和策略：

```bash
roslaunch go2w_policy_deployment policy_pipeline.launch \
  model:=/absolute/path/to/planner.onnx
```

发布 odom 坐标系局部目标：

```bash
rostopic pub -1 /local_goal geometry_msgs/PoseStamped \
  "header: {frame_id: odom}
pose: {position: {x: 3.0, y: 0.0, z: 0.0}, orientation: {w: 1.0}}"
```

确认地图、里程计、目标和底层急停均正常后再使能：

```bash
rosservice call /go2w_policy/set_enabled "data: true"
```

禁用会立即输出零速度：

```bash
rosservice call /go2w_policy/set_enabled "data: false"
```

状态可从 `/go2w_policy/status` 和 `/go2w_policy/safety_stop` 查看。不要在未架空轮子、未限制场地、没有人工急停的条件下首次使能。

## 接口约束

| 接口 | 类型 | 说明 |
| --- | --- | --- |
| `/local_environment/grid_map` | `grid_map_msgs/GridMap` | 必须含 `ground_height`、`height_range`、`observed_mask` |
| `/Odometry` | `nav_msgs/Odometry` | 位姿为 odom；`twist.linear.x/angular.z` 是实测机体速度 |
| `/local_environment/ground_reference` | `geometry_msgs/PointStamped` | `point.z` 为该地图的绝对地面参考 |
| `/local_goal` | `geometry_msgs/PoseStamped` | 目标必须与地图处于同一 frame，默认 `odom` |
| `/cmd_vel` | `geometry_msgs/Twist` | 物理单位 m/s、rad/s；可通过 launch 参数改话题 |

地图、里程计和高度参考通过近似时间同步消费。输入超过 0.5 秒未更新时 watchdog 会持续发布零速度。
