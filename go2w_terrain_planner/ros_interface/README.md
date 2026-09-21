# Go2W ROS1 C++ 推理接口

该工作区把 `local_map_ws` 的滚动 GridMap、FAST-LIO `/Odometry`、地面参考和
RViz 目标转换为训练时完全相同的 `policy_observation [1,280045]`，再使用
TensorRT 执行 ONNX Actor。ONNX 已包含动作映射，输出直接是物理单位
`[linear.x, angular.z]`。

默认仅发布影子话题：

- `/go2w_terrain_planner/cmd_vel_shadow` (`geometry_msgs/Twist`)
- `/go2w_terrain_planner/cmd_vel_shadow_stamped` (`geometry_msgs/TwistStamped`)
- `/go2w_terrain_planner/diagnostics` (`diagnostic_msgs/DiagnosticArray`)
- `/go2w_terrain_planner/predicted_motion` (`visualization_msgs/Marker`)

节点不会发布机器人的 `/cmd_vel`。rosbag 回放不会受模型输出影响，所以这里只能检查
策略的影子指令、地图输入、推理耗时和行为趋势，不能据此计算闭环导航成功率。

## 1. 导出 ONNX

在包含训练镜像的机器上，从仓库根目录执行：

```bash
mkdir -p data/export
docker run --rm \
  -v "$PWD/data:/workspace/data" \
  go2w-terrain-planner:0.1.0 \
  export \
  --checkpoint /workspace/data/model_1400.pt \
  --output /workspace/data/export/planner.onnx \
  --opset-version 16

docker run --rm \
  -v "$PWD/data:/workspace/data" \
  go2w-terrain-planner:0.1.0 \
  verify-onnx --model /workspace/data/export/planner.onnx
```

TensorRT 8.5 尚不支持 opset 17 的原生 `LayerNormalization`，因此该 Jetson
部署固定使用 opset 16；LayerNorm 会被等价展开为基础算子。训练与导出必须使用
同一套 observation、action 和 model 配置。固定接口为：

```text
policy_observation [B, 280045] -> velocity_command [B, 2]
```

## 2. 编译

```bash
cd /home/allgo/mydrive/graduation_pro/go2w_terrain_planner/ros_interface
chmod +x build.sh start_shadow_inference.sh
./build.sh -j2
```

依赖 ROS Noetic、已编译的 `local_map_ws`、CUDA、TensorRT、
`libnvinfer-dev`、TensorRT plugin runtime 和 `libnvonnxparsers-dev`。第一次启动时
节点会把 ONNX 构建为 `data/export/planner_fp32.engine`；该 engine 只可在兼容的
TensorRT/GPU 环境使用。

当前 `model_1400.pt` 在 TensorRT 8.5 全层 FP16 下会产生非有限输出，因此默认使用
已经通过 rosbag 实测的 FP32 engine。不要把未经验证的 `planner_fp16.engine` 用于控制。
在当前 Orin 上，FP32 单次推理约 5 ms，足以满足 10 Hz 地图输入。

## 3. rosbag 影子回放

终端 1：

```bash
source /opt/ros/noetic/setup.bash
roscore
```

终端 2：

```bash
source /opt/ros/noetic/setup.bash
rosparam set /use_sim_time true
cd /home/allgo/mydrive/graduation_pro/local_map_ws
./start_local_environment.sh fast_lio_rviz:=true
```

终端 3：

```bash
cd /home/allgo/mydrive/graduation_pro/go2w_terrain_planner/ros_interface
./start_shadow_inference.sh
```

终端 4（先暂停，确保所有订阅者启动后再按空格）：

```bash
cd /home/allgo/mydrive/graduation_pro
rosbag play --clock --pause -r 0.5 bag/floorex.bag \
  --topics /Imu /lidar_points
```

在 RViz 中把 Fixed Frame 设为 `odom`，使用 “2D Nav Goal” 在机器人 10 m
范围内设置目标；节点订阅 `/move_base_simple/goal`。添加 Marker 显示
`/go2w_terrain_planner/predicted_motion`，绿色表示安全门控通过，红色表示零速停止。

可记录轻量结果：

```bash
rosbag record --lz4 -O bag/model_1400_shadow.bag \
  /go2w_terrain_planner/cmd_vel_shadow_stamped \
  /go2w_terrain_planner/diagnostics \
  /go2w_terrain_planner/predicted_motion \
  /move_base_simple/goal
```

## 输入对齐

实现固定复现以下训练接口：

- 将 GridMap 循环缓冲和 odom 轴地图恢复为机器人前向/左向地图；
- `ground_height / 1.0`、`height_range / 3.0`，无效高度填 0；
- 12 帧 SE(2) 与地面参考对齐、稳健融合；
- 四通道融合图 + recent change + age + confidence，共 7 通道；
- 目标 `[distance/10, sin(bearing), cos(bearing)]`；
- 当前速度和 8 步命令分别除以 `[1.2, 1.0]`；
- 8 步运动 `[dx/5, dy/5, dyaw/pi]`。

时间倒退（重新播放 bag）时历史会自动清空。地图观测率低于 5%、有效高度率低于
1%、没有目标、到达目标或模型输出异常时，影子指令固定为零。
