# Go2W 局部环境表示

当前工作空间已经形成一条完整 ROS1 流水线：

```text
XT16 /lidar_points + IMU /Imu
              ↓
FAST-LIO2 /cloud_registered + /Odometry
              ↓
ROG-Map /rog_map_node/rog_map/{occ,unk}
              ↓
双层高度—净空投影
              ↓
/local_environment/grid_map + /local_environment/blocked
```

所有中间地图和最终地图统一使用 `odom` 坐标系。ROG-Map 同时发布占据体素和未知体素，使投影模块能够区分已观测自由空间与未知空间。

## 编译

先确保 `lio_ws` 已经编译，然后执行：

```bash
cd /path/to/graduation_project/local_map_ws
./build.sh
```

工作空间仅保留运行所需的 `ROG-Map/rog_map` 核心库，只编译 ROG-Map 核心库和本项目节点。

## 启动完整流水线

```bash
cd /path/to/graduation_project/local_map_ws
./start_local_environment.sh
```

默认同时启动：

- `/laserMapping`：FAST-LIO2；
- `/rog_map_node`：机器人中心局部三维占据地图；
- `/local_environment`：双层高度—净空投影。

需要显示 FAST-LIO 原有 RViz 配置时：

```bash
./start_local_environment.sh fast_lio_rviz:=true
```

如果 FAST-LIO 已经在其他终端运行，只启动 ROG-Map 和投影模块：

```bash
./start_local_environment.sh start_fast_lio:=false
```

可覆盖的主要启动参数包括：

- `lidar_input_topic`，默认 `/lidar_points`；
- `cloud_topic`，默认 `/cloud_registered`；
- `odometry_topic`，默认 `/Odometry`；
- `rog_map_config` 和 `projector_config`；
- `visualize_grid_map`，系统安装 `grid_map_visualization` 后可设为 `true`。

## 输出

`/local_environment/grid_map` 为 `grid_map_msgs/GridMap`，包含以下固定二维通道：

- `ground_height`：地面高度；
- `ceiling_height`：顶部障碍高度；
- `clearance`：垂直净空；
- `blocked`：净空或未知约束形成的阻塞标记；
- `observed`、`ground_observed`、`ceiling_observed`：观测有效性；
- `unknown_fraction`：机器人所需净空范围内的未知体素比例；
- `occupied_count`：当前二维栅格柱内的占据体素数量。

`/local_environment/blocked` 是便于传统 ROS 工具显示的 `nav_msgs/OccupancyGrid` 阻塞图。

主要参数位于：

- `config/rog_map_go2w.yaml`：三维占据地图范围、分辨率、射线更新和可视化输出；
- `config/projector.yaml`：二维地图尺寸、地面搜索范围、Go2W 最小净空等投影参数。

`minimum_clearance` 和 `nominal_ground_offset` 当前为初始值，实机测试时需要根据 Go2W 姿态和可通过高度继续标定。

## ROG-Map 体素更新 V1.1

本项目在上游 ROG-Map 的概率地图和零拷贝滚动结构上，增加了以下观测更新约束：

- `/cloud_registered` 按消息时间戳匹配 `/Odometry`，必要时在相邻位姿间插值；
- 射线起点使用 LiDAR 在 `odom` 中的真实位置，而不是直接使用机体原点；
- 丢弃非有限点，并支持在 `base_link` 下配置机器人本体过滤包围盒；
- 相同终点体素只执行一条射线，同一批次内每个体素的 hit/miss 证据均可限幅；
- 当前概率参数要求三次独立 miss 才把初始未知体素确认为自由，并可在三帧自由观测后清除单次命中的动态障碍。

相关参数位于 `config/rog_map_go2w.yaml` 的 `ros_callback`、`self_filter` 和
`raycasting` 段。其中 `sensor_origin_in_body` 必须与 FAST-LIO 的
`extrinsic_T` 保持一致；`self_filter` 在实机测量 Go2W 相对 `base_link`
的包围盒之前保持关闭。

## ROG-Map 实时性优化 V1.2

- 局部概率地图缩小为 `10 m × 10 m × 5 m`，最大射线距离为 `12 m`；
- 输入点类型精简为 ROG 实际使用的 `XYZI`，最新点云通过只读共享指针交给更新线程，不再进行第二次深拷贝；
- `point_filt_num` 设为 `2`，在多帧概率融合条件下限制单帧射线数量；
- 射线端点和更新候选使用可复用的连续缓存，避免逐帧队列分配；
- 占据与未知体素在一次遍历中同时提取，并使用相同时间戳发布；
- `PointCloud2` 直接写入，消息转换和发布移到地图锁之外，减少点云到达时的等待。
