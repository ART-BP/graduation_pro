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
cd /home/allgo/mydrive/graduation_pro/local_map_ws
./build.sh
```

工作空间中的 `ROG-Map/examples` 已通过 `CATKIN_IGNORE` 排除，只编译 ROG-Map 核心库和本项目节点。

## 启动完整流水线

```bash
cd /home/allgo/mydrive/graduation_pro/local_map_ws
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
