# Go2W 局部环境表示工作区（第一版）

该工作区与 `lio_ws` 平级，负责把机器人周围的三维占据体素投影为机器人中心的双层高度—净空地图。

当前第一版的数据链路为：

```text
FAST-LIO2 /cloud_registered + /Odometry
                 │
                 ▼
             ROG-Map
      occupied / unknown voxel clouds
                 │
                 ▼
    go2w_local_environment
                 │
                 ├── /local_environment/grid_map
                 └── /local_environment/blocked
```

## 已实现

- 机器人中心、世界坐标轴对齐的滚动局部 Grid Map。
- 按垂直体素列提取地面高度和最低顶部障碍高度。
- 计算已知垂直净空、未知比例、阻塞状态和观测状态。
- 分别接收 ROG-Map 的占据体素点云和未知体素点云。
- 未接入 ROG-Map 时，可把 FAST-LIO2 注册点云临时作为占据点输入，用于投影链路调试。
- 投影核心与 ROS 节点分离，并带有基础单元测试。

## Grid Map 图层

| 图层 | 含义 |
| --- | --- |
| `ground_height` | 地面占据体素上表面的世界坐标高度，单位 m |
| `ceiling_height` | 地面上方最低障碍体素的下表面高度，单位 m；未发现时为 NaN |
| `clearance` | 地面到最低顶部障碍或搜索上界的垂直距离，单位 m |
| `blocked` | 已观测单元是否因净空不足而阻塞，0/1；无有效地面时为 NaN |
| `observed` | 地面有效且机器人高度范围内未知比例满足阈值，0/1 |
| `ground_observed` | 是否提取到地面，0/1 |
| `ceiling_observed` | 是否发现地面上方障碍，0/1 |
| `unknown_fraction` | 机器人所需高度范围内未知体素比例，范围 0~1；无未知点云时为 NaN |
| `occupied_count` | 当前二维单元中的占据体素数量 |

## 依赖

运行环境为 ROS Noetic。构建所需的 Grid Map 1.6.4 最小源码包已经放在 `src/vendor`，因此当前机器不需要 sudo 安装即可编译。

如果希望使用 launch 文件中的可选 Grid Map 可视化节点，还需要系统提供 `grid_map_visualization`；默认 `visualize:=false` 不依赖它。

ROG-Map 需要单独放入本工作区或其他已 source 的 catkin 工作区。官方 ROS 1 仓库为：

```text
https://github.com/hku-mars/ROG-Map
```

## 编译

```bash
cd /home/zezhao/graduation_project/local_map_ws
./build.sh
```

## 运行投影节点

先启动 FAST-LIO2 和 ROG-Map，再运行：

```bash
cd /home/zezhao/graduation_project/local_map_ws
./start_local_environment.sh
```

默认输入话题为：

- `/Odometry`
- `/rog_map_node/rog_map/occ`
- `/rog_map_node/rog_map/unk`

如果 ROG-Map 节点名称不同，可在启动时覆盖：

```bash
roslaunch go2w_local_environment local_environment.launch \
  occupied_cloud_topic:=/实际的占据点云话题 \
  unknown_cloud_topic:=/实际的未知点云话题
```

仅调试投影链路时可以直接使用 FAST-LIO2 点云：

```bash
roslaunch go2w_local_environment local_environment.launch \
  occupied_cloud_topic:=/cloud_registered \
  use_unknown_cloud:=false
```

这种模式没有射线更新产生的自由/未知体素，只用于验证高度投影，不等价于完整局部占据地图。

## ROG-Map 配置

`config/rog_map_go2w.yaml` 已按当前 FAST-LIO2 话题和 `odom` 坐标系配置，并开启未知体素可视化发布。该配置需要加载到名为 `rog_map_node` 的 ROG-Map 节点私有命名空间。

## 第一版边界

- 当前采用逐二维单元的垂直列投影，没有加入邻域地面连续性、坡度、粗糙度和时序滤波。
- 地面候选由搜索范围内最接近标称机体—地面高度差的占据体素确定；墙面和高障碍会通过极小净空被标为阻塞，但其地面高度本身不应用于规划。
- Grid Map 保持世界坐标轴方向，地图中心随机器人平移，不随机器人偏航旋转。
- 输入新鲜度使用消息到达本节点的时间判断，因此可以直接处理保留原始时间戳的 rosbag 回放数据。
- 后续应在实测 bag 上标定地面搜索范围、最小净空和未知比例阈值，再增加地形几何特征层。
