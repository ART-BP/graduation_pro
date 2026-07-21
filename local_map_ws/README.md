# Go2W 局部高程环境表示（第一版）

本工作区与 lio_ws 平级。当前版本不使用 ROG-Map，也不构建三维占据体素，直接把 FAST-LIO2 运动补偿后的注册点云投影为机器人周围的局部二维高程栅格。

数据链路：

    XT16 /lidar_points + IMU /Imu
                    ↓
    FAST-LIO2 /cloud_registered + /Odometry
                    ↓
    时间同步、15 m × 15 m 点云裁剪
    10 m × 10 m 滚动地图增量更新
                    ↓
    /local_environment/grid_map

## 输出通道

Grid Map 只包含三个固定图层：

| 图层 | 含义 |
| --- | --- |
| ground_height | 相对机器人当前脚下参考地面的高程，平地约为 0，单位 m |
| height_range | 最近有效观测窗口内最大高度减最小高度，单位 m |
| observed_mask | 单元有未过期的回波或被激光束穿过时为 1，否则为 0 |

未观测单元的 ground_height 和 height_range 保持 NaN。第一版不发布坡度、粗糙度、地面法向量、净空或阻塞图层。

滚动地图内部使用 odom 绝对高程完成增量融合，发布前才转换为相对高程。脚下地面参考优先由机器人周围的历史地面栅格中值确定，观测不足时回退到 FAST-LIO IMU 位姿和标定的 IMU 离地高度。请在 `config/projector.yaml` 中校准 `height_reference/imu_to_ground_height`。

节点同时发布 `/local_environment/ground_reference`（`geometry_msgs/PointStamped`）；其时间戳与 Grid Map
一致，`point.z` 为该帧相对高程所使用的 odom 系绝对地面参考值。

回退参考考虑当前 IMU 姿态：`z_ref = z_imu + [R_odom_body * (0, 0, -h_imu_ground)]_z`，最终发布值为 `ground_height = z_cell_odom - z_ref`。因此不会把 IMU 腹部原点误当成地面零点。

## 算法流程

1. 使用消息时间戳近似同步 /cloud_registered 和 /Odometry。FAST-LIO2 当前会为二者写入相同的扫描结束时间。
2. 每帧先按机器人中心裁剪 15 m × 15 m 的水平点云，并保留机器人高度 [-1.0 m, 1.5 m] 内的点。
3. 持久维护 10 m × 10 m、0.20 m 分辨率的滚动 Grid Map；机器人跨过一个栅格时移动循环缓存，只清空新进入地图的边缘。
4. 根据 FAST-LIO 位姿和 XT16 外参计算雷达原点，对雷达原点到回波点执行二维栅格射线遍历；射线穿过的单元刷新 observed_mask。
5. 按 XY 坐标将当前点云终点写入地图覆盖范围内的单元，并更新地面高程与高度差。
6. 每个终点栅格最多融合最近 5 次有效观测；地面高度对逐帧低分位估计取均值，高度差使用窗口内总最大值减总最小值。
7. 未被当前帧命中或穿过的栅格继续保留；超过 1 秒没有任何观测后清空并将 observed_mask 置为 0。

地图中心随机器人平移，坐标轴和高度仍使用 odom 世界坐标系。这里的 5 次观测用于地图内部的增量稳定；后续学习型局部规划器仍应按地图消息时间戳缓存最近 3 至 5 张完整高程图。

## 编译

先编译 lio_ws，再执行：

    cd /home/zezhao/graduation_project/local_map_ws
    ./build.sh

本工作区只依赖 ROS Noetic、PCL、message_filters 和随源码保留的最小 Grid Map 包，不再依赖或编译 ROG-Map。

## 启动

同时启动 FAST-LIO2 和局部高程节点：

    cd /home/zezhao/graduation_project/local_map_ws
    ./start_local_environment.sh

默认同时启动 RViz。本项目的 RViz 配置使用 ground_height 作为显示高度、height_range 作为颜色，能够直接观察滚动高程图。关闭显示：

    ./start_local_environment.sh fast_lio_rviz:=false

如果 FAST-LIO2 已经在其他终端运行：

    ./start_local_environment.sh start_fast_lio:=false

可覆盖的话题参数：

    ./start_local_environment.sh \
      cloud_topic:=/cloud_registered \
      odometry_topic:=/Odometry

主要参数位于 src/go2w_local_environment/config/projector.yaml。

## 第一版边界

- 只有射线穿过、没有回波终点的单元会标记为已观测，但 ground_height 和 height_range 仍为 NaN；射线只能证明束线位置自由，不能凭空确定地面高度。
- 当前使用二维射线投影更新观测掩码，不构建完整的三维自由空间占据状态。
- height_range 同时响应障碍物、台阶、坑洼边缘和地面起伏，但第一版不对这些语义做手工分类。
- ground_percentile 和 minimum_points_per_cell 需要根据 XT16 实测点密度继续标定；默认最小点数为 1，以避免 0.20 m 栅格下远处区域过于稀疏。
- 15 m × 15 m 是输入点云裁剪范围；真正写入和发布的高程图始终为内部 10 m × 10 m 区域。
- 学习规划器使用的历史帧堆叠、局部目标和当前速度不在本环境表示节点中实现。
