# go2w_local_environment

该 ROS1 包把 FAST-LIO2 注册点云直接增量更新到滚动局部高程 Grid Map，不依赖 ROG-Map。

输入：

- /cloud_registered：已完成运动补偿并注册到 odom 的 XT16 点云；
- /Odometry：与点云同一时间戳的 FAST-LIO2 位姿。

输出 `/local_environment/grid_map`。为保持第一版学习规划器的输入兼容，原有三层名称与含义继续保留：

- ground_height：相对于机器人脚下参考地面的高程；平地约为 0；
- height_range：栅格内稳健上下分位点之间的垂向跨度，用于表征台阶、障碍物及地面突变；
- observed_mask：存在有效回波或有激光束穿过时为 1，否则为 0。

同时增加四个状态层，以消除“射线穿过、真实测高、插值补全和可用高度”之间的歧义：

- ray_observed_mask：最近有效期内有二维激光射线穿过；
- height_measured_mask：栅格内保留有真实回波历史，包括尚未确认的单点观测；
- height_inferred_mask：当前发布高度来自边缘约束的孔洞补全；
- height_valid_mask：当前 ground_height 与 height_range 均为有限值，可作为高度输入。

典型状态为：纯射线自由空间仅 `ray_observed_mask=1`、高度仍为 NaN；直接测高单元
`height_measured_mask=1` 且在通过可靠性判定后 `height_valid_mask=1`；插值单元
`height_inferred_mask=1` 且 `height_valid_mask=1`。旧规划器仍可只读取前三层，新模块建议直接使用
`height_valid_mask` 判断高度有效性。

同步输出 `/local_environment/ground_reference`（`geometry_msgs/PointStamped`）。消息时间戳与 Grid Map
一致，`point.z` 是本帧使用的 odom 系绝对脚下地面参考高度，`point.x/y` 是对应机器人位置。学习型规划器
使用该标量统一历史相对高程图的垂直零点。

每帧输入先裁剪为机器人中心 15 m × 15 m，输出地图固定为 10 m × 10 m。雷达原点到回波点之间执行二维栅格射线遍历，穿越单元更新观测掩码，终点单元增量更新高度。地图滚动时保留重叠区域并清空新边缘，地图坐标轴保持 odom 方向。

单帧内先去除非有限点，再用低分位数估计地面，并用上下分位数计算稳健垂向跨度，避免孤立离群点直接放大 `height_range`。每个终点栅格保留最近 5 帧，选择地面高度最一致的稠密观测簇进行融合。单帧至少含两个点时可直接形成有效高度；只有一个点时先作为暂定测量，需另一帧在 0.12 m 高度容差内重复观测后才发布，降低 0.05 m 小栅格中稀疏回波造成的跳变。

高度历史和射线状态采用独立有效期：普通地形实测高度默认保留 10 s，射线穿越状态保留 1 s；机器人周围 0.8 m 内的普通地形历史不因超时清除，用于覆盖雷达近场盲区。高垂向跨度单元可能来自移动障碍，使用 2 s 的较短有效期且不受近场保护。若同一地面高度连续两次得到低跨度观测，则主动移除旧的高跨度历史，减轻动态物体离开后的残影。二维射线穿越本身不作为清除高度的依据，避免把被遮挡地形误判为空地。

过期检查每帧只处理 8000 格，约 0.5 s 完成一次全图轮询。高程端点仍按 0.05 m 落格，射线穿越独立按 0.10 m 去重，同一帧每个穿越格只写一次。

发布前对半径 2 格内、至少具有 3 个有效邻居且邻域高差不超过 0.08 m 的孤立孔洞进行中值补全。补全必须在至少一个方向上得到两侧支撑，遇到台阶或障碍边缘时保持未知。补全由后台线程默认以 2 Hz 计算，主回调逐帧复用最近完成且与当前滚动地图版本一致的结果。补全值不写回真实测量历史；它只设置 `height_inferred_mask` 与 `height_valid_mask`，`observed_mask` 仍只表示真实回波或射线穿越。

点云投影使用按栅格索引排序的连续样本缓存，避免为 200 × 200 个栅格逐帧创建独立容器。

内部历史始终保存 odom 坐标系下的绝对高程，发布前才统一减去当前脚下地面参考高度，避免不同时刻的参考值污染多帧融合。参考高度优先取机器人周围低高度差栅格的中值；数据不足时使用经过当前姿态旋转的 `IMU -> 地面` 标称偏移。`height_reference/imu_to_ground_height` 必须根据实机正常站立状态标定。

RViz 配置位于 rviz/local_elevation.rviz，参数见 config/projector.yaml。

依赖方面仅在 `src/vendor` 保留 `grid_map_core` 与 `grid_map_msgs`。消息序列化已在本包内以轻量转换器重写，不再依赖 `grid_map_ros`、`grid_map_cv` 或 OpenCV；发布的消息仍是标准 `grid_map_msgs/GridMap`，因此现有 RViz GridMap 插件接口不变。
