/**
* This file is part of ROG-Map
*
* Copyright 2024 Yunfan REN, MaRS Lab, University of Hong Kong, <mars.hku.hk>
* Developed by Yunfan REN <renyf at connect dot hku dot hk>
* for more information see <https://github.com/hku-mars/ROG-Map>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* ROG-Map is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ROG-Map is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with ROG-Map. If not, see <http://www.gnu.org/licenses/>.
*/

#include "rog_map/rog_map.h"

#include <algorithm>
#include <chrono>
#include <cmath>

using namespace rog_map;

ROGMap::ROGMap(const ros::NodeHandle& nh) :nh_(nh) {

    cfg_ = rog_map::Config(nh);
    initProbMap();

    map_info_log_file_.open(DEBUG_FILE_DIR("rm_info_log.csv"), std::ios::out | std::ios::trunc);
    time_log_file_.open(DEBUG_FILE_DIR("rm_performance_log.csv"), std::ios::out | std::ios::trunc);

    vm_.vizcfg.use_body_center = true;
    vm_.vizcfg.box_min = -cfg_.visualization_range / 2;
    vm_.vizcfg.box_max = cfg_.visualization_range / 2;

    vm_.vizcfg.callback_func = boost::bind(&ROGMap::VizCfgCallback, this, _1, _2);
    vm_.vizcfg.vizcfgserver.setCallback(vm_.vizcfg.callback_func);

    robot_state_.p = cfg_.fix_map_origin;

    if (cfg_.map_sliding_en) {
        mapSliding(Vec3f(0, 0, 0));
        inf_map_->mapSliding(Vec3f(0, 0, 0));
    }
    else {
        /// if disable map sliding, fix map origin to (0,0,0)
        /// update the local map bound as
        local_map_bound_min_d_ = -cfg_.half_map_size_d + cfg_.fix_map_origin;
        local_map_bound_max_d_ = cfg_.half_map_size_d + cfg_.fix_map_origin;
        mapSliding(cfg_.fix_map_origin);
        inf_map_->mapSliding(cfg_.fix_map_origin);
        vm_.vizcfg.box_min += cfg_.fix_map_origin;
        vm_.vizcfg.box_max += cfg_.fix_map_origin;
    }

    /// Initialize visualization module
    if (cfg_.visualization_en) {
        vm_.occ_pub = nh_.advertise<sensor_msgs::PointCloud2>("rog_map/occ", 1);
        vm_.unknown_pub = nh_.advertise<sensor_msgs::PointCloud2>("rog_map/unk", 1);
        vm_.occ_inf_pub = nh_.advertise<sensor_msgs::PointCloud2>("rog_map/inf_occ", 1);
        vm_.unknown_inf_pub = nh_.advertise<sensor_msgs::PointCloud2>("rog_map/inf_unk", 1);

        if (cfg_.frontier_extraction_en) {
            vm_.frontier_pub = nh_.advertise<sensor_msgs::PointCloud2>("rog_map/frontier", 1);
        }

        if (cfg_.esdf_en) {
            vm_.esdf_pub = nh_.advertise<sensor_msgs::PointCloud2>("rog_map/esdf", 1);
            vm_.esdf_neg_pub = nh_.advertise<sensor_msgs::PointCloud2>("rog_map/esdf/neg", 1);
            vm_.esdf_occ_pub = nh_.advertise<sensor_msgs::PointCloud2>("rog_map/esdf/occ", 1);
        }

        if (cfg_.viz_time_rate > 0) {
            vm_.viz_timer = nh_.createTimer(ros::Duration(1.0 / cfg_.viz_time_rate), &ROGMap::vizCallback,
                                            this);
        }
    }
    vm_.mkr_arr_pub = nh_.advertise<visualization_msgs::MarkerArray>("rog_map/map_bound", 1);

    if (cfg_.ros_callback_en) {
        rc_.odom_sub = nh_.subscribe(cfg_.odom_topic, 1, &ROGMap::odomCallback, this);
        rc_.cloud_sub = nh_.subscribe(cfg_.cloud_topic, 1, &ROGMap::cloudCallback, this);
        rc_.update_timer = nh_.createTimer(ros::Duration(0.001), &ROGMap::updateCallback, this);
    }

    writeMapInfoToLog(map_info_log_file_);
    map_info_log_file_.close();
    for (std::size_t i = 0; i < time_consuming_name_.size(); ++i) {
        time_log_file_ << time_consuming_name_[i];
        if (i != time_consuming_name_.size() - 1) {
            time_log_file_ << ", ";
        }
    }
    time_log_file_ << endl;


    if (cfg_.load_pcd_en) {
        string pcd_path = cfg_.pcd_name;
        PointCloud::Ptr pcd_map(new PointCloud);
        if (pcl::io::loadPCDFile(pcd_path, *pcd_map) == -1) {
            cout << RED << "Load pcd file failed!" << RESET << endl;
            exit(-1);
        }
        Pose cur_pose;
        cur_pose.first = Vec3f(0, 0, 0);
        updateOccPointCloud(*pcd_map);
        if (cfg_.esdf_en) {
            esdf_map_->updateESDF3D(robot_state_.p);
        }
        cout << BLUE << " -- [ROGMap]Load pcd file success with " << pcd_map->size() << " pts." << RESET << endl;
        map_empty_ = false;
    }
}

bool ROGMap::isLineFree(const rog_map::Vec3f& start_pt, const rog_map::Vec3f& end_pt,
                        const bool& use_inf_map, const bool& use_unk_as_occ) const {
    std::lock_guard<std::mutex> map_lock(map_mutex_);
    if(start_pt.array().isNaN().any() || end_pt.array().isNaN().any() ) {
        cout<<RED<<" -- [ROGMap] Call isLineFree with NaN in start or end pt, return false."<<RESET<<endl;
        return false;
    }
    raycaster::RayCaster raycaster;
    if (use_inf_map) {
        raycaster.setResolution(cfg_.inflation_resolution);
    }
    else {
        raycaster.setResolution(cfg_.resolution);
    }
    Vec3f ray_pt;
    raycaster.setInput(start_pt, end_pt);
    while (raycaster.step(ray_pt)) {
        if (!use_unk_as_occ) {
            // allow both unk and free
            if (use_inf_map) {
                if (isOccupiedInflate(ray_pt)) {
                    return false;
                }
            }
            else {
                if (isOccupied(ray_pt)) {
                    return false;
                }
            }
        }
        else {
            // only allow known free
            if (use_inf_map) {
                if ((isUnknownInflate(ray_pt) || isOccupiedInflate(ray_pt)))
                    return false;
            }
            else {
                if (!isKnownFree(ray_pt)) {
                    return false;
                }
            }
        }
    }
    return true;
}

bool ROGMap::isLineFree(const Vec3f& start_pt, const Vec3f& end_pt, const double& max_dis,
                        const vec_Vec3i& neighbor_list) const {
    std::lock_guard<std::mutex> map_lock(map_mutex_);
    raycaster::RayCaster raycaster;
    raycaster.setResolution(cfg_.resolution);
    Vec3f ray_pt;
    raycaster.setInput(start_pt, end_pt);
    while (raycaster.step(ray_pt)) {
        if (max_dis > 0 && (ray_pt - start_pt).norm() > max_dis) {
            return false;
        }

        if (neighbor_list.empty()) {
            if (isOccupied(ray_pt)) {
                return false;
            }
        }
        else {
            Vec3i ray_pt_id_g;
            posToGlobalIndex(ray_pt, ray_pt_id_g);
            for (const auto& nei : neighbor_list) {
                Vec3i shift_tmp = ray_pt_id_g + nei;
                if (isOccupied(shift_tmp)) {
                    return false;
                }
            }
        }
    }
    return true;
}

bool ROGMap::isLineFree(const Vec3f& start_pt, const Vec3f& end_pt, Vec3f& free_local_goal, const double& max_dis,
                        const vec_Vec3i& neighbor_list) const {
    std::lock_guard<std::mutex> map_lock(map_mutex_);
    raycaster::RayCaster raycaster;
    raycaster.setResolution(cfg_.resolution);
    Vec3f ray_pt;
    raycaster.setInput(start_pt, end_pt);
    free_local_goal = start_pt;
    while (raycaster.step(ray_pt)) {
        free_local_goal = ray_pt;
        if (max_dis > 0 && (ray_pt - start_pt).norm() > max_dis) {
            return false;
        }

        if (neighbor_list.empty()) {
            if (isOccupied(ray_pt)) {
                return false;
            }
        }
        else {
            Vec3i ray_pt_id_g;
            posToGlobalIndex(ray_pt, ray_pt_id_g);
            for (const auto& nei : neighbor_list) {
                Vec3i shift_tmp = ray_pt_id_g + nei;
                if (isOccupied(shift_tmp)) {
                    return false;
                }
            }
        }
    }
    free_local_goal = end_pt;
    return true;
}

void ROGMap::updateMap(const PointCloud& cloud, const Pose& pose) {
    TimeConsuming ssss("sss", false);
    if (cfg_.ros_callback_en) {
        std::cout << RED << "ROS callback is enabled, can not insert map from updateMap API." << RESET
            << std::endl;
        return;
    }

    if (cloud.empty()) {
        static int local_cnt = 0;
        if (local_cnt++ > 100) {
            cout << YELLOW << "No cloud input, please check the input topic." << RESET << endl;
            local_cnt = 0;
        }
        return;
    }

    Pose normalized_pose = pose;
    if (normalized_pose.second.norm() > 1e-9) {
        normalized_pose.second.normalize();
    } else {
        normalized_pose.second = Quatf::Identity();
    }
    updateRobotState(normalized_pose, ros::Time::now());
    const Vec3f sensor_origin = sensorOriginFromBodyPose(normalized_pose);
    std::lock_guard<std::mutex> map_lock(map_mutex_);
    updateProbMap(cloud, normalized_pose, sensor_origin);

    writeTimeConsumingToLog(time_log_file_);
}

RobotState ROGMap::getRobotState() const {
    std::lock_guard<std::mutex> state_lock(state_mutex_);
    return robot_state_;
}

void ROGMap::updateRobotState(const Pose& input_pose, const ros::Time& input_stamp) {
    Pose pose = input_pose;
    if (pose.second.norm() > 1e-9) {
        pose.second.normalize();
    } else {
        pose.second = Quatf::Identity();
    }
    const ros::Time stamp = input_stamp.isZero() ? ros::Time::now() : input_stamp;

    std::lock_guard<std::mutex> state_lock(state_mutex_);
    robot_state_.p = pose.first;
    robot_state_.q = pose.second;
    robot_state_.rcv_time = ros::Time::now().toSec();
    robot_state_.rcv = true;
    robot_state_.yaw = get_yaw_from_quaternion<double>(pose.second);

    const auto insertion_point = std::lower_bound(
            odom_history_.begin(), odom_history_.end(), stamp,
            [](const TimedPose& sample, const ros::Time& value) {
                return sample.stamp < value;
            });
    if (insertion_point != odom_history_.end() && insertion_point->stamp == stamp) {
        insertion_point->pose = pose;
    } else {
        odom_history_.insert(insertion_point, TimedPose{stamp, pose});
    }

    const ros::Time newest_stamp = odom_history_.back().stamp;
    while (!odom_history_.empty() &&
           (newest_stamp - odom_history_.front().stamp).toSec() >
                   cfg_.odom_history_duration) {
        odom_history_.pop_front();
    }
}

bool ROGMap::findPoseAt(const ros::Time& stamp, Pose& pose) const {
    std::lock_guard<std::mutex> state_lock(state_mutex_);
    if (!robot_state_.rcv || odom_history_.empty() ||
        ros::Time::now().toSec() - robot_state_.rcv_time > cfg_.odom_timeout) {
        return false;
    }

    if (stamp.isZero()) {
        pose = odom_history_.back().pose;
        return true;
    }

    const auto after = std::lower_bound(
            odom_history_.begin(), odom_history_.end(), stamp,
            [](const TimedPose& sample, const ros::Time& value) {
                return sample.stamp < value;
            });

    if (after == odom_history_.begin()) {
        if (std::abs((after->stamp - stamp).toSec()) > cfg_.pose_sync_tolerance) {
            return false;
        }
        pose = after->pose;
        return true;
    }
    if (after == odom_history_.end()) {
        const TimedPose& latest = odom_history_.back();
        if (std::abs((stamp - latest.stamp).toSec()) > cfg_.pose_sync_tolerance) {
            return false;
        }
        pose = latest.pose;
        return true;
    }
    if (after->stamp == stamp) {
        pose = after->pose;
        return true;
    }

    const TimedPose& before = *std::prev(after);
    const double interval = (after->stamp - before.stamp).toSec();
    const double before_distance = (stamp - before.stamp).toSec();
    const double after_distance = (after->stamp - stamp).toSec();
    if (interval > 0.0 && before_distance <= cfg_.pose_sync_tolerance &&
        after_distance <= cfg_.pose_sync_tolerance) {
        const double ratio = before_distance / interval;
        pose.first = before.pose.first + ratio * (after->pose.first - before.pose.first);
        pose.second = before.pose.second.slerp(ratio, after->pose.second).normalized();
        return true;
    }

    const TimedPose& nearest = before_distance <= after_distance ? before : *after;
    if (std::min(before_distance, after_distance) > cfg_.pose_sync_tolerance) {
        return false;
    }
    pose = nearest.pose;
    return true;
}

Vec3f ROGMap::sensorOriginFromBodyPose(const Pose& body_pose) const {
    Quatf orientation = body_pose.second;
    if (orientation.norm() > 1e-9) {
        orientation.normalize();
    } else {
        orientation = Quatf::Identity();
    }
    return body_pose.first + orientation * cfg_.sensor_origin_in_body;
}


void ROGMap::odomCallback(const nav_msgs::OdometryConstPtr& odom_msg) {
    updateRobotState(std::make_pair(
        Vec3f(odom_msg->pose.pose.position.x, odom_msg->pose.pose.position.y,
              odom_msg->pose.pose.position.z),
        Quatf(odom_msg->pose.pose.orientation.w, odom_msg->pose.pose.orientation.x,
              odom_msg->pose.pose.orientation.y, odom_msg->pose.pose.orientation.z)),
        odom_msg->header.stamp);


    if (!cfg_.publish_tf) {
        return;
    }

    static tf2_ros::TransformBroadcaster br_map_ego;
    geometry_msgs::TransformStamped transformStamped;
    transformStamped.header.stamp =
            odom_msg->header.stamp.isZero() ? ros::Time::now() : odom_msg->header.stamp;
    transformStamped.header.frame_id = cfg_.frame_id;
    transformStamped.child_frame_id = cfg_.body_frame_id;
    transformStamped.transform.translation.x = odom_msg->pose.pose.position.x;
    transformStamped.transform.translation.y = odom_msg->pose.pose.position.y;
    transformStamped.transform.translation.z = odom_msg->pose.pose.position.z;
    transformStamped.transform.rotation.x = odom_msg->pose.pose.orientation.x;
    transformStamped.transform.rotation.y = odom_msg->pose.pose.orientation.y;
    transformStamped.transform.rotation.z = odom_msg->pose.pose.orientation.z;
    transformStamped.transform.rotation.w = odom_msg->pose.pose.orientation.w;
    br_map_ego.sendTransform(transformStamped);
}

void ROGMap::cloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg) {
    PointCloud::Ptr temp_pc(new PointCloud);
    pcl::fromROSMsg(*cloud_msg, *temp_pc);
    std::lock_guard<std::mutex> update_lock(rc_.update_lock);
    rc_.pc = temp_pc;
    rc_.pc_stamp = cloud_msg->header.stamp;
    rc_.pc_received = ros::WallTime::now();
    rc_.unfinished_frame_cnt++;
}

void ROGMap::updateCallback(const ros::TimerEvent& event) {
    ros::Time cloud_stamp;
    ros::WallTime cloud_received;
    int pending_frames = 0;
    {
        std::lock_guard<std::mutex> update_lock(rc_.update_lock);
        pending_frames = rc_.unfinished_frame_cnt;
        cloud_stamp = rc_.pc_stamp;
        cloud_received = rc_.pc_received;
    }

    if (pending_frames == 0) {
        std::lock_guard<std::mutex> map_lock(map_mutex_);
        if (!map_empty_) {
            return;
        }
        static double last_print_t = ros::Time::now().toSec();
        double cur_t = ros::Time::now().toSec();
        if (cfg_.ros_callback_en && (cur_t - last_print_t > 1.0)) {
            std::cout << YELLOW << " -- [ROG WARN] No point cloud input, check the topic name." << RESET << std::endl;
            last_print_t = cur_t;
        }
        return;
    }

    Pose synchronized_pose;
    if (!findPoseAt(cloud_stamp, synchronized_pose)) {
        if (!cloud_received.isZero() &&
            (ros::WallTime::now() - cloud_received).toSec() > cfg_.odom_timeout) {
            std::lock_guard<std::mutex> update_lock(rc_.update_lock);
            if (rc_.pc_stamp == cloud_stamp) {
                rc_.unfinished_frame_cnt = 0;
            }
            ROS_WARN_THROTTLE(
                    1.0,
                    "Dropping point cloud: no odometry within pose_sync_tolerance.");
        }
        return;
    }

    if (pending_frames > 1) {
        ROS_WARN_THROTTLE(
                1.0,
                "ROG-Map dropped intermediate point clouds: %d frames arrived "
                "before the latest frame was consumed.",
                pending_frames);
    }

    PointCloud::ConstPtr temp_pc;
    {
        std::lock_guard<std::mutex> update_lock(rc_.update_lock);
        if (rc_.pc_stamp != cloud_stamp) {
            return;
        }
        temp_pc = rc_.pc;
        rc_.unfinished_frame_cnt = 0;
    }

    const Vec3f sensor_origin = sensorOriginFromBodyPose(synchronized_pose);
    if (!temp_pc) {
        return;
    }
    const auto lock_request = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> map_lock(map_mutex_);
    const auto lock_acquired = std::chrono::steady_clock::now();
    updateProbMap(*temp_pc, synchronized_pose, sensor_origin);
    const auto update_finished = std::chrono::steady_clock::now();
    map_lock.unlock();

    const double lock_wait_ms =
            std::chrono::duration<double, std::milli>(
                    lock_acquired - lock_request).count();
    const double update_ms =
            std::chrono::duration<double, std::milli>(
                    update_finished - lock_acquired).count();
    if (lock_wait_ms + update_ms > 80.0) {
        ROS_WARN_THROTTLE(
                1.0,
                "ROG-Map slow frame: total=%.1f ms, map_lock=%.1f ms, "
                "core=%.1f ms, raycast=%.1f ms, cache=%.1f ms, points=%zu.",
                lock_wait_ms + update_ms,
                lock_wait_ms,
                update_ms,
                1000.0 * time_consuming_[1],
                1000.0 * time_consuming_[2],
                temp_pc->size());
    }

    writeTimeConsumingToLog(time_log_file_);
}

void ROGMap::vecEVec3fToPC2(const vec_E<Vec3f>& points, sensor_msgs::PointCloud2& cloud) const {
    sensor_msgs::PointCloud2Modifier modifier(cloud);
    modifier.setPointCloud2FieldsByString(1, "xyz");
    modifier.resize(points.size());
    sensor_msgs::PointCloud2Iterator<float> x(cloud, "x");
    sensor_msgs::PointCloud2Iterator<float> y(cloud, "y");
    sensor_msgs::PointCloud2Iterator<float> z(cloud, "z");
    for (const Vec3f& point : points) {
        *x = static_cast<float>(point.x());
        *y = static_cast<float>(point.y());
        *z = static_cast<float>(point.z());
        ++x;
        ++y;
        ++z;
    }
    cloud.is_dense = true;
    cloud.header.stamp = ros::Time::now();
    cloud.header.frame_id = cfg_.frame_id;
}

void ROGMap::vizCallback(const ros::TimerEvent& event) {
    if (!cfg_.visualization_en) {
        return;
    }
    std::unique_lock<std::mutex> callback_lock(
            vm_.callback_mutex, std::try_to_lock);
    if (!callback_lock.owns_lock()) {
        return;
    }

    const RobotState robot_state = getRobotState();
    const bool publish_unknown = cfg_.pub_unknown_map_en &&
            vm_.unknown_pub.getNumSubscribers() > 0;
    const bool publish_occupied = vm_.occ_pub.getNumSubscribers() > 0;
    const bool publish_unknown_inflated = publish_unknown &&
            cfg_.unk_inflation_en &&
            vm_.unknown_inf_pub.getNumSubscribers() > 0;
    const bool publish_occupied_inflated =
            vm_.occ_inf_pub.getNumSubscribers() > 0;
    const bool publish_frontier = cfg_.frontier_extraction_en &&
            vm_.frontier_pub.getNumSubscribers() > 0;
    const bool publish_esdf_positive = cfg_.esdf_en &&
            vm_.esdf_pub.getNumSubscribers() > 0;
    const bool publish_esdf_negative = cfg_.esdf_en &&
            vm_.esdf_neg_pub.getNumSubscribers() > 0;

    Vec3f box_min, box_max;
    if (cfg_.use_dynamic_reconfigure) {
        box_min = vm_.vizcfg.box_min;
        box_max = vm_.vizcfg.box_max;
        if (vm_.vizcfg.use_body_center) {
            box_min += robot_state.p;
            box_max += robot_state.p;
        }
    } else {
        box_max = robot_state.p + cfg_.visualization_range / 2;
        box_min = robot_state.p - cfg_.visualization_range / 2;
    }

    Vec3f local_map_min, local_map_max, local_map_origin;
    Vec3f update_box_min, update_box_max;
    Vec3f esdf_box_min, esdf_box_max;
    sensor_msgs::PointCloud2 esdf_positive_msg, esdf_negative_msg;
#ifdef ESDF_MAP_DEBUG
    const bool publish_esdf_occupied = cfg_.esdf_en &&
            vm_.esdf_occ_pub.getNumSubscribers() > 0;
    sensor_msgs::PointCloud2 esdf_occupied_msg;
#endif
    {
        std::lock_guard<std::mutex> map_lock(map_mutex_);
        if (map_empty_) {
            return;
        }
        boundBoxByLocalMap(box_min, box_max);
        if ((box_max - box_min).minCoeff() <= 0) {
            ROS_WARN_THROTTLE(1.0, "ROG-Map visualization range is outside the local map.");
            return;
        }

        if (publish_occupied || publish_unknown) {
            // Copy only the compact probability buffer while holding the map
            // lock. Classification, point generation, serialization and ROS
            // publication happen after the update thread is unblocked.
            copyProbabilityBox(
                    box_min, box_max, vm_.probability_snapshot);
        }
        if (publish_unknown_inflated) {
            boxSearchInflate(
                    box_min, box_max, UNKNOWN,
                    vm_.unknown_inflated_points);
        }
        if (publish_occupied_inflated) {
            boxSearchInflate(
                    box_min, box_max, OCCUPIED,
                    vm_.occupied_inflated_points);
        }
        if (publish_frontier) {
            boxSearch(box_min, box_max, FRONTIER, vm_.frontier_points);
        }
        if (publish_esdf_positive) {
            esdf_map_->getPositiveESDFPC2(
                    box_min, box_max, robot_state.p.z() - 0.5,
                    esdf_positive_msg);
        }
        if (publish_esdf_negative) {
            esdf_map_->getNegativeESDFPC2(
                    box_min, box_max, robot_state.p.z() - 0.5,
                    esdf_negative_msg);
        }
#ifdef ESDF_MAP_DEBUG
        if (publish_esdf_occupied) {
            esdf_map_->getESDFOccPC2(
                    box_min, box_max, esdf_occupied_msg);
        }
#endif

        local_map_min = local_map_bound_min_d_;
        local_map_max = local_map_bound_max_d_;
        local_map_origin = local_map_origin_d_;
        update_box_min = raycast_data_.cache_box_min;
        update_box_max = raycast_data_.cache_box_max;
        if (cfg_.esdf_en) {
            esdf_map_->getUpdatedBbox(esdf_box_min, esdf_box_max);
        }
    }

    if (publish_occupied || publish_unknown) {
        probabilityBoxSnapshotToPoints(
                vm_.probability_snapshot,
                vm_.occupied_points, vm_.unknown_points);
    }

    const ros::Time stamp = ros::Time::now();
    sensor_msgs::PointCloud2 cloud_msg;
    if (publish_unknown) {
        vecEVec3fToPC2(vm_.unknown_points, cloud_msg);
        cloud_msg.header.stamp = stamp;
        vm_.unknown_pub.publish(cloud_msg);
    }
    if (publish_occupied) {
        vecEVec3fToPC2(vm_.occupied_points, cloud_msg);
        cloud_msg.header.stamp = stamp;
        vm_.occ_pub.publish(cloud_msg);
    }
    if (publish_unknown_inflated) {
        vecEVec3fToPC2(vm_.unknown_inflated_points, cloud_msg);
        cloud_msg.header.stamp = stamp;
        vm_.unknown_inf_pub.publish(cloud_msg);
    }
    if (publish_occupied_inflated) {
        vecEVec3fToPC2(vm_.occupied_inflated_points, cloud_msg);
        cloud_msg.header.stamp = stamp;
        vm_.occ_inf_pub.publish(cloud_msg);
    }
    if (publish_frontier) {
        vecEVec3fToPC2(vm_.frontier_points, cloud_msg);
        cloud_msg.header.stamp = stamp;
        vm_.frontier_pub.publish(cloud_msg);
    }
    if (publish_esdf_positive) {
        esdf_positive_msg.header.stamp = stamp;
        vm_.esdf_pub.publish(esdf_positive_msg);
    }
    if (publish_esdf_negative) {
        esdf_negative_msg.header.stamp = stamp;
        vm_.esdf_neg_pub.publish(esdf_negative_msg);
    }
#ifdef ESDF_MAP_DEBUG
    if (publish_esdf_occupied) {
        esdf_occupied_msg.header.stamp = stamp;
        vm_.esdf_occ_pub.publish(esdf_occupied_msg);
    }
#endif

    /* Publish visualization range */
    vm_.mkr_arr.markers.clear();
    visualizeBoundingBox(vm_.mkr_arr, box_min, box_max, "Visualization Range", Color::Purple());
    visualizeText(vm_.mkr_arr, "Visualization Range Text", "Visualization Range", box_max + Vec3f(0, 0, 0.5),
                  Color::Purple(), 0.6, 0);

    /* Publish local map range */
    visualizeBoundingBox(vm_.mkr_arr, local_map_min, local_map_max, "Local Map Range",
                         Color::Orange());
    visualizeText(vm_.mkr_arr, "Local Map Range Text", "Local Map Range", local_map_max + Vec3f(0, 0, 1.0),
                  Color::Orange(),
                  0.6, 0);

    /* Publish Ray-casting range */
    visualizeBoundingBox(vm_.mkr_arr, update_box_min, update_box_max,
                         "Updating Range",
                         Color::Green());
    visualizeText(vm_.mkr_arr, "Updating Range Text", "Updating Range",
                  update_box_max + Vec3f(0, 0, 0.5),
                  Color::Green(), 0.6, 0);

    /* Publish Local map origin */
    visualizePoint(vm_.mkr_arr, local_map_origin, Color::Red(), "Local Map Origin", 0.2, 0);

    if (cfg_.esdf_en) {
        visualizeText(vm_.mkr_arr, "ESDF Map Text", "ESDF Map", esdf_box_max + Vec3f(0, 0, 1.0),
                      Color::Blue(),
                      0.6, 0);
        visualizeBoundingBox(vm_.mkr_arr, esdf_box_min, esdf_box_max, "ESDF Updating Range",
                             Color::Blue());
    }

    vm_.mkr_arr_pub.publish(vm_.mkr_arr);

}

void ROGMap::VizCfgCallback(rog_map::VizConfig& config, uint32_t level) {
    std::lock_guard<std::mutex> callback_lock(vm_.callback_mutex);
    vm_.vizcfg.use_body_center = config.use_body_center;
    vm_.vizcfg.box_min.x() = config.x_lower_bound;
    vm_.vizcfg.box_min.y() = config.y_lower_bound;
    vm_.vizcfg.box_min.z() = config.z_lower_bound;
    vm_.vizcfg.box_max.x() = config.x_upper_bound;
    vm_.vizcfg.box_max.y() = config.y_upper_bound;
    vm_.vizcfg.box_max.z() = config.z_upper_bound;
}
