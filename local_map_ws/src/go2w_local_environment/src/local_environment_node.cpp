#include "go2w_local_environment/elevation_projector.hpp"

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <boost/bind/bind.hpp>
#include <Eigen/Geometry>
#include <grid_map_core/BufferRegion.hpp>
#include <grid_map_core/GridMap.hpp>
#include <grid_map_core/GridMapMath.hpp>
#include <grid_map_core/iterators/CircleIterator.hpp>
#include <grid_map_core/iterators/LineIterator.hpp>
#include <grid_map_msgs/GridMap.h>
#include <grid_map_ros/GridMapRosConverter.hpp>
#include <geometry_msgs/PointStamped.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <nav_msgs/Odometry.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

namespace go2w_local_environment {
namespace {

constexpr char kGroundHeight[] = "ground_height";
constexpr char kHeightRange[] = "height_range";
constexpr char kObservedMask[] = "observed_mask";

const std::vector<std::string> kPublishedLayers{
    kGroundHeight,
    kHeightRange,
    kObservedMask,
};

float quietNaN() {
  return std::numeric_limits<float>::quiet_NaN();
}

template <typename T>
void loadParameter(
    const ros::NodeHandle& private_node,
    const std::string& name,
    T& value) {
  private_node.param(name, value, value);
}

std::string normalizeFrame(const std::string& frame) {
  if (!frame.empty() && frame.front() == '/') {
    return frame.substr(1);
  }
  return frame;
}

}  // namespace

class LocalEnvironmentNode {
 public:
  LocalEnvironmentNode()
      : node_(),
        private_node_("~"),
        projector_(loadProjectionParameters()) {
    loadNodeParameters();
    initializeRollingMap();

    grid_map_publisher_ =
        node_.advertise<grid_map_msgs::GridMap>(grid_map_topic_, 1);
    ground_reference_publisher_ =
        node_.advertise<geometry_msgs::PointStamped>(
            ground_reference_topic_, 1);

    cloud_subscriber_.reset(new CloudSubscriber(
        node_, cloud_topic_, subscriber_queue_size_));
    odometry_subscriber_.reset(new OdometrySubscriber(
        node_, odometry_topic_, subscriber_queue_size_));

    SyncPolicy sync_policy(sync_queue_size_);
    sync_policy.setMaxIntervalDuration(
        ros::Duration(sync_tolerance_));
    synchronizer_.reset(new Synchronizer(sync_policy));
    synchronizer_->connectInput(
        *cloud_subscriber_, *odometry_subscriber_);
    synchronizer_->registerCallback(
        boost::bind(
            &LocalEnvironmentNode::synchronizedCallback,
            this,
            boost::placeholders::_1,
            boost::placeholders::_2));

    if (hole_filling_enabled_) {
      hole_fill_worker_ = std::thread(
          &LocalEnvironmentNode::holeFillWorkerLoop, this);
    }

    ROS_INFO_STREAM(
        "Rolling local elevation map started. cloud=" << cloud_topic_
        << ", map=" << map_length_x_ << " x " << map_length_y_
        << " m @ " << map_resolution_ << " m"
        << ", input_crop=" << input_crop_length_x_ << " x "
        << input_crop_length_y_ << " m"
        << ", history=" << history_length_ << " frames"
        << ", elevation_ttl=" << elevation_stale_after_ << " s"
        << ", ray_ttl=" << ray_stale_after_ << " s"
        << ", imu_to_ground=" << imu_to_ground_height_ << " m");
  }

  ~LocalEnvironmentNode() {
    {
      std::lock_guard<std::mutex> lock(hole_fill_mutex_);
      stop_hole_fill_worker_ = true;
    }
    hole_fill_condition_.notify_one();
    if (hole_fill_worker_.joinable()) {
      hole_fill_worker_.join();
    }
  }

 private:
  using SyncPolicy =
      message_filters::sync_policies::ApproximateTime<
          sensor_msgs::PointCloud2,
          nav_msgs::Odometry>;
  using Synchronizer = message_filters::Synchronizer<SyncPolicy>;
  using CloudSubscriber =
      message_filters::Subscriber<sensor_msgs::PointCloud2>;
  using OdometrySubscriber =
      message_filters::Subscriber<nav_msgs::Odometry>;

  struct IndexedHeightSample {
    std::size_t cell_index;
    float height;
  };

  struct HoleFillSnapshot {
    grid_map::Size map_size{grid_map::Size::Zero()};
    grid_map::Index buffer_start{grid_map::Index::Zero()};
    std::uint64_t map_revision{0U};
    std::vector<unsigned char> measured;
    std::vector<float> ground_height;
    std::vector<float> height_range;
  };

  struct HoleFillWorkResult {
    std::uint64_t map_revision{0U};
    std::vector<float> ground_height;
    std::vector<float> height_range;
    std::size_t filled_cell_count{0U};
    double computation_ms{0.0};
  };

  ElevationProjectionParameters loadProjectionParameters() {
    ElevationProjectionParameters parameters;
    loadParameter(
        private_node_,
        "projection/vertical_min_offset",
        parameters.vertical_min_offset);
    loadParameter(
        private_node_,
        "projection/vertical_max_offset",
        parameters.vertical_max_offset);
    loadParameter(
        private_node_,
        "projection/ground_percentile",
        parameters.ground_percentile);

    int minimum_points =
        static_cast<int>(parameters.minimum_points_per_cell);
    loadParameter(
        private_node_,
        "projection/minimum_points_per_cell",
        minimum_points);
    if (minimum_points <= 0) {
      throw std::runtime_error(
          "projection/minimum_points_per_cell must be positive");
    }
    parameters.minimum_points_per_cell =
        static_cast<std::size_t>(minimum_points);
    return parameters;
  }

  void loadNodeParameters() {
    loadParameter(private_node_, "map/frame_id", map_frame_);
    loadParameter(private_node_, "map/length_x", map_length_x_);
    loadParameter(private_node_, "map/length_y", map_length_y_);
    loadParameter(private_node_, "map/resolution", map_resolution_);

    loadParameter(
        private_node_, "input_crop/length_x", input_crop_length_x_);
    loadParameter(
        private_node_, "input_crop/length_y", input_crop_length_y_);

    private_node_.getParam(
        "sensor/origin_in_body", sensor_origin_in_body_);

    loadParameter(
        private_node_, "height_reference/imu_to_ground_height",
        imu_to_ground_height_);
    loadParameter(
        private_node_, "height_reference/local_ground_radius",
        ground_reference_radius_);
    int minimum_ground_cells = static_cast<int>(
        ground_reference_parameters_.minimum_samples);
    loadParameter(
        private_node_, "height_reference/minimum_ground_cells",
        minimum_ground_cells);
    loadParameter(
        private_node_, "height_reference/maximum_cell_height_range",
        ground_reference_maximum_cell_range_);
    loadParameter(
        private_node_, "height_reference/maximum_nominal_difference",
        ground_reference_parameters_.maximum_nominal_difference);
    if (minimum_ground_cells <= 0) {
      throw std::runtime_error(
          "height_reference/minimum_ground_cells must be positive");
    }
    ground_reference_parameters_.minimum_samples =
        static_cast<std::size_t>(minimum_ground_cells);
    ground_reference_estimator_ =
        GroundReferenceEstimator(ground_reference_parameters_);

    int history_length = static_cast<int>(history_length_);
    loadParameter(
        private_node_, "fusion/history_length", history_length);
    loadParameter(
        private_node_, "fusion/elevation_stale_after",
        elevation_stale_after_);
    loadParameter(
        private_node_, "fusion/ray_stale_after", ray_stale_after_);
    loadParameter(
        private_node_, "fusion/robot_history_keep_radius",
        robot_history_keep_radius_);
    int expiration_cells_per_frame = static_cast<int>(
        expiration_cells_per_frame_);
    loadParameter(
        private_node_, "fusion/expiration_cells_per_frame",
        expiration_cells_per_frame);
    if (history_length <= 0) {
      throw std::runtime_error(
          "fusion/history_length must be positive");
    }
    if (expiration_cells_per_frame <= 0) {
      throw std::runtime_error(
          "fusion/expiration_cells_per_frame must be positive");
    }
    history_length_ = static_cast<std::size_t>(history_length);
    expiration_cells_per_frame_ =
        static_cast<std::size_t>(expiration_cells_per_frame);

    loadParameter(
        private_node_, "raycasting/resolution", raycast_resolution_);

    int hole_fill_radius = hole_fill_parameters_.radius_cells;
    int hole_fill_minimum_neighbors = static_cast<int>(
        hole_fill_parameters_.minimum_neighbors);
    loadParameter(
        private_node_, "hole_filling/enabled", hole_filling_enabled_);
    loadParameter(
        private_node_, "hole_filling/radius_cells", hole_fill_radius);
    loadParameter(
        private_node_, "hole_filling/minimum_neighbors",
        hole_fill_minimum_neighbors);
    loadParameter(
        private_node_, "hole_filling/maximum_ground_height_difference",
        hole_fill_parameters_.maximum_ground_height_difference);
    loadParameter(
        private_node_, "hole_filling/update_interval",
        hole_fill_update_interval_);
    if (hole_fill_radius <= 0 || hole_fill_minimum_neighbors <= 0) {
      throw std::runtime_error(
          "hole-filling radius and minimum neighbors must be positive");
    }
    if (hole_fill_update_interval_ <= 0.0) {
      throw std::runtime_error(
          "hole_filling/update_interval must be positive");
    }
    hole_fill_parameters_.radius_cells = hole_fill_radius;
    hole_fill_parameters_.minimum_neighbors =
        static_cast<std::size_t>(hole_fill_minimum_neighbors);
    hole_filler_ = ElevationHoleFiller(hole_fill_parameters_);

    loadParameter(
        private_node_, "sync/queue_size", sync_queue_size_);
    loadParameter(
        private_node_, "sync/subscriber_queue_size",
        subscriber_queue_size_);
    loadParameter(
        private_node_, "sync/tolerance", sync_tolerance_);
    loadParameter(
        private_node_, "diagnostics/slow_processing_threshold",
        slow_processing_threshold_);

    loadParameter(private_node_, "topics/cloud", cloud_topic_);
    loadParameter(
        private_node_, "topics/odometry", odometry_topic_);
    loadParameter(
        private_node_, "topics/grid_map", grid_map_topic_);
    loadParameter(
        private_node_, "topics/ground_reference",
        ground_reference_topic_);

    if (map_frame_.empty()) {
      throw std::runtime_error("map/frame_id must not be empty");
    }
    if (map_length_x_ <= 0.0 || map_length_y_ <= 0.0 ||
        map_resolution_ <= 0.0) {
      throw std::runtime_error(
          "map lengths and resolution must be positive");
    }
    if (input_crop_length_x_ < map_length_x_ ||
        input_crop_length_y_ < map_length_y_) {
      throw std::runtime_error(
          "input crop must cover the complete rolling map");
    }
    if (sensor_origin_in_body_.size() != 3U) {
      throw std::runtime_error(
          "sensor/origin_in_body must contain exactly three values");
    }
    if (imu_to_ground_height_ <= 0.0 ||
        ground_reference_radius_ <= 0.0 ||
        ground_reference_maximum_cell_range_ < 0.0) {
      throw std::runtime_error(
          "height-reference distances must be valid and positive");
    }
    if (elevation_stale_after_ <= 0.0 || ray_stale_after_ <= 0.0) {
      throw std::runtime_error(
          "fusion elevation/ray stale times must be positive");
    }
    if (robot_history_keep_radius_ < 0.0) {
      throw std::runtime_error(
          "fusion/robot_history_keep_radius must not be negative");
    }
    if (raycast_resolution_ < map_resolution_) {
      throw std::runtime_error(
          "raycasting/resolution must be at least map/resolution");
    }
    if (sync_queue_size_ <= 0 || subscriber_queue_size_ <= 0) {
      throw std::runtime_error("sync queue sizes must be positive");
    }
    if (sync_tolerance_ < 0.0) {
      throw std::runtime_error("sync/tolerance must not be negative");
    }
    if (slow_processing_threshold_ < 0.0) {
      throw std::runtime_error(
          "diagnostics/slow_processing_threshold must not be negative");
    }
  }

  void initializeRollingMap() {
    map_ = grid_map::GridMap(kPublishedLayers);
    map_.setFrameId(map_frame_);
    map_.setGeometry(
        grid_map::Length(map_length_x_, map_length_y_),
        map_resolution_,
        grid_map::Position::Zero());

    const grid_map::Size size = map_.getSize();
    const std::size_t cell_count =
        static_cast<std::size_t>(size(0)) *
        static_cast<std::size_t>(size(1));
    cell_histories_.resize(cell_count);
    last_ray_observed_seconds_.resize(cell_count, quietNaN());
    inferred_ground_height_.resize(cell_count, quietNaN());
    inferred_height_range_.resize(cell_count, quietNaN());
    ray_observed_this_frame_.resize(cell_count, 0U);
    const int ray_crop_cells_x = std::max(
        1, static_cast<int>(std::ceil(
               input_crop_length_x_ / raycast_resolution_)));
    const int ray_crop_cells_y = std::max(
        1, static_cast<int>(std::ceil(
               input_crop_length_y_ / raycast_resolution_)));
    ray_endpoint_seen_.resize(
        static_cast<std::size_t>(ray_crop_cells_x) *
            static_cast<std::size_t>(ray_crop_cells_y),
        0U);
    clearAllCells();
  }

  void clearAllCells() {
    map_[kGroundHeight].setConstant(quietNaN());
    map_[kHeightRange].setConstant(quietNaN());
    map_[kObservedMask].setZero();
    for (IncrementalElevationCell& history : cell_histories_) {
      history.clear();
    }
    std::fill(
        last_ray_observed_seconds_.begin(),
        last_ray_observed_seconds_.end(),
        std::numeric_limits<double>::quiet_NaN());
    std::fill(
        inferred_ground_height_.begin(),
        inferred_ground_height_.end(),
        quietNaN());
    std::fill(
        inferred_height_range_.begin(),
        inferred_height_range_.end(),
        quietNaN());
    last_hole_fill_update_seconds_ =
        std::numeric_limits<double>::quiet_NaN();
    expiration_cursor_ = 0U;
    ++map_revision_;
  }

  void synchronizedCallback(
      const sensor_msgs::PointCloud2ConstPtr& cloud_message,
      const nav_msgs::OdometryConstPtr& odometry) {
    const ros::WallTime start_time = ros::WallTime::now();
    const double stamp_difference = std::abs(
        (cloud_message->header.stamp -
         odometry->header.stamp).toSec());
    if (stamp_difference > sync_tolerance_) {
      ROS_WARN_THROTTLE(
          2.0,
          "Registered cloud and odometry differ by %.3f s; frame skipped.",
          stamp_difference);
      return;
    }

    if (!frameMatches(cloud_message->header.frame_id) ||
        !frameMatches(odometry->header.frame_id)) {
      ROS_ERROR_THROTTLE(
          2.0,
          "Cloud/odometry frame does not match map/frame_id='%s'.",
          map_frame_.c_str());
      return;
    }

    ros::Time stamp = cloud_message->header.stamp;
    if (stamp.isZero()) {
      stamp = odometry->header.stamp;
    }
    if (stamp.isZero()) {
      stamp = ros::Time::now();
    }
    const double stamp_seconds = stamp.toSec();

    if (std::isfinite(last_stamp_seconds_)) {
      if (stamp_seconds + sync_tolerance_ < last_stamp_seconds_) {
        ROS_WARN(
            "Input time moved backwards; resetting the rolling elevation map.");
        clearAllCells();
        map_.setPosition(grid_map::Position(
            odometry->pose.pose.position.x,
            odometry->pose.pose.position.y));
      } else if (stamp_seconds < last_stamp_seconds_) {
        ROS_WARN_THROTTLE(
            2.0,
            "Slightly out-of-order synchronized frame skipped.");
        return;
      }
    }
    last_stamp_seconds_ = stamp_seconds;
    consumeHoleFillResult();

    const grid_map::Position robot_position(
        odometry->pose.pose.position.x,
        odometry->pose.pose.position.y);
    moveMap(robot_position);
    expireOldCells(stamp_seconds, robot_position);
    const ros::WallTime after_expiration = ros::WallTime::now();

    pcl::PointCloud<pcl::PointXYZ> cloud;
    pcl::fromROSMsg(*cloud_message, cloud);
    updateMapIncrementally(cloud, odometry, stamp_seconds);
    const ros::WallTime after_update = ros::WallTime::now();

    scheduleHoleFillIfNeeded(stamp_seconds);
    map_.setTimestamp(stamp.toNSec());
    grid_map::GridMap published_map = map_;
    const std::size_t filled_cell_count =
        applyHoleFillCache(published_map);
    const double ground_reference_z = estimateGroundReferenceZ(
        *odometry, robot_position);
    makeGroundHeightRelative(published_map, ground_reference_z);
    geometry_msgs::PointStamped ground_reference_message;
    ground_reference_message.header.stamp = stamp;
    ground_reference_message.header.frame_id = map_frame_;
    ground_reference_message.point.x = robot_position.x();
    ground_reference_message.point.y = robot_position.y();
    ground_reference_message.point.z = ground_reference_z;
    ground_reference_publisher_.publish(ground_reference_message);
    const ros::WallTime after_hole_filling = ros::WallTime::now();
    grid_map_msgs::GridMap message;
    grid_map::GridMapRosConverter::toMessage(
        published_map, kPublishedLayers, message);
    grid_map_publisher_.publish(message);
    const ros::WallTime finish_time = ros::WallTime::now();

    const double elapsed_ms = (finish_time - start_time).toSec() * 1000.0;
    if (slow_processing_threshold_ > 0.0 &&
        elapsed_ms > slow_processing_threshold_) {
      ROS_WARN_THROTTLE(
          1.0,
          "Rolling elevation update is slow: total=%.1f ms, "
          "expire=%.1f ms, core=%.1f ms, fill=%.1f ms, "
          "publish=%.1f ms, input points=%zu, filled cells=%zu.",
          elapsed_ms,
          (after_expiration - start_time).toSec() * 1000.0,
          (after_update - after_expiration).toSec() * 1000.0,
          (after_hole_filling - after_update).toSec() * 1000.0,
          (finish_time - after_hole_filling).toSec() * 1000.0,
          cloud.size(),
          filled_cell_count);
    }
    ROS_DEBUG_STREAM_THROTTLE(
        1.0,
        "Ground-height reference: z=" << ground_reference_z
        << " m (IMU z=" << odometry->pose.pose.position.z << " m)");
  }

  double nominalGroundReferenceZ(
      const nav_msgs::Odometry& odometry) const {
    const auto& orientation = odometry.pose.pose.orientation;
    Eigen::Quaterniond rotation(
        orientation.w,
        orientation.x,
        orientation.y,
        orientation.z);
    if (rotation.norm() < 1e-6) {
      rotation = Eigen::Quaterniond::Identity();
    } else {
      rotation.normalize();
    }
    const Eigen::Vector3d imu_to_ground_body(
        0.0, 0.0, -imu_to_ground_height_);
    return odometry.pose.pose.position.z +
           (rotation * imu_to_ground_body).z();
  }

  double estimateGroundReferenceZ(
      const nav_msgs::Odometry& odometry,
      const grid_map::Position& robot_position) {
    const float nominal_reference = static_cast<float>(
        nominalGroundReferenceZ(odometry));
    ground_reference_samples_.clear();
    try {
      for (grid_map::CircleIterator iterator(
               map_, robot_position, ground_reference_radius_);
           !iterator.isPastEnd(); ++iterator) {
        const grid_map::Index index = *iterator;
        const float ground_height = map_.at(kGroundHeight, index);
        const float height_range = map_.at(kHeightRange, index);
        if (!std::isfinite(ground_height) ||
            !std::isfinite(height_range) ||
            height_range > ground_reference_maximum_cell_range_) {
          continue;
        }
        ground_reference_samples_.push_back(ground_height);
      }
    } catch (const std::invalid_argument&) {
      return nominal_reference;
    }
    return ground_reference_estimator_.estimate(
        ground_reference_samples_, nominal_reference);
  }

  void makeGroundHeightRelative(
      grid_map::GridMap& output_map,
      const double ground_reference_z) const {
    output_map[kGroundHeight].array() -=
        static_cast<float>(ground_reference_z);
  }

  void moveMap(const grid_map::Position& robot_position) {
    std::vector<grid_map::BufferRegion> new_regions;
    if (!map_.move(robot_position, new_regions)) {
      return;
    }
    ++map_revision_;
    for (const grid_map::BufferRegion& region : new_regions) {
      clearBufferRegion(region);
    }
  }

  void clearBufferRegion(const grid_map::BufferRegion& region) {
    const grid_map::Index start = region.getStartIndex();
    const grid_map::Size size = region.getSize();
    for (int row = start(0); row < start(0) + size(0); ++row) {
      for (int column = start(1);
           column < start(1) + size(1); ++column) {
        const grid_map::Index index(row, column);
        clearCell(index);
      }
    }
  }

  void clearCell(const grid_map::Index& index) {
    clearElevation(index);
    last_ray_observed_seconds_[historyIndex(index)] =
        std::numeric_limits<double>::quiet_NaN();
    map_.at(kObservedMask, index) = 0.0F;
  }

  void clearElevation(const grid_map::Index& index) {
    const std::size_t linear_index = historyIndex(index);
    map_.at(kGroundHeight, index) = quietNaN();
    map_.at(kHeightRange, index) = quietNaN();
    cell_histories_[linear_index].clear();
    inferred_ground_height_[linear_index] = quietNaN();
    inferred_height_range_[linear_index] = quietNaN();
  }

  void expireOldCells(
      const double current_stamp,
      const grid_map::Position& robot_position) {
    const double oldest_elevation_stamp =
        current_stamp - elevation_stale_after_;
    const double oldest_ray_stamp = current_stamp - ray_stale_after_;
    const double protected_radius_squared =
        robot_history_keep_radius_ * robot_history_keep_radius_;
    const std::size_t cell_count = cell_histories_.size();
    const std::size_t cells_to_process = std::min(
        expiration_cells_per_frame_, cell_count);
    for (std::size_t processed = 0U;
         processed < cells_to_process; ++processed) {
      const std::size_t linear_index = expiration_cursor_;
      expiration_cursor_ = (expiration_cursor_ + 1U) % cell_count;
      const grid_map::Index index = indexFromHistory(linear_index);
      IncrementalElevationCell& history =
          cell_histories_[linear_index];
      bool protect_elevation_history = false;
      if (!history.empty() && robot_history_keep_radius_ > 0.0) {
        grid_map::Position cell_position;
        protect_elevation_history =
            map_.getPosition(index, cell_position) &&
            (cell_position - robot_position).squaredNorm() <=
                protected_radius_squared;
      }
      if (!protect_elevation_history &&
          history.removeOlderThan(oldest_elevation_stamp)) {
        if (history.empty()) {
          clearElevation(index);
        } else {
          writeFusedCell(index, history.fused());
        }
      }

      double& last_ray_observed =
          last_ray_observed_seconds_[linear_index];
      if (std::isfinite(last_ray_observed) &&
          last_ray_observed < oldest_ray_stamp) {
        last_ray_observed =
            std::numeric_limits<double>::quiet_NaN();
      }
      refreshObservedMask(index);
    }
  }

  std::size_t applyHoleFillCache(
      grid_map::GridMap& output_map) const {
    if (!hole_filling_enabled_) {
      return 0U;
    }

    std::size_t filled_cell_count = 0U;
    const std::size_t cell_count = cell_histories_.size();
    for (std::size_t linear_index = 0U;
         linear_index < cell_count; ++linear_index) {
      if (!cell_histories_[linear_index].empty() ||
          !std::isfinite(inferred_ground_height_[linear_index]) ||
          !std::isfinite(inferred_height_range_[linear_index])) {
        continue;
      }
      const grid_map::Index index = indexFromHistory(linear_index);
      output_map.at(kGroundHeight, index) =
          inferred_ground_height_[linear_index];
      output_map.at(kHeightRange, index) =
          inferred_height_range_[linear_index];
      ++filled_cell_count;
    }
    return filled_cell_count;
  }

  void scheduleHoleFillIfNeeded(const double stamp_seconds) {
    if (!hole_filling_enabled_ ||
        (std::isfinite(last_hole_fill_update_seconds_) &&
         stamp_seconds - last_hole_fill_update_seconds_ <
             hole_fill_update_interval_)) {
      return;
    }

    {
      std::lock_guard<std::mutex> lock(hole_fill_mutex_);
      if (hole_fill_request_pending_ || hole_fill_work_in_progress_ ||
          hole_fill_result_ready_) {
        return;
      }
    }

    HoleFillSnapshot snapshot;
    snapshot.map_size = map_.getSize();
    snapshot.buffer_start = map_.getStartIndex();
    snapshot.map_revision = map_revision_;
    const std::size_t cell_count = cell_histories_.size();
    snapshot.measured.resize(cell_count, 0U);
    snapshot.ground_height.resize(cell_count, quietNaN());
    snapshot.height_range.resize(cell_count, quietNaN());
    for (std::size_t linear_index = 0U;
         linear_index < cell_count; ++linear_index) {
      if (cell_histories_[linear_index].empty()) {
        continue;
      }
      const grid_map::Index index = indexFromHistory(linear_index);
      snapshot.measured[linear_index] = 1U;
      snapshot.ground_height[linear_index] =
          map_.at(kGroundHeight, index);
      snapshot.height_range[linear_index] =
          map_.at(kHeightRange, index);
    }

    {
      std::lock_guard<std::mutex> lock(hole_fill_mutex_);
      if (hole_fill_request_pending_ || hole_fill_work_in_progress_ ||
          hole_fill_result_ready_) {
        return;
      }
      pending_hole_fill_snapshot_ = std::move(snapshot);
      hole_fill_request_pending_ = true;
      last_hole_fill_update_seconds_ = stamp_seconds;
    }
    hole_fill_condition_.notify_one();
  }

  void consumeHoleFillResult() {
    if (!hole_filling_enabled_) {
      return;
    }

    HoleFillWorkResult result;
    {
      std::lock_guard<std::mutex> lock(hole_fill_mutex_);
      if (!hole_fill_result_ready_) {
        return;
      }
      result = std::move(completed_hole_fill_result_);
      hole_fill_result_ready_ = false;
    }

    if (result.map_revision != map_revision_ ||
        result.ground_height.size() != cell_histories_.size() ||
        result.height_range.size() != cell_histories_.size()) {
      return;
    }

    inferred_ground_height_ = std::move(result.ground_height);
    inferred_height_range_ = std::move(result.height_range);
    ROS_INFO_STREAM_ONCE(
        "Asynchronous elevation hole filling initialized: cells="
        << result.filled_cell_count << ", worker="
        << result.computation_ms << " ms");
    ROS_DEBUG_STREAM_THROTTLE(
        1.0,
        "Asynchronous hole fill completed: cells="
        << result.filled_cell_count << ", worker="
        << result.computation_ms << " ms");
  }

  void holeFillWorkerLoop() {
    while (true) {
      HoleFillSnapshot snapshot;
      {
        std::unique_lock<std::mutex> lock(hole_fill_mutex_);
        hole_fill_condition_.wait(
            lock,
            [this]() {
              return stop_hole_fill_worker_ ||
                     hole_fill_request_pending_;
            });
        if (stop_hole_fill_worker_) {
          return;
        }
        snapshot = std::move(pending_hole_fill_snapshot_);
        hole_fill_request_pending_ = false;
        hole_fill_work_in_progress_ = true;
      }

      const ros::WallTime start_time = ros::WallTime::now();
      HoleFillWorkResult result = computeHoleFill(snapshot);
      result.computation_ms =
          (ros::WallTime::now() - start_time).toSec() * 1000.0;

      {
        std::lock_guard<std::mutex> lock(hole_fill_mutex_);
        hole_fill_work_in_progress_ = false;
        if (stop_hole_fill_worker_) {
          return;
        }
        completed_hole_fill_result_ = std::move(result);
        hole_fill_result_ready_ = true;
      }
    }
  }

  HoleFillWorkResult computeHoleFill(
      const HoleFillSnapshot& snapshot) const {
    HoleFillWorkResult result;
    result.map_revision = snapshot.map_revision;
    const grid_map::Size map_size = snapshot.map_size;
    const grid_map::Index buffer_start = snapshot.buffer_start;
    const std::size_t cell_count = snapshot.measured.size();
    result.ground_height.resize(cell_count, quietNaN());
    result.height_range.resize(cell_count, quietNaN());

    const int radius = hole_filler_.parameters().radius_cells;
    const std::size_t maximum_neighbors =
        static_cast<std::size_t>((2 * radius + 1) *
                                 (2 * radius + 1) - 1);
    const auto linear_index = [&map_size](const grid_map::Index& index) {
      return static_cast<std::size_t>(index(0)) *
                 static_cast<std::size_t>(map_size(1)) +
             static_cast<std::size_t>(index(1));
    };
    std::vector<unsigned char> is_candidate(cell_count, 0U);
    std::vector<std::size_t> candidates;
    candidates.reserve(cell_count / 2U);

    // Generate candidates only around directly measured cells. This avoids
    // running a neighborhood query for every empty cell in a fine map.
    for (int row = 0; row < map_size(0); ++row) {
      for (int column = 0; column < map_size(1); ++column) {
        const grid_map::Index measured_index(row, column);
        if (snapshot.measured[linear_index(measured_index)] == 0U) {
          continue;
        }
        const grid_map::Index measured_unwrapped =
            grid_map::getIndexFromBufferIndex(
                measured_index, map_size, buffer_start);
        for (int x_offset = -radius;
             x_offset <= radius; ++x_offset) {
          for (int y_offset = -radius;
               y_offset <= radius; ++y_offset) {
            if (x_offset == 0 && y_offset == 0) {
              continue;
            }
            const grid_map::Index candidate_unwrapped =
                measured_unwrapped +
                grid_map::Index(x_offset, y_offset);
            if (!grid_map::checkIfIndexInRange(
                    candidate_unwrapped, map_size)) {
              continue;
            }
            const grid_map::Index candidate_index =
                grid_map::getBufferIndexFromIndex(
                    candidate_unwrapped, map_size, buffer_start);
            if (snapshot.measured[linear_index(candidate_index)] != 0U) {
              continue;
            }
            const std::size_t candidate_linear_index =
                static_cast<std::size_t>(candidate_index(0)) *
                    static_cast<std::size_t>(map_size(1)) +
                static_cast<std::size_t>(candidate_index(1));
            if (is_candidate[candidate_linear_index] == 0U) {
              is_candidate[candidate_linear_index] = 1U;
              candidates.push_back(candidate_linear_index);
            }
          }
        }
      }
    }

    std::vector<ElevationNeighbor> neighbors;
    neighbors.reserve(maximum_neighbors);
    std::vector<float> ground_height_scratch;
    ground_height_scratch.reserve(maximum_neighbors);
    for (const std::size_t candidate : candidates) {
      const std::size_t columns =
          static_cast<std::size_t>(map_size(1));
      const grid_map::Index index(
          static_cast<int>(candidate / columns),
          static_cast<int>(candidate % columns));
      const grid_map::Index unwrapped_index =
          grid_map::getIndexFromBufferIndex(
              index, map_size, buffer_start);

      neighbors.clear();
      for (int x_offset = -radius;
           x_offset <= radius; ++x_offset) {
        for (int y_offset = -radius;
             y_offset <= radius; ++y_offset) {
          if (x_offset == 0 && y_offset == 0) {
            continue;
          }

          const grid_map::Index neighbor_unwrapped =
              unwrapped_index + grid_map::Index(x_offset, y_offset);
          if (!grid_map::checkIfIndexInRange(
                  neighbor_unwrapped, map_size)) {
            continue;
          }
          const grid_map::Index neighbor_index =
              grid_map::getBufferIndexFromIndex(
                  neighbor_unwrapped, map_size, buffer_start);
          const std::size_t neighbor_linear_index =
              linear_index(neighbor_index);
          if (snapshot.measured[neighbor_linear_index] == 0U) {
            continue;
          }

          neighbors.push_back({
              x_offset,
              y_offset,
              snapshot.ground_height[neighbor_linear_index],
              snapshot.height_range[neighbor_linear_index],
          });
        }
      }

      const ElevationHoleFillResult filled =
          hole_filler_.fill(neighbors, ground_height_scratch);
      if (!filled.filled) {
        continue;
      }
      result.ground_height[candidate] = filled.ground_height;
      result.height_range[candidate] = filled.height_range;
      ++result.filled_cell_count;
    }
    return result;
  }

  void updateMapIncrementally(
      const pcl::PointCloud<pcl::PointXYZ>& cloud,
      const nav_msgs::OdometryConstPtr& odometry,
      const double stamp_seconds) {
    const grid_map::Size map_size = map_.getSize();
    const std::size_t column_count =
        static_cast<std::size_t>(map_size(0)) *
        static_cast<std::size_t>(map_size(1));
    frame_height_samples_.clear();
    if (frame_height_samples_.capacity() < cloud.points.size()) {
      frame_height_samples_.reserve(cloud.points.size());
    }

    const double robot_x = odometry->pose.pose.position.x;
    const double robot_y = odometry->pose.pose.position.y;
    const double robot_z = odometry->pose.pose.position.z;
    const double half_crop_x = 0.5 * input_crop_length_x_;
    const double half_crop_y = 0.5 * input_crop_length_y_;
    const double minimum_z =
        robot_z + projector_.parameters().vertical_min_offset;
    const double maximum_z =
        robot_z + projector_.parameters().vertical_max_offset;
    const grid_map::Position sensor_origin =
        sensorOriginInMap(*odometry);

    const int crop_cells_x = std::max(
        1, static_cast<int>(std::ceil(
               input_crop_length_x_ / raycast_resolution_)));
    const int crop_cells_y = std::max(
        1, static_cast<int>(std::ceil(
               input_crop_length_y_ / raycast_resolution_)));
    const std::size_t ray_endpoint_cell_count =
        static_cast<std::size_t>(crop_cells_x) *
        static_cast<std::size_t>(crop_cells_y);
    if (ray_endpoint_seen_.size() != ray_endpoint_cell_count) {
      ray_endpoint_seen_.resize(ray_endpoint_cell_count);
    }
    std::fill(
        ray_endpoint_seen_.begin(), ray_endpoint_seen_.end(), 0U);
    if (ray_observed_this_frame_.size() != column_count) {
      ray_observed_this_frame_.resize(column_count);
    }
    std::fill(
        ray_observed_this_frame_.begin(),
        ray_observed_this_frame_.end(),
        0U);

    std::size_t cropped_point_count = 0U;
    std::size_t integrated_point_count = 0U;
    std::size_t unique_ray_count = 0U;
    std::size_t ray_observed_cell_count = 0U;
    for (const pcl::PointXYZ& point : cloud.points) {
      if (!std::isfinite(point.x) ||
          !std::isfinite(point.y) ||
          !std::isfinite(point.z) ||
          std::abs(point.x - robot_x) > half_crop_x ||
          std::abs(point.y - robot_y) > half_crop_y ||
          point.z < minimum_z || point.z > maximum_z) {
        continue;
      }
      ++cropped_point_count;

      const int crop_x = std::min(
          crop_cells_x - 1,
          std::max(
              0,
              static_cast<int>(std::floor(
                  (point.x - (robot_x - half_crop_x)) /
                  raycast_resolution_))));
      const int crop_y = std::min(
          crop_cells_y - 1,
          std::max(
              0,
              static_cast<int>(std::floor(
                  (point.y - (robot_y - half_crop_y)) /
                  raycast_resolution_))));
      const std::size_t crop_index =
          static_cast<std::size_t>(crop_x) *
              static_cast<std::size_t>(crop_cells_y) +
          static_cast<std::size_t>(crop_y);
      if (ray_endpoint_seen_[crop_index] == 0U) {
        ray_endpoint_seen_[crop_index] = 1U;
        ++unique_ray_count;
        ray_observed_cell_count += markRayObserved(
            sensor_origin,
            grid_map::Position(point.x, point.y),
            stamp_seconds,
            ray_observed_this_frame_);
      }

      grid_map::Index index;
      if (!map_.getIndex(
              grid_map::Position(point.x, point.y), index)) {
        continue;
      }

      const std::size_t linear_index = historyIndex(index);
      frame_height_samples_.push_back({linear_index, point.z});
      ++integrated_point_count;
    }

    std::sort(
        frame_height_samples_.begin(),
        frame_height_samples_.end(),
        [](const IndexedHeightSample& left,
           const IndexedHeightSample& right) {
          return left.cell_index < right.cell_index;
        });

    std::size_t updated_cell_count = 0U;
    std::size_t sample_begin = 0U;
    while (sample_begin < frame_height_samples_.size()) {
      const std::size_t linear_index =
          frame_height_samples_[sample_begin].cell_index;
      frame_cell_heights_.clear();
      std::size_t sample_end = sample_begin;
      while (sample_end < frame_height_samples_.size() &&
             frame_height_samples_[sample_end].cell_index == linear_index) {
        frame_cell_heights_.push_back(
            frame_height_samples_[sample_end].height);
        ++sample_end;
      }
      const ElevationCell frame_measurement =
          projector_.project(frame_cell_heights_);
      if (!frame_measurement.observed) {
        sample_begin = sample_end;
        continue;
      }

      IncrementalElevationCell& history =
          cell_histories_[linear_index];
      history.add(
          frame_measurement, stamp_seconds, history_length_);
      const grid_map::Index index = indexFromHistory(linear_index);
      writeFusedCell(index, history.fused());
      ++updated_cell_count;
      sample_begin = sample_end;
    }

    ROS_INFO_STREAM_ONCE(
        "Published first rolling elevation map. layers=["
        << kGroundHeight << ", " << kHeightRange << ", "
        << kObservedMask << "], 15x15_cropped_points="
        << cropped_point_count << ", integrated_points="
        << integrated_point_count << ", unique_rays="
        << unique_ray_count << ", ray_observed_cells="
        << ray_observed_cell_count);
    ROS_DEBUG_STREAM_THROTTLE(
        1.0,
        "Rolling elevation update: crop=" << cropped_point_count
        << "/" << cloud.size()
        << ", integrated=" << integrated_point_count
        << ", rays=" << unique_ray_count
        << ", ray_cells=" << ray_observed_cell_count
        << ", updated_cells=" << updated_cell_count);
  }

  grid_map::Position sensorOriginInMap(
      const nav_msgs::Odometry& odometry) const {
    const auto& orientation = odometry.pose.pose.orientation;
    Eigen::Quaterniond rotation(
        orientation.w,
        orientation.x,
        orientation.y,
        orientation.z);
    if (rotation.norm() < 1e-6) {
      rotation = Eigen::Quaterniond::Identity();
    } else {
      rotation.normalize();
    }

    const Eigen::Vector3d body_origin(
        odometry.pose.pose.position.x,
        odometry.pose.pose.position.y,
        odometry.pose.pose.position.z);
    const Eigen::Vector3d sensor_offset(
        sensor_origin_in_body_[0],
        sensor_origin_in_body_[1],
        sensor_origin_in_body_[2]);
    const Eigen::Vector3d sensor_origin =
        body_origin + rotation * sensor_offset;
    return grid_map::Position(sensor_origin.x(), sensor_origin.y());
  }

  std::size_t markRayObserved(
      const grid_map::Position& sensor_origin,
      const grid_map::Position& endpoint,
      const double stamp_seconds,
      std::vector<unsigned char>& observed_this_frame) {
    std::size_t newly_observed_count = 0U;
    try {
      for (grid_map::LineIterator iterator(
               map_, sensor_origin, endpoint);
           !iterator.isPastEnd(); ++iterator) {
        const grid_map::Index index = *iterator;
        const std::size_t linear_index = historyIndex(index);
        if (observed_this_frame[linear_index] != 0U) {
          continue;
        }
        observed_this_frame[linear_index] = 1U;
        ++newly_observed_count;
        last_ray_observed_seconds_[linear_index] = stamp_seconds;
        map_.at(kObservedMask, index) = 1.0F;
      }
    } catch (const std::invalid_argument&) {
      // The sensor is normally inside the rolling map. A line can fail to
      // intersect only during a discontinuous pose jump; skip that ray.
    }
    return newly_observed_count;
  }

  void writeFusedCell(
      const grid_map::Index& index,
      const ElevationCell& elevation) {
    if (!elevation.observed) {
      clearElevation(index);
      refreshObservedMask(index);
      return;
    }
    map_.at(kGroundHeight, index) = elevation.ground_height;
    map_.at(kHeightRange, index) = elevation.height_range;
    map_.at(kObservedMask, index) = 1.0F;
    const std::size_t linear_index = historyIndex(index);
    inferred_ground_height_[linear_index] = quietNaN();
    inferred_height_range_[linear_index] = quietNaN();
  }

  void refreshObservedMask(const grid_map::Index& index) {
    const std::size_t linear_index = historyIndex(index);
    const bool elevation_observed =
        !cell_histories_[linear_index].empty();
    const bool ray_observed =
        std::isfinite(last_ray_observed_seconds_[linear_index]);
    map_.at(kObservedMask, index) =
        (elevation_observed || ray_observed) ? 1.0F : 0.0F;
  }

  std::size_t historyIndex(const grid_map::Index& index) const {
    return static_cast<std::size_t>(index(0)) *
               static_cast<std::size_t>(map_.getSize()(1)) +
           static_cast<std::size_t>(index(1));
  }

  grid_map::Index indexFromHistory(
      const std::size_t linear_index) const {
    const std::size_t columns =
        static_cast<std::size_t>(map_.getSize()(1));
    return grid_map::Index(
        static_cast<int>(linear_index / columns),
        static_cast<int>(linear_index % columns));
  }

  bool frameMatches(const std::string& frame) const {
    return frame.empty() ||
           normalizeFrame(frame) == normalizeFrame(map_frame_);
  }

  ros::NodeHandle node_;
  ros::NodeHandle private_node_;
  ElevationProjector projector_;
  ElevationHoleFillParameters hole_fill_parameters_;
  ElevationHoleFiller hole_filler_;
  GroundReferenceParameters ground_reference_parameters_;
  GroundReferenceEstimator ground_reference_estimator_;
  grid_map::GridMap map_;
  std::vector<IncrementalElevationCell> cell_histories_;
  std::vector<double> last_ray_observed_seconds_;
  std::vector<float> inferred_ground_height_;
  std::vector<float> inferred_height_range_;
  std::vector<unsigned char> ray_endpoint_seen_;
  std::vector<unsigned char> ray_observed_this_frame_;
  std::vector<IndexedHeightSample> frame_height_samples_;
  std::vector<float> frame_cell_heights_;
  std::vector<float> ground_reference_samples_;

  std::thread hole_fill_worker_;
  std::mutex hole_fill_mutex_;
  std::condition_variable hole_fill_condition_;
  HoleFillSnapshot pending_hole_fill_snapshot_;
  HoleFillWorkResult completed_hole_fill_result_;
  bool stop_hole_fill_worker_{false};
  bool hole_fill_request_pending_{false};
  bool hole_fill_work_in_progress_{false};
  bool hole_fill_result_ready_{false};
  std::uint64_t map_revision_{0U};

  ros::Publisher grid_map_publisher_;
  ros::Publisher ground_reference_publisher_;
  std::unique_ptr<CloudSubscriber> cloud_subscriber_;
  std::unique_ptr<OdometrySubscriber> odometry_subscriber_;
  std::unique_ptr<Synchronizer> synchronizer_;

  std::string map_frame_{"odom"};
  double map_length_x_{10.0};
  double map_length_y_{10.0};
  double map_resolution_{0.20};
  double input_crop_length_x_{15.0};
  double input_crop_length_y_{15.0};
  std::vector<double> sensor_origin_in_body_{0.171, 0.0, 0.0908};
  double imu_to_ground_height_{0.35};
  double ground_reference_radius_{1.0};
  float ground_reference_maximum_cell_range_{0.15F};
  std::size_t history_length_{5U};
  double elevation_stale_after_{10.0};
  double ray_stale_after_{1.0};
  double robot_history_keep_radius_{0.8};
  std::size_t expiration_cells_per_frame_{8000U};
  std::size_t expiration_cursor_{0U};
  double raycast_resolution_{0.10};
  bool hole_filling_enabled_{true};
  double hole_fill_update_interval_{0.5};
  double last_hole_fill_update_seconds_{
      std::numeric_limits<double>::quiet_NaN()};
  double last_stamp_seconds_{
      std::numeric_limits<double>::quiet_NaN()};
  int sync_queue_size_{10};
  int subscriber_queue_size_{5};
  double sync_tolerance_{0.05};
  double slow_processing_threshold_{50.0};

  std::string cloud_topic_{"/cloud_registered"};
  std::string odometry_topic_{"/Odometry"};
  std::string grid_map_topic_{"/local_environment/grid_map"};
  std::string ground_reference_topic_{
      "/local_environment/ground_reference"};
};

}  // namespace go2w_local_environment

int main(int argc, char** argv) {
  ros::init(argc, argv, "local_environment");

  try {
    go2w_local_environment::LocalEnvironmentNode node;
    ros::spin();
  } catch (const std::exception& exception) {
    ROS_FATAL_STREAM(
        "Failed to start local environment node: "
        << exception.what());
    return 1;
  }

  return 0;
}
