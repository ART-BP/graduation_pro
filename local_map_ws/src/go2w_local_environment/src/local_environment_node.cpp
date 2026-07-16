#include "go2w_local_environment/dual_layer_projector.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <grid_map_core/GridMap.hpp>
#include <grid_map_msgs/GridMap.h>
#include <grid_map_ros/GridMapRosConverter.hpp>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Odometry.h>
#include <pcl/common/point_tests.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

namespace go2w_local_environment {
namespace {

constexpr char kGroundHeight[] = "ground_height";
constexpr char kCeilingHeight[] = "ceiling_height";
constexpr char kClearance[] = "clearance";
constexpr char kBlocked[] = "blocked";
constexpr char kObserved[] = "observed";
constexpr char kGroundObserved[] = "ground_observed";
constexpr char kCeilingObserved[] = "ceiling_observed";
constexpr char kUnknownFraction[] = "unknown_fraction";
constexpr char kOccupiedCount[] = "occupied_count";

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

}  // namespace

class LocalEnvironmentNode {
 public:
  LocalEnvironmentNode()
      : node_(),
        private_node_("~"),
        projector_(loadProjectionParameters()) {
    loadNodeParameters();

    grid_map_publisher_ =
        node_.advertise<grid_map_msgs::GridMap>(grid_map_topic_, 1);
    blocked_map_publisher_ =
        node_.advertise<nav_msgs::OccupancyGrid>(blocked_map_topic_, 1);

    odometry_subscriber_ = node_.subscribe(
        odometry_topic_,
        5,
        &LocalEnvironmentNode::odometryCallback,
        this);
    occupied_subscriber_ = node_.subscribe(
        occupied_cloud_topic_,
        1,
        &LocalEnvironmentNode::occupiedCloudCallback,
        this);

    if (use_unknown_cloud_) {
      unknown_subscriber_ = node_.subscribe(
          unknown_cloud_topic_,
          1,
          &LocalEnvironmentNode::unknownCloudCallback,
          this);
    }

    projection_timer_ = node_.createTimer(
        ros::Duration(1.0 / publish_rate_),
        &LocalEnvironmentNode::projectionTimerCallback,
        this);

    ROS_INFO_STREAM(
        "Local environment projector started. occupied="
        << occupied_cloud_topic_ << ", unknown="
        << (use_unknown_cloud_ ? unknown_cloud_topic_ : "disabled")
        << ", odometry=" << odometry_topic_);
  }

 private:
  ProjectionParameters loadProjectionParameters() {
    ProjectionParameters parameters;
    loadParameter(
        private_node_,
        "projection/voxel_resolution",
        parameters.voxel_resolution);
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
        "projection/ground_search_min_offset",
        parameters.ground_search_min_offset);
    loadParameter(
        private_node_,
        "projection/ground_search_max_offset",
        parameters.ground_search_max_offset);
    loadParameter(
        private_node_,
        "projection/nominal_ground_offset",
        parameters.nominal_ground_offset);
    loadParameter(
        private_node_,
        "projection/minimum_clearance",
        parameters.minimum_clearance);
    loadParameter(
        private_node_,
        "projection/maximum_unknown_fraction",
        parameters.maximum_unknown_fraction);
    loadParameter(
        private_node_,
        "projection/unknown_is_blocked",
        parameters.unknown_is_blocked);
    return parameters;
  }

  void loadNodeParameters() {
    loadParameter(private_node_, "map/frame_id", map_frame_);
    loadParameter(private_node_, "map/length_x", map_length_x_);
    loadParameter(private_node_, "map/length_y", map_length_y_);
    loadParameter(private_node_, "map/resolution", map_resolution_);
    loadParameter(private_node_, "publish_rate", publish_rate_);
    loadParameter(
        private_node_,
        "maximum_input_age",
        maximum_input_age_);
    loadParameter(
        private_node_,
        "use_unknown_cloud",
        use_unknown_cloud_);

    loadParameter(
        private_node_,
        "topics/odometry",
        odometry_topic_);
    loadParameter(
        private_node_,
        "topics/occupied_cloud",
        occupied_cloud_topic_);
    loadParameter(
        private_node_,
        "topics/unknown_cloud",
        unknown_cloud_topic_);
    loadParameter(
        private_node_,
        "topics/grid_map",
        grid_map_topic_);
    loadParameter(
        private_node_,
        "topics/blocked_map",
        blocked_map_topic_);

    if (map_length_x_ <= 0.0 || map_length_y_ <= 0.0 ||
        map_resolution_ <= 0.0) {
      throw std::runtime_error(
          "map lengths and resolution must be positive");
    }
    if (publish_rate_ <= 0.0) {
      throw std::runtime_error("publish_rate must be positive");
    }
    if (maximum_input_age_ <= 0.0) {
      throw std::runtime_error("maximum_input_age must be positive");
    }
  }

  void odometryCallback(const nav_msgs::OdometryConstPtr& message) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    latest_odometry_ = message;
    latest_odometry_received_ = ros::WallTime::now();
  }

  void occupiedCloudCallback(
      const sensor_msgs::PointCloud2ConstPtr& message) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    latest_occupied_cloud_ = message;
    latest_occupied_received_ = ros::WallTime::now();
  }

  void unknownCloudCallback(
      const sensor_msgs::PointCloud2ConstPtr& message) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    latest_unknown_cloud_ = message;
    latest_unknown_received_ = ros::WallTime::now();
  }

  void projectionTimerCallback(const ros::TimerEvent&) {
    nav_msgs::OdometryConstPtr odometry;
    sensor_msgs::PointCloud2ConstPtr occupied_message;
    sensor_msgs::PointCloud2ConstPtr unknown_message;
    ros::WallTime odometry_received;
    ros::WallTime occupied_received;
    ros::WallTime unknown_received;

    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      odometry = latest_odometry_;
      occupied_message = latest_occupied_cloud_;
      unknown_message = latest_unknown_cloud_;
      odometry_received = latest_odometry_received_;
      occupied_received = latest_occupied_received_;
      unknown_received = latest_unknown_received_;
    }

    if (!odometry || !occupied_message) {
      ROS_WARN_THROTTLE(
          5.0,
          "Waiting for odometry and occupied voxel cloud.");
      return;
    }

    const ros::WallTime now = ros::WallTime::now();
    if (!isFresh(odometry_received, now) ||
        !isFresh(occupied_received, now)) {
      ROS_WARN_THROTTLE(
          5.0,
          "Odometry or occupied cloud is stale; projection skipped.");
      return;
    }

    bool unknown_available = false;
    if (use_unknown_cloud_ && unknown_message) {
      unknown_available = isFresh(unknown_received, now);
    }

    if (use_unknown_cloud_ && !unknown_available) {
      ROS_WARN_THROTTLE(
          5.0,
          "Unknown voxel cloud is unavailable or stale. "
          "Publishing occupied-only projection.");
    }

    if (!frameMatches(odometry->header.frame_id) ||
        !frameMatches(occupied_message->header.frame_id) ||
        (unknown_available &&
         !frameMatches(unknown_message->header.frame_id))) {
      ROS_ERROR_THROTTLE(
          5.0,
          "Input frames do not match map/frame_id. "
          "This first version does not transform input clouds.");
      return;
    }

    pcl::PointCloud<pcl::PointXYZ> occupied_cloud;
    pcl::fromROSMsg(*occupied_message, occupied_cloud);

    pcl::PointCloud<pcl::PointXYZ> unknown_cloud;
    if (unknown_available) {
      pcl::fromROSMsg(*unknown_message, unknown_cloud);
    }

    ros::Time output_stamp = occupied_message->header.stamp;
    if (output_stamp.isZero()) {
      output_stamp = odometry->header.stamp;
    }
    if (output_stamp.isZero()) {
      output_stamp = ros::Time::now();
    }

    publishProjection(
        odometry,
        occupied_cloud,
        unknown_cloud,
        unknown_available,
        output_stamp);
  }

  void publishProjection(
      const nav_msgs::OdometryConstPtr& odometry,
      const pcl::PointCloud<pcl::PointXYZ>& occupied_cloud,
      const pcl::PointCloud<pcl::PointXYZ>& unknown_cloud,
      const bool unknown_available,
      const ros::Time& stamp) {
    grid_map::GridMap map({
        kGroundHeight,
        kCeilingHeight,
        kClearance,
        kBlocked,
        kObserved,
        kGroundObserved,
        kCeilingObserved,
        kUnknownFraction,
        kOccupiedCount,
    });
    map.setFrameId(map_frame_);
    map.setGeometry(
        grid_map::Length(map_length_x_, map_length_y_),
        map_resolution_,
        grid_map::Position(
            odometry->pose.pose.position.x,
            odometry->pose.pose.position.y));
    map.setTimestamp(stamp.toNSec());

    for (const std::string& layer : map.getLayers()) {
      map[layer].setConstant(quietNaN());
    }

    const grid_map::Size size = map.getSize();
    const std::size_t cell_count =
        static_cast<std::size_t>(size(0)) *
        static_cast<std::size_t>(size(1));
    std::vector<std::vector<float>> occupied_columns(cell_count);
    std::vector<std::vector<float>> unknown_columns(cell_count);

    const double robot_z = odometry->pose.pose.position.z;
    addCloudToColumns(
        map,
        occupied_cloud,
        robot_z,
        occupied_columns);
    if (unknown_available) {
      addCloudToColumns(
          map,
          unknown_cloud,
          robot_z,
          unknown_columns);
    }

    for (int row = 0; row < size(0); ++row) {
      for (int column = 0; column < size(1); ++column) {
        const grid_map::Index index(row, column);
        const std::size_t linear_index =
            static_cast<std::size_t>(row) *
                static_cast<std::size_t>(size(1)) +
            static_cast<std::size_t>(column);

        const ColumnProjection projection = projector_.project(
            robot_z,
            occupied_columns[linear_index],
            unknown_columns[linear_index],
            unknown_available);
        writeProjection(map, index, projection);
      }
    }

    grid_map_msgs::GridMap message;
    grid_map::GridMapRosConverter::toMessage(map, message);
    grid_map_publisher_.publish(message);

    nav_msgs::OccupancyGrid blocked_message;
    grid_map::GridMapRosConverter::toOccupancyGrid(
        map,
        kBlocked,
        0.0,
        1.0,
        blocked_message);
    blocked_message.header.stamp = stamp;
    blocked_map_publisher_.publish(blocked_message);
  }

  void addCloudToColumns(
      const grid_map::GridMap& map,
      const pcl::PointCloud<pcl::PointXYZ>& cloud,
      const double robot_z,
      std::vector<std::vector<float>>& columns) const {
    const grid_map::Size size = map.getSize();
    const double minimum_z =
        robot_z + projector_.parameters().vertical_min_offset;
    const double maximum_z =
        robot_z + projector_.parameters().vertical_max_offset;

    for (const pcl::PointXYZ& point : cloud.points) {
      if (!pcl::isFinite(point) ||
          point.z < minimum_z ||
          point.z > maximum_z) {
        continue;
      }

      grid_map::Index index;
      if (!map.getIndex(grid_map::Position(point.x, point.y), index)) {
        continue;
      }

      const std::size_t linear_index =
          static_cast<std::size_t>(index(0)) *
              static_cast<std::size_t>(size(1)) +
          static_cast<std::size_t>(index(1));
      columns[linear_index].push_back(point.z);
    }
  }

  void writeProjection(
      grid_map::GridMap& map,
      const grid_map::Index& index,
      const ColumnProjection& projection) const {
    map.at(kGroundObserved, index) =
        projection.ground_observed ? 1.0F : 0.0F;
    map.at(kCeilingObserved, index) =
        projection.ceiling_observed ? 1.0F : 0.0F;
    map.at(kObserved, index) =
        projection.observed ? 1.0F : 0.0F;
    map.at(kOccupiedCount, index) =
        static_cast<float>(projection.occupied_count);

    if (std::isfinite(projection.unknown_fraction)) {
      map.at(kUnknownFraction, index) =
          static_cast<float>(projection.unknown_fraction);
    }
    if (!projection.ground_observed) {
      return;
    }

    map.at(kGroundHeight, index) =
        static_cast<float>(projection.ground_height);
    map.at(kClearance, index) =
        static_cast<float>(projection.clearance);
    map.at(kBlocked, index) =
        projection.blocked ? 1.0F : 0.0F;

    if (projection.ceiling_observed) {
      map.at(kCeilingHeight, index) =
          static_cast<float>(projection.ceiling_height);
    }
  }

  bool isFresh(
      const ros::WallTime& received,
      const ros::WallTime& now) const {
    if (received.isZero()) {
      return false;
    }
    return (now - received).toSec() <= maximum_input_age_;
  }

  bool frameMatches(const std::string& frame) const {
    if (frame.empty()) {
      return true;
    }
    const std::string normalized_frame =
        frame.front() == '/' ? frame.substr(1) : frame;
    const std::string normalized_map_frame =
        !map_frame_.empty() && map_frame_.front() == '/'
            ? map_frame_.substr(1)
            : map_frame_;
    return normalized_frame == normalized_map_frame;
  }

  ros::NodeHandle node_;
  ros::NodeHandle private_node_;

  DualLayerProjector projector_;

  ros::Subscriber odometry_subscriber_;
  ros::Subscriber occupied_subscriber_;
  ros::Subscriber unknown_subscriber_;
  ros::Publisher grid_map_publisher_;
  ros::Publisher blocked_map_publisher_;
  ros::Timer projection_timer_;

  mutable std::mutex data_mutex_;
  nav_msgs::OdometryConstPtr latest_odometry_;
  sensor_msgs::PointCloud2ConstPtr latest_occupied_cloud_;
  sensor_msgs::PointCloud2ConstPtr latest_unknown_cloud_;
  ros::WallTime latest_odometry_received_;
  ros::WallTime latest_occupied_received_;
  ros::WallTime latest_unknown_received_;

  std::string map_frame_{"odom"};
  double map_length_x_{20.0};
  double map_length_y_{20.0};
  double map_resolution_{0.20};
  double publish_rate_{5.0};
  double maximum_input_age_{1.0};
  bool use_unknown_cloud_{true};

  std::string odometry_topic_{"/Odometry"};
  std::string occupied_cloud_topic_{
      "/rog_map_node/rog_map/occ"};
  std::string unknown_cloud_topic_{
      "/rog_map_node/rog_map/unk"};
  std::string grid_map_topic_{
      "/local_environment/grid_map"};
  std::string blocked_map_topic_{
      "/local_environment/blocked"};
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
