#include <diagnostic_msgs/DiagnosticArray.h>
#include <diagnostic_msgs/DiagnosticStatus.h>
#include <diagnostic_msgs/KeyValue.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/TwistStamped.h>
#include <grid_map_msgs/GridMap.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <visualization_msgs/Marker.h>

#include <boost/bind/bind.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <exception>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "go2w_terrain_planner_ros/grid_map_decoder.hpp"
#include "go2w_terrain_planner_ros/policy_observation_builder.hpp"
#include "go2w_terrain_planner_ros/tensorrt_engine.hpp"

// grid_map_msgs/GridMap keeps its standard ROS header inside `info`, so the
// generated message does not advertise HasHeader. Teach message_filters where
// to obtain the timestamp without copying or wrapping the 7-layer map.
namespace ros {
namespace message_traits {
template <>
struct TimeStamp<grid_map_msgs::GridMap> {
  static ros::Time* pointer(grid_map_msgs::GridMap& message) {
    return &message.info.header.stamp;
  }
  static const ros::Time* pointer(const grid_map_msgs::GridMap& message) {
    return &message.info.header.stamp;
  }
  static ros::Time value(const grid_map_msgs::GridMap& message) {
    return message.info.header.stamp;
  }
};
}  // namespace message_traits
}  // namespace ros

namespace go2w_terrain_planner_ros {
namespace {

std::string normalizedFrame(std::string frame) {
  while (!frame.empty() && frame.front() == '/') {
    frame.erase(frame.begin());
  }
  return frame;
}

float quaternionYaw(const geometry_msgs::Quaternion& quaternion) {
  const double numerator = 2.0 *
      (quaternion.w * quaternion.z + quaternion.x * quaternion.y);
  const double denominator = 1.0 - 2.0 *
      (quaternion.y * quaternion.y + quaternion.z * quaternion.z);
  return static_cast<float>(std::atan2(numerator, denominator));
}

diagnostic_msgs::KeyValue keyValue(
    const std::string& key, const std::string& value) {
  diagnostic_msgs::KeyValue output;
  output.key = key;
  output.value = value;
  return output;
}

std::string number(const double value, const int precision = 4) {
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(precision) << value;
  return stream.str();
}

}  // namespace

class InferenceNode {
 public:
  InferenceNode() : node_(), private_node_("~") {
    std::string onnx_path;
    std::string engine_path;
    bool enable_fp16 = false;
    int workspace_megabytes = 512;
    private_node_.param<std::string>("model/onnx_path", onnx_path, "");
    private_node_.param<std::string>("model/engine_path", engine_path, "");
    private_node_.param("model/enable_fp16", enable_fp16, false);
    private_node_.param("model/workspace_megabytes", workspace_megabytes, 512);
    if (workspace_megabytes <= 0) {
      throw std::invalid_argument("model/workspace_megabytes must be positive");
    }
    ROS_INFO("Loading Go2W planner model (this can take several minutes when building an engine)...");
    engine_.initialize(
        onnx_path, engine_path, enable_fp16,
        static_cast<std::size_t>(workspace_megabytes) * 1024U * 1024U);
    ROS_INFO("Go2W TensorRT planner is ready");

    private_node_.param<std::string>(
        "topics/grid_map", grid_map_topic_, "/local_environment/grid_map");
    private_node_.param<std::string>(
        "topics/ground_reference", ground_reference_topic_,
        "/local_environment/ground_reference");
    private_node_.param<std::string>(
        "topics/odometry", odometry_topic_, "/Odometry");
    private_node_.param<std::string>(
        "topics/goal", goal_topic_, "/move_base_simple/goal");
    private_node_.param<std::string>(
        "topics/cmd_vel", cmd_vel_topic_,
        "/go2w_terrain_planner/cmd_vel_shadow");
    private_node_.param<std::string>(
        "topics/cmd_vel_stamped", cmd_vel_stamped_topic_,
        "/go2w_terrain_planner/cmd_vel_shadow_stamped");
    private_node_.param<std::string>(
        "topics/diagnostics", diagnostics_topic_,
        "/go2w_terrain_planner/diagnostics");
    private_node_.param<std::string>(
        "topics/prediction_marker", marker_topic_,
        "/go2w_terrain_planner/predicted_motion");
    private_node_.param("safety/minimum_observed_ratio",
                        minimum_observed_ratio_, 0.05);
    private_node_.param("safety/minimum_height_valid_ratio",
                        minimum_height_valid_ratio_, 0.01);
    private_node_.param("safety/goal_tolerance_m", goal_tolerance_m_, 0.30);
    private_node_.param("sync/tolerance_s", sync_tolerance_s_, 0.03);
    private_node_.param("sync/queue_size", sync_queue_size_, 10);
    if (minimum_observed_ratio_ < 0.0 || minimum_observed_ratio_ > 1.0 ||
        minimum_height_valid_ratio_ < 0.0 ||
        minimum_height_valid_ratio_ > 1.0 || goal_tolerance_m_ < 0.0 ||
        sync_tolerance_s_ < 0.0 || sync_queue_size_ <= 0) {
      throw std::invalid_argument("Invalid inference-node safety/sync parameters");
    }

    cmd_vel_publisher_ = node_.advertise<geometry_msgs::Twist>(
        cmd_vel_topic_, 1);
    cmd_vel_stamped_publisher_ = node_.advertise<geometry_msgs::TwistStamped>(
        cmd_vel_stamped_topic_, 1);
    diagnostics_publisher_ = node_.advertise<diagnostic_msgs::DiagnosticArray>(
        diagnostics_topic_, 1);
    marker_publisher_ = node_.advertise<visualization_msgs::Marker>(
        marker_topic_, 1);
    goal_subscriber_ = node_.subscribe(
        goal_topic_, 1, &InferenceNode::goalCallback, this);

    map_subscriber_.subscribe(node_, grid_map_topic_, 2);
    ground_reference_subscriber_.subscribe(
        node_, ground_reference_topic_, 2);
    odometry_subscriber_.subscribe(node_, odometry_topic_, 5);
    synchronizer_.reset(new Synchronizer(
        SynchronizationPolicy(sync_queue_size_), map_subscriber_,
        ground_reference_subscriber_, odometry_subscriber_));
    synchronizer_->setMaxIntervalDuration(ros::Duration(sync_tolerance_s_));
    synchronizer_->registerCallback(boost::bind(
        &InferenceNode::synchronizedCallback, this,
        boost::placeholders::_1, boost::placeholders::_2,
        boost::placeholders::_3));
    ROS_INFO_STREAM(
        "Go2W planner shadow output: " << cmd_vel_topic_
        << "; set a goal in frame 'odom' through " << goal_topic_);
  }

 private:
  using SynchronizationPolicy =
      message_filters::sync_policies::ApproximateTime<
          grid_map_msgs::GridMap, geometry_msgs::PointStamped,
          nav_msgs::Odometry>;
  using Synchronizer = message_filters::Synchronizer<SynchronizationPolicy>;

  void goalCallback(const geometry_msgs::PoseStampedConstPtr& message) {
    if (!std::isfinite(message->pose.position.x) ||
        !std::isfinite(message->pose.position.y)) {
      ROS_WARN("Ignoring a non-finite planner goal");
      return;
    }
    goal_xy_ = {{static_cast<float>(message->pose.position.x),
                 static_cast<float>(message->pose.position.y)}};
    goal_frame_ = normalizedFrame(message->header.frame_id);
    goal_available_ = true;
    ROS_INFO("Planner goal set to (%.3f, %.3f) in frame '%s'",
             goal_xy_[0], goal_xy_[1], goal_frame_.c_str());
  }

  void synchronizedCallback(
      const grid_map_msgs::GridMapConstPtr& map,
      const geometry_msgs::PointStampedConstPtr& ground_reference,
      const nav_msgs::OdometryConstPtr& odometry) {
    const auto processing_start = std::chrono::steady_clock::now();
    const ros::Time stamp = map->info.header.stamp;
    if (!last_stamp_.isZero() && stamp <= last_stamp_) {
      if (stamp < last_stamp_) {
        ROS_WARN("ROS time moved backwards; resetting planner histories");
        observation_builder_.clear();
        last_command_ = {{0.0F, 0.0F}};
      } else {
        return;
      }
    }
    last_stamp_ = stamp;

    const std::string map_frame = normalizedFrame(map->info.header.frame_id);
    const std::string odometry_frame = normalizedFrame(odometry->header.frame_id);
    const std::string reference_frame =
        normalizedFrame(ground_reference->header.frame_id);
    Pose2d pose;
    pose.x = static_cast<float>(odometry->pose.pose.position.x);
    pose.y = static_cast<float>(odometry->pose.pose.position.y);
    pose.yaw = quaternionYaw(odometry->pose.pose.orientation);
    std::string failure_reason;
    std::uint8_t diagnostic_level = diagnostic_msgs::DiagnosticStatus::OK;

    if (map_frame.empty() || map_frame != odometry_frame ||
        map_frame != reference_frame) {
      failure_reason = "map, ground reference, and odometry frames differ";
      diagnostic_level = diagnostic_msgs::DiagnosticStatus::ERROR;
      publishResult(stamp, map_frame, pose, {{0.0F, 0.0F}}, false,
                    ObservationStatistics(), 0.0, 0.0,
                    diagnostic_level, failure_reason);
      last_command_ = {{0.0F, 0.0F}};
      return;
    }

    FourChannelMap raw_map;
    if (!decoder_.decodeRobotCentric(*map, pose, raw_map, failure_reason)) {
      diagnostic_level = diagnostic_msgs::DiagnosticStatus::ERROR;
      publishResult(stamp, map_frame, pose, {{0.0F, 0.0F}}, false,
                    ObservationStatistics(), 0.0, 0.0,
                    diagnostic_level, failure_reason);
      last_command_ = {{0.0F, 0.0F}};
      return;
    }

    const bool goal_frame_valid = goal_available_ &&
        (goal_frame_.empty() || goal_frame_ == map_frame);
    const std::array<float, 2> effective_goal = goal_frame_valid
        ? goal_xy_ : std::array<float, 2>{{pose.x, pose.y}};
    const std::array<float, 2> current_velocity{{
        static_cast<float>(odometry->twist.twist.linear.x),
        static_cast<float>(odometry->twist.twist.angular.z),
    }};
    ObservationStatistics statistics;
    std::vector<float> observation;
    try {
      observation = observation_builder_.update(
          raw_map, pose, static_cast<float>(ground_reference->point.z),
          last_command_, current_velocity, effective_goal, &statistics);
    } catch (const std::exception& exception) {
      failure_reason = std::string("observation assembly failed: ") +
                       exception.what();
      diagnostic_level = diagnostic_msgs::DiagnosticStatus::ERROR;
      publishResult(stamp, map_frame, pose, {{0.0F, 0.0F}}, false,
                    statistics, 0.0, elapsedMs(processing_start),
                    diagnostic_level, failure_reason);
      last_command_ = {{0.0F, 0.0F}};
      return;
    }

    bool safe = true;
    if (!goal_available_) {
      safe = false;
      failure_reason = "waiting for a goal";
    } else if (!goal_frame_valid) {
      safe = false;
      failure_reason = "goal frame does not match map frame";
    } else if (statistics.observed_ratio < minimum_observed_ratio_) {
      safe = false;
      failure_reason = "observed map ratio is below threshold";
    } else if (statistics.height_valid_ratio <
               minimum_height_valid_ratio_) {
      safe = false;
      failure_reason = "valid-height map ratio is below threshold";
    } else if (statistics.goal_distance_m <= goal_tolerance_m_) {
      safe = false;
      failure_reason = "goal reached";
    }

    std::array<float, 2> command{{0.0F, 0.0F}};
    double inference_ms = 0.0;
    if (safe) {
      try {
        const auto inference_start = std::chrono::steady_clock::now();
        command = engine_.infer(observation);
        inference_ms = elapsedMs(inference_start);
        if (!std::isfinite(command[0]) || !std::isfinite(command[1]) ||
            command[0] < -0.2001F || command[0] > 1.2001F ||
            command[1] < -1.0001F || command[1] > 1.0001F) {
          safe = false;
          failure_reason = "model output is non-finite or outside action limits";
          command = {{0.0F, 0.0F}};
          diagnostic_level = diagnostic_msgs::DiagnosticStatus::ERROR;
        } else {
          command[0] = std::max(-0.2F, std::min(command[0], 1.2F));
          command[1] = std::max(-1.0F, std::min(command[1], 1.0F));
        }
      } catch (const std::exception& exception) {
        safe = false;
        command = {{0.0F, 0.0F}};
        failure_reason = std::string("TensorRT inference failed: ") +
                         exception.what();
        diagnostic_level = diagnostic_msgs::DiagnosticStatus::ERROR;
      }
    }
    if (!safe && diagnostic_level == diagnostic_msgs::DiagnosticStatus::OK) {
      diagnostic_level = diagnostic_msgs::DiagnosticStatus::WARN;
    }
    last_command_ = command;
    publishResult(
        stamp, map_frame, pose, command, safe, statistics, inference_ms,
        elapsedMs(processing_start), diagnostic_level,
        safe ? "shadow inference active" : failure_reason);
  }

  template <typename TimePoint>
  static double elapsedMs(const TimePoint& start) {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
  }

  void publishResult(
      const ros::Time& stamp,
      const std::string& frame,
      const Pose2d& pose,
      const std::array<float, 2>& command,
      const bool safe,
      const ObservationStatistics& statistics,
      const double inference_ms,
      const double processing_ms,
      const std::uint8_t diagnostic_level,
      const std::string& message) {
    geometry_msgs::Twist twist;
    twist.linear.x = command[0];
    twist.angular.z = command[1];
    cmd_vel_publisher_.publish(twist);
    geometry_msgs::TwistStamped stamped;
    stamped.header.stamp = stamp;
    stamped.header.frame_id = frame;
    stamped.twist = twist;
    cmd_vel_stamped_publisher_.publish(stamped);

    diagnostic_msgs::DiagnosticArray diagnostics;
    diagnostics.header.stamp = stamp;
    diagnostic_msgs::DiagnosticStatus status;
    status.level = diagnostic_level;
    status.name = "go2w_terrain_planner/shadow_inference";
    status.hardware_id = "go2w_tensorrt";
    status.message = message;
    status.values.push_back(keyValue("safe_to_move", safe ? "true" : "false"));
    status.values.push_back(keyValue(
        "observed_ratio", number(statistics.observed_ratio)));
    status.values.push_back(keyValue(
        "height_valid_ratio", number(statistics.height_valid_ratio)));
    status.values.push_back(keyValue(
        "goal_distance_m", number(statistics.goal_distance_m)));
    status.values.push_back(keyValue("linear_command_mps", number(command[0])));
    status.values.push_back(keyValue("angular_command_radps", number(command[1])));
    status.values.push_back(keyValue("inference_ms", number(inference_ms, 2)));
    status.values.push_back(keyValue("processing_ms", number(processing_ms, 2)));
    diagnostics.status.push_back(status);
    diagnostics_publisher_.publish(diagnostics);
    publishPrediction(stamp, frame, pose, command, safe);
  }

  void publishPrediction(
      const ros::Time& stamp,
      const std::string& frame,
      const Pose2d& pose,
      const std::array<float, 2>& command,
      const bool safe) {
    visualization_msgs::Marker marker;
    marker.header.stamp = stamp;
    marker.header.frame_id = frame;
    marker.ns = "go2w_policy_shadow";
    marker.id = 0;
    marker.type = visualization_msgs::Marker::LINE_STRIP;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.05;
    marker.color.a = 1.0;
    marker.color.r = safe ? 0.1F : 1.0F;
    marker.color.g = safe ? 1.0F : 0.1F;
    marker.color.b = 0.1F;
    marker.lifetime = ros::Duration(0.3);
    float x = pose.x;
    float y = pose.y;
    float yaw = pose.yaw;
    constexpr float kDt = 0.1F;
    for (int step = 0; step <= 20; ++step) {
      geometry_msgs::Point point;
      point.x = x;
      point.y = y;
      point.z = 0.10;
      marker.points.push_back(point);
      x += command[0] * std::cos(yaw) * kDt;
      y += command[0] * std::sin(yaw) * kDt;
      yaw += command[1] * kDt;
    }
    marker_publisher_.publish(marker);
  }

  ros::NodeHandle node_;
  ros::NodeHandle private_node_;
  std::string grid_map_topic_;
  std::string ground_reference_topic_;
  std::string odometry_topic_;
  std::string goal_topic_;
  std::string cmd_vel_topic_;
  std::string cmd_vel_stamped_topic_;
  std::string diagnostics_topic_;
  std::string marker_topic_;
  double minimum_observed_ratio_{0.05};
  double minimum_height_valid_ratio_{0.01};
  double goal_tolerance_m_{0.30};
  double sync_tolerance_s_{0.03};
  int sync_queue_size_{10};

  TensorRtEngine engine_;
  GridMapDecoder decoder_;
  PolicyObservationBuilder observation_builder_;
  std::array<float, 2> last_command_{{0.0F, 0.0F}};
  std::array<float, 2> goal_xy_{{0.0F, 0.0F}};
  std::string goal_frame_;
  bool goal_available_{false};
  ros::Time last_stamp_;

  ros::Publisher cmd_vel_publisher_;
  ros::Publisher cmd_vel_stamped_publisher_;
  ros::Publisher diagnostics_publisher_;
  ros::Publisher marker_publisher_;
  ros::Subscriber goal_subscriber_;
  message_filters::Subscriber<grid_map_msgs::GridMap> map_subscriber_;
  message_filters::Subscriber<geometry_msgs::PointStamped>
      ground_reference_subscriber_;
  message_filters::Subscriber<nav_msgs::Odometry> odometry_subscriber_;
  std::unique_ptr<Synchronizer> synchronizer_;
};

}  // namespace go2w_terrain_planner_ros

int main(int argc, char** argv) {
  ros::init(argc, argv, "go2w_policy_inference");
  try {
    go2w_terrain_planner_ros::InferenceNode node;
    ros::spin();
  } catch (const std::exception& exception) {
    ROS_FATAL("Go2W inference node failed: %s", exception.what());
    return 1;
  }
  return 0;
}
