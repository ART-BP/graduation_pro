#include "go2w_policy_deployment/observation_builder.hpp"
#include "go2w_policy_deployment/onnx_policy.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <boost/bind.hpp>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Twist.h>
#include <grid_map_core/GridMap.hpp>
#include <grid_map_msgs/GridMap.h>
#include <ros/message_traits.h>

// grid_map_msgs/GridMap keeps its Header inside `info`, so ROS does not infer
// the timestamp trait needed by message_filters automatically.
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

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <std_msgs/String.h>
#include <std_srvs/SetBool.h>
#include <tf2/utils.h>

namespace go2w_policy_deployment {
namespace {

double clampDouble(const double value, const double minimum, const double maximum) {
  return std::max(minimum, std::min(maximum, value));
}

int layerIndex(const grid_map_msgs::GridMap& message, const std::string& name) {
  const auto iterator = std::find(
      message.layers.begin(), message.layers.end(), name);
  if (iterator == message.layers.end()) {
    throw std::runtime_error("grid map is missing layer: " + name);
  }
  return static_cast<int>(std::distance(message.layers.begin(), iterator));
}

grid_map::Matrix matrixFromMessage(
    const std_msgs::Float32MultiArray& message) {
  if (message.layout.dim.size() != 2U) {
    throw std::runtime_error("grid-map layer must have two layout dimensions");
  }
  const auto& first = message.layout.dim[0];
  const auto& second = message.layout.dim[1];
  const bool column_major =
      first.label == "column_index" && second.label == "row_index";
  const bool row_major =
      first.label == "row_index" && second.label == "column_index";
  if (!column_major && !row_major) {
    throw std::runtime_error("unsupported grid-map layer memory layout");
  }
  const int rows = static_cast<int>(column_major ? second.size : first.size);
  const int columns = static_cast<int>(column_major ? first.size : second.size);
  if (rows <= 0 || columns <= 0 ||
      message.data.size() != static_cast<std::size_t>(rows * columns)) {
    throw std::runtime_error("grid-map layer data size is invalid");
  }
  grid_map::Matrix matrix(rows, columns);
  if (column_major) {
    std::copy(message.data.begin(), message.data.end(), matrix.data());
  } else {
    for (int row = 0; row < rows; ++row) {
      for (int column = 0; column < columns; ++column) {
        matrix(row, column) = message.data[
            static_cast<std::size_t>(row) * columns + column];
      }
    }
  }
  return matrix;
}

grid_map::GridMap gridMapFromMessage(
    const grid_map_msgs::GridMap& message) {
  if (message.info.resolution <= 0.0 || message.info.length_x <= 0.0 ||
      message.info.length_y <= 0.0) {
    throw std::runtime_error("grid-map geometry is invalid");
  }
  const auto& orientation = message.info.pose.orientation;
  if (std::abs(orientation.x) > 1.0e-6 ||
      std::abs(orientation.y) > 1.0e-6 ||
      std::abs(orientation.z) > 1.0e-6 ||
      std::abs(orientation.w - 1.0) > 1.0e-6) {
    throw std::runtime_error("rotated GridMapInfo poses are not supported");
  }

  const std::vector<std::string> required_layers{
      "ground_height", "height_range", "observed_mask"};
  grid_map::GridMap map(required_layers);
  map.setGeometry(
      grid_map::Length(message.info.length_x, message.info.length_y),
      message.info.resolution,
      grid_map::Position(
          message.info.pose.position.x,
          message.info.pose.position.y));
  map.setFrameId(message.info.header.frame_id);
  map.setTimestamp(message.info.header.stamp.toNSec());
  for (const std::string& layer : required_layers) {
    const int index = layerIndex(message, layer);
    map.add(layer, matrixFromMessage(message.data.at(
        static_cast<std::size_t>(index))));
  }
  map.setStartIndex(grid_map::Index(
      static_cast<int>(message.outer_start_index),
      static_cast<int>(message.inner_start_index)));
  return map;
}

}  // namespace

class PolicyNode {
 public:
  PolicyNode() : private_node_("~") {
    loadParameters();
    observation_builder_.reset(new ObservationBuilder(observation_config_));
    policy_.reset(new OnnxPolicy(
        model_path_, observation_builder_->expectedDimension(),
        intra_op_threads_));

    command_publisher_ = node_.advertise<geometry_msgs::Twist>(command_topic_, 1U);
    safety_publisher_ = private_node_.advertise<std_msgs::Bool>("safety_stop", 1U, true);
    status_publisher_ = private_node_.advertise<std_msgs::String>("status", 1U, true);
    goal_subscriber_ = node_.subscribe(
        goal_topic_, 1U, &PolicyNode::goalCallback, this);
    enable_service_ = private_node_.advertiseService(
        "set_enabled", &PolicyNode::setEnabled, this);

    map_subscriber_.subscribe(node_, map_topic_, subscriber_queue_size_);
    odometry_subscriber_.subscribe(node_, odometry_topic_, subscriber_queue_size_);
    ground_reference_subscriber_.subscribe(
        node_, ground_reference_topic_, subscriber_queue_size_);
    synchronizer_.reset(new Synchronizer(
        SyncPolicy(sync_queue_size_),
        map_subscriber_, odometry_subscriber_, ground_reference_subscriber_));
    synchronizer_->setMaxIntervalDuration(ros::Duration(sync_tolerance_s_));
    synchronizer_->registerCallback(boost::bind(
        &PolicyNode::inputCallback, this, _1, _2, _3));
    watchdog_timer_ = node_.createTimer(
        ros::Duration(std::max(0.05, 0.5 * input_timeout_s_)),
        &PolicyNode::watchdogCallback,
        this);

    publishStop(start_enabled_ ? "waiting_for_inputs" : "disabled", false);
    enabled_ = start_enabled_;
    ROS_INFO_STREAM(
        "Go2W policy loaded from " << model_path_
        << "; expected input dimension="
        << observation_builder_->expectedDimension()
        << "; enabled=" << std::boolalpha << enabled_);
  }

 private:
  using SyncPolicy = message_filters::sync_policies::ApproximateTime<
      grid_map_msgs::GridMap,
      nav_msgs::Odometry,
      geometry_msgs::PointStamped>;
  using Synchronizer = message_filters::Synchronizer<SyncPolicy>;

  template <typename Value>
  void requiredParam(const std::string& name, Value& value) {
    if (!private_node_.getParam(name, value)) {
      throw std::runtime_error("missing required ROS parameter: ~" + name);
    }
  }

  void loadParameters() {
    requiredParam("model/path", model_path_);
    private_node_.param("runtime/intra_op_threads", intra_op_threads_, 1);

    private_node_.param("map/size", observation_config_.map_size, 200);
    private_node_.param("map/channels", observation_config_.map_channels, 7);
    private_node_.param("map/extent_m", observation_config_.map_extent_m, 10.0);
    private_node_.param(
        "map/ground_height_scale_m",
        observation_config_.ground_height_scale_m, 1.0);
    private_node_.param(
        "map/height_range_scale_m",
        observation_config_.height_range_scale_m, 3.0);
    private_node_.param(
        "map/maximum_ground_deviation",
        observation_config_.maximum_ground_deviation, 0.06);
    private_node_.param(
        "map/fusion_span_percentile",
        observation_config_.fusion_span_percentile, 0.75);
    private_node_.param(
        "history/observation_fusion_length",
        observation_config_.observation_fusion_length, 12);
    private_node_.param(
        "history/command_length",
        observation_config_.command_history_length, 8);
    private_node_.param(
        "history/motion_length",
        observation_config_.motion_history_length, 8);
    private_node_.param(
        "goal/maximum_distance_m",
        observation_config_.goal_maximum_distance_m, 10.0);
    private_node_.param(
        "action/linear_velocity_scale_mps",
        observation_config_.linear_velocity_scale_mps, 1.2);
    private_node_.param(
        "action/angular_velocity_scale_radps",
        observation_config_.angular_velocity_scale_radps, 1.0);

    private_node_.param("action/linear_min_mps", linear_min_mps_, -0.2);
    private_node_.param("action/linear_max_mps", linear_max_mps_, 1.2);
    private_node_.param("action/angular_min_radps", angular_min_radps_, -1.0);
    private_node_.param("action/angular_max_radps", angular_max_radps_, 1.0);
    private_node_.param(
        "action/linear_acceleration_limit_mps2",
        linear_acceleration_limit_mps2_, 0.8);
    private_node_.param(
        "action/angular_acceleration_limit_radps2",
        angular_acceleration_limit_radps2_, 1.0);
    private_node_.param(
        "action/apply_acceleration_limits",
        apply_acceleration_limits_, true);

    private_node_.param("safety/start_enabled", start_enabled_, false);
    private_node_.param("safety/input_timeout_s", input_timeout_s_, 0.5);
    private_node_.param("safety/goal_tolerance_m", goal_tolerance_m_, 0.30);
    private_node_.param(
        "safety/minimum_observed_ratio", minimum_observed_ratio_, 0.05);
    private_node_.param(
        "safety/minimum_height_valid_ratio",
        minimum_height_valid_ratio_, 0.01);

    private_node_.param<std::string>(
        "topics/map", map_topic_, "/local_environment/grid_map");
    private_node_.param<std::string>(
        "topics/odometry", odometry_topic_, "/Odometry");
    private_node_.param<std::string>(
        "topics/ground_reference",
        ground_reference_topic_, "/local_environment/ground_reference");
    private_node_.param<std::string>(
        "topics/goal", goal_topic_, "/local_goal");
    private_node_.param<std::string>(
        "topics/command", command_topic_, "/cmd_vel");
    private_node_.param("sync/queue_size", sync_queue_size_, 10);
    private_node_.param("sync/subscriber_queue_size", subscriber_queue_size_, 5);
    private_node_.param("sync/tolerance_s", sync_tolerance_s_, 0.05);

    if (model_path_.empty() || intra_op_threads_ <= 0 ||
        input_timeout_s_ <= 0.0 || goal_tolerance_m_ < 0.0 ||
        minimum_observed_ratio_ < 0.0 || minimum_observed_ratio_ > 1.0 ||
        minimum_height_valid_ratio_ < 0.0 ||
        minimum_height_valid_ratio_ > 1.0 ||
        linear_min_mps_ >= 0.0 || linear_max_mps_ <= 0.0 ||
        angular_min_radps_ >= 0.0 || angular_max_radps_ <= 0.0 ||
        linear_acceleration_limit_mps2_ <= 0.0 ||
        angular_acceleration_limit_radps2_ <= 0.0 ||
        sync_queue_size_ <= 0 || subscriber_queue_size_ <= 0 ||
        sync_tolerance_s_ <= 0.0) {
      throw std::runtime_error("invalid Go2W policy ROS parameters");
    }
  }

  void goalCallback(const geometry_msgs::PoseStampedConstPtr& message) {
    if (!std::isfinite(message->pose.position.x) ||
        !std::isfinite(message->pose.position.y)) {
      ROS_ERROR_THROTTLE(1.0, "Ignoring non-finite local goal");
      return;
    }
    goal_ = *message;
    has_goal_ = true;
  }

  bool setEnabled(
      std_srvs::SetBool::Request& request,
      std_srvs::SetBool::Response& response) {
    enabled_ = request.data;
    observation_builder_->reset();
    last_linear_command_ = 0.0;
    last_angular_command_ = 0.0;
    has_last_command_stamp_ = false;
    publishStop(enabled_ ? "waiting_for_fresh_inputs" : "disabled", false);
    response.success = true;
    response.message = enabled_ ? "policy enabled" : "policy disabled";
    return true;
  }

  PolicyMap robotCentricMap(
      const grid_map_msgs::GridMap& message,
      const Pose2d& pose) const {
    grid_map::GridMap source = gridMapFromMessage(message);
    const double expected_resolution =
        observation_config_.map_extent_m / observation_config_.map_size;
    if (std::abs(source.getResolution() - expected_resolution) > 1.0e-6 ||
        std::abs(source.getLength().x() - observation_config_.map_extent_m) >
            expected_resolution ||
        std::abs(source.getLength().y() - observation_config_.map_extent_m) >
            expected_resolution ||
        source.getSize()(0) != observation_config_.map_size ||
        source.getSize()(1) != observation_config_.map_size) {
      throw std::runtime_error(
          "incoming grid-map geometry does not match policy training config");
    }

    PolicyMap result(observation_config_.map_size);
    const double half_extent = 0.5 * observation_config_.map_extent_m;
    const double cosine = std::cos(pose.yaw);
    const double sine = std::sin(pose.yaw);
    for (int row = 0; row < observation_config_.map_size; ++row) {
      const double local_x = -half_extent +
          (static_cast<double>(row) + 0.5) * expected_resolution;
      for (int column = 0; column < observation_config_.map_size; ++column) {
        const double local_y = -half_extent +
            (static_cast<double>(column) + 0.5) * expected_resolution;
        const grid_map::Position world_position(
            pose.x + cosine * local_x - sine * local_y,
            pose.y + sine * local_x + cosine * local_y);
        if (!source.isInside(world_position)) {
          continue;
        }
        const float ground = source.atPosition(
            "ground_height", world_position);
        const float height_range = source.atPosition(
            "height_range", world_position);
        const float observed = source.atPosition(
            "observed_mask", world_position);
        result.at(2, row, column) = std::isfinite(observed)
            ? static_cast<float>(clampDouble(observed, 0.0, 1.0))
            : 0.0F;
        if (std::isfinite(ground) && std::isfinite(height_range)) {
          result.at(0, row, column) = static_cast<float>(clampDouble(
              ground / observation_config_.ground_height_scale_m,
              -1.0, 1.0));
          result.at(1, row, column) = static_cast<float>(clampDouble(
              height_range / observation_config_.height_range_scale_m,
              0.0, 1.0));
          result.at(3, row, column) = 1.0F;
        }
      }
    }
    return result;
  }

  void inputCallback(
      const grid_map_msgs::GridMapConstPtr& map_message,
      const nav_msgs::OdometryConstPtr& odometry,
      const geometry_msgs::PointStampedConstPtr& ground_reference) {
    last_input_time_ = map_message->info.header.stamp;
    has_input_ = true;
    try {
      if (!inputIsFresh(last_input_time_)) {
        observation_builder_->reset();
        publishStop("stale_input", false);
        return;
      }
      const std::string& frame = map_message->info.header.frame_id;
      if (frame.empty() || odometry->header.frame_id != frame ||
          ground_reference->header.frame_id != frame) {
        throw std::runtime_error(
            "map, odometry, and ground-reference frames must match");
      }
      const Pose2d pose{
          odometry->pose.pose.position.x,
          odometry->pose.pose.position.y,
          tf2::getYaw(odometry->pose.pose.orientation)};
      const PolicyMap current_map = robotCentricMap(*map_message, pose);

      double goal_x = pose.x;
      double goal_y = pose.y;
      if (has_goal_) {
        if (!goal_.header.frame_id.empty() && goal_.header.frame_id != frame) {
          throw std::runtime_error(
              "local goal frame must match the grid-map frame");
        }
        goal_x = goal_.pose.position.x;
        goal_y = goal_.pose.position.y;
      }
      ObservationResult observation = observation_builder_->build(
          current_map,
          pose,
          ground_reference->point.z,
          goal_x,
          goal_y,
          odometry->twist.twist.linear.x,
          odometry->twist.twist.angular.z);

      if (!enabled_) {
        stopForDecision("disabled");
        return;
      }
      if (!has_goal_) {
        stopForDecision("waiting_for_goal");
        return;
      }
      const double goal_distance = std::hypot(goal_x - pose.x, goal_y - pose.y);
      if (goal_distance <= goal_tolerance_m_) {
        stopForDecision("goal_reached");
        return;
      }
      if (observation.observed_ratio < minimum_observed_ratio_ ||
          observation.height_valid_ratio < minimum_height_valid_ratio_) {
        stopForDecision("unhealthy_map");
        return;
      }

      const std::vector<float> output = policy_->infer(observation.values);
      // Do not publish a command computed from an input that became stale
      // while preprocessing or inference was running.
      if (!inputIsFresh(last_input_time_)) {
        stopForDecision("inference_timeout");
        return;
      }
      if (output[0] < linear_min_mps_ - 1.0e-3 ||
          output[0] > linear_max_mps_ + 1.0e-3 ||
          output[1] < angular_min_radps_ - 1.0e-3 ||
          output[1] > angular_max_radps_ + 1.0e-3) {
        throw std::runtime_error("policy output exceeds configured physical limits");
      }
      double linear = clampDouble(output[0], linear_min_mps_, linear_max_mps_);
      double angular = clampDouble(output[1], angular_min_radps_, angular_max_radps_);
      const ros::Time command_stamp = map_message->info.header.stamp;
      if (apply_acceleration_limits_ && has_last_command_stamp_) {
        double dt = (command_stamp - last_command_stamp_).toSec();
        if (!std::isfinite(dt) || dt <= 0.0 || dt > 1.0) {
          dt = 0.1;
        }
        linear = clampDouble(
            linear,
            last_linear_command_ - linear_acceleration_limit_mps2_ * dt,
            last_linear_command_ + linear_acceleration_limit_mps2_ * dt);
        angular = clampDouble(
            angular,
            last_angular_command_ - angular_acceleration_limit_radps2_ * dt,
            last_angular_command_ + angular_acceleration_limit_radps2_ * dt);
      }
      publishCommand(linear, angular, false, "running");
      observation_builder_->recordCommand(linear, angular);
      last_command_stamp_ = command_stamp;
      has_last_command_stamp_ = true;
    } catch (const std::exception& exception) {
      ROS_ERROR_THROTTLE(1.0, "Go2W policy stopped: %s", exception.what());
      observation_builder_->reset();
      publishStop(std::string("error: ") + exception.what(), false);
    }
  }

  void watchdogCallback(const ros::TimerEvent&) {
    if (!enabled_) {
      return;
    }
    if (!has_input_ || !inputIsFresh(last_input_time_)) {
      publishStop("input_timeout", false);
    }
  }

  bool inputIsFresh(const ros::Time& stamp) const {
    if (stamp.isZero()) {
      return false;
    }
    const double age = (ros::Time::now() - stamp).toSec();
    return std::isfinite(age) && age >= -sync_tolerance_s_ &&
           age <= input_timeout_s_;
  }

  void stopForDecision(const std::string& reason) {
    publishStop(reason, true);
  }

  void publishStop(const std::string& reason, const bool record_command) {
    publishCommand(0.0, 0.0, true, reason);
    if (record_command && observation_builder_) {
      observation_builder_->recordCommand(0.0, 0.0);
    }
    last_linear_command_ = 0.0;
    last_angular_command_ = 0.0;
  }

  void publishCommand(
      const double linear,
      const double angular,
      const bool safety_stop,
      const std::string& status) {
    geometry_msgs::Twist command;
    command.linear.x = linear;
    command.angular.z = angular;
    command_publisher_.publish(command);

    std_msgs::Bool safety_message;
    safety_message.data = safety_stop;
    safety_publisher_.publish(safety_message);
    std_msgs::String status_message;
    status_message.data = status;
    status_publisher_.publish(status_message);

    last_linear_command_ = linear;
    last_angular_command_ = angular;
  }

  ros::NodeHandle node_;
  ros::NodeHandle private_node_;
  ros::Publisher command_publisher_;
  ros::Publisher safety_publisher_;
  ros::Publisher status_publisher_;
  ros::Subscriber goal_subscriber_;
  ros::ServiceServer enable_service_;
  ros::Timer watchdog_timer_;

  message_filters::Subscriber<grid_map_msgs::GridMap> map_subscriber_;
  message_filters::Subscriber<nav_msgs::Odometry> odometry_subscriber_;
  message_filters::Subscriber<geometry_msgs::PointStamped>
      ground_reference_subscriber_;
  std::unique_ptr<Synchronizer> synchronizer_;

  ObservationConfig observation_config_;
  std::unique_ptr<ObservationBuilder> observation_builder_;
  std::unique_ptr<OnnxPolicy> policy_;
  geometry_msgs::PoseStamped goal_;
  bool has_goal_{false};
  bool has_input_{false};
  bool enabled_{false};
  bool start_enabled_{false};
  bool apply_acceleration_limits_{true};
  bool has_last_command_stamp_{false};
  ros::Time last_input_time_;
  ros::Time last_command_stamp_;
  double last_linear_command_{0.0};
  double last_angular_command_{0.0};

  std::string model_path_;
  std::string map_topic_;
  std::string odometry_topic_;
  std::string ground_reference_topic_;
  std::string goal_topic_;
  std::string command_topic_;
  int intra_op_threads_{1};
  int sync_queue_size_{10};
  int subscriber_queue_size_{5};
  double sync_tolerance_s_{0.05};
  double input_timeout_s_{0.5};
  double goal_tolerance_m_{0.30};
  double minimum_observed_ratio_{0.05};
  double minimum_height_valid_ratio_{0.01};
  double linear_min_mps_{-0.2};
  double linear_max_mps_{1.2};
  double angular_min_radps_{-1.0};
  double angular_max_radps_{1.0};
  double linear_acceleration_limit_mps2_{0.8};
  double angular_acceleration_limit_radps2_{1.0};
};

}  // namespace go2w_policy_deployment

int main(int argc, char** argv) {
  ros::init(argc, argv, "go2w_policy");
  try {
    go2w_policy_deployment::PolicyNode node;
    ros::spin();
  } catch (const std::exception& exception) {
    ROS_FATAL("Failed to start Go2W policy node: %s", exception.what());
    return 1;
  }
  return 0;
}
