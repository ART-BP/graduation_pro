#include <cmath>
#include <algorithm>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Pose.h>
#include <nav_msgs/Odometry.h>
#include <pcl/common/common.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Bool.h>
#include <std_msgs/String.h>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/MarkerArray.h>

namespace {

using PointT = pcl::PointXYZ;
using CloudT = pcl::PointCloud<PointT>;

struct Keyframe {
  int id = 0;
  ros::Time stamp;
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  CloudT::Ptr cloud;
  std::vector<float> descriptor;
};

Eigen::Isometry3d odomToEigen(const nav_msgs::Odometry& odom) {
  const auto& p = odom.pose.pose.position;
  const auto& q = odom.pose.pose.orientation;

  Eigen::Quaterniond quat(q.w, q.x, q.y, q.z);
  if (quat.norm() < 1e-6) {
    quat = Eigen::Quaterniond::Identity();
  } else {
    quat.normalize();
  }

  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.linear() = quat.toRotationMatrix();
  pose.translation() = Eigen::Vector3d(p.x, p.y, p.z);
  return pose;
}

double yawDistance(const Eigen::Isometry3d& a, const Eigen::Isometry3d& b) {
  const Eigen::Matrix3d rotation_delta = a.linear().transpose() * b.linear();
  Eigen::AngleAxisd angle_axis(rotation_delta);
  return std::abs(angle_axis.angle());
}

CloudT::Ptr downsampleCloud(const sensor_msgs::PointCloud2& msg,
                            double voxel_leaf_size) {
  CloudT::Ptr raw(new CloudT);
  pcl::fromROSMsg(msg, *raw);

  CloudT::Ptr finite(new CloudT);
  finite->reserve(raw->size());
  for (const auto& p : raw->points) {
    if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z)) {
      finite->push_back(p);
    }
  }

  if (voxel_leaf_size <= 0.0 || finite->empty()) {
    return finite;
  }

  CloudT::Ptr filtered(new CloudT);
  pcl::VoxelGrid<PointT> voxel;
  voxel.setLeafSize(voxel_leaf_size, voxel_leaf_size, voxel_leaf_size);
  voxel.setInputCloud(finite);
  voxel.filter(*filtered);
  return filtered;
}

geometry_msgs::Point toPointMsg(const Eigen::Vector3d& p) {
  geometry_msgs::Point msg;
  msg.x = p.x();
  msg.y = p.y();
  msg.z = p.z();
  return msg;
}

geometry_msgs::Pose toPoseMsg(const Eigen::Isometry3d& pose) {
  geometry_msgs::Pose msg;
  msg.position = toPointMsg(pose.translation());
  Eigen::Quaterniond q(pose.linear());
  q.normalize();
  msg.orientation.w = q.w();
  msg.orientation.x = q.x();
  msg.orientation.y = q.y();
  msg.orientation.z = q.z();
  return msg;
}

void setIdentityMarkerPose(visualization_msgs::Marker& marker) {
  marker.pose.orientation.w = 1.0;
  marker.pose.orientation.x = 0.0;
  marker.pose.orientation.y = 0.0;
  marker.pose.orientation.z = 0.0;
}

tf::Transform toTfTransform(const Eigen::Isometry3d& pose) {
  Eigen::Quaterniond q(pose.linear());
  q.normalize();
  tf::Transform transform;
  transform.setOrigin(tf::Vector3(pose.translation().x(),
                                  pose.translation().y(),
                                  pose.translation().z()));
  transform.setRotation(tf::Quaternion(q.x(), q.y(), q.z(), q.w()));
  return transform;
}

Eigen::Isometry3d interpolatePose(const Eigen::Isometry3d& from,
                                  const Eigen::Isometry3d& to,
                                  double alpha) {
  alpha = std::max(0.0, std::min(1.0, alpha));
  Eigen::Quaterniond q_from(from.linear());
  Eigen::Quaterniond q_to(to.linear());
  q_from.normalize();
  q_to.normalize();

  Eigen::Isometry3d result = Eigen::Isometry3d::Identity();
  result.translation() =
      (1.0 - alpha) * from.translation() + alpha * to.translation();
  result.linear() = q_from.slerp(alpha, q_to).toRotationMatrix();
  return result;
}

}  // namespace

class LoopCorrectionNode {
 public:
  LoopCorrectionNode() : nh_(), pnh_("~") {
    pnh_.param<std::string>("odom_topic", odom_topic_, "/Odometry");
    pnh_.param<std::string>("cloud_topic", cloud_topic_, "/cloud_registered_body");
    pnh_.param<std::string>("map_frame", map_frame_, "map");
    pnh_.param<std::string>("odom_frame", odom_frame_, "odom");
    pnh_.param<std::string>("body_frame", body_frame_, "base_link");
    pnh_.param<std::string>("world_frame", world_frame_, map_frame_);
    pnh_.param("keyframe_distance", keyframe_distance_, 1.0);
    pnh_.param("keyframe_angle_deg", keyframe_angle_deg_, 15.0);
    pnh_.param("history_skip", history_skip_, 30);
    pnh_.param("min_loop_interval", min_loop_interval_, 10.0);
    pnh_.param("search_radius", search_radius_, 4.0);
    pnh_.param("voxel_leaf_size", voxel_leaf_size_, 0.35);
    pnh_.param("min_cloud_points", min_cloud_points_, 200);
    pnh_.param("max_odom_cloud_dt", max_odom_cloud_dt_, 0.2);
    pnh_.param("icp_max_iterations", icp_max_iterations_, 40);
    pnh_.param("icp_max_correspondence_distance",
               icp_max_correspondence_distance_, 1.0);
    pnh_.param("icp_fitness_threshold", icp_fitness_threshold_, 0.35);
    pnh_.param("loop_cooldown_keyframes", loop_cooldown_keyframes_, 20);
    pnh_.param("descriptor_rings", descriptor_rings_, 20);
    pnh_.param("descriptor_sectors", descriptor_sectors_, 60);
    pnh_.param("descriptor_max_radius", descriptor_max_radius_, 30.0);
    pnh_.param("descriptor_distance_threshold",
               descriptor_distance_threshold_, 0.35);
    pnh_.param("descriptor_top_k", descriptor_top_k_, 5);
    pnh_.param("publish_map_to_odom", publish_map_to_odom_, true);
    pnh_.param("correction_alpha", correction_alpha_, 0.08);
    pnh_.param("tf_publish_rate", tf_publish_rate_, 50.0);
    pnh_.param("max_correction_translation", max_correction_translation_, 50.0);

    keyframe_angle_rad_ = keyframe_angle_deg_ * M_PI / 180.0;

    odom_sub_ = nh_.subscribe(
        odom_topic_, 200, &LoopCorrectionNode::odomCallback, this,
        ros::TransportHints().tcpNoDelay());
    cloud_sub_ = nh_.subscribe(
        cloud_topic_, 20, &LoopCorrectionNode::cloudCallback, this,
        ros::TransportHints().tcpNoDelay());

    loop_pub_ = nh_.advertise<std_msgs::Bool>("/loop_closure_detected", 10);
    info_pub_ = nh_.advertise<std_msgs::String>("/loop_closure_info", 10);
    corrected_odom_pub_ =
        nh_.advertise<nav_msgs::Odometry>("/loop_corrected_odometry", 50);
    marker_pub_ =
        nh_.advertise<visualization_msgs::MarkerArray>("/loop_closure_markers",
                                                       10, true);
    tf_timer_ = nh_.createTimer(ros::Duration(1.0 / tf_publish_rate_),
                                &LoopCorrectionNode::tfTimerCallback, this);

    ROS_INFO_STREAM("lio_loop_correction: odom=" << odom_topic_
                    << ", cloud=" << cloud_topic_
                    << ", descriptor_distance_threshold="
                    << descriptor_distance_threshold_
                    << ", icp_fitness_threshold=" << icp_fitness_threshold_);
  }

 private:
  void odomCallback(const nav_msgs::OdometryConstPtr& msg) {
    latest_odom_ = *msg;
    latest_pose_ = odomToEigen(*msg);
    has_odom_ = true;
    updateSmoothedCorrection();
    publishMapToOdom(msg->header.stamp);
    publishCorrectedOdom(msg->header.stamp);
  }

  void tfTimerCallback(const ros::TimerEvent&) {
    if (!has_odom_) {
      return;
    }
    publishMarkers(latest_odom_.header.stamp);
  }

  void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
    if (!has_odom_) {
      ROS_WARN_THROTTLE(2.0, "No odometry received yet; skip loop detector.");
      return;
    }

    const double dt =
        std::abs((msg->header.stamp - latest_odom_.header.stamp).toSec());
    if (dt > max_odom_cloud_dt_) {
      ROS_WARN_THROTTLE(
          2.0,
          "Cloud/odom time gap %.3f s is larger than %.3f s; skip keyframe.",
          dt, max_odom_cloud_dt_);
      return;
    }

    publishMapToOdom(msg->header.stamp);

    if (!needNewKeyframe(latest_pose_)) {
      return;
    }

    CloudT::Ptr cloud = downsampleCloud(*msg, voxel_leaf_size_);
    if (static_cast<int>(cloud->size()) < min_cloud_points_) {
      ROS_WARN_THROTTLE(2.0,
                        "Loop detector keyframe cloud has too few points: %zu",
                        cloud->size());
      return;
    }

    Keyframe keyframe;
    keyframe.id = static_cast<int>(keyframes_.size());
    keyframe.stamp = msg->header.stamp;
    keyframe.pose = latest_pose_;
    keyframe.cloud = cloud;
    keyframe.descriptor = makeDescriptor(*cloud);
    keyframes_.push_back(keyframe);

    ROS_INFO_STREAM("Added loop keyframe " << keyframe.id << " with "
                    << keyframe.cloud->size() << " points.");

    publishMarkers(msg->header.stamp);
    detectLoop(keyframes_.back());
  }

  bool needNewKeyframe(const Eigen::Isometry3d& pose) const {
    if (keyframes_.empty()) {
      return true;
    }
    const Eigen::Vector3d delta =
        pose.translation() - keyframes_.back().pose.translation();
    const double distance = delta.head<2>().norm();
    const double angle = yawDistance(keyframes_.back().pose, pose);
    return distance >= keyframe_distance_ || angle >= keyframe_angle_rad_;
  }

  void detectLoop(const Keyframe& current) {
    if (current.id <= history_skip_) {
      return;
    }
    if (current.id - last_loop_keyframe_id_ < loop_cooldown_keyframes_) {
      return;
    }

    struct Candidate {
      int id = -1;
      double descriptor_distance = std::numeric_limits<double>::max();
      int yaw_shift = 0;
    };

    std::vector<Candidate> candidates;

    for (int i = 0; i < current.id - history_skip_; ++i) {
      const Keyframe& candidate = keyframes_[i];
      const double age = (current.stamp - candidate.stamp).toSec();
      if (age < min_loop_interval_) {
        continue;
      }

      int yaw_shift = 0;
      const double distance = descriptorDistance(current.descriptor,
                                                 candidate.descriptor,
                                                 &yaw_shift);
      if (distance <= descriptor_distance_threshold_) {
        candidates.push_back({i, distance, yaw_shift});
      }
    }

    if (candidates.empty()) {
      return;
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) {
                return a.descriptor_distance < b.descriptor_distance;
              });

    const int try_count =
        std::min<int>(descriptor_top_k_, static_cast<int>(candidates.size()));
    for (int idx = 0; idx < try_count; ++idx) {
      const Candidate& loop_candidate = candidates[idx];
      const Keyframe& candidate = keyframes_[loop_candidate.id];
      pcl::IterativeClosestPoint<PointT, PointT> icp;
      icp.setInputSource(current.cloud);
      icp.setInputTarget(candidate.cloud);
      icp.setMaxCorrespondenceDistance(icp_max_correspondence_distance_);
      icp.setMaximumIterations(icp_max_iterations_);
      icp.setTransformationEpsilon(1e-6);
      icp.setEuclideanFitnessEpsilon(1e-5);

      const double yaw =
          -2.0 * M_PI * static_cast<double>(loop_candidate.yaw_shift) /
          static_cast<double>(descriptor_sectors_);
      Eigen::Matrix4f initial_guess = Eigen::Matrix4f::Identity();
      initial_guess.block<3, 3>(0, 0) =
          Eigen::AngleAxisf(static_cast<float>(yaw),
                            Eigen::Vector3f::UnitZ())
              .toRotationMatrix();

      CloudT aligned;
      icp.align(aligned, initial_guess);
      const double fitness = icp.getFitnessScore();
      if (!icp.hasConverged() || fitness > icp_fitness_threshold_) {
        ROS_INFO_STREAM("Loop candidate rejected: current=" << current.id
                        << ", candidate=" << candidate.id
                        << ", descriptor_distance="
                        << loop_candidate.descriptor_distance
                        << ", fitness=" << fitness);
        continue;
      }

      publishLoop(current, candidate, loop_candidate.descriptor_distance,
                  fitness, icp.getFinalTransformation());
      return;
    }
  }

  std::vector<float> makeDescriptor(const CloudT& cloud) const {
    std::vector<float> descriptor(descriptor_rings_ * descriptor_sectors_,
                                  0.0f);
    if (descriptor_rings_ <= 0 || descriptor_sectors_ <= 0 ||
        descriptor_max_radius_ <= 0.0) {
      return descriptor;
    }

    int used_points = 0;
    for (const auto& point : cloud.points) {
      const double radius = std::hypot(point.x, point.y);
      if (radius <= 0.01 || radius > descriptor_max_radius_) {
        continue;
      }
      double angle = std::atan2(point.y, point.x);
      if (angle < 0.0) {
        angle += 2.0 * M_PI;
      }
      const int ring = std::min(
          descriptor_rings_ - 1,
          static_cast<int>(radius / descriptor_max_radius_ *
                           static_cast<double>(descriptor_rings_)));
      const int sector = std::min(
          descriptor_sectors_ - 1,
          static_cast<int>(angle / (2.0 * M_PI) *
                           static_cast<double>(descriptor_sectors_)));
      descriptor[ring * descriptor_sectors_ + sector] += 1.0f;
      ++used_points;
    }

    if (used_points <= 0) {
      return descriptor;
    }
    const float inv_points = 1.0f / static_cast<float>(used_points);
    for (auto& value : descriptor) {
      value *= inv_points;
    }
    return descriptor;
  }

  double descriptorDistance(const std::vector<float>& current,
                            const std::vector<float>& candidate,
                            int* best_shift) const {
    if (current.size() != candidate.size() || current.empty() ||
        descriptor_rings_ <= 0 || descriptor_sectors_ <= 0) {
      if (best_shift) {
        *best_shift = 0;
      }
      return std::numeric_limits<double>::max();
    }

    double best_similarity = -1.0;
    int best = 0;
    for (int shift = 0; shift < descriptor_sectors_; ++shift) {
      double dot = 0.0;
      double norm_current = 0.0;
      double norm_candidate = 0.0;
      for (int ring = 0; ring < descriptor_rings_; ++ring) {
        for (int sector = 0; sector < descriptor_sectors_; ++sector) {
          const int candidate_index = ring * descriptor_sectors_ + sector;
          const int current_sector = (sector + shift) % descriptor_sectors_;
          const int current_index = ring * descriptor_sectors_ + current_sector;
          const double a = current[current_index];
          const double b = candidate[candidate_index];
          dot += a * b;
          norm_current += a * a;
          norm_candidate += b * b;
        }
      }
      if (norm_current <= 1e-12 || norm_candidate <= 1e-12) {
        continue;
      }
      const double similarity =
          dot / (std::sqrt(norm_current) * std::sqrt(norm_candidate));
      if (similarity > best_similarity) {
        best_similarity = similarity;
        best = shift;
      }
    }

    if (best_shift) {
      *best_shift = best;
    }
    if (best_similarity < 0.0) {
      return std::numeric_limits<double>::max();
    }
    return 1.0 - best_similarity;
  }

  void publishLoop(const Keyframe& current, const Keyframe& candidate,
                   double descriptor_distance, double fitness,
                   const Eigen::Matrix4f& transform) {
    const Eigen::Isometry3d candidate_from_current =
        Eigen::Isometry3d(transform.cast<double>());
    const Eigen::Isometry3d candidate_pose_map =
        map_to_odom_target_ * candidate.pose;
    const Eigen::Isometry3d corrected_current_pose_map =
        candidate_pose_map * candidate_from_current;
    const Eigen::Isometry3d new_map_to_odom =
        corrected_current_pose_map * current.pose.inverse();

    const double correction_translation =
        (new_map_to_odom.translation() - map_to_odom_target_.translation())
            .norm();
    if (correction_translation > max_correction_translation_) {
      ROS_WARN_STREAM("Loop correction rejected: translation jump "
                      << correction_translation << " m is larger than "
                      << max_correction_translation_ << " m.");
      return;
    }

    map_to_odom_target_ = new_map_to_odom;
    last_loop_keyframe_id_ = current.id;
    last_loop_candidate_id_ = candidate.id;
    last_loop_fitness_ = fitness;

    std_msgs::Bool loop_msg;
    loop_msg.data = true;
    loop_pub_.publish(loop_msg);

    std_msgs::String info;
    std::ostringstream ss;
    ss << "loop current_id=" << current.id
       << " candidate_id=" << candidate.id
       << " descriptor_distance=" << descriptor_distance
       << " icp_fitness=" << fitness
       << " correction_translation=" << correction_translation
       << " transform=" << transform;
    info.data = ss.str();
    info_pub_.publish(info);

    ROS_WARN_STREAM("Loop detected: current=" << current.id
                    << ", candidate=" << candidate.id
                    << ", descriptor_distance=" << descriptor_distance
                    << ", icp_fitness=" << fitness
                    << ", correction_translation=" << correction_translation);
    updateSmoothedCorrection();
    publishMapToOdom(current.stamp);
    publishCorrectedOdom(current.stamp);
    publishMarkers(current.stamp);
  }

  void updateSmoothedCorrection() {
    map_to_odom_smoothed_ = interpolatePose(
        map_to_odom_smoothed_, map_to_odom_target_, correction_alpha_);
  }

  void publishMapToOdom(const ros::Time& stamp) {
    if (!publish_map_to_odom_) {
      return;
    }
    if (has_last_tf_stamp_) {
      if (stamp < last_tf_stamp_) {
        has_last_tf_stamp_ = false;
      } else if (stamp == last_tf_stamp_) {
        return;
      }
    }
    tf_broadcaster_.sendTransform(tf::StampedTransform(
        toTfTransform(map_to_odom_smoothed_), stamp, map_frame_, odom_frame_));
    last_tf_stamp_ = stamp;
    has_last_tf_stamp_ = true;
  }

  void publishCorrectedOdom(const ros::Time& stamp) {
    if (!has_odom_) {
      return;
    }

    nav_msgs::Odometry corrected = latest_odom_;
    corrected.header.stamp = stamp;
    corrected.header.frame_id = map_frame_;
    corrected.child_frame_id = body_frame_;
    corrected.pose.pose = toPoseMsg(map_to_odom_smoothed_ * latest_pose_);
    corrected_odom_pub_.publish(corrected);
  }

  void publishMarkers(const ros::Time& stamp) const {
    visualization_msgs::MarkerArray markers;

    visualization_msgs::Marker keyframe_marker;
    keyframe_marker.header.frame_id = world_frame_;
    keyframe_marker.header.stamp = stamp;
    keyframe_marker.ns = "loop_keyframes";
    keyframe_marker.id = 0;
    keyframe_marker.type = visualization_msgs::Marker::SPHERE_LIST;
    keyframe_marker.action = keyframes_.empty()
                                  ? visualization_msgs::Marker::DELETE
                                  : visualization_msgs::Marker::ADD;
    setIdentityMarkerPose(keyframe_marker);
    keyframe_marker.scale.x = 0.25;
    keyframe_marker.scale.y = 0.25;
    keyframe_marker.scale.z = 0.25;
    keyframe_marker.color.r = 0.1;
    keyframe_marker.color.g = 0.7;
    keyframe_marker.color.b = 1.0;
    keyframe_marker.color.a = 0.8;
    for (const auto& keyframe : keyframes_) {
      keyframe_marker.points.push_back(
          toPointMsg((map_to_odom_smoothed_ * keyframe.pose).translation()));
    }
    markers.markers.push_back(keyframe_marker);

    visualization_msgs::Marker loop_marker;
    loop_marker.header.frame_id = world_frame_;
    loop_marker.header.stamp = stamp;
    loop_marker.ns = "loop_edges";
    loop_marker.id = 1;
    loop_marker.type = visualization_msgs::Marker::LINE_LIST;
    loop_marker.action = visualization_msgs::Marker::DELETE;
    setIdentityMarkerPose(loop_marker);
    loop_marker.scale.x = 0.08;
    loop_marker.color.r = 1.0;
    loop_marker.color.g = 0.2;
    loop_marker.color.b = 0.1;
    loop_marker.color.a = 1.0;

    if (last_loop_keyframe_id_ >= 0 && last_loop_candidate_id_ >= 0 &&
        last_loop_keyframe_id_ < static_cast<int>(keyframes_.size()) &&
        last_loop_candidate_id_ < static_cast<int>(keyframes_.size())) {
      loop_marker.action = visualization_msgs::Marker::ADD;
      loop_marker.points.push_back(
          toPointMsg((map_to_odom_smoothed_ *
                      keyframes_[last_loop_keyframe_id_].pose)
                         .translation()));
      loop_marker.points.push_back(
          toPointMsg((map_to_odom_smoothed_ *
                      keyframes_[last_loop_candidate_id_].pose)
                         .translation()));
    }
    markers.markers.push_back(loop_marker);

    visualization_msgs::Marker text_marker;
    text_marker.header.frame_id = world_frame_;
    text_marker.header.stamp = stamp;
    text_marker.ns = "loop_text";
    text_marker.id = 2;
    text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    text_marker.action = visualization_msgs::Marker::ADD;
    setIdentityMarkerPose(text_marker);
    text_marker.scale.z = 0.6;
    text_marker.color.r = 1.0;
    text_marker.color.g = 1.0;
    text_marker.color.b = 1.0;
    text_marker.color.a = 1.0;
    if (last_loop_keyframe_id_ >= 0 && last_loop_keyframe_id_ <
                                          static_cast<int>(keyframes_.size())) {
      text_marker.pose.position =
          toPointMsg((map_to_odom_smoothed_ *
                      keyframes_[last_loop_keyframe_id_].pose)
                         .translation() +
                     Eigen::Vector3d(0.0, 0.0, 1.0));
      std::ostringstream ss;
      ss << "Loop " << last_loop_candidate_id_ << " -> "
         << last_loop_keyframe_id_ << "\nfitness " << last_loop_fitness_;
      text_marker.text = ss.str();
    } else {
      text_marker.action = visualization_msgs::Marker::DELETE;
    }
    markers.markers.push_back(text_marker);

    marker_pub_.publish(markers);
  }

  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  ros::Subscriber odom_sub_;
  ros::Subscriber cloud_sub_;
  ros::Publisher loop_pub_;
  ros::Publisher info_pub_;
  ros::Publisher corrected_odom_pub_;
  ros::Publisher marker_pub_;
  ros::Timer tf_timer_;
  tf::TransformBroadcaster tf_broadcaster_;

  std::string odom_topic_;
  std::string cloud_topic_;
  std::string world_frame_;
  std::string map_frame_;
  std::string odom_frame_;
  std::string body_frame_;
  double keyframe_distance_ = 1.0;
  double keyframe_angle_deg_ = 15.0;
  double keyframe_angle_rad_ = M_PI / 12.0;
  int history_skip_ = 30;
  double min_loop_interval_ = 10.0;
  double search_radius_ = 4.0;
  double voxel_leaf_size_ = 0.35;
  int min_cloud_points_ = 200;
  double max_odom_cloud_dt_ = 0.2;
  int icp_max_iterations_ = 40;
  double icp_max_correspondence_distance_ = 1.0;
  double icp_fitness_threshold_ = 0.35;
  int loop_cooldown_keyframes_ = 20;
  int descriptor_rings_ = 20;
  int descriptor_sectors_ = 60;
  double descriptor_max_radius_ = 30.0;
  double descriptor_distance_threshold_ = 0.35;
  int descriptor_top_k_ = 5;
  bool publish_map_to_odom_ = true;
  double correction_alpha_ = 0.08;
  double tf_publish_rate_ = 50.0;
  double max_correction_translation_ = 50.0;

  bool has_odom_ = false;
  bool has_last_tf_stamp_ = false;
  ros::Time last_tf_stamp_;
  nav_msgs::Odometry latest_odom_;
  Eigen::Isometry3d latest_pose_ = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d map_to_odom_target_ = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d map_to_odom_smoothed_ = Eigen::Isometry3d::Identity();
  std::vector<Keyframe> keyframes_;
  int last_loop_keyframe_id_ = -1000000;
  int last_loop_candidate_id_ = -1;
  double last_loop_fitness_ = 0.0;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "lio_loop_correction");
  LoopCorrectionNode node;
  ros::spin();
  return 0;
}
