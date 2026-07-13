#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>

class HesaiFastlioAdapter {
public:
  HesaiFastlioAdapter() : pnh_("~") {
    pnh_.param<std::string>("input_topic", input_topic_, "/lidar_points");
    pnh_.param<std::string>("output_topic", output_topic_, "/lidar_points_fastlio");
    pnh_.param("max_scan_duration", max_scan_duration_, 0.2);

    pub_ = nh_.advertise<sensor_msgs::PointCloud2>(output_topic_, 5);
    sub_ = nh_.subscribe(
        input_topic_, 5, &HesaiFastlioAdapter::cloudCallback, this,
        ros::TransportHints().tcpNoDelay());

    ROS_INFO_STREAM("hesai_fastlio_adapter: " << input_topic_
                    << " -> " << output_topic_);
  }

private:
  static bool hasField(const sensor_msgs::PointCloud2& msg,
                       const std::string& name) {
    for (const auto& field : msg.fields) {
      if (field.name == name) {
        return true;
      }
    }
    return false;
  }

  void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
    if (msg->width * msg->height == 0) {
      return;
    }

    const char* required_fields[] = {
        "x", "y", "z", "intensity", "ring", "timestamp"};
    for (const char* field : required_fields) {
      if (!hasField(*msg, field)) {
        ROS_ERROR_THROTTLE(
            2.0,
            "Input PointCloud2 is missing required field '%s'. "
            "Expected Hesai fields: x y z intensity ring timestamp.",
            field);
        return;
      }
    }

    sensor_msgs::PointCloud2ConstIterator<float> in_x(*msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> in_y(*msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> in_z(*msg, "z");
    sensor_msgs::PointCloud2ConstIterator<float> in_intensity(*msg, "intensity");
    sensor_msgs::PointCloud2ConstIterator<uint16_t> in_ring(*msg, "ring");
    sensor_msgs::PointCloud2ConstIterator<double> in_timestamp(*msg, "timestamp");

    const double first_point_time = *in_timestamp;
    const double msg_stamp = msg->header.stamp.toSec();
    const bool header_time_valid =
        std::isfinite(msg_stamp) && msg_stamp > 1e-6;
    const bool point_time_valid = std::isfinite(first_point_time);
    const bool point_time_is_absolute =
        point_time_valid && first_point_time > 1e6;

    double frame_start_time = header_time_valid ? msg_stamp : 0.0;
    if (!header_time_valid && point_time_is_absolute) {
      frame_start_time = first_point_time;
    }

    sensor_msgs::PointCloud2 out;
    out.header = msg->header;
    out.header.stamp = ros::Time().fromSec(frame_start_time);
    out.height = 1;
    out.width = msg->width * msg->height;
    out.is_bigendian = false;
    out.is_dense = msg->is_dense;

    sensor_msgs::PointCloud2Modifier modifier(out);
    modifier.setPointCloud2Fields(
        6,
        "x", 1, sensor_msgs::PointField::FLOAT32,
        "y", 1, sensor_msgs::PointField::FLOAT32,
        "z", 1, sensor_msgs::PointField::FLOAT32,
        "intensity", 1, sensor_msgs::PointField::FLOAT32,
        "ring", 1, sensor_msgs::PointField::UINT16,
        "time", 1, sensor_msgs::PointField::FLOAT32);
    modifier.resize(out.width);

    sensor_msgs::PointCloud2Iterator<float> out_x(out, "x");
    sensor_msgs::PointCloud2Iterator<float> out_y(out, "y");
    sensor_msgs::PointCloud2Iterator<float> out_z(out, "z");
    sensor_msgs::PointCloud2Iterator<float> out_intensity(out, "intensity");
    sensor_msgs::PointCloud2Iterator<uint16_t> out_ring(out, "ring");
    sensor_msgs::PointCloud2Iterator<float> out_time(out, "time");

    const std::size_t point_count =
        static_cast<std::size_t>(msg->width) * msg->height;

    for (std::size_t i = 0; i < point_count;
         ++i, ++in_x, ++in_y, ++in_z, ++in_intensity, ++in_ring,
         ++in_timestamp, ++out_x, ++out_y, ++out_z, ++out_intensity,
         ++out_ring, ++out_time) {
      double relative_time = 0.0;
      if (point_time_is_absolute) {
        relative_time = *in_timestamp - frame_start_time;
      } else {
        // Some Hesai bags store per-point timestamp as scan-relative seconds.
        // In that case the ROS header is still the frame start time.
        relative_time = *in_timestamp;
      }

      if (!std::isfinite(relative_time)) {
        relative_time = 0.0;
      }

      // Permit tiny floating-point negatives, but reject clearly invalid values.
      if (relative_time < 0.0 && relative_time > -1e-4) {
        relative_time = 0.0;
      }

      if (relative_time < 0.0 || relative_time > max_scan_duration_) {
        ROS_WARN_THROTTLE(
            2.0,
            "Abnormal Hesai point relative time %.6f s; clamping to valid range.",
            relative_time);
        relative_time =
            std::max(0.0, std::min(relative_time, max_scan_duration_));
      }

      *out_x = *in_x;
      *out_y = *in_y;
      *out_z = *in_z;
      *out_intensity = *in_intensity;
      *out_ring = *in_ring;
      *out_time = static_cast<float>(relative_time);
    }

    pub_.publish(out);
  }

  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  ros::Subscriber sub_;
  ros::Publisher pub_;
  std::string input_topic_;
  std::string output_topic_;
  double max_scan_duration_;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "hesai_fastlio_adapter");
  HesaiFastlioAdapter node;
  ros::spin();
  return 0;
}
