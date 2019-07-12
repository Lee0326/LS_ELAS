#ifndef SINEVA_AUTOWARE_ROS_SRC_COMPUTING_PERCEPTION_LOCALIZATION_PACKAGES_SINEVA_STEREO_UTILS_INCLUDE_PRECOMP_H_
#define SINEVA_AUTOWARE_ROS_SRC_COMPUTING_PERCEPTION_LOCALIZATION_PACKAGES_SINEVA_STEREO_UTILS_INCLUDE_PRECOMP_H_

// standard cpp includes
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <iomanip>

// opencv includes
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// ros includes
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

// boost includes
#include <boost/format.hpp>

// namepaces
using std::string;
using cv::Mat;
using cv::Size;
using cv::INTER_CUBIC;
using cv::INTER_NEAREST;

#endif // SINEVA_AUTOWARE_ROS_SRC_COMPUTING_PERCEPTION_LOCALIZATION_PACKAGES_SINEVA_STEREO_UTILS_INCLUDE_PRECOMP_H_