#ifndef SINEVA_AUTOWARE_ROS_SRC_COMPUTING_PERCEPTION_LOCALIZATION_PACKAGES_SINEVA_STEREO_INCLUDE_STEREOMATCH_H_
#define SINEVA_AUTOWARE_ROS_SRC_COMPUTING_PERCEPTION_LOCALIZATION_PACKAGES_SINEVA_STEREO_INCLUDE_STEREOMATCH_H_

#include <opencv2/core/mat.hpp>

#include "include/elas.h"

class StereoMatch
{
 public:
  StereoMatch() {}
  explicit StereoMatch(Elas::parameters &param) : mParam(param) {}
  ~StereoMatch() {}

  cv::Mat run(cv::Mat &left, const cv::Mat &right);
  cv::Mat convertDisparityToDepth(const cv::Mat &disp, const float &baseline, const float &fx);

 public:
  Elas::parameters mParam;
};

#endif  // SINEVA_AUTOWARE_ROS_SRC_COMPUTING_PERCEPTION_LOCALIZATION_PACKAGES_SINEVA_STEREO_INCLUDE_STEREOMATCH_H_
