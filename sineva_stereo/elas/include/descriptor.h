#ifndef SINEVA_AUTOWARE_ROS_SRC_COMPUTING_PERCEPTION_LOCALIZATION_PACKAGES_SINEVA_STEREO_INCLUDE_DESCRIPTOR_H_
#define SINEVA_AUTOWARE_ROS_SRC_COMPUTING_PERCEPTION_LOCALIZATION_PACKAGES_SINEVA_STEREO_INCLUDE_DESCRIPTOR_H_

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

#include <opencv2/opencv.hpp>

#include <iostream>

using namespace cv;

class Descriptor
{
 public:
  Descriptor() = delete;
  // deconstructor releases memory
  ~Descriptor();

  // constructor creates filters
  Descriptor(const Mat &image, const int32_t &width, const int32_t &height);

  Mat CreateDescriptor();

 private:
  Mat descriptor;
  Mat grad_x, grad_y;
  int32_t width_, height_;

  Mat image;
};

#endif  // SINEVA_AUTOWARE_ROS_SRC_COMPUTING_PERCEPTION_LOCALIZATION_PACKAGES_SINEVA_STEREO_INCLUDE_DESCRIPTOR_H_
