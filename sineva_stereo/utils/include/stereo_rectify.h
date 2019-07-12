#ifndef SINEVA_AUTOWARE_ROS_SRC_COMPUTING_PERCEPTION_LOCALIZATION_PACKAGES_SINEVA_STEREO_UTILS_STEREORECTIFY_H_
#define SINEVA_AUTOWARE_ROS_SRC_COMPUTING_PERCEPTION_LOCALIZATION_PACKAGES_SINEVA_STEREO_UTILS_STEREORECTIFY_H_

#include <opencv2/opencv.hpp>

class StereoRectify
{
public:
    StereoRectify(const std::string &left_calib_path, const std::string &right_calib_path);

    ~StereoRectify();

    void stereoRectify(const cv::Mat &left, const cv::Mat &right, cv::Mat &rec_left, cv::Mat &rec_right);

    void loadCalibrationData(const std::string &filename, cv::Mat &K, cv::Mat &D, cv::Mat &R, cv::Mat &P, cv::Mat &mapx,
                             cv::Mat &mapy);

public:
    cv::Mat K1, D1, R1, P1;
    cv::Mat mapx1, mapy1;
    cv::Mat K2, D2, R2, P2;
    cv::Mat mapx2, mapy2;
    cv::Size image_size;
};

#endif // SINEVA_AUTOWARE_ROS_SRC_COMPUTING_PERCEPTION_LOCALIZATION_PACKAGES_SINEVA_STEREO_UTILS_STEREORECTIFY_H_