#ifndef SINEVA_AUTOWARE_ROS_SRC_COMPUTING_PERCEPTION_LOCALIZATION_PACKAGES_SINEVA_STEREO_INCLUDE_ELAS_H_
#define SINEVA_AUTOWARE_ROS_SRC_COMPUTING_PERCEPTION_LOCALIZATION_PACKAGES_SINEVA_STEREO_INCLUDE_ELAS_H_

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <vector>
#include "orb_extractor.h"

using namespace cv;
using namespace std;

class Elas
{
public:
    // parameter settings
    struct parameters
    {
        int32_t disp_min = 10;           // min disparity
        int32_t disp_max = 255;          // max disparity
        int32_t grid_size = 20;          // size of neighborhood for additional support point extrapolation
        int32_t step_size = 2;
        float beta = 0.02;               // image likelihood parameter
        float gamma = 3;                 // prior constant
        float sigma = 1;                 // prior sigma
        float sradius = 2;               // prior sigma radius
        int32_t lr_threshold = 1;        // disparity threshold for left/right consistency check
        int32_t speckle_size = 200;      // maximal size of a speckle (small speckles get removed)
        int32_t ipol_gap_width = 10;     // interpolate small gaps (left<->right, top<->bottom)
    };

    // constructor, input: parameters
    explicit Elas(parameters param) : param_(param)
    {
    }

    // deconstructor
    ~Elas()
    {
    }

    bool process(const Mat &image_left, const Mat &image_right, Mat &disparity_left, Mat &disparity_right);

    void ComputeSupportMatches(const Mat &image_left, const Mat &image_right, vector<Point3i> &support_points);

private:
    vector<Vec6f>
    ComputeDelaunayTriangulation(const vector<Point3i> &support_points, const bool &right_image, const int32_t &width,
                                 const int32_t &height);

    vector<Vec6f> ComputeDisparityPlanes(const vector<Point3i> &support_points, const vector<Vec6f> &triangulate_points,
                                         vector<Vec3d> &triangulate_d,
                                         const bool &right_image);

    void CreateGrid(const vector<Point3i> &support_points, Mat &disparity_grid, const Size &disparity_grid_size,
                    const bool &right_image);

    void
    ComputeDisparity(const Mat &descriptor_left, const Mat &descriptor_right, const vector<Vec6f> &triangulate_points,
                     const vector<Vec6f> &plane_params, const Mat &disparity_grid, const Size &disparity_grid_size,
                     const bool &right_image, const vector<Vec3d> &triangulate_d, const int32_t &width,
                     const int32_t &height, Mat &disparity);

    void FindMatch(const Mat &descriptor_left, const Mat &descriptor_right, const int32_t &u, const int32_t &v,
                   const float &plane_a, const float &plane_b, const float &plane_c, const Mat &disparity_grid,
                   const Size &disparity_grid_size, int32_t *P, const int32_t &plane_radius, const bool &valid,
                   const bool &right_image, const int32_t &width, const int32_t &height, Mat &disparity);

    void GapInterpolation(const int32_t &width, const int32_t &height, Mat &disparity);

    void LeftRightConsistencyCheck(Mat &disparity_left, Mat &disparity_right, const int& lr_threshold);

    void ComputeDisparity(vector<Point3i> support_points, const Mat &descriptor_left, const Mat &descriptor_right, const bool &is_right_image, Mat &disparity);

    // parameter set
    parameters param_;

};

#endif // SINEVA_AUTOWARE_ROS_SRC_COMPUTING_PERCEPTION_LOCALIZATION_PACKAGES_SINEVA_STEREO_INCLUDE_ELAS_H_
