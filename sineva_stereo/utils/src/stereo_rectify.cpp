#include <stereo_rectify.h>

StereoRectify::StereoRectify(const std::string &left_calib_path, const std::string &right_calib_path)
{
    loadCalibrationData(left_calib_path, K1, D1, R1, P1, mapx1, mapy1);
    loadCalibrationData(right_calib_path, K2, D2, R2, P2, mapx2, mapy2);

    initUndistortRectifyMap(K1, D1, R1, P1, image_size, CV_32F, mapx1, mapy1);
    initUndistortRectifyMap(K2, D2, R2, P2, image_size, CV_32F, mapx2, mapy2);
}

StereoRectify::~StereoRectify() {}

void StereoRectify::stereoRectify(const cv::Mat &left, const cv::Mat &right, cv::Mat &rec_left, cv::Mat &rec_right)
{
    cv::remap(left, rec_left, mapx1, mapy1, cv::INTER_CUBIC);
    cv::remap(right, rec_right, mapx2, mapy2, cv::INTER_CUBIC);
}

void StereoRectify::loadCalibrationData(const std::string &filename, cv::Mat &K, cv::Mat &D, cv::Mat &R, cv::Mat &P,
                                        cv::Mat &mapx,
                                        cv::Mat &mapy)
{
    cv::FileStorage fileStorage(filename, cv::FileStorage::READ);
    if (!fileStorage.isOpened())
    {
        std::cout << "Open file " << filename << " error." << std::endl;
        exit(0);
    }

    fileStorage["K"] >> K;
    fileStorage["D"] >> D;
    fileStorage["R"] >> R;
    fileStorage["P"] >> P;
    fileStorage["imgsize"] >> image_size;
    fileStorage.release();
}