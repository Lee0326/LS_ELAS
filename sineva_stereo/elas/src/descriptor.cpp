#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "../include/descriptor.h"

#include <future>

#include "boost/progress.hpp"

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

Descriptor::Descriptor(const Mat &img, const int32_t &width, const int32_t &height)
        : width_(width), height_(height), image(img.clone())
{
    Mat filtered_x, filtered_y;

    future<void> ft1 = async(launch::async, [&] {
        Sobel(img, filtered_x, CV_16S, 1, 0, 3, 1, 1, BORDER_DEFAULT);
        img.convertTo(grad_x, CV_8U);
    });

    future<void> ft2 = async(launch::async, [&] {
        Sobel(img, filtered_y, CV_16S, 0, 1, 3, 1, 1, BORDER_DEFAULT);
        img.convertTo(grad_y, CV_8U);
    });

    ft1.wait();
    ft2.wait();
}

Descriptor::~Descriptor()
{
}

Mat Descriptor::CreateDescriptor()
{
    Mat descriptor(width_ * height_, 16, CV_8U, Scalar(0));

    int32_t step = grad_x.step;

    for (int32_t v = 2; v < height_ - 2; ++v)
    {
        const uchar *data_x = grad_x.ptr<uchar>(v);
        const uchar *data_y = grad_y.ptr<uchar>(v);

        for (int32_t u = 2; u < width_ - 2; ++u)
        {
            uchar *data_desc = descriptor.ptr<uchar>(v * width_ + u);
            data_desc[0] = (int32_t) data_x[u - 2 * step];
            data_desc[1] = (int32_t) data_x[u - step - 2];
            data_desc[2] = (int32_t) data_x[u - step];
            data_desc[3] = (int32_t) data_x[u - step + 2];
            data_desc[4] = (int32_t) data_x[u - 1];
            data_desc[5] = (int32_t) data_x[u];
            data_desc[6] = (int32_t) data_x[u];
            data_desc[7] = (int32_t) data_x[u + 1];
            data_desc[8] = (int32_t) data_x[u + step - 2];
            data_desc[9] = (int32_t) data_x[u + step];
            data_desc[10] = (int32_t) data_x[u + step + 2];
            data_desc[11] = (int32_t) data_x[u + 2 * step];
            data_desc[12] = (int32_t) data_y[u - step];
            data_desc[13] = (int32_t) data_y[u - 1];
            data_desc[14] = (int32_t) data_y[u + 1];
            data_desc[15] = (int32_t) data_y[u + step];
        }
    }

    return descriptor.clone();
}
