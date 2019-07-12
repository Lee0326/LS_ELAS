#include "elas.h"
#include "descriptor.h"

int main(int argc, char *argv[])
{
    string elas_para_file = argv[1];

    Elas::parameters param;
    FileStorage elas_para(elas_para_file, FileStorage::READ);

    elas_para["disp_min"] >> param.disp_min;
    elas_para["disp_max"] >> param.disp_max;
    elas_para["grid_size"] >> param.grid_size;
    elas_para["beta"] >> param.beta;
    elas_para["gamma"] >> param.gamma;
    elas_para["sigma"] >> param.sigma;
    elas_para["sradius"] >> param.sradius;
    elas_para["lr_threshold"] >> param.lr_threshold;
    elas_para["speckle_size"] >> param.speckle_size;
    elas_para["ipol_gap_width"] >> param.ipol_gap_width;

    Elas elas(param);

    Mat test_img_left(1280, 720, CV_8U, Scalar(0));
    Mat test_img_right(1280, 720, CV_8U, Scalar(0));
    
    for (size_t i = 0; i < 1280; i++)
    {
        for (size_t j = 0; j < 720; j++)
        {
            if (i % 100 < 50 && j % 100 < 50)
                test_img_left.at<uchar>(j, i) = 255;

            if (i % 100 >= 50 && j % 100 < 50)
                test_img_right.at<uchar>(j, i) = 255;
        }
    }
}
