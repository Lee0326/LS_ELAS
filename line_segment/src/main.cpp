#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace std;

Mat edgeMap, dirMap;
vector<Point> seedlist;

Mat mat2gray(const cv::Mat& src)
{
    Mat dst;
    normalize(src, dst, 0.0, 255.0, cv::NORM_MINMAX, CV_8U);

    return dst;
}
Mat orientationMap(const cv::Mat& mag, const cv::Mat& ori, double thresh = 1.0)
{
    Mat oriMap = Mat::zeros(ori.size(), CV_8UC3); 
    Vec3b red(0, 0, 255);
    Vec3b cyan(255, 255, 0);
    Vec3b green(0, 255, 0);
    Vec3b yellow(0, 255, 255);
    for(int i = 0; i < mag.rows*mag.cols; i++)
    {
        float* magPixel = reinterpret_cast<float*>(mag.data + i*sizeof(float));
        if(*magPixel > thresh)
        {
            float* oriPixel = reinterpret_cast<float*>(ori.data + i*sizeof(float));
            Vec3b* mapPixel = reinterpret_cast<Vec3b*>(oriMap.data + i*3*sizeof(char));
            if(*oriPixel < 90.0)
                *mapPixel = red;
            else if(*oriPixel >= 90.0 && *oriPixel < 180.0)
                *mapPixel = cyan;
            else if(*oriPixel >= 180.0 && *oriPixel < 270.0)
                *mapPixel = green;
            else if(*oriPixel >= 270.0 && *oriPixel < 360.0)
                *mapPixel = yellow;
        }
    }

    return oriMap;
}

void reverse(float &direction)
{
    if (direction <= 90)
    {
        direction += 180;
    }
    else
    {
        direction -+ 180;
    } 
}

int findAdj(Point &adj, float dir)
{

}
void extractLineSeg(const Point &seed, vector<Point> &lineSeg, const float &direction)
{
    Point adj;
    adj.x = seed.x;
    adj.y = seed.y;
    float dir = direction;
    lineSeg.push_back(seed);
    int have_adj = 1;
    while (have_adj>0)
    {
        have_adj = findAdj(adj, dir);
        dir = dirMap.at<float>(adj.x, adj.y);
        remove(seedlist.begin(),seedlist.end(),adj);
    }
}

int main(int argc, char *argv[])
{
    string data_dir = "/home/colin/catkin_ls_ws/src/line_segment/img/left.jpg";
    Mat src, src_gray;
    Mat grad;
    char* window_name = "Sobel Edge Detection";
    int scale = 1;
    int delta = 0;
    int ddepth = CV_32F;

    int c;

    // Load an image
    src = imread( data_dir );

    if ( !src.data )
    { return -1; }
    /// Convert the image to gray
    GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT); 
    cvtColor( src, src_gray, CV_BGR2GRAY );

    /// Create Window
    namedWindow( window_name, CV_WINDOW_AUTOSIZE );

    /// Generate grad_x and grad_y
    Mat grad_x, grad_y;
    /// Mat abs_grad_x, abs_grad_y;

    /// Gradient X
    Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
    /// convertScaleAbs( grad_x, abs_grad_x );

    Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
    /// convertScaleAbs( grad_y, abs_grad_y );
      /// Total Gradient (approximate)
    bool useDegree = true; 
    Mat magtitude; 
    cartToPolar(grad_x, grad_y, magtitude, dirMap, useDegree);
    /// addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
    // Mat ori;
    // phase(grad_x, grad_y, ori, true);
    Mat oriMap = orientationMap(magtitude, dirMap, 1.0);
    


    edgeMap = src_gray.clone();
    Canny( edgeMap, edgeMap, 10, 100, 3);
    ///imshow( window_name, grad );
    ///imshow( "cart results", mat2gray(magtitude) ); 
    ///imshow( "orientation map", oriMap);
    imshow( "canny results", edgeMap);


    cv::findNonZero(edgeMap, seedlist);
    cout << "Edge points in total: " << seedlist.size() << endl;

    vector<vector<Point>> lineSegments;
    for (auto seed:seedlist)
    {
        float direction = dirMap.at<float>(seed.x, seed.y);
        vector<Point> lineSeg;
        extractLineSeg(seed, lineSeg, direction);
        reverse(direction);
    }
    waitKey(0);
    return 0;
}

