#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace std;

Mat edgeMap, dirMap ,dst;
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


int findAdj(Point &adj, float dir)
{
    edgeMap.at<uchar>(adj.y, adj.x) = 0;
    if (abs(dir-45)<=22.5 && (int)edgeMap.at<uchar>(adj.y-1, adj.x-1) > 0 )
    {
        adj.x -= 1; adj.y -= 1; return 1;
    }
    else if ((dir<22.5 || dir >337.5) && (int)edgeMap.at<uchar>(adj.y-1, adj.x) > 0)
    {
        adj.y -= 1; return 1;
    }
    else if (abs(dir-315)<=22.5 && (int)edgeMap.at<uchar>(adj.y-1, adj.x+1) > 0)
    {
        adj.x += 1; adj.y -= 1; return 1;
    }
    else if (abs(dir-90)<=22.5 && (int)edgeMap.at<uchar>(adj.y, adj.x+1) > 0)
    {
        adj.x += 1; return 1;
    }
    else if (abs(dir-135)<=22.5 && (int)edgeMap.at<uchar>(adj.y+1, adj.x+1) > 0)
    {
        adj.x += 1; adj.y += 1;  return 1;
    }
    else if (abs(dir-180)<=22.5 && (int)edgeMap.at<uchar>(adj.y+1, adj.x) > 0)
    {
        adj.y += 1; return 1;
    }
    else if (abs(dir-225)<=22.5 && (int)edgeMap.at<uchar>(adj.y+1, adj.x-1) > 0)
    {
        adj.x -= 1; adj.y += 1; return 1;
    }
    else if (abs(dir-270)<=22.5 && (int)edgeMap.at<uchar>(adj.y, adj.x-1) > 0)
    {
        adj.x -= 1; return 1;
    }
    else
        return -1;
}

void extractLineSeg(const Point &seed, vector<Point> &lineSeg, const float &direction, bool reverse)
{
    Point adj;
    adj.x = seed.x;
    adj.y = seed.y;
    float dir = direction;
    int have_adj = 1;
    while (have_adj>0)
    {
        have_adj = findAdj(adj, dir);
        dir = dirMap.at<float>(adj.y, adj.x);
        if (reverse)
            dir = (dir <= 180) ? dir + 180 : dir - 180; 
        // remove(seedlist.begin(),seedlist.end(),adj);
        if (have_adj>0)
            lineSeg.push_back(adj);
        cout << "the x coordinate of the adjacent point: " << adj.x << endl; 
        cout << "the y coordinate of the adjacent point: " << adj.y << endl;
        cout << "the direction of the adjacent point: " << dir << endl; 
    }
}

int main(int argc, char *argv[])
{
    string data_dir = "/home/lee/Sineva/LS_ELAS/line_segment/img/baby1.png";
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
    // namedWindow( window_name, CV_WINDOW_AUTOSIZE );

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
    Canny( edgeMap, edgeMap, 50, 150, 3);
    Mat ini_edge = edgeMap.clone();
    ///imshow( window_name, grad );
    ///imshow( "cart results", mat2gray(magtitude) ); 
    ///imshow( "orientation map", oriMap);


    cv::findNonZero(edgeMap, seedlist);
    cout << "Edge points in total: " << seedlist.size() << endl;
    vector<vector<Point>> lineSegments;
    for (int i = 0; i<seedlist.size(); i++)
    {
        Point seed = seedlist[i];

        float direction = dirMap.at<float>(seed.y, seed.x);
        float rev_direction = (direction<=180) ? direction+180 : direction-180;
        vector<Point> lineSeg;
        extractLineSeg(seed, lineSeg, direction, false);
        extractLineSeg(seed, lineSeg, rev_direction, true);
        cout <<  "the size of the line segments is: " << lineSegments.size() << endl;
        if (lineSeg.size()>2)
            lineSegments.push_back(lineSeg);
        int rd1 = rand()%255;
        int rd2 = rand()%255;
        int rd3 = rand()%255;
        for (auto point:lineSeg)
        {
            src.at<Vec3b>(point.y,point.x)[0] = rd1;
            src.at<Vec3b>(point.y,point.x)[1] = rd2;
            src.at<Vec3b>(point.y,point.x)[2] = rd3;
        }
            
    }


    imshow( "segment results", src);
    waitKey(0);
    return 0;
}

