#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>

#include <algorithm>
#include <cassert>
#include <vector>
#include <future>
#include <stdlib.h>
#include <stdio.h>
#include <algorithm>

#include "elas.h"
#include "descriptor.h"
#include "boost/progress.hpp"


using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace GEOM_FADE2D;

bool Elas::process(Mat &image_left, const Mat &image_right, Mat &disparity_left, Mat &disparity_right)
{
  int32_t width = image_left.cols, height = image_left.rows;
  vector<Point3i> support_points;
  vector<vector<Point>> lineSegments_left;
  vector<vector<Point>> lineSegments_right;
  Mat descriptor_left, descriptor_right;
  Mat edgeMap_left, dirMap_left, edgeMap_right, dirMap_right;
  vector<Segment2> vSegments;

  
  ExtractEdgeSegment(image_left, edgeMap_left, dirMap_left, lineSegments_left, width, height);
  ExtractEdgeSegment(image_right, edgeMap_right, dirMap_right, lineSegments_right, width, height);

  // cout << "there are " << lineSegments_left.size() << " line segments in total." << endl;

  Descriptor descriptor_L(image_left, width, height);
  descriptor_left = descriptor_L.CreateDescriptor();

  Descriptor descriptor_R(image_right, width, height);
  descriptor_right = descriptor_R.CreateDescriptor();

   
  ComputeSupportMatches(image_left, image_right, lineSegments_left, lineSegments_right, support_points, vSegments);
  // cout << "there are " << support_points.size() << " support points in total." << endl; 

  // visulise the segment, triangulation and support points results
  for (auto point:support_points)
  {
    Point sp;
    sp.x = point.x;
    sp.y = point.y;
    circle(image_left, sp, 2.5, Scalar(0,0,255), -1);
    circle(image_left, sp, 2.5, Scalar(0,0,255), -1);
  }

  for (auto line:lineSegments_left)
  {
    int rd1 = rand()%255;
    int rd2 = rand()%255;
    int rd3 = rand()%255;
    for (auto point:line)
    {
      image_left.at<Vec3b>(point.y,point.x)[0] = rd3;
      image_left.at<Vec3b>(point.y,point.x)[1] = rd2;
      image_left.at<Vec3b>(point.y,point.x)[2] = rd1;
    }
  }
    
  future<void> compute_disparity_left = async(launch::async, [&] {
    ComputeDisparity(image_left, support_points, descriptor_left, descriptor_right, false, disparity_left, vSegments);
  });

  future<void> compute_disparity_right = async(launch::async, [&] {
    ComputeDisparity(image_left, support_points, descriptor_left, descriptor_right, true, disparity_right,vSegments);
  });

  compute_disparity_left.wait();
  compute_disparity_right.wait();

  // LeftRightConsistencyCheck(disparity_left, disparity_right, param_.lr_threshold);

  GapInterpolation(width, height, disparity_left);

  Mat bilateraled_disparity_left;
  bilateralFilter(disparity_left, bilateraled_disparity_left, 9, 9 * 2, 9 / 2);

  cv::medianBlur(bilateraled_disparity_left, bilateraled_disparity_left, 5);
  disparity_left = bilateraled_disparity_left.clone();

  return true;
}

void Elas::ExtractEdgeSegment(const Mat &image, Mat &edgeMap, Mat &dirMap, 
vector<vector<Point>> &lineSegments, const int32_t &width, const int32_t &height)
{
  Mat src, grad_x, grad_y, magtitude, edge_temp;
  vector<Point> seedlist;
  bool useDegree = true; 
  int scale = 1;
  int delta = 0;
  int ddepth = CV_32F;
  src = image.clone();

  GaussianBlur(src, edgeMap, Size(5,5), 0, 0, BORDER_DEFAULT);
  //cvtColor(src, edgeMap, CV_BGR2GRAY);
  Sobel( edgeMap, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
  Sobel( edgeMap, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
  cartToPolar(grad_x, grad_y, magtitude, dirMap, useDegree);
  Canny( edgeMap, edgeMap, 20, 100, 5);
  edge_temp = edgeMap.clone();
  findNonZero(edge_temp, seedlist);
  cout << "Edge points in total: " << seedlist.size() << endl;
  for (int i = 0; i<seedlist.size(); i++)
  {
    Point seed = seedlist[i];
    float direction = dirMap.at<float>(seed.y, seed.x);
    float rev_direction = (direction<=180) ? direction+180 : direction-180;
    vector<Point> lineSeg;
    ExtractLineSeg(seed, lineSeg, direction, false, edge_temp, dirMap, width, height);
    ExtractLineSeg(seed, lineSeg, direction, true, edge_temp, dirMap, width, height);
    if (lineSeg.size()>15)
      lineSegments.push_back(lineSeg);
  } 
}

void Elas::ExtractLineSeg(const Point &seed, vector<Point> &lineSeg, const float &direction, 
bool reverse, Mat &edgeMap, const Mat &dirMap, const int32_t &width, const int32_t &height)
{
    Point adj;
    adj.x = seed.x;
    adj.y = seed.y;
    float dir = direction;
    int have_adj = 1;     
    while (have_adj>0)
    {       
        have_adj = findAdj(adj, dir, edgeMap, width, height);
        dir = dirMap.at<float>(adj.y, adj.x);
        if (reverse)
            dir = (dir <= 180) ? dir + 180 : dir - 180; 
        if (have_adj>0)
            lineSeg.push_back(adj);
        // cout << "the x coordinate of the adjacent point: " << adj.x << endl;
        // cout << "the y coordinate of the adjacent point: " << adj.y << endl;

    }
}

int Elas::findAdj(Point &adj, float dir, Mat &edgeMap, const int32_t &width, const int32_t &height)
{
    edgeMap.at<uchar>(adj.y, adj.x) = 0;
    vector<Point> adjcentPoints(8,adj);
    adjcentPoints[0].y -= 1;adjcentPoints[1].x += 1;
    adjcentPoints[1].y -= 1;adjcentPoints[2].x += 1;
    adjcentPoints[3].x += 1;adjcentPoints[3].y += 1;
    adjcentPoints[4].y += 1;adjcentPoints[5].x -= 1;
    adjcentPoints[5].y += 1;adjcentPoints[6].x -= 1;
    adjcentPoints[7].x -= 1;adjcentPoints[7].y -= 1;
    for (int i=0; i<adjcentPoints.size(); i++)
    {
        Point adjpt = adjcentPoints[i];
        if ((int)edgeMap.at<uchar>(adjpt.y, adjpt.x) > 0 && (adjpt.x > 0) && (adjpt.x < width) && (adjpt.y>0) && (adjpt.y < height))
        {
            adj.x = adjpt.x;
            adj.y = adjpt.y;
            return 1;
        }
    }
    return -1;
}

void Elas::ComputeDisparity(Mat &image_left, vector<Point3i> support_points, const Mat &descriptor_left, const Mat &descriptor_right,
                            const bool &is_right_image, Mat &disparity, vector<Segment2> &vSegments)
{
  
  vector<Vec6f> triangulate_points, plane_param, constrained_triangulate_points;
  vector<Vec3d> triangulate_points_d;

  int32_t grid_width = (int32_t)ceil(disparity.cols / param_.grid_size);
  int32_t grid_height = (int32_t)ceil(disparity.rows / param_.grid_size);

  Mat disparity_grid_left(grid_height * grid_width, param_.disp_max + 2, CV_16S, Scalar(0));
  Mat disparity_grid_right(grid_height * grid_width, param_.disp_max + 2, CV_16S, Scalar(0));

  // triangulate_points = ComputeDelaunayTriangulation(support_points, is_right_image, disparity.cols, disparity.rows);


  Fade_2D dt;

  ConstraintGraph2* pCG = dt.createConstraint(vSegments, CIS_CONSTRAINED_DELAUNAY);
  dt.applyConstraintsAndZones();
	Visualizer2 vis2("Triangulation_with_constraints.ps");
	dt.show(&vis2);
  vector<Triangle2*> vAllTriangles;
  dt.getTrianglePointers(vAllTriangles);
  
  triangulate_points = ComputeConstrainedDelaunayTriangulation(vAllTriangles, is_right_image, disparity.cols, disparity.rows);



 
  if (!is_right_image)
  {
    int point_num_in_tri = 3;
    for (int i=0; i<triangulate_points.size(); i++)
    {
      Point supportPoints[3];
      for (int k=0; k<3; k++)
      {
        int32_t u = (int32_t)triangulate_points[i][2*k];
        int32_t v = (int32_t)triangulate_points[i][2*k+1];
        supportPoints[k] = Point(u,v);
      }
      const cv::Point* ppt = supportPoints;
      polylines(image_left, &ppt, &point_num_in_tri, 1, true , cv::Scalar(100,100,100));
    }
  }

  plane_param = ComputeDisparityPlanes(support_points, triangulate_points, triangulate_points_d, is_right_image);

  CreateGrid(support_points, disparity_grid_left, Size(grid_width, grid_height), is_right_image);

  ComputeDisparity(descriptor_left, descriptor_right, triangulate_points, plane_param, disparity_grid_left,
                   Size(grid_width, grid_height), is_right_image, triangulate_points_d, disparity.cols,
                   disparity.rows, disparity);
}

void Elas::LeftRightConsistencyCheck(Mat &disparity_left, Mat &disparity_right, const int &lr_threshold)
{
  // get disparity image dimensions
  int32_t D_width = disparity_left.cols, D_height = disparity_left.rows;

  // loop variables
  float u_warp, d;

  // for all image points do
  for (int32_t u = 0; u < D_width; u++)
  {
    for (int32_t v = 0; v < D_height; v++)
    {
      // compute address (u,v) and disparity value
      d = disparity_left.at<float>(v, u);
      u_warp = (float)u - d;

      // check if left disparity is valid
      if (d >= 0 && u_warp >= 0)
      {
        if (fabs(disparity_right.at<float>(v, u_warp) - d) > lr_threshold)
          disparity_left.at<float>(v, u) = -1;
      }
      else
      {
        disparity_left.at<float>(v, u) = -1;
      }
    }
  }
}


// Using ORB descriptor for LS-ELAS support matching
void Elas::ComputeSupportMatches(const Mat &image_left, const Mat &image_right, const vector<vector<Point>> &lineSegments_left, 
const vector<vector<Point>> &lineSegments_right, vector<Point3i> &support_points, vector<Segment2> &vSegments)
{
  
  Ptr<DescriptorExtractor> extractor;

  extractor = ORB::create(500, 1.2f, 8, 2);
  
  Mat d_descriptorsL, d_descriptorsR;

  vector<KeyPoint> keyPoints_1, keyPoints_2;


  int candidate_stepsize = param_.candidate_stepsize;

  assert(candidate_stepsize>0);

  int apertureSize = 5;
  int min_y = 100000;
  int max_y = 0;
// Select keypoints from line segments results in left image
  for (auto line:lineSegments_left)
  { 
    for (int i = candidate_stepsize; i < line.size(); i += candidate_stepsize)
    {
      int u = line[i].x; 
      int v = line[i].y;
      KeyPoint newKeyPoint;
      newKeyPoint.pt = Point2f(line[i].x, line[i].y);
      newKeyPoint.size = 2*apertureSize;
      keyPoints_1.push_back(newKeyPoint);
    }
  }

// Select keypoints from line segments results in right image
  for (auto line:lineSegments_right)
  { 
    for (int i = candidate_stepsize; i < line.size(); i += candidate_stepsize)
    {
      // cout << "the index of the line: " << i << endl; 
      int u = line[i].x; 
      int v = line[i].y;
      KeyPoint newKeyPoint;
      newKeyPoint.pt = Point2f(line[i].x, line[i].y);
      newKeyPoint.size = 2*apertureSize;
      keyPoints_2.push_back(newKeyPoint);
    }
  }

  extractor->compute(image_left, keyPoints_1, d_descriptorsL);
  extractor->compute(image_right, keyPoints_2, d_descriptorsR);

  bool crossCheck = false;
  Ptr<DescriptorMatcher> matcher;
  vector<DMatch> matches;
  int normType = cv::NORM_HAMMING; 
  matcher = BFMatcher::create(normType, crossCheck);
  matcher->match(d_descriptorsL, d_descriptorsR, matches);

  for (int i = 0; i < keyPoints_1.size(); i++)
  {
    if (keyPoints_1[i].pt.y < min_y) min_y = keyPoints_1[i].pt.y;
    if (keyPoints_1[i].pt.y > max_y) max_y = keyPoints_1[i].pt.y;

    if (abs(keyPoints_1[i].pt.y - keyPoints_2[matches[i].trainIdx].pt.y) < 2)
    {
      support_points.push_back(Point3i(keyPoints_1[i].pt.x, keyPoints_1[i].pt.y,
      keyPoints_1[i].pt.x - keyPoints_2[matches[i].trainIdx].pt.x));
    }
  }

  for (auto line:lineSegments_left)
  {
    int last_valid = -1;
    for (int i = candidate_stepsize; i < line.size(); i += candidate_stepsize)
    {
      for (int k = 0; k < support_points.size(); k++)
      {
        if ((line[i].x==support_points[k].x)&&(line[i].y==support_points[k].y))
        {
          if (last_valid < 0) last_valid=i;
          else 
          { 
            Point2 strt = Point2(line[last_valid].x, line[last_valid].y);
            Point2 end = Point2(line[i].x, line[i].y);
            vSegments.push_back(Segment2(strt, end));
            last_valid = i;
          }
        }
      }
    }
  }
}



int Elas::ComputeMatchingDisparity(const Mat &descriptor_left, const Mat &descriptor_right, const int &u, const int &v, 
const bool &right_image, const int32_t &width, const int32_t &height)
{
    if (u < 1 || u > width-2 || v < 1 || v > height-2)
        return -1;

    const int up_shift = (v - 1) * width + u;
    const int left_shift = v * width + u - 1;
    const int right_shift = v * width + u + 1;
    const int bottom_shift = (v + 1) * width + u;

    int disp_min_valid = max(param_.disp_min, 0);
    int disp_max_valid = param_.disp_max;

    if (right_image)
        disp_max_valid = min(param_.disp_max, width - u);
    else
        disp_max_valid = min(param_.disp_max, u);

    const uchar *left_up, *left_left, *left_right, *left_bottom, *right_up, *right_left, *right_right, *right_bottom;
    int sum, min_energy = INT_MAX, sec_min_energy = INT_MAX, min_disparity = -1, sec_min_disparity = -1;

    for (int d = disp_min_valid; d <= disp_max_valid; ++d)
    {
        if (right_image)
        {
            left_up = descriptor_left.ptr<uchar>(up_shift + d);
            left_left = descriptor_left.ptr<uchar>(left_shift + d);
            left_right = descriptor_left.ptr<uchar>(right_shift + d);
            left_bottom = descriptor_left.ptr<uchar>(bottom_shift + d);

            right_up = descriptor_left.ptr<uchar>(up_shift);
            right_left = descriptor_left.ptr<uchar>(left_shift);
            right_right = descriptor_left.ptr<uchar>(right_shift);
            right_bottom = descriptor_left.ptr<uchar>(bottom_shift);
        }
        else
        {
            left_up = descriptor_left.ptr<uchar>(up_shift);
            left_left = descriptor_left.ptr<uchar>(left_shift);
            left_right = descriptor_left.ptr<uchar>(right_shift);
            left_bottom = descriptor_left.ptr<uchar>(bottom_shift);

            right_up = descriptor_left.ptr<uchar>(up_shift - d);
            right_left = descriptor_left.ptr<uchar>(left_shift - d);
            right_right = descriptor_left.ptr<uchar>(right_shift - d);
            right_bottom = descriptor_left.ptr<uchar>(bottom_shift - d);
        }

        sum = 0;
        for (int i = 0; i < 16; ++i)
        {
            sum += abs(left_up[i] - right_up[i]);
            sum += abs(left_left[i] - right_left[i]);
            sum += abs(left_right[i] - right_right[i]);
            sum += abs(left_bottom[i] - right_bottom[i]);
        }

        if (sum == 0)
            continue;

        if (sum < min_energy)
        {
            sec_min_energy = min_energy;
            sec_min_disparity = min_disparity;
            min_energy = sum;
            min_disparity = d;
        }
        else if (sum < sec_min_energy)
        {
            sec_min_energy = sum;
            sec_min_disparity = d;
        }
    }

    if (min_disparity >= 0 && sec_min_disparity >= 0 && (float)min_energy < param_.support_threshold * (float)sec_min_energy)
        return min_disparity;
    else
        return -1;
}
vector<Vec6f>
Elas::ComputeDelaunayTriangulation(const vector<Point3i> &support_points, const bool &right_image, const int32_t &width,
                                   const int32_t &height)
{
  Subdiv2D subdiv(Rect(0, 0, width, height));

  for (const auto point : support_points)
  {
    if (right_image)
      subdiv.insert(Point2f(static_cast<float>(point.x - point.z), static_cast<float>(point.y)));
    else
      subdiv.insert(Point2f(static_cast<float>(point.x), static_cast<float>(point.y)));
  }

  vector<Vec6f> triangles;
  subdiv.getTriangleList(triangles);
  for (vector<Vec6f>::iterator it = triangles.begin(); it != triangles.end();)
  {
    if ((*it)[0] > width || (*it)[2] > width || (*it)[4] > width ||
        (*it)[1] > height || (*it)[3] > height || (*it)[5] > height ||
        (*it)[0] < 0 || (*it)[2] < 0 || (*it)[4] < 0 ||
        (*it)[1] < 0 || (*it)[3] < 0 || (*it)[5] < 0)
    {
      it = triangles.erase(it);
      continue;
    }
    it++;
  }

  return triangles;
}

vector<Vec6f>
Elas::ComputeConstrainedDelaunayTriangulation(vector<Triangle2*> &vAllTriangles, const bool &right_image, const int32_t &width,
                                   const int32_t &height)
{
  vector<Vec6f> triangles;
  for (vector<Triangle2*>::iterator it(vAllTriangles.begin());it!=vAllTriangles.end();++it)
  {
    Triangle2* pT(*it);
    Vec6f triangle;
    bool valid = true;
    for (int i=0; i<3; ++i)
    {
      Point2* pCorner(pT->getCorner(i));
      // if (pCorner->x() > width || pCorner->y() > height
      // || pCorner->x() < 0 || pCorner->y() < 0) 
      // {
      //   valid = false;
      //   break;
      // }
      // else 
      triangle[2*i] = (float)pCorner->x();
      triangle[2*i+1] = (float)pCorner->y(); 
      //cout << "the cordinate of x is: "<< triangle[2*i] << endl;
    }
  triangles.push_back(triangle);
  }
  
  return triangles;
}

vector<Vec6f>
Elas::ComputeDisparityPlanes(const vector<Point3i> &support_points, const vector<Vec6f> &triangulate_points,
                             vector<Vec3d> &triangulate_d, const bool &right_image)
{
  vector<Vec6f> result;

  for (auto tri_point : triangulate_points)
  {
    Mat A_right, b, A_left;
    double d1, d2, d3;

    for (auto point : support_points)
    {
      int32_t search_x = right_image ? point.x - point.z : point.x;

      if (search_x == tri_point[0] && point.y == tri_point[1])
        d1 = point.z;
      if (search_x == tri_point[2] && point.y == tri_point[3])
        d2 = point.z;
      if (search_x == tri_point[4] && point.y == tri_point[5])
        d3 = point.z;
    }
    b = (Mat_<double>(3, 1) << d1, d2, d3);

    Mat result_left, result_right;
    Vec6f plane_param(0, 0, 0, 0, 0, 0);
    Vec3d tri_d(0, 0, 0);

    if (right_image)
    {
      A_right = (Mat_<double>(3, 3) << tri_point[0], tri_point[1], 1, tri_point[2], tri_point[3], 1, tri_point[4], tri_point[5], 1);
      A_left = (Mat_<double>(3, 3) << tri_point[0] + d1, tri_point[1], 1, tri_point[2] + d2, tri_point[3], 1,
                tri_point[4] + d3, tri_point[5], 1);
    }
    else
    {
      A_left = (Mat_<double>(3, 3) << tri_point[0], tri_point[1], 1, tri_point[2], tri_point[3], 1, tri_point[4], tri_point[5], 1);
      A_right = (Mat_<double>(3, 3) << tri_point[0] - d1, tri_point[1], 1, tri_point[2] - d2, tri_point[3], 1,
                 tri_point[4] - d3, tri_point[5], 1);
    }

    if (solve(A_left, b, result_left) && solve(A_right, b, result_right))
    {
      plane_param[0] = result_left.at<double>(0, 0);
      plane_param[1] = result_left.at<double>(0, 1);
      plane_param[2] = result_left.at<double>(0, 2);

      plane_param[3] = result_right.at<double>(0, 0);
      plane_param[4] = result_right.at<double>(0, 1);
      plane_param[5] = result_right.at<double>(0, 2);

      tri_d[0] = d1;
      tri_d[1] = d2;
      tri_d[2] = d3;
    }

    result.push_back(plane_param);
    triangulate_d.push_back(tri_d);
  }

  return result;
}

void Elas::CreateGrid(const vector<Point3i> &support_points, Mat &disparity_grid, const Size &disparity_grid_size,
                      const bool &right_image)
{
  int32_t grid_width = disparity_grid_size.width;
  int32_t grid_height = disparity_grid_size.height;

  Mat disparity_grid_temp(grid_height * grid_width, param_.disp_max + 1, CV_16S, Scalar(0));

  for (auto &point : support_points)
  {
    int32_t x_curr = point.x, y_curr = point.y, d_curr = point.z;
    int32_t d_min = max(d_curr - 1, 0), d_max = min(d_curr + 1, param_.disp_max);
    int32_t x, y = floor(y_curr / param_.grid_size);

    x = right_image ? floor((x_curr - d_curr) / param_.grid_size) : floor(x_curr / param_.grid_size);

    if (x >= 0 && x < grid_width && y >= 0 && y < grid_height)
    {
      for (int32_t d = d_min; d < d_max; ++d)
      {
        disparity_grid_temp.ptr<int16_t>(x + grid_width * y)[d] = 1;
      }
    }
  }

  for (int32_t y = 1; y < grid_height - 1; y++)
  {
    for (int32_t x = 1; x < grid_width - 1; x++)
    {
      int32_t curr_ind = 1;
      const uint16_t *tl = disparity_grid_temp.ptr<uint16_t>((x - 1) + grid_width * (y - 1));
      const uint16_t *tc = disparity_grid_temp.ptr<uint16_t>(x + grid_width * (y - 1));
      const uint16_t *tr = disparity_grid_temp.ptr<uint16_t>((x + 1) + grid_width * (y - 1));
      const uint16_t *cl = disparity_grid_temp.ptr<uint16_t>((x - 1) + grid_width * y);
      const uint16_t *cc = disparity_grid_temp.ptr<uint16_t>(x + grid_width * y);
      const uint16_t *cr = disparity_grid_temp.ptr<uint16_t>((x + 1) + grid_width * y);
      const uint16_t *bl = disparity_grid_temp.ptr<uint16_t>((x - 1) + grid_width * (y + 1));
      const uint16_t *bc = disparity_grid_temp.ptr<uint16_t>(x + grid_width * (y + 1));
      const uint16_t *br = disparity_grid_temp.ptr<uint16_t>((x + 1) + grid_width * (y + 1));

      for (int32_t d = 0; d <= param_.disp_max; d++)
      {
        if (tl[d] || tc[d] || tr[d] || cl[d] || cc[d] || cr[d] || bl[d] || bc[d] || br[d])
        {
          disparity_grid.at<int16_t>(x + grid_width * y, curr_ind) = d;
          curr_ind++;
        }
      }
      disparity_grid.at<int16_t>(x + grid_width * y, 0) = curr_ind - 1;
    }
  }
}

void Elas::ComputeDisparity(const Mat &descriptor_left, const Mat &descriptor_right, const vector<Vec6f> &triangulate_points,
                            const vector<Vec6f> &plane_params, const Mat &disparity_grid, const Size &disparity_grid_size,
                            const bool &right_image, const vector<Vec3d> &triangulate_d, const int32_t &width,
                            const int32_t &height, Mat &disparity)
{
  // pre-compute prior
  float two_sigma_squared = 2 * param_.sigma * param_.sigma;
  int32_t *P = new int32_t[param_.disp_max + 1];
  for (int32_t delta_d = 0; delta_d < param_.disp_max + 1; delta_d++)
    P[delta_d] = (int32_t)((-log(param_.gamma + exp(-delta_d * delta_d / two_sigma_squared)) + log(param_.gamma)) /
                           param_.beta);
  int32_t plane_radius = (int32_t)max(static_cast<float>(ceil(param_.sigma * param_.sradius)),
                                      static_cast<float>(2.0));

  float plane_a, plane_b, plane_c, plane_d;

  for (int32_t i = 0; i < plane_params.size(); i++)
  {
    if (triangulate_d[i][0] == 0 && triangulate_d[i][1] == 0 && triangulate_d[i][2] == 0)
      continue;

    float tri_u[3], tri_v[3] = {triangulate_points[i][1], triangulate_points[i][3], triangulate_points[i][5]};

    plane_a = right_image ? plane_params[i][3] : plane_params[i][0];
    plane_b = right_image ? plane_params[i][4] : plane_params[i][1];
    plane_c = right_image ? plane_params[i][5] : plane_params[i][2];
    plane_d = right_image ? plane_params[i][0] : plane_params[i][3];

    tri_u[0] = right_image ? (triangulate_points[i][0] - triangulate_d[i][0]) : triangulate_points[i][0];
    tri_u[1] = right_image ? (triangulate_points[i][2] - triangulate_d[i][1]) : triangulate_points[i][2];
    tri_u[2] = right_image ? (triangulate_points[i][4] - triangulate_d[i][2]) : triangulate_points[i][4];

    for (uint32_t j = 0; j < 3; j++)
    {
      for (uint32_t k = 0; k < j; k++)
      {
        if (tri_u[k] > tri_u[j])
        {
          float tri_u_temp = tri_u[j];
          tri_u[j] = tri_u[k];
          tri_u[k] = tri_u_temp;
          float tri_v_temp = tri_v[j];
          tri_v[j] = tri_v[k];
          tri_v[k] = tri_v_temp;
        }
      }
    }

    // rename corners
    float A_u = tri_u[0];
    float A_v = tri_v[0];
    float B_u = tri_u[1];
    float B_v = tri_v[1];
    float C_u = tri_u[2];
    float C_v = tri_v[2];

    // compute straight lines connecting triangle corners
    float AB_a = ((int32_t)(A_u) == (int32_t)(B_u)) ? 0 : (A_v - B_v) / (A_u - B_u);
    float AC_a = ((int32_t)(A_u) == (int32_t)(C_u)) ? 0 : (A_v - C_v) / (A_u - C_u);
    float BC_a = ((int32_t)(B_u) == (int32_t)(C_u)) ? 0 : (B_v - C_v) / (B_u - C_u);

    float AB_b = A_v - AB_a * A_u;
    float AC_b = A_v - AC_a * A_u;
    float BC_b = B_v - BC_a * B_u;

    // a plane is only valid if itself and its projection
    // into the other image is not too much slanted
    bool valid = fabs(plane_a) < 0.7 && fabs(plane_d) < 0.7;

    // first part (triangle corner A->B)
    if (AB_a != 0 || BC_a != 0)
    {
      for (int32_t u = max((int32_t)A_u, 0); u < min((int32_t)C_u, width); u++)
      {
        int32_t v_1 = (uint32_t)(AC_a * static_cast<float>(u) + AC_b);
        int32_t v_2 = u < B_u ? (uint32_t)(AB_a * static_cast<float>(u) + AB_b) : (uint32_t)(BC_a * static_cast<float>(u) + BC_b);
        for (int32_t v = min(v_1, v_2); v < max(v_1, v_2); v++)
        {
          FindMatch(descriptor_left, descriptor_right, u, v, plane_a, plane_b, plane_c, disparity_grid,
                    disparity_grid_size, P, plane_radius,
                    valid, right_image, width, height, disparity);
        }
      }
    }
  }
}

void Elas::FindMatch(const Mat &descriptor_left, const Mat &descriptor_right, const int32_t &u, const int32_t &v,
                     const float &plane_a, const float &plane_b, const float &plane_c, const Mat &disparity_grid,
                     const Size &disparity_grid_size, int32_t *P, const int32_t &plane_radius, const bool &valid,
                     const bool &right_image, const int32_t &width, const int32_t &height, Mat &disparity)
{
  uint32_t grid_width = disparity_grid_size.width;
  const int32_t window_size = 2;
  if (u < window_size || u >= width - window_size)
    return;

  int32_t sum = 0;
  const uchar *data_desc1, *data_desc2;

  data_desc1 = right_image ? descriptor_right.ptr<uchar>(v * width + u) : descriptor_left.ptr<uchar>(v * width + u);

  // compute disparity, min disparity and max disparity of plane prior
  int32_t d_plane = (int32_t)(plane_a * static_cast<float>(u) + plane_b * static_cast<float>(v) + plane_c);
  int32_t d_plane_min = max(d_plane - plane_radius, 0);
  int32_t d_plane_max = min(d_plane + plane_radius, param_.disp_max);

  // get grid pointer
  int32_t grid_x = (int32_t)floor(static_cast<float>(u) / static_cast<float>(param_.grid_size));
  int32_t grid_y = (int32_t)floor(static_cast<float>(v) / static_cast<float>(param_.grid_size));

  int32_t num_grid = disparity_grid.at<int16_t>(grid_y * grid_width + grid_x, 0);
  const uint16_t *d_grid = disparity_grid.ptr<uint16_t>(grid_y * grid_width + grid_x);
  int32_t u_warp, min_val = INT_MAX, min_d = -1;

  uint16_t d_curr;
  for (int32_t i = 0; i < num_grid; i++)
  {
    d_curr = d_grid[i + 1];
    if (d_curr < d_plane_min || d_curr > d_plane_max)
    {
      u_warp = right_image ? u + d_curr : u - d_curr;
      if (u_warp < window_size || u_warp >= width - window_size)
        continue;
      data_desc2 = descriptor_right.ptr<uchar>(v * width + u - d_curr);

      sum = 0;
      for (int32_t i = 0; i < 16; ++i)
      {
        sum += abs(data_desc1[i] - data_desc2[i]);
      }

      if (sum < min_val)
      {
        min_val = sum;
        min_d = d_curr;
      }

      // set disparity value
      if (min_d >= param_.disp_min && min_d < param_.disp_max)
        disparity.at<float>(v, u) = min_d; // MAP value (min neg-Log probability)
      else
        disparity.at<float>(v, u) = -1; // invalid disparity
    }
  }

  for (d_curr = d_plane_min; d_curr <= d_plane_max; d_curr++)
  {
    sum = 0;
    u_warp = right_image ? u + d_curr : u - d_curr;
    if (u_warp < window_size || u_warp >= width - window_size)
      continue;
    data_desc2 = descriptor_right.ptr<uchar>(v * width + u - d_curr);

    for (int32_t i = 0; i < 16; ++i)
    {
      sum += abs(data_desc1[i] - data_desc2[i]);
    }

    sum += valid ? *(P + abs(d_curr - d_plane)) : 0;
    if (sum < min_val)
    {
      min_val = sum;
      min_d = d_curr;
    }
  }
  // set disparity value
  if (min_d >= param_.disp_min && min_d < param_.disp_max)
    disparity.at<float>(v, u) = min_d; // MAP value (min neg-Log probability)
  else
    disparity.at<float>(v, u) = -1; // invalid disparity
}

void Elas::GapInterpolation(const int32_t &width, const int32_t &height, Mat &disparity)
{
  int32_t D_ipol_gap_width = param_.ipol_gap_width;

  // discontinuity threshold
  float discon_threshold = 3.0;

  // declare loop variables
  int32_t loop_first, loop_last;
  float d1, d2, d_ipol;

  // for both row-wise and column-wise
  bool flag = true;
  for (int32_t v = 0; v < height; v++)
  {
    // init counter
    int count = 0;

    // for each element of the row do
    for (int32_t u = 0; u < width; u++)
    {
      // if disparity valid
      if (disparity.at<float>(v, u) >= 0)
      {
        // check if speckle is small enough
        if (count >= 1 && count <= D_ipol_gap_width)
        {
          // first and last value for interpolation
          if (flag)
          {
            loop_first = u - count;
            loop_last = u - 1;
            flag = false;
          }
          else
          {
            loop_first = v - count;
            loop_last = v - 1;
          }

          // if value in range
          if (loop_first > 0 && loop_last < width - 1)
          {
            d1 = disparity.at<float>(v, loop_first - 1);
            d2 = disparity.at<float>(v, loop_last + 1);
            if (fabs(d1 - d2) < discon_threshold)
              d_ipol = (d1 + d2) / 2;
            else
              d_ipol = min(d1, d2);

            // set all values to d_ipol
            for (int32_t curr = loop_first; curr <= loop_last; curr++)
              disparity.at<float>(v, curr) = d_ipol;
          }
        }

        // reset counter
        count = 0;

        // otherwise increment counter
      }
      else
      {
        count++;
      }
    }
  }
}
