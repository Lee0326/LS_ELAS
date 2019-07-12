#include <unistd.h>
#include <sys/time.h>
#include <dirent.h>
#include <string>

#include <cv.hpp>
#include <opencv2/imgcodecs.hpp>

#include "stereo_match.h"
#include "stereo_rectify.h"

#define BOOST_ENABLE_ASSERT_HANDLER
#include <boost/format.hpp>
#include <boost/assert.hpp>

using boost::format;
using cv::COLORMAP_HSV;
using cv::cvtColor;
using cv::FileStorage;
using cv::imread;
using cv::Mat;
using cv::Size;
using std::cout;
using std::endl;
using std::string;
using std::vector;

bool kTestFailed = false;
void boost::assertion_failed(char const *expr, char const *function, char const *file, long line)
{
  format fmt("Assertion failed!\n Expression: %s\nFunction:%s\nFile: %s\nLine: %ld\n\n");
  fmt % expr % function % file % line;
  cout << fmt << endl;
  kTestFailed = true;
}

vector<string> GetDirFile(const string dirPath, const char *extenStr)
{
  vector<string> files_name;
  DIR *dir = opendir(dirPath.c_str());
  dirent *pDirent = NULL;
  while ((pDirent = readdir(dir)) != NULL)
  {
    if (strstr(pDirent->d_name, extenStr))
    {
      files_name.push_back(string(pDirent->d_name));
    }
  }
  closedir(dir);
  return files_name;
}

int analysizeDepth(const cv::Mat &depth_map, const vector<float> &ranges, vector<float> &cumu_dist)
{
  if (depth_map.empty())
  {
    cout << "Depth map is empty, please check..." << endl;
    return 0;
  }

  int n_valid_points = 0;
  for (int i = 0; i < depth_map.rows; i++)
  {
    for (int j = 0; j < depth_map.cols; j++)
    {
      float d = depth_map.at<unsigned short>(i, j) / 1000.;
      if (d > ranges[0] && d < ranges[ranges.size() - 1])
        n_valid_points++;

      for (int v = 0; v < ranges.size(); ++v)
      {
        double min_dist = v == 0 ? 0 : ranges[v - 1];
        if (d > min_dist && d <= ranges[v])
          cumu_dist[v]++;
      }
    }
  }

  int N = depth_map.rows * depth_map.cols;
  for (int v = 0; v < cumu_dist.size(); ++v)
  {
    cumu_dist[v] /= N;
  }

  return n_valid_points;
}

int main(int argc, char *argv[])
{
  string test_setting_file = argv[1];
  string elas_para_file = argv[2];
  FileStorage fs(test_setting_file, FileStorage::READ);
  if (!fs.isOpened())
  {
    cout << "Open setting file error. " << endl;
    return 1;
  }

  float baseline;
  float max_tolerate_percent;
  string data_dir;
  string save_depth_dir;
  string save_disp_dir;
  string format;
  string calib_file_left;
  string calib_file_right;

  vector<float> ranges;

  fs["data_dir"] >> data_dir;
  fs["save_depth_dir"] >> save_depth_dir;
  fs["save_disp_dir"] >> save_disp_dir;
  fs["format"] >> format;
  fs["baseline"] >> baseline;
  fs["ranges"] >> ranges;
  fs["calib_file_left"] >> calib_file_left;
  fs["calib_file_right"] >> calib_file_right;
  fs["max_tolerate_percent"] >> max_tolerate_percent;
  fs.release();

  StereoRectify rectifier(calib_file_left, calib_file_right);

  float fx = rectifier.P1.at<double>(0, 0);

  Size image_size = rectifier.image_size;

  FileStorage elas_para(elas_para_file, FileStorage::READ);
  if (!elas_para.isOpened())
  {
    cout << "Open ELAS para file error. " << endl;
    return 1;
  }

  Elas::parameters param;

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

  StereoMatch stereo_match(param);

  // Load images to be processed
  vector<string> all_files = GetDirFile(data_dir + "/left/", format.c_str());
  sort(all_files.begin(), all_files.end());
  cout << "Load " << all_files.size() << " images from " << data_dir << endl;

  // for time counting
  float total_time = 0.;
  vector<float> total_cumu_dists = vector<float>(ranges.size(), 0.);

  int npoints = image_size.height * image_size.width;
  int n_valid_frames = 0;
  max_tolerate_percent *= all_files.size();

  for (int i = 0; i < all_files.size(); ++i)
  {
    Mat left = imread(data_dir + "/left/" + all_files[i]);
    Mat right = imread(data_dir + "/right/" + all_files[i]);

    Mat rec_left, rec_right;
    rectifier.stereoRectify(left, right, rec_left, rec_right);

    Mat left_gray, right_gray;
    if (rec_left.channels() > 1)
    {
      cvtColor(rec_left, left_gray, CV_BGR2GRAY);
      cvtColor(rec_right, right_gray, CV_BGR2GRAY);
    }
    else
    {
      left_gray = rec_left.clone();
      right_gray = rec_right.clone();
    }

    struct timeval tv;
    gettimeofday(&tv, NULL);
    unsigned int time_count_begin = tv.tv_sec * 1000 + tv.tv_usec / 1000;

    Mat dispMap = stereo_match.run(left_gray, right_gray);
    if (dispMap.empty())
    {
      max_tolerate_percent--;
      BOOST_ASSERT(max_tolerate_percent < 0);
      continue;
    }
    struct timeval tv1;
    gettimeofday(&tv1, NULL);
    Mat depth_map = stereo_match.convertDisparityToDepth(dispMap, baseline, fx);

    unsigned int time_count_end = tv1.tv_sec * 1000 + tv1.tv_usec / 1000;
    float ti = (time_count_end - time_count_begin) / 1000.;

    vector<float> cumu_dist = vector<float>(ranges.size(), 0.);
    int nvalid = analysizeDepth(depth_map, ranges, cumu_dist);
    if (nvalid == 0)
    {
      max_tolerate_percent--;
      BOOST_ASSERT(max_tolerate_percent < 0);
    }
    for (int v = 0; v < ranges.size(); ++v)
    {
      total_cumu_dists[v] += cumu_dist[v];
    }

    cout << "Processing frame " << i + 1 << "/" << all_files.size() << " in " << ti << "s with " << nvalid
         << " valid points..." << endl;
    total_time += ti;

    if (!save_depth_dir.empty())
      imwrite(save_depth_dir + "/" + all_files[i], depth_map);
    if (!save_disp_dir.empty())
    {
      Mat dispMap8u, dispMapColor;
      dispMap.convertTo(dispMap8u, CV_8U);
      applyColorMap(dispMap8u, dispMapColor, COLORMAP_HSV);
      imwrite(save_disp_dir + "/" + all_files[i], dispMapColor);
    }

    n_valid_frames++;
  }

  double sum_percent = 0;

  cout << "Number of valid frame is: " << n_valid_frames << " Percentage of all frames: " << n_valid_frames / all_files.size() << endl;
  for (int v = 0; v < total_cumu_dists.size(); ++v)
  {
    float dmax = ranges[v];
    float percent = total_cumu_dists[v] / n_valid_frames;
    double min_dist = v == 0 ? 0 : ranges[v - 1];
    cout << "for depth within " << min_dist << "m --" << dmax << "m, avg points " << int(percent * npoints) << ", percent "
         << percent << endl;
    sum_percent += percent;
  }

  bool minimal_distance_point = (total_cumu_dists[0] / n_valid_frames) * npoints > 0;
  bool max_distance_point = (total_cumu_dists.back() / n_valid_frames) * npoints > 0;

  bool point_per_frame = sum_percent * npoints > 5000;

  BOOST_ASSERT(minimal_distance_point && max_distance_point && point_per_frame);

  return kTestFailed;
}
