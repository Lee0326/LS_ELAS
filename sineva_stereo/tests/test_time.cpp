#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <dirent.h>
#include <string>
#include <cassert>

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
  vector<string> file_names;
  DIR *dir = opendir(dirPath.c_str());
  dirent *pDirent = NULL;
  while ((pDirent = readdir(dir)) != NULL)
  {
    if (strstr(pDirent->d_name, extenStr))
    {
      file_names.push_back(string(pDirent->d_name));
    }
  }
  closedir(dir);
  return file_names;
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

  int n_valid_frames = 0;

  for (int i = 0; i < all_files.size(); ++i)
  {
    Mat left = imread(data_dir + "/left/" + all_files[i]);
    Mat right = imread(data_dir + "/right/" + all_files[i]);

    struct timeval tv;
    gettimeofday(&tv, NULL);
    unsigned int time_count_begin = tv.tv_sec * 1000 + tv.tv_usec / 1000;

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

    Mat disparity_map = stereo_match.run(left_gray, right_gray);
    if (disparity_map.empty())
      continue;
    struct timeval tv1;
    gettimeofday(&tv1, NULL);
    Mat depth_map = stereo_match.convertDisparityToDepth(disparity_map, baseline, fx);

    unsigned int time_count_end = tv1.tv_sec * 1000 + tv1.tv_usec / 1000;
    float ti = (time_count_end - time_count_begin) / 1000.;
    BOOST_ASSERT(ti < 0.1 && "Process time over 100ms");

    cout << "Processing frame " << i + 1 << "/" << all_files.size() << " in " << ti << "s" << endl;
    total_time += ti;
    n_valid_frames++;
  }

  double avg_time = total_time / n_valid_frames;
  cout << "Processing  " << n_valid_frames << " image pairs with avg time " << avg_time
       << "s ..." << endl;

  BOOST_ASSERT(avg_time < 0.1 && "Average process time over 100ms");
  return kTestFailed;
}
