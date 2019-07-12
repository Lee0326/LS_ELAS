#include <sensor_msgs/CameraInfo.h>
#include <ros/topic.h>

#include "precomp.h"
#include "stereo_rectify.h"
#include "stereo_match.h"

#include <string>

using namespace std;

class ELASWrapper
{

public:
  std::string config_path_;
  StereoRectify *rectifier;
  StereoMatch *elas;
  float baseline, fx;

  ros::Publisher mDepthPublish;
  ros::Publisher mRecedPublish;

  ELASWrapper(const std::string &config_path, ros::NodeHandle &nh)
      : config_path_(config_path)
  {

    std::string calibfileleft, calibfileright;

    mDepthPublish = nh.advertise<sensor_msgs::Image>("/elas/depth", 1);
    mRecedPublish = nh.advertise<sensor_msgs::Image>("/elas/rec_left", 1);

    cv::FileStorage fileStorage(config_path, cv::FileStorage::READ);
    assert(fileStorage.isOpened());

    fileStorage["calib_file_left"] >> calibfileleft;
    fileStorage["calib_file_right"] >> calibfileright;
    fileStorage["baseline"] >> baseline;
    fileStorage.release();

    rectifier = new StereoRectify(calibfileleft, calibfileright);

    elas = new StereoMatch();

    fx = rectifier->P1.at<double>(0, 0);
  }

  void processELAS(const sensor_msgs::ImageConstPtr &msg_left, const sensor_msgs::ImageConstPtr &msg_right);
};

int main(int argc, char *argv[])
{
  ros::init(argc, argv, "sineva_stereo");
  ros::start();
  ros::NodeHandle node_handle;
  string topic_left, topic_right, config_path;

  node_handle.getParam("topic_left", topic_left);
  node_handle.getParam("topic_right", topic_right);
  node_handle.getParam("config_path", config_path);

  ELASWrapper elas(config_path, node_handle);

  message_filters::Subscriber<sensor_msgs::Image> left_sub(node_handle, topic_left, 1);
  message_filters::Subscriber<sensor_msgs::Image> right_sub(node_handle, topic_right, 1);
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> sync_policy;
  message_filters::Synchronizer<sync_policy> sync(sync_policy(100), left_sub, right_sub);
  sync.registerCallback(boost::bind(&ELASWrapper::processELAS, elas, _1, _2));

  ros::spin();
  ros::shutdown();

  return 0;
}

void ELASWrapper::processELAS(const sensor_msgs::ImageConstPtr &msg_left, const sensor_msgs::ImageConstPtr &msg_right)
{
  Mat image_left, image_right;
  try
  {
    image_left = cv_bridge::toCvShare(msg_left, "bgr8")->image;
  }
  catch (cv_bridge::Exception &e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
  try
  {
    image_right = cv_bridge::toCvShare(msg_right, "bgr8")->image;
  }
  catch (cv_bridge::Exception &e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  assert(image_left.size() == cv::Size(640, 480) || image_left.size() == cv::Size(1280, 720));

  // calibration
  Mat rec_left, rec_right;
  rectifier->stereoRectify(image_left, image_right, rec_left, rec_right);
  // ELAS
  Mat gray_left, gray_right, img_depth, img_disp;

  cvtColor(rec_left, gray_left, CV_BGR2GRAY);
  cvtColor(rec_right, gray_right, CV_BGR2GRAY);

  img_disp = elas->run(gray_left, gray_right);
  img_depth = elas->convertDisparityToDepth(img_disp, baseline, fx);

  sensor_msgs::Image depth_image;
  depth_image.header.stamp = msg_left->header.stamp;
  depth_image.header.frame_id = "world";

  sensor_msgs::ImagePtr depth_msg = cv_bridge::CvImage(depth_image.header, "mono8", img_depth).toImageMsg();

  mDepthPublish.publish(depth_msg);

  sensor_msgs::Image reced_img;
  reced_img.header.stamp = msg_left->header.stamp;
  reced_img.header.frame_id = "world";

  sensor_msgs::ImagePtr reced_msg = cv_bridge::CvImage(reced_img.header, "rgb8", rec_left).toImageMsg();

  mRecedPublish.publish(reced_msg);
}
