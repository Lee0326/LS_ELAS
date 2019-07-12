cmake_minimum_required(VERSION 2.8.3)
project(sineva_stereo)
add_compile_options(-std=c++11)
add_compile_options(-O3)
add_compile_options("$<$<CONFIG:RELEASE>: -O3>")

find_package(PkgConfig)
find_package(Boost COMPONENTS program_options system REQUIRED)
find_package(OpenCV REQUIRED)
find_package(catkin REQUIRED
  roscpp
  sensor_msgs
  std_msgs
  cv_bridge
  tf
)

catkin_package()

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

add_library(elas SHARED
  elas/src/descriptor.cpp
  elas/src/elas.cpp
  elas/src/stereo_match.cpp
  utils/src/orb_extractor.cpp
)

target_link_libraries(elas
        ${OpenCV_LIBS}
)

include_directories(
  utils/include
  elas/include
  elas
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIR}
  ${EIGEN3_INCLUDE_DIR}
)

add_library(stereo_rectify SHARED
  utils/src/stereo_rectify.cpp
)

target_link_libraries(stereo_rectify
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  elas
)

add_executable(${PROJECT_NAME}_node
  nodes/sineva_stereo_node.cpp
)

target_link_libraries(${PROJECT_NAME}_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  stereo_rectify
  elas
)

install(TARGETS ${PROJECT_NAME}_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

add_executable(${PROJECT_NAME}_TEST_TIME
  tests/test_time.cpp
)

target_link_libraries(${PROJECT_NAME}_TEST_TIME
  ${OpenCV_LIBS}
  elas
  stereo_rectify
)

add_executable(${PROJECT_NAME}_TEST_POINTCLOUD
  tests/test_pointcloud.cpp
)

target_link_libraries(${PROJECT_NAME}_TEST_POINTCLOUD
  ${OpenCV_LIBS}
  elas
  stereo_rectify
)

add_executable(${PROJECT_NAME}_TEST_SUPPORT_POINT
  tests/test_support_points.cpp
)

target_link_libraries(${PROJECT_NAME}_TEST_SUPPORT_POINT
  ${OpenCV_LIBS}
  elas
)

add_test(NAME ${PROJECT_NAME}_TEST_TIME
  COMMAND ${PROJECT_NAME}_TEST_TIME ${PROJECT_SOURCE_DIR}/tests/test_data/test_setting.yml ${PROJECT_SOURCE_DIR}/tests/test_data/elas_para.yml
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)

add_test(NAME ${PROJECT_NAME}_TEST_POINTCLOUD
  COMMAND ${PROJECT_NAME}_TEST_POINTCLOUD ${PROJECT_SOURCE_DIR}/tests/test_data/test_setting.yml ${PROJECT_SOURCE_DIR}/tests/test_data/elas_para.yml
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)

add_test(NAME ${PROJECT_NAME}_TEST_SUPPORT_POINT
  COMMAND ${PROJECT_NAME}_TEST_SUPPORT_POINT ${PROJECT_SOURCE_DIR}/tests/test_data/elas_para.yml
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)
