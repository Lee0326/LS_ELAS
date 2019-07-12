# project
cmake_minimum_required (VERSION 3.12.0)
add_compile_options(-std=c++11)
project (line_segment)

find_package(PkgConfig)
find_package(OpenCV REQUIRED)


# directories
set (LINE_SEGMENT_SRC_DIR src)

# include directory
include_directories("${LINE_SEGMENT_SRC_DIR}")
# use sse3 instruction set
SET(CMAKE_CXX_FLAGS "-msse3")

# sources
FILE(GLOB LINE_SEGMENT_SRC_FILES "src/*.cpp")

# make release version
set(CMAKE_BUILD_TYPE Release)
# build demo program
add_executable(line_segment ${LINE_SEGMENT_SRC_FILES})
target_link_libraries(line_segment ${OpenCV_LIBS})
