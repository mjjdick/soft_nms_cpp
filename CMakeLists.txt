cmake_minimum_required(VERSION 2.8.3)
project(soft_nms)

SET(CMAKE_BUILD_TYPE "Debug")

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
        roscpp
        )

find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)

#INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})

include_directories(
        include
        ${catkin_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
        )

catkin_package(
        CATKIN_DEPENDS roscpp
        DEPENDS EIGEN3 PCL OpenCV
        INCLUDE_DIRS include
        LIBRARIES ${PROJECT_NAME}
        )

add_library(${PROJECT_NAME}
        include/soft_nms/soft_nms.h
        src/soft_nms/soft_nms.cpp
        )

add_dependencies(${PROJECT_NAME} ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})
