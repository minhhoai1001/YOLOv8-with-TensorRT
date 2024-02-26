cmake_minimum_required(VERSION 3.12)

set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 86)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_VERBOSE_MAKEFILE OFF)

project(lice_det LANGUAGES CXX CUDA)

# Set C++ standard to C++14
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3 -g -Wno-narrowing")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

# CUDA
find_package(CUDA REQUIRED)
message(STATUS "CUDA Libs: \n${CUDA_LIBRARIES}\n")
message(STATUS "CUDA Headers: \n${CUDA_INCLUDE_DIRS}\n")

# OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV Libs: \n${OpenCV_LIBS}\n")
message(STATUS "OpenCV Libraries: \n${OpenCV_LIBRARIES}\n")
message(STATUS "OpenCV Headers: \n${OpenCV_INCLUDE_DIRS}\n")

# TensorRT
set(TensorRT_INCLUDE_DIRS /usr/include/aarch64-linux-gnu)
set(TensorRT_LIBRARIES /usr/lib/aarch64-linux-gnu)

message(STATUS "TensorRT Libs: \n${TensorRT_LIBRARIES}\n")
message(STATUS "TensorRT Headers: \n${TensorRT_INCLUDE_DIRS}\n")

# Find the GStreamer package
find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)

# Find yaml-cpp package
find_package(yaml-cpp REQUIRED)
message(STATUS "yaml-cpp: \n${YAML_CPP_INCLUDE_DIR}\n")

# Find PostgreSQL and include its required components
# set(PostgreSQL_DIR /usr/include/postgresql)
# set(PostgreSQL_TYPE_INCLUDE_DIR /usr/include/postgresql)
# find_package(PostgreSQL REQUIRED)
# include_directories(${PostgreSQL_INCLUDE_DIRS})

list(APPEND INCLUDE_DIRS
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIRS}
    ${GSTREAMER_INCLUDE_DIRS}
	./include
    )

list(APPEND ALL_LIBS
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${TensorRT_LIBRARIES}
    )

include_directories(${INCLUDE_DIRS})

# Add source files
set(SOURCES
    src/yolov8-seg.cpp
    src/yolov8.cpp
    src/fish_seg.cpp
    src/lice_det.cpp
    src/filter.cpp
    main.cpp
)

# Add executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Link libraries
link_directories(${ALL_LIBS})
target_link_libraries(${PROJECT_NAME} PRIVATE nvinfer nvinfer_plugin pthread  ${CUDA_LIBRARIES} ${OpenCV_LIBS} ${GSTREAMER_LIBRARIES} ${YAML_CPP_INCLUDE_DIR})

if(${OpenCV_VERSION} VERSION_GREATER_EQUAL 4.7.0)
    message(STATUS "Build with -DBATCHED_NMS")
    add_definitions(-DBATCHED_NMS)
endif()

# print LOG_LEVEL
# option(LOG_LEVEL "Enable LOG_LEVEL" OFF)
if(LOG_LEVEL)
    message(STATUS "Build with -DLOG_LEVEL=${LOG_LEVEL}\n")
    add_definitions(-DLOG_LEVEL=${LOG_LEVEL})
endif()
