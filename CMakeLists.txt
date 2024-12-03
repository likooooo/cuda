cmake_minimum_required (VERSION 3.14)
project(cpp_cuda 
    DESCRIPTION "cuda cpp wrapper"
    HOMEPAGE_URL "https://github.com/likooooo/cuda"
    LANGUAGES CXX CUDA
)
find_package(py_visualizer REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE Release)
endif()
if(CMAKE_CUDA_ARCHITECTURES LESS 60)
  set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86 90)
endif()
set(CMAKE_PROJECT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_SRC_DIR ${CMAKE_PROJECT_DIR}/cmake)
set(CMAKE_INC_DIR ${CMAKE_PROJECT_DIR}/include)
# make target : cpp_cuda
add_subdirectory(src)
