cmake_minimum_required (VERSION 3.14)
project(cpp_cuda 
    DESCRIPTION "cuda cpp wrapper"
    HOMEPAGE_URL "https://github.com/likooooo/cuda"
    LANGUAGES CXX CUDA
)
find_package(py_visualizer REQUIRED)
find_package(type_traist_notebook REQUIRED)
find_package(CUDAToolkit REQUIRED)
if(CUDAToolkit_FOUND)
    message(STATUS "Found CUDA Toolkit v${CUDAToolkit_VERSION}")
    message(STATUS "CUDA include path: ${CUDAToolkit_INCLUDE_DIRS}")
    message(STATUS "CUDA libraries: ${CUDAToolkit_LIBRARIES}")
else()
    message(WARNING "CUDA Toolkit not found!")
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_CUDA_ARCHITECTURES LESS 60)
  set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86 90)
endif()
set(CMAKE_PROJECT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_SRC_DIR ${CMAKE_PROJECT_DIR}/cmake)
set(CMAKE_INC_DIR ${CMAKE_PROJECT_DIR}/include)
add_subdirectory(src)

# 导出目标
install(EXPORT cpp_cudaTargets
    FILE cpp_cudaTargets.cmake
    NAMESPACE cpp_cuda::
    DESTINATION cmake
)
# 安装目标和头文件
install(TARGETS cpp_cuda cuda_device_info
    EXPORT cpp_cudaTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

file(GLOB headers ${CMAKE_INC_DIR}/*.h ${CMAKE_INC_DIR}/*.hpp)
install(FILES ${headers}
    DESTINATION include/cpp_cuda
)
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cpp_cudaConfig.cmake"
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cpp_cudaConfig-version.cmake"
    DESTINATION cmake
)
include(${PROJECT_SOURCE_DIR}/cmake/auto_ctest.cmake)
