# cpp_cudaConfig.cmake - Configuration file for the cpp_cuda package.

# Ensure that this script is included only once.
if(TARGET cpp_cuda)
    return()
endif()

# Get the directory where this file is located.
get_filename_component(_current_dir "${CMAKE_CURRENT_LIST_FILE}" PATH)

# # Include the exported targets file.
include("${_current_dir}/cpp_cudaTargets.cmake")
# Set the package version variables.
set(cpp_cuda_VERSION_MAJOR 1) # Replace with your major version
set(cpp_cuda_VERSION_MINOR 0) # Replace with your minor version
set(cpp_cuda_VERSION_PATCH 0) # Replace with your patch version
set(cpp_cuda_VERSION "${cpp_cuda_VERSION_MAJOR}.${cpp_cuda_VERSION_MINOR}.${cpp_cuda_VERSION_PATCH}")

# Check if the requested version is compatible.
if(NOT "${cpp_cuda_FIND_VERSION}" STREQUAL "")
    if(NOT "${cpp_cuda_FIND_VERSION}" VERSION_LESS "${cpp_cuda_VERSION}")
        set(cpp_cuda_VERSION_COMPATIBLE TRUE)
    endif()

    if("${cpp_cuda_FIND_VERSION}" VERSION_EQUAL "${cpp_cuda_VERSION}")
        set(cpp_cuda_VERSION_EXACT TRUE)
    endif()
endif()

find_package(type_traist_notebook REQUIRED)
find_package(py_visualizer REQUIRED)
find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
    message(STATUS "Found CUDA Toolkit v${CUDAToolkit_VERSION}")
    message(STATUS "CUDA include path: ${CUDAToolkit_INCLUDE_DIRS}")
    message(STATUS "CUDA libraries: ${CUDAToolkit_LIBRARIES}")
else()
    message(WARNING "CUDA Toolkit not found!")
endif()

# include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${py_visualizer_INCLUDE_DIRS})
# Mark the package as found.
set(cpp_cuda_FOUND TRUE)