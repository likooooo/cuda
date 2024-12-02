cmake_minimum_required (VERSION 3.14)
if(POLICY CMP0146)
  cmake_policy(SET CMP0146 OLD) # 继续使用 FindCUDA 模块
endif()

project(cuda)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    message(STATUS "CUDA found: ${CUDA_VERSION} at ${CUDA_TOOLKIT_ROOT_DIR}")
else()
    message(FATAL_ERROR "CUDA not found")
endif()

file(GLOB cuda_source ${CMAKE_CURRENT_LIST_DIR}/*.cu)
cuda_add_library(cuda ${cuda_source})

target_include_directories(cuda
  PUBLIC ${CUDA_TOOLKIT_ROOT_DIR}/include
)
target_link_libraries(cuda 
     ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcublas.so 
     ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcufft.so
     ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcudart.so
)

add_subdirectory(tools)

install(TARGETS cuda EXPORT CudaTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    PUBLIC_HEADER DESTINATION include
)
install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
    DESTINATION include
    FILES_MATCHING 
        PATTERN "*.hpp"
        PATTERN "*.h"
        PATTERN "build" EXCLUDE
)

install(EXPORT CudaTargets FILE cuda.cmake DESTINATION cmake)
