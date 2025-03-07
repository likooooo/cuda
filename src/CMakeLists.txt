add_library(cpp_cuda)
file(GLOB cuda_source ${CMAKE_CURRENT_LIST_DIR}/*.cu)
target_sources(cpp_cuda PRIVATE ${cuda_source})
include_directories(${CMAKE_INC_DIR})
target_include_directories(cpp_cuda PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${py_visualizer_INCLUDE_DIRS})
target_link_libraries(cpp_cuda PUBLIC CUDA::cublas CUDA::cufft CUDA::cudart py_visualizer::py_visualizer)


file(GLOB tools ${CMAKE_CURRENT_LIST_DIR}/cuda_test_*.cpp)
list(APPEND tools cuda_device_info.cpp)
set(TOOLS_TARGETS "" CACHE INTERNAL "List of tool targets")
foreach(tool IN LISTS tools)
    get_filename_component(test_WLE ${tool} NAME_WLE)
    add_executable(${test_WLE} ${tool})
    target_link_libraries(${test_WLE} PUBLIC cpp_cuda)
    # target_include_directories(${test_WLE} PUBLIC ${py_visualizer_INCLUDE_DIRS})
    list(APPEND TOOLS_TARGETS ${test_WLE})
endforeach()

copy_visualizer_files(cpp_cuda "$<TARGET_FILE_DIR:cpp_cuda>")

add_custom_command(
    TARGET cpp_cuda ${TOOLS_TARGETS} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_LIST_DIR}/cuda_test_fft_golden.py"
            "$<TARGET_FILE_DIR:cpp_cuda>/cuda_test_fft_golden.py"
    COMMENT "Copying cuda_test_fft_golden.py to $<TARGET_FILE_DIR:cpp_cuda>/cuda_test_fft_golden.py"
)

# 安装目标和头文件
install(TARGETS cpp_cuda ${TOOLS_TARGETS}
    EXPORT cpp_cudaTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY ${CMAKE_INC_DIR} DESTINATION ${CMAKE_INSTALL_PREFIX})

# 生成配置文件
include(CMakePackageConfigHelpers)

set(PROJECT_VERSION "1.0.0")
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/cpp_cudaConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)
configure_package_config_file(
    "${CMAKE_SRC_DIR}/cpp_cudaConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cpp_cudaConfig.cmake"
    INSTALL_DESTINATION cmake
)

# 安装配置文件
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/cpp_cudaConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/cpp_cudaConfigVersion.cmake"
    DESTINATION cmake
)
install(FILES
    "$<TARGET_FILE_DIR:cpp_cuda>/cuda_test_fft_golden.py"
    DESTINATION bin
)

# 导出目标
install(EXPORT cpp_cudaTargets
    FILE cpp_cudaTargets.cmake
    NAMESPACE cpp_cuda::
    DESTINATION cmake
)