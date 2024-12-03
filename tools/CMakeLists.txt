file(GLOB tools ${CMAKE_CURRENT_LIST_DIR}/*.cpp)
include_directories(../)

find_package(py_visualizer REQUIRED)
foreach(tool IN LISTS tools)
    get_filename_component(test_WLE ${tool} NAME_WLE)
    add_executable(${test_WLE} ${tool})
    target_link_libraries(${test_WLE} cuda py_visualizer::py_visualizer)
    target_include_directories(${test_WLE} PRIVATE ${py_visualizer_INCLUDE_DIRS})
endforeach()

set(test_WLE cuda_test_fft)
add_custom_command(
    TARGET ${test_WLE} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "/usr/local/include/visualizer.py"
            "$<TARGET_FILE_DIR:${test_WLE}>/visualizer.py"
    COMMENT "Copying visualizer.py"
)
add_custom_command(
    TARGET ${test_WLE} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_LIST_DIR}/cuda_test_fft_golden.py"
            "$<TARGET_FILE_DIR:${test_WLE}>/cuda_test_fft_golden.py"
    COMMENT "Copying cuda_test_fft_golden.py to $<TARGET_FILE_DIR:${test_WLE}>/cuda_test_fft_golden.py"
)

