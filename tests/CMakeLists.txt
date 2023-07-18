project(digit_recognizer_tests)

include_directories(
    neural_network/include
    third_party/include)
file(GLOB_RECURSE SOURCES *.c)
add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} neural_network_lib)