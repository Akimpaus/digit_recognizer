project(digit_recognizer_tests)

include_directories(include third_party/include)
file(GLOB_RECURSE SOURCES sources/*.c)
add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} digit_recognizer_lib)