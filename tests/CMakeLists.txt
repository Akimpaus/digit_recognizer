project(digit_recognizer_tests)

file(GLOB_RECURSE SOURCES *.c)

add_executable(${PROJECT_NAME} ${SOURCES})

include_directories(
    digit_recognizer/include
    include/neural_network
    third_party/include)

target_link_libraries(${PROJECT_NAME} PUBLIC
    digit_recognizer_lib
    m #math
)