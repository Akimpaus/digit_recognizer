project(digit_recognizer)

add_subdirectory(neural_network)

add_executable(${PROJECT_NAME} main.c)
include_directories(
    application/include
    ../general/include)
target_link_libraries(${PROJECT_NAME} PRIVATE neural_network_lib)