set(NEURAL_NETWORK_LIB_NAME neural_network_lib)
file(GLOB_RECURSE SOURCES sources/*.c)
add_library(${NEURAL_NETWORK_LIB_NAME} STATIC ${SOURCES})
target_include_directories(${NEURAL_NETWORK_LIB_NAME} PUBLIC include ../general/include)
target_link_libraries(${NEURAL_NETWORK_LIB_NAME} PUBLIC m) # math