project(digit_recognizer)

# lib
file(GLOB_RECURSE SOURCES sources/*.c)
set(DIGIT_RECOGNIZER_LIB ${PROJECT_NAME}_lib)
add_library(${DIGIT_RECOGNIZER_LIB} STATIC ${SOURCES})
target_include_directories(${DIGIT_RECOGNIZER_LIB} PUBLIC include)
target_link_libraries(${DIGIT_RECOGNIZER_LIB} PRIVATE m)

# executable
add_executable(${PROJECT_NAME} sources/main.c)
target_link_libraries(${PROJECT_NAME} PRIVATE ${DIGIT_RECOGNIZER_LIB})