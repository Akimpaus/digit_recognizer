project(digit_recognizer)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Raylib
set(RAYLIB_VERSION 4.5.0)
find_package(raylib ${RAYLIB_VERSION} QUIET)
if (NOT raylib_FOUND)
  message("Raylib not found. Downloading...")
  include(FetchContent)
  FetchContent_Declare(
    raylib
    DOWNLOAD_EXTRACT_TIMESTAMP OFF
    URL https://github.com/raysan5/raylib/archive/refs/tags/${RAYLIB_VERSION}.tar.gz
  )
  FetchContent_GetProperties(raylib)
  if (NOT raylib_POPULATED)
    message("Raylib not populated. Populating...")
    set(FETCHCONTENT_QUIET NO)
    FetchContent_Populate(raylib)
    set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    add_subdirectory(${raylib_SOURCE_DIR} ${raylib_BINARY_DIR})
  endif()
endif()

# Project lib
set(PROJECT_LIB_NAME ${PROJECT_NAME}_lib)
file(GLOB_RECURSE SOURCES sources/*.c)
list(REMOVE_ITEM SOURCES sources/main.c)
add_library(${PROJECT_LIB_NAME} ${SOURCES})
target_include_directories(${PROJECT_LIB_NAME} PUBLIC
  include
  third_party/include
  )

# Project execuatable
add_executable(${PROJECT_NAME} sources/main.c)
target_link_libraries(${PROJECT_NAME} PUBLIC
  m #math
  ${PROJECT_LIB_NAME}
  raylib
  )

# OS specific
if (APPLE)
    target_link_libraries(${PROJECT_NAME} "-framework IOKit")
    target_link_libraries(${PROJECT_NAME} "-framework Cocoa")
    target_link_libraries(${PROJECT_NAME} "-framework OpenGL")
endif()