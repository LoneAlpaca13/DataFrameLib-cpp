cmake_minimum_required(VERSION 3.14)
project(DataFrameLib)

set(CMAKE_CXX_STANDARD 20)

find_package(PkgConfig REQUIRED)
pkg_check_modules(ARROW REQUIRED arrow)

add_executable(main src/main.cpp src/dataframe.cpp src/lazy.cpp src/utils.cpp)

target_include_directories(main PRIVATE ${ARROW_INCLUDE_DIRS})
target_link_libraries(main PRIVATE ${ARROW_LIBRARIES})