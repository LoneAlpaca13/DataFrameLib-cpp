cmake_minimum_required(VERSION 3.14)
project(DataFrameLib)

# =====================
# C++ STANDARD
# =====================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Extra safety (some environments ignore standard)
add_compile_options(-std=c++20)

# =====================
# DEPENDENCIES
# =====================
find_package(PkgConfig REQUIRED)

pkg_check_modules(ARROW REQUIRED arrow)
pkg_check_modules(PARQUET REQUIRED parquet)

# =====================
# LIBRARY
# =====================
add_library(dataframelib
    src/dataframe.cpp
    src/lazy.cpp
    src/utils.cpp
)

# =====================
# INCLUDE DIRECTORIES
# =====================
target_include_directories(dataframelib PUBLIC
    ${ARROW_INCLUDE_DIRS}
    ${PARQUET_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# =====================
# LINK LIBRARIES
# =====================
target_link_libraries(dataframelib PUBLIC
    ${ARROW_LIBRARIES}
    ${PARQUET_LIBRARIES}
)

# =====================
# FORCE C++20 (again)
# =====================
target_compile_features(dataframelib PUBLIC cxx_std_20)

# =====================
# TEST EXECUTABLE (OPTIONAL)
# =====================
add_executable(main src/main.cpp)

target_link_libraries(main PRIVATE dataframelib)