cmake_minimum_required(VERSION 3.16)
project(DataFrameLib CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# =====================
# FIND ARROW & PARQUET
# =====================
find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)

# =====================
# LIBRARY
# =====================
add_library(dataframelib STATIC
  src/expr.cpp
  src/dataframe.cpp
  src/lazy.cpp
  src/utils.cpp
)

target_include_directories(dataframelib PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(dataframelib PUBLIC
  Arrow::arrow_shared
  Parquet::parquet_shared
)

target_compile_features(dataframelib PUBLIC cxx_std_20)

# =====================
# OPTIONAL: MAIN EXECUTABLE
# =====================
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")
  add_executable(main src/main.cpp)
  target_link_libraries(main PRIVATE dataframelib)
endif()
