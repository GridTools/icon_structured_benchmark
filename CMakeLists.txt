cmake_minimum_required(VERSION 3.28.1)

project(Nabla4Benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(benchmark src/wrapper.cpp)
target_include_directories(benchmark PRIVATE include)

add_executable(nabla4_benchmark src/main.cpp)
target_link_libraries(nabla4_benchmark PRIVATE benchmark)
target_include_directories(nabla4_benchmark PRIVATE include)

find_package(Python 3.9.6 COMPONENTS Interpreter Development.Module REQUIRED)

# Find nanobind module
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

# or
# add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ext/nanobind)

nanobind_add_module(icon_benchmark python/extensions.cpp)
target_include_directories(icon_benchmark PRIVATE include)
target_link_libraries(icon_benchmark PRIVATE benchmark)
