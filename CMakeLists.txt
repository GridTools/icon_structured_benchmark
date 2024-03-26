cmake_minimum_required(VERSION 3.27.0)

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

# Find or fetch GridTools
if(NOT _gridtools_already_fetched)
    find_package(GridTools "2" QUIET)
endif()
if(NOT GridTools_FOUND)
  set(_gridtools_repository "https://github.com/GridTools/gridtools.git")
  set(_gridtools_tag        "v2.3.2")
  if(NOT _gridtools_already_fetched)
    message(STATUS "Fetching GridTools ${_gridtools_tag} from ${_gridtools_repository}")
  endif()
  include(FetchContent)
  FetchContent_Declare(
    gridtools
    GIT_REPOSITORY ${_gridtools_repository}
    GIT_TAG        ${_gridtools_tag}
    OVERRIDE_FIND_PACKAGE
  )
  FetchContent_MakeAvailable(gridtools)
  set(_gridtools_already_fetched ON CACHE INTERNAL "")
endif()

nanobind_add_module(nabla4_gtfn python/nabla4_gtfn_bindings.cpp)
target_include_directories(nabla4_gtfn PRIVATE include GridTools)
target_link_libraries(nabla4_gtfn PRIVATE benchmark GridTools::gridtools GridTools::stencil_naive GridTools::storage_cpu_ifirst GridTools::storage_cpu_kfirst)
