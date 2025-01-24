cmake_minimum_required(VERSION 3.27.0)

project(Nabla4Benchmark)

option(IS_GPU "Enable GPU execution" OFF)
option(BUILD_SLIDES "Build slides" OFF)

set(INDEX_TYPE "int" CACHE STRING "Index type")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(IS_GPU)
  include(CheckLanguage)
  check_language(CUDA)
  if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_EXTENSIONS OFF)
  endif()
endif()

# Find or fetch GridTools
if(NOT _gridtools_already_fetched)
    find_package(GridTools "2" QUIET)
endif()
if(NOT GridTools_FOUND)
  set(_gridtools_repository "https://github.com/GridTools/gridtools.git")
  set(_gridtools_tag        "master")
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
else()
  message(STATUS "Using GridTools from ${GridTools_DIR}")
endif()

add_library(benchmark src/wrapper.cpp)
set_property(TARGET benchmark PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(benchmark PRIVATE include GridTools)
target_link_libraries(benchmark PUBLIC GridTools::gridtools GridTools::stencil_naive GridTools::storage_cpu_ifirst GridTools::storage_cpu_kfirst)
if (${INDEX_TYPE} STREQUAL "sizet")
  message(STATUS "Using size_t as index type")
  target_compile_definitions(benchmark PUBLIC INDEX_TYPE_SIZE_T)
elseif(${INDEX_TYPE} STREQUAL "int64")
  message(STATUS "Using int64 as index type")
  target_compile_definitions(benchmark PUBLIC INDEX_TYPE_INT64)
elseif(${INDEX_TYPE} STREQUAL "uint32")
  message(STATUS "Using uint32 as index type")
  target_compile_definitions(benchmark PUBLIC INDEX_TYPE_UINT32_T)
elseif(${INDEX_TYPE} STREQUAL "int")
  message(STATUS "Using int as index type")
  target_compile_definitions(benchmark PUBLIC INDEX_TYPE_INT)
else()
  message(FATAL_ERROR "Invalid value for INDEX_TYPE ${INDEX_TYPE}. Valid values are sizet, uint32, or int.")
endif()

if(IS_GPU)
  target_link_libraries(benchmark PUBLIC GridTools::storage_gpu)
  set_source_files_properties(src/wrapper.cpp PROPERTIES LANGUAGE CUDA)
endif()
add_executable(nabla4_benchmark src/main.cpp)
target_link_libraries(nabla4_benchmark PRIVATE benchmark)
target_include_directories(nabla4_benchmark PRIVATE include GridTools)

find_package(Python 3.9.6 COMPONENTS Interpreter Development.Module REQUIRED)

# Find nanobind module
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

# Test if we have the right version.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --version
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_VER)
if(${NB_VER} VERSION_LESS 2.0.0)
  message(FATAL_ERROR "Require nanobind version greater than 2.0.0, but found ${NB_VER}")
endif()

nanobind_add_module(icon_benchmark python/extensions.cpp)
target_include_directories(icon_benchmark PRIVATE include)
target_link_libraries(icon_benchmark PRIVATE benchmark)

nanobind_add_module(nabla4_gtfn python/nabla4_gtfn_bindings.cpp)
target_include_directories(nabla4_gtfn PRIVATE include GridTools)
target_link_libraries(nabla4_gtfn PRIVATE benchmark GridTools::gridtools GridTools::stencil_naive GridTools::storage_cpu_ifirst GridTools::storage_cpu_kfirst)

if(IS_GPU)
  target_compile_definitions(nabla4_gtfn PRIVATE IS_GPU)
  target_link_libraries(nabla4_gtfn PUBLIC GridTools::storage_gpu)
  set_source_files_properties(python/nabla4_gtfn_bindings.cpp PROPERTIES LANGUAGE CUDA)
endif()

if(BUILD_SLIDES)
  include(slides-support/Marp.cmake)
  add_marp_slides(icon-structured-slides icon_structured_slides.md)
  add_marp_slides(icon-structured-slides-3kernels icon_structured_slides_3kernels.md)
endif()

install(TARGETS nabla4_benchmark icon_benchmark nabla4_gtfn benchmark
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)
