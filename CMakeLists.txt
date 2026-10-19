# =============================================================================
#  CryoSPH-CUDA — GPU-accelerated multiphase SPH for microgravity cryogenics
# =============================================================================
#  Build configurations
#    * CUDA backend  (default when nvcc is available):  kernels run on the GPU,
#      neighbour sorting uses thrust::sort_by_key.
#    * CPU backend   (-DCRYOSPH_ENABLE_CUDA=OFF or no nvcc found): the very same
#      physics sources (*.cu) are compiled as C++ and parallelised with OpenMP.
#
#  Typical use
#    cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
#    cmake --build build -j
#    ctest --test-dir build --output-on-failure
# =============================================================================
cmake_minimum_required(VERSION 3.24)

project(CryoSPH-CUDA
        VERSION 0.2.0
        DESCRIPTION "Microgravity cryogenic multiphase SPH solver"
        LANGUAGES CXX)

# ---------------------------------------------------------------- options ----
option(CRYOSPH_ENABLE_CUDA        "Build the CUDA GPU backend"                     ON)
option(CRYOSPH_BUILD_TESTS        "Build GoogleTest unit and integration tests"    ON)
option(CRYOSPH_BUILD_PYTHON       "Build the pybind11 Python module 'CryoSPH'"     ON)
option(CRYOSPH_BUILD_BENCHMARKS   "Build the C++ performance benchmark driver"     ON)
option(CRYOSPH_DOUBLE_PRECISION   "Use double precision for particle data"         OFF)
option(CRYOSPH_FETCH_DEPENDENCIES "Download GoogleTest/pybind11 if not installed"  ON)
option(CRYOSPH_FAST_MATH          "Enable fast-math intrinsics on the GPU"         OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ------------------------------------------------------------------ CUDA ----
set(CRYOSPH_USE_CUDA OFF)
if(CRYOSPH_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    # Turing (7.5) and newer by default.  Override on the command line, e.g.
    #   -DCMAKE_CUDA_ARCHITECTURES=61     (Pascal)   or   =native
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    find_package(CUDAToolkit REQUIRED)
    set(CRYOSPH_USE_CUDA ON)
    message(STATUS "CryoSPH: CUDA backend enabled (archs: ${CMAKE_CUDA_ARCHITECTURES})")
  else()
    message(WARNING "CryoSPH: no CUDA compiler found - building the CPU/OpenMP backend")
  endif()
else()
  message(STATUS "CryoSPH: CUDA disabled - building the CPU/OpenMP backend")
endif()

find_package(OpenMP)
find_package(Threads REQUIRED)

# --------------------------------------------------------- core library ----
set(CRYOSPH_CU_SOURCES
  src/core/spatial_hash.cu
  src/core/telemetry.cu
  src/physics/SPH_kernels.cu
  src/physics/navier_stokes.cu
  src/physics/surface_tension.cu
  src/physics/phase_change.cu
  src/solvers/sph_solver.cu)

set(CRYOSPH_CXX_SOURCES
  src/core/cuda_utils.cpp
  src/core/particle_system.cpp
  src/core/geometry.cpp
  src/core/io.cpp
  src/physics/fluid_properties.cpp)

if(NOT CRYOSPH_USE_CUDA)
  # Same sources, compiled as ordinary C++ (CMP0119 makes CMake pass -x c++).
  set_source_files_properties(${CRYOSPH_CU_SOURCES} PROPERTIES LANGUAGE CXX)
endif()

add_library(cryosph_core STATIC ${CRYOSPH_CU_SOURCES} ${CRYOSPH_CXX_SOURCES})
add_library(CryoSPH::core ALIAS cryosph_core)

target_include_directories(cryosph_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

target_compile_features(cryosph_core PUBLIC cxx_std_20)
target_link_libraries(cryosph_core PUBLIC Threads::Threads)

if(OpenMP_CXX_FOUND)
  target_link_libraries(cryosph_core PUBLIC OpenMP::OpenMP_CXX)
endif()

if(CRYOSPH_DOUBLE_PRECISION)
  target_compile_definitions(cryosph_core PUBLIC CRYOSPH_DOUBLE_PRECISION=1)
endif()

if(CRYOSPH_USE_CUDA)
  target_compile_definitions(cryosph_core PUBLIC CRYOSPH_WITH_CUDA=1)
  target_link_libraries(cryosph_core PUBLIC CUDA::cudart)
  set_target_properties(cryosph_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  target_compile_options(cryosph_core PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr -lineinfo>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${OpenMP_CXX_FOUND}>>:-Xcompiler=-fopenmp>)
  if(CRYOSPH_FAST_MATH)
    target_compile_options(cryosph_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
  endif()
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(cryosph_core PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Wno-unknown-pragmas>)
elseif(MSVC)
  target_compile_options(cryosph_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive->)
endif()

# ------------------------------------------------------ dependencies -----
include(FetchContent)

if(CRYOSPH_BUILD_TESTS)
  find_package(GTest CONFIG QUIET)
  if(NOT GTest_FOUND AND CRYOSPH_FETCH_DEPENDENCIES)
    message(STATUS "CryoSPH: fetching GoogleTest")
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(googletest
      URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_MakeAvailable(googletest)
    set(GTest_FOUND TRUE)
  endif()
  if(GTest_FOUND)
    enable_testing()
    add_subdirectory(tests)
  else()
    message(WARNING "CryoSPH: GoogleTest not found - tests disabled")
  endif()
endif()

if(CRYOSPH_BUILD_PYTHON)
  find_package(Python 3.9 COMPONENTS Interpreter Development.Module)
  if(Python_FOUND)
    find_package(pybind11 CONFIG QUIET)
    if(NOT pybind11_FOUND AND CRYOSPH_FETCH_DEPENDENCIES)
      message(STATUS "CryoSPH: fetching pybind11")
      FetchContent_Declare(pybind11
        URL https://github.com/pybind/pybind11/archive/refs/tags/v2.13.6.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
      FetchContent_MakeAvailable(pybind11)
      set(pybind11_FOUND TRUE)
    endif()
  endif()
  if(pybind11_FOUND)
    pybind11_add_module(CryoSPH src/python/bindings.cpp)
    target_link_libraries(CryoSPH PRIVATE cryosph_core)
    # Put the module next to the build root so `PYTHONPATH=build` works.
    set_target_properties(CryoSPH PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python)
    install(TARGETS CryoSPH LIBRARY DESTINATION python)
  else()
    message(WARNING "CryoSPH: pybind11/Python not found - Python module disabled")
  endif()
endif()

if(CRYOSPH_BUILD_BENCHMARKS)
  add_executable(cryosph_perf benchmarks/perf_scaling.cpp)
  target_link_libraries(cryosph_perf PRIVATE cryosph_core)
endif()

install(TARGETS cryosph_core ARCHIVE DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
