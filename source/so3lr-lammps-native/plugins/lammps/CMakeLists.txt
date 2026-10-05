cmake_minimum_required(VERSION 3.20)
project(so3lr_lammps_native_plugin VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT LAMMPS_HEADER_DIR)
  message(FATAL_ERROR "Set LAMMPS_HEADER_DIR to the exact LAMMPS src directory")
endif()
if(NOT SO3LR_KOKKOS_SOURCE)
  message(FATAL_ERROR "Set SO3LR_KOKKOS_SOURCE to the bundled LAMMPS Kokkos source")
endif()

find_package(MPI REQUIRED COMPONENTS CXX)

add_library(lammps_headers INTERFACE)
target_include_directories(lammps_headers INTERFACE "${LAMMPS_HEADER_DIR}")
target_compile_definitions(lammps_headers INTERFACE
  LAMMPS_SMALLBIG MPICH_SKIP_MPICXX OMPI_SKIP_MPICXX)
target_link_libraries(lammps_headers INTERFACE MPI::MPI_CXX)

get_filename_component(SO3LR_NATIVE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(SO3LR_ENABLE_KOKKOS ON CACHE BOOL "" FORCE)
set(SO3LR_KOKKOS_SOURCE "${SO3LR_KOKKOS_SOURCE}" CACHE PATH "" FORCE)
add_subdirectory("${SO3LR_NATIVE_ROOT}" "${CMAKE_BINARY_DIR}/native")

add_library(so3lrnativeplugin MODULE
  so3lrnativeplugin.cpp
  pair_so3lr_native.cpp
  pair_so3lr_native_mpi.cpp)
target_link_libraries(so3lrnativeplugin PRIVATE
  lammps_headers so3lr_lammps_neighbor_adapter so3lr_evaluator
  so3lr_local_cartesian_forces so3lr_feature_exchange
  so3lr_distributed_physical_lr so3lr_physical_zbl MPI::MPI_CXX)
target_compile_options(so3lrnativeplugin PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(so3lrnativeplugin PROPERTIES PREFIX "" SUFFIX ".so")
if(UNIX AND NOT APPLE)
  target_link_options(so3lrnativeplugin PRIVATE -rdynamic)
endif()

add_library(so3lradapterplugin MODULE
  so3lradapterplugin.cpp
  pair_so3lr_adapter.cpp)
target_link_libraries(so3lradapterplugin PRIVATE
  lammps_headers so3lr_lammps_neighbor_adapter)
target_compile_options(so3lradapterplugin PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(so3lradapterplugin PROPERTIES PREFIX "" SUFFIX ".so")
if(UNIX AND NOT APPLE)
  target_link_options(so3lradapterplugin PRIVATE -rdynamic)
endif()
