# Native SO3LR integration for LAMMPS 11Feb2026.
# This file is installed into cmake/Modules/Packages by the release helper.

if(NOT PKG_KOKKOS)
  message(FATAL_ERROR "PKG_SO3LR requires PKG_KOKKOS=ON")
endif()
if(NOT BUILD_MPI)
  message(FATAL_ERROR "PKG_SO3LR requires BUILD_MPI=ON")
endif()
if(NOT Kokkos_ENABLE_CUDA)
  message(FATAL_ERROR "SO3LR is a CUDA/Kokkos implementation; set Kokkos_ENABLE_CUDA=ON")
endif()

set(SO3LR_PACKAGE_ROOT "${LAMMPS_SOURCE_DIR}/SO3LR")
if(NOT EXISTS "${SO3LR_PACKAGE_ROOT}/lammps/pair_so3lr_native_mpi.cpp")
  message(FATAL_ERROR "Incomplete SO3LR source integration under ${SO3LR_PACKAGE_ROOT}")
endif()

file(GLOB SO3LR_NATIVE_SOURCES CONFIGURE_DEPENDS
  "${SO3LR_PACKAGE_ROOT}/src/[^.]*.cpp")

RegisterStyles("${SO3LR_PACKAGE_ROOT}/lammps")
target_sources(lammps PRIVATE
  ${SO3LR_NATIVE_SOURCES}
  "${SO3LR_PACKAGE_ROOT}/lammps/pair_so3lr_native_mpi.cpp"
  "${SO3LR_PACKAGE_ROOT}/lammps/pair_so3lr_turbo.cpp")
target_include_directories(lammps PRIVATE
  "${SO3LR_PACKAGE_ROOT}/include"
  "${SO3LR_PACKAGE_ROOT}/lammps")

find_package(CUDAToolkit REQUIRED)
target_link_libraries(lammps PRIVATE CUDA::cublas)
target_compile_features(lammps PRIVATE cxx_std_20)

message(STATUS "SO3LR native pair styles enabled: so3lr/native/mpi, so3lr/turbo")
