#!/bin/bash
# Build SO3LR as a loadable module against the exact matching LAMMPS source.
set -Eeuo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 LAMMPS_SOURCE BUILD_DIRECTORY INSTALL_PREFIX" >&2
  exit 2
fi

LAMMPS_SOURCE=$(cd "$1" && pwd)
BUILD_DIRECTORY=$2
INSTALL_PREFIX=$3
SOURCE_ROOT=$(cd "$(dirname "$0")/../source/so3lr-lammps-native" && pwd)
MPI_CXX=${SO3LR_MPI_CXX:-$(command -v mpicxx || true)}
CUDA_HOST_CXX=${SO3LR_CUDA_HOST_CXX:-$(command -v nvc++ || true)}
KOKKOS_ARCH=${SO3LR_KOKKOS_ARCH:-AMPERE80}

[[ -f "${LAMMPS_SOURCE}/src/lammps.h" ]] || {
  echo "ERROR: ${LAMMPS_SOURCE} is not a LAMMPS source tree" >&2
  exit 3
}
[[ -x "${LAMMPS_SOURCE}/lib/kokkos/bin/nvcc_wrapper" ]] || {
  echo "ERROR: bundled LAMMPS Kokkos nvcc_wrapper is absent" >&2
  exit 4
}
[[ -n "${MPI_CXX}" && -n "${CUDA_HOST_CXX}" ]] || {
  echo "ERROR: set SO3LR_MPI_CXX and SO3LR_CUDA_HOST_CXX" >&2
  exit 5
}

export KOKKOS_WRAPPER_CXX="${MPI_CXX}"
export OMPI_CXX="${CUDA_HOST_CXX}"
cmake -S "${SOURCE_ROOT}/plugins/lammps" -B "${BUILD_DIRECTORY}" \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_CXX_COMPILER="${LAMMPS_SOURCE}/lib/kokkos/bin/nvcc_wrapper" \
  -D MPI_CXX_COMPILER="${MPI_CXX}" \
  -D LAMMPS_HEADER_DIR="${LAMMPS_SOURCE}/src" \
  -D SO3LR_KOKKOS_SOURCE="${LAMMPS_SOURCE}/lib/kokkos" \
  -D Kokkos_ENABLE_CUDA=ON -D Kokkos_ENABLE_SERIAL=ON \
  -D "Kokkos_ARCH_${KOKKOS_ARCH}=ON"
cmake --build "${BUILD_DIRECTORY}" --target so3lrnativeplugin \
  --parallel "${SO3LR_BUILD_JOBS:-8}"
mkdir -p "${INSTALL_PREFIX}/lib"
install -m 0755 "${BUILD_DIRECTORY}/so3lrnativeplugin.so" \
  "${INSTALL_PREFIX}/lib/"
echo "SO3LR_EXTERNAL_PLUGIN_BUILD=PASS"
