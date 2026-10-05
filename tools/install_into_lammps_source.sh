#!/bin/bash
# Vendor the production SO3LR source into an unmodified LAMMPS source tree.
set -Eeuo pipefail

usage() {
  echo "Usage: $0 /path/to/lammps-source" >&2
  echo "Tested target: LAMMPS patch_11Feb2026 (commit b1ef9f45934826495efe9b442b39e9c74e20095d)." >&2
}

if [[ $# -ne 1 ]]; then
  usage
  exit 2
fi

LAMMPS_SOURCE=$(cd "$1" && pwd)
RELEASE_ROOT=$(cd "$(dirname "$0")/.." && pwd)
NATIVE_ROOT="${RELEASE_ROOT}/source/so3lr-lammps-native"
INTEGRATION_ROOT="${RELEASE_ROOT}/integration/lammps_builtin"
TARGET_ROOT="${LAMMPS_SOURCE}/src/SO3LR"
CMAKE_FILE="${LAMMPS_SOURCE}/cmake/CMakeLists.txt"
PACKAGE_FILE="${LAMMPS_SOURCE}/cmake/Modules/Packages/SO3LR.cmake"

[[ -f "${LAMMPS_SOURCE}/src/version.h" && -f "${CMAKE_FILE}" ]] || {
  echo "ERROR: ${LAMMPS_SOURCE} is not a LAMMPS source tree" >&2
  exit 3
}
[[ -d "${NATIVE_ROOT}/src" && -d "${NATIVE_ROOT}/include" ]] || {
  echo "ERROR: release source is incomplete under ${NATIVE_ROOT}" >&2
  exit 4
}

if ! grep -Eq 'LAMMPS_VERSION[[:space:]]+"11 Feb 2026"' "${LAMMPS_SOURCE}/src/version.h"; then
  if [[ "${SO3LR_ALLOW_UNTESTED_LAMMPS:-0}" != 1 ]]; then
    echo "ERROR: this release is validated only with LAMMPS 11Feb2026." >&2
    echo "Set SO3LR_ALLOW_UNTESTED_LAMMPS=1 to perform an unsupported compatibility trial." >&2
    exit 5
  fi
  echo "WARNING: installing into an untested LAMMPS version" >&2
fi

if [[ -e "${TARGET_ROOT}" || -e "${PACKAGE_FILE}" ]] || \
   grep -q 'option(PKG_SO3LR ' "${CMAKE_FILE}"; then
  echo "ERROR: SO3LR already appears to be installed in ${LAMMPS_SOURCE}" >&2
  echo "Use a clean LAMMPS source tree for a reproducible build." >&2
  exit 6
fi

PATCH_FILE="${INTEGRATION_ROOT}/lammps-11Feb2026-so3lr.patch"
if ! patch --dry-run --batch --forward -d "${LAMMPS_SOURCE}" -p1 < "${PATCH_FILE}" >/dev/null; then
  echo "ERROR: the SO3LR CMake patch does not apply cleanly." >&2
  echo "Use the exact patch_11Feb2026 source or review the patch for a newer LAMMPS release." >&2
  exit 7
fi

patch --batch --forward -d "${LAMMPS_SOURCE}" -p1 < "${PATCH_FILE}"
mkdir -p "${TARGET_ROOT}/lammps" "$(dirname "${PACKAGE_FILE}")"
cp -R "${NATIVE_ROOT}/include" "${TARGET_ROOT}/"
cp -R "${NATIVE_ROOT}/src" "${TARGET_ROOT}/"
cp "${NATIVE_ROOT}/plugins/lammps/pair_so3lr_native_mpi.cpp" "${TARGET_ROOT}/lammps/"
cp "${NATIVE_ROOT}/plugins/lammps/pair_so3lr_native_mpi.h" "${TARGET_ROOT}/lammps/"
cp "${NATIVE_ROOT}/plugins/lammps/pair_so3lr_turbo.cpp" "${TARGET_ROOT}/lammps/"
cp "${NATIVE_ROOT}/plugins/lammps/pair_so3lr_turbo.h" "${TARGET_ROOT}/lammps/"
cp "${NATIVE_ROOT}/plugins/lammps/so3lr_element_map.h" "${TARGET_ROOT}/lammps/"
cp "${INTEGRATION_ROOT}/cmake/Packages/SO3LR.cmake" "${PACKAGE_FILE}"

echo "SO3LR source installed into ${LAMMPS_SOURCE}"
echo "Configure LAMMPS with PKG_SO3LR=ON, PKG_KOKKOS=ON, BUILD_MPI=ON, and Kokkos_ENABLE_CUDA=ON."
echo "SO3LR_LAMMPS_SOURCE_INSTALL=PASS"
