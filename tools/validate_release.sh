#!/bin/bash
# Local/static validation that does not require CUDA or an HPC allocation.
set -Eeuo pipefail

RELEASE_ROOT=$(cd "$(dirname "$0")/.." && pwd)
CXX=${CXX:-$(command -v c++ || true)}
PYTHON=${SO3LR_VALIDATION_PYTHON:-$(command -v python3 || true)}
TMP_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/so3lr-release-validation.XXXXXX")
trap 'rm -rf "${TMP_ROOT}"' EXIT

for script in \
    "${RELEASE_ROOT}/tools/install_into_lammps_source.sh" \
    "${RELEASE_ROOT}/tools/build_external_plugin.sh" \
    "${RELEASE_ROOT}/tools/update_manifest.sh" \
    "${RELEASE_ROOT}/examples/meluxina/00_install_meluxina.sh" \
    "${RELEASE_ROOT}/examples/meluxina/02_prepare_lammps_so3lr_source.sh" \
    "${RELEASE_ROOT}/examples/meluxina/03_build_lammps_so3lr.sh" \
    "${RELEASE_ROOT}/examples/meluxina/04_verify_installation.sh" \
    "${RELEASE_ROOT}/examples/meluxina/submit_convert_checkpoint.sh" \
    "${RELEASE_ROOT}/examples/turbo/run_meluxina.sh" \
    "${RELEASE_ROOT}/examples/turbo/submit_1gpu.sh" \
    "${RELEASE_ROOT}/examples/turbo/submit_4gpu.sh" \
    "${RELEASE_ROOT}/examples/turbo/bind_gpu.sh" \
    "${RELEASE_ROOT}/tests/turbo/job_turbo_validate.sh"; do
  bash -n "${script}"
done
echo "shell_syntax=PASS"

if grep -REn -- '(mpirun|srun).*(-k on|-sf kk|-pk kokkos)' \
    "${RELEASE_ROOT}/README.md" \
    "${RELEASE_ROOT}/docs" \
    "${RELEASE_ROOT}/examples/README.md"; then
  echo "ERROR: obsolete global LAMMPS Kokkos launch command remains" >&2
  exit 2
fi
echo "runtime_documentation=PASS"

for path in \
    source/so3lr-lammps-native/plugins/lammps/pair_so3lr_turbo.cpp \
    source/so3lr-lammps-native/plugins/lammps/pair_so3lr_turbo.h \
    source/so3lr-lammps-native/src/kokkos_packed_so3lr_evaluator.cpp \
    source/so3lr-lammps-native/src/so3lr_repulsion_setup.cpp \
    source/so3lr-lammps-native/src/kokkos_turbo_replica_bridge.cpp \
    source/so3lr-lammps-native/src/mpi_turbo_replica_broker.cpp; do
  [[ -f "${RELEASE_ROOT}/${path}" ]] || {
    echo "ERROR: missing turbo source ${path}" >&2
    exit 2
  }
done
grep -Fq '"${SO3LR_PACKAGE_ROOT}/lammps/pair_so3lr_turbo.cpp"' \
    "${RELEASE_ROOT}/integration/lammps_builtin/cmake/Packages/SO3LR.cmake"
echo "turbo_source_integration=PASS"

[[ -n "${CXX}" ]] || { echo "ERROR: no C++ compiler" >&2; exit 2; }
"${CXX}" -std=c++20 -Wall -Wextra -Wpedantic \
  -I "${RELEASE_ROOT}/source/so3lr-lammps-native/plugins/lammps" \
  "${RELEASE_ROOT}/tests/test_element_map.cpp" \
  -o "${TMP_ROOT}/test_element_map"
"${TMP_ROOT}/test_element_map"

if [[ -n "${PYTHON}" ]]; then
  PYTHONPYCACHEPREFIX="${TMP_ROOT}/pycache" "${PYTHON}" -m py_compile \
    "${RELEASE_ROOT}/tools/checkpoint_conversion/convert_so3lr_checkpoint.py" \
    "${RELEASE_ROOT}/tools/checkpoint_conversion/export_flax_so3lr.py" \
    "${RELEASE_ROOT}/tools/checkpoint_conversion/so3lr_native_format.py" \
    "${RELEASE_ROOT}/examples/scaling/generate_water_boxes.py" \
    "${RELEASE_ROOT}/examples/scaling/analyze_scaling.py" \
    "${RELEASE_ROOT}/source/so3lr-lammps-native/tests/test_capability_rejection.py" \
    "${RELEASE_ROOT}/examples/turbo/generate_images.py" \
    "${RELEASE_ROOT}/tests/turbo/compare_turbo.py" \
    "${RELEASE_ROOT}/tests/jaxref/so3lr_reference.py" \
    "${RELEASE_ROOT}/tests/jaxref/compare_native.py"
  echo "converter_syntax=PASS"
fi

if command -v sha256sum >/dev/null 2>&1; then
  (cd "${RELEASE_ROOT}" && sha256sum -c MANIFEST.sha256) \
    > "${TMP_ROOT}/manifest-check.txt"
elif command -v shasum >/dev/null 2>&1; then
  (cd "${RELEASE_ROOT}" && shasum -a 256 -c MANIFEST.sha256) \
    > "${TMP_ROOT}/manifest-check.txt"
else
  echo "ERROR: no SHA-256 manifest checker available" >&2
  exit 2
fi
echo "manifest_integrity=PASS"

if [[ $# -eq 1 ]]; then
  LAMMPS_SOURCE=$(cd "$1" && pwd)
  cp -R "${LAMMPS_SOURCE}/." "${TMP_ROOT}/lammps"
  "${RELEASE_ROOT}/tools/install_into_lammps_source.sh" "${TMP_ROOT}/lammps"
  grep -q 'PairStyle(so3lr/native/mpi,PairSO3LRNativeMPI)' \
    "${TMP_ROOT}/lammps/src/SO3LR/lammps/pair_so3lr_native_mpi.h"
  grep -q 'PairStyle(so3lr/turbo,PairSO3LRTurbo)' \
    "${TMP_ROOT}/lammps/src/SO3LR/lammps/pair_so3lr_turbo.h"
  grep -q 'include(Packages/SO3LR)' "${TMP_ROOT}/lammps/cmake/CMakeLists.txt"
  echo "builtin_source_integration=PASS"
elif [[ $# -gt 1 ]]; then
  echo "Usage: $0 [clean-lammps-source]" >&2
  exit 2
fi

echo "SO3LR_RELEASE_STATIC_VALIDATION=PASS"
