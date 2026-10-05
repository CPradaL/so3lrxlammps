#!/bin/bash -l
#SBATCH --job-name=so3lr-source
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=so3lr-source-%j.out
#SBATCH --error=so3lr-source-%j.err

# Installation Step 2: source preparation.
#
# Submit from the release's examples/meluxina directory. MeluXina requires an
# account on the sbatch command line:
#
#   sbatch --account=pXXXXXX \
#     --export=ALL,SO3LR_INSTALL_ROOT=/project/home/pXXXXXX/$USER/software/so3lr-lammps-0.4.0 \
#     02_prepare_lammps_so3lr_source.sh
#
# If the script is submitted from another directory, also export the absolute
# path to the unpacked release as SO3LR_RELEASE_ROOT.
set -Eeuo pipefail
umask 0027

readonly LAMMPS_COMMIT=b1ef9f45934826495efe9b442b39e9c74e20095d
readonly LAMMPS_TAG=patch_11Feb2026

: "${SO3LR_INSTALL_ROOT:?Export SO3LR_INSTALL_ROOT=/project/home/PROJECT/USER/software/so3lr-lammps-0.4.0}"

SUBMIT_DIR=${SLURM_SUBMIT_DIR:-$PWD}
SO3LR_RELEASE_ROOT=${SO3LR_RELEASE_ROOT:-$(cd "${SUBMIT_DIR}/../.." && pwd)}
SO3LR_RELEASE_ROOT=$(readlink -f "${SO3LR_RELEASE_ROOT}")
SO3LR_INSTALL_ROOT=$(readlink -m "${SO3LR_INSTALL_ROOT}")
LAMMPS_SOURCE=${LAMMPS_SOURCE:-${SO3LR_INSTALL_ROOT}/lammps-11Feb2026}
LAMMPS_SOURCE=$(readlink -m "${LAMMPS_SOURCE}")
INSTALLER=${SO3LR_RELEASE_ROOT}/tools/install_into_lammps_source.sh

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -x "${INSTALLER}" ]] || fail "release installer not found: ${INSTALLER}"
command -v git >/dev/null || fail "git is not available"
command -v patch >/dev/null || fail "patch is not available"

echo "host=$(hostname)"
echo "release_root=${SO3LR_RELEASE_ROOT}"
echo "install_root=${SO3LR_INSTALL_ROOT}"
echo "lammps_source=${LAMMPS_SOURCE}"
echo "lammps_tag=${LAMMPS_TAG}"
echo "lammps_commit=${LAMMPS_COMMIT}"
git --version

mkdir -p "${SO3LR_INSTALL_ROOT}"

if [[ ! -e "${LAMMPS_SOURCE}" ]]; then
  mkdir -p "${LAMMPS_SOURCE}"
  git -C "${LAMMPS_SOURCE}" init
  git -C "${LAMMPS_SOURCE}" remote add origin https://github.com/lammps/lammps.git
  git -C "${LAMMPS_SOURCE}" fetch --depth=1 origin "${LAMMPS_COMMIT}"
  git -C "${LAMMPS_SOURCE}" checkout --detach FETCH_HEAD
elif [[ ! -d "${LAMMPS_SOURCE}/.git" ]]; then
  fail "${LAMMPS_SOURCE} exists but is not a Git checkout; choose a fresh path"
fi

actual_commit=$(git -C "${LAMMPS_SOURCE}" rev-parse HEAD)
[[ "${actual_commit}" == "${LAMMPS_COMMIT}" ]] || \
  fail "wrong LAMMPS commit in ${LAMMPS_SOURCE}: ${actual_commit}"

if [[ -d "${LAMMPS_SOURCE}/src/SO3LR" ]] || \
   [[ -e "${LAMMPS_SOURCE}/cmake/Modules/Packages/SO3LR.cmake" ]] || \
   grep -q 'option(PKG_SO3LR ' "${LAMMPS_SOURCE}/cmake/CMakeLists.txt"; then
  echo "SO3LR is already present; verifying that it matches this release."
  diff -qr \
    "${SO3LR_RELEASE_ROOT}/source/so3lr-lammps-native/include" \
    "${LAMMPS_SOURCE}/src/SO3LR/include" >/dev/null || \
    fail "installed SO3LR headers differ from this release; use a fresh LAMMPS tree"
  diff -qr \
    "${SO3LR_RELEASE_ROOT}/source/so3lr-lammps-native/src" \
    "${LAMMPS_SOURCE}/src/SO3LR/src" >/dev/null || \
    fail "installed SO3LR sources differ from this release; use a fresh LAMMPS tree"
  cmp -s \
    "${SO3LR_RELEASE_ROOT}/source/so3lr-lammps-native/plugins/lammps/pair_so3lr_native_mpi.cpp" \
    "${LAMMPS_SOURCE}/src/SO3LR/lammps/pair_so3lr_native_mpi.cpp" || \
    fail "installed pair source differs from this release; use a fresh LAMMPS tree"
  cmp -s \
    "${SO3LR_RELEASE_ROOT}/source/so3lr-lammps-native/plugins/lammps/pair_so3lr_turbo.cpp" \
    "${LAMMPS_SOURCE}/src/SO3LR/lammps/pair_so3lr_turbo.cpp" || \
    fail "installed turbo source differs from this release; use a fresh LAMMPS tree"
  cmp -s \
    "${SO3LR_RELEASE_ROOT}/integration/lammps_builtin/cmake/Packages/SO3LR.cmake" \
    "${LAMMPS_SOURCE}/cmake/Modules/Packages/SO3LR.cmake" || \
    fail "installed SO3LR CMake module differs from this release; use a fresh LAMMPS tree"
else
  "${INSTALLER}" "${LAMMPS_SOURCE}"
fi

PAIR_SOURCE=${LAMMPS_SOURCE}/src/SO3LR/lammps/pair_so3lr_native_mpi.cpp
grep -Fq 'IndexView(std::string(prefix) + suffix, capacity)' "${PAIR_SOURCE}" || \
  fail "the first Kokkos 5 label compatibility fix is missing"
grep -Fq 'IndexView result(std::string(label), values.size())' "${PAIR_SOURCE}" || \
  fail "the second Kokkos 5 label compatibility fix is missing"

cat > "${SO3LR_INSTALL_ROOT}/installation-paths.env" <<EOF
export SO3LR_RELEASE_ROOT='${SO3LR_RELEASE_ROOT}'
export SO3LR_INSTALL_ROOT='${SO3LR_INSTALL_ROOT}'
export LAMMPS_SOURCE='${LAMMPS_SOURCE}'
export BUILD_DIR='${SO3LR_INSTALL_ROOT}/build'
export INSTALL_PREFIX='${SO3LR_INSTALL_ROOT}/install'
EOF

git -C "${LAMMPS_SOURCE}" status --short
echo "paths_file=${SO3LR_INSTALL_ROOT}/installation-paths.env"
echo "SO3LR_MELUXINA_SOURCE_PREPARE=PASS"
