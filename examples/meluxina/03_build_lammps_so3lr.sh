#!/bin/bash -l
#SBATCH --job-name=so3lr-build
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --output=so3lr-build-%j.out
#SBATCH --error=so3lr-build-%j.err

# Step 3: build and install LAMMPS with the compiled-in SO3LR pair styles.
#
# Compilation needs no GPU, so this runs on a CPU node, which usually starts
# far sooner than a GPU node. CPU nodes have no CUDA driver library, so lmp is
# linked against the CUDA toolkit's driver stub; on a GPU node the real driver
# is loaded at run time. Step 4 (04_verify_installation.sh) then checks the
# installation on a GPU.
#
# The job is resumable: submit it again with the same SO3LR_INSTALL_ROOT if a
# time limit or node failure interrupts compilation.
#
#   sbatch --account=pXXXXXX \
#     --export=ALL,SO3LR_INSTALL_ROOT=/project/home/pXXXXXX/$USER/software/so3lr-lammps-0.4.0 \
#     03_build_lammps_so3lr.sh
#
# Optional: SO3LR_KOKKOS_ARCH (default AMPERE80, the MeluXina A100),
# SO3LR_CUDA_ARCH (default: the number in SO3LR_KOKKOS_ARCH), SO3LR_BUILD_JOBS.
set -Eeuo pipefail
umask 0027

readonly LAMMPS_COMMIT=b1ef9f45934826495efe9b442b39e9c74e20095d

: "${SO3LR_INSTALL_ROOT:?Export SO3LR_INSTALL_ROOT=/project/home/PROJECT/USER/software/so3lr-lammps-0.4.0}"
SO3LR_INSTALL_ROOT=$(readlink -m "${SO3LR_INSTALL_ROOT}")

if [[ -f "${SO3LR_INSTALL_ROOT}/installation-paths.env" ]]; then
  # Produced by Step 2. It contains only paths selected by the submitting user.
  source "${SO3LR_INSTALL_ROOT}/installation-paths.env"
fi

LAMMPS_SOURCE=${LAMMPS_SOURCE:-${SO3LR_INSTALL_ROOT}/lammps-11Feb2026}
BUILD_DIR=${BUILD_DIR:-${SO3LR_INSTALL_ROOT}/build}
INSTALL_PREFIX=${INSTALL_PREFIX:-${SO3LR_INSTALL_ROOT}/install}
LAMMPS_SOURCE=$(readlink -f "${LAMMPS_SOURCE}")
BUILD_DIR=$(readlink -m "${BUILD_DIR}")
INSTALL_PREFIX=$(readlink -m "${INSTALL_PREFIX}")

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -f "${LAMMPS_SOURCE}/src/version.h" ]] || \
  fail "LAMMPS source is missing; complete Step 2 first"
[[ -f "${LAMMPS_SOURCE}/src/SO3LR/lammps/pair_so3lr_native_mpi.cpp" ]] || \
  fail "SO3LR source package is missing; complete Step 2 first"
[[ -f "${LAMMPS_SOURCE}/src/SO3LR/lammps/pair_so3lr_turbo.cpp" ]] || \
  fail "SO3LR turbo source is missing; complete Step 2 with this release"
[[ "$(git -C "${LAMMPS_SOURCE}" rev-parse HEAD)" == "${LAMMPS_COMMIT}" ]] || \
  fail "LAMMPS source is not the validated patch_11Feb2026 commit"

PAIR_SOURCE=${LAMMPS_SOURCE}/src/SO3LR/lammps/pair_so3lr_native_mpi.cpp
grep -Fq 'IndexView(std::string(prefix) + suffix, capacity)' "${PAIR_SOURCE}" || \
  fail "Kokkos 5 compatibility fix 1 is absent; rerun Step 2 with the current release"
grep -Fq 'IndexView result(std::string(label), values.size())' "${PAIR_SOURCE}" || \
  fail "Kokkos 5 compatibility fix 2 is absent; rerun Step 2 with the current release"

export MODULEPATH=/apps/USE/system/modules:/apps/USE/easybuild/release/2025.1/modules/all
source /usr/share/lmod/lmod/init/bash
module purge
module load GCCcore/14.2.0
module load CUDA/12.8.0
module load NVHPC/25.3-CUDA-12.8.0
module load OpenMPI/5.0.7-NVHPC-25.3-CUDA-12.8.0
module load CMake/3.31.3-GCCcore-14.2.0

for command_name in cmake mpicc mpicxx nvc++ nvcc; do
  command -v "${command_name}" >/dev/null || fail "${command_name} is unavailable"
done

JOB_LABEL=${SLURM_JOB_ID:-manual}
LOG_DIR=${SO3LR_INSTALL_ROOT}/logs/build-${JOB_LABEL}
BUILD_JOBS=${SO3LR_BUILD_JOBS:-${SLURM_CPUS_PER_TASK:-16}}
mkdir -p "${BUILD_DIR}" "${INSTALL_PREFIX}" "${LOG_DIR}"

KOKKOS_ARCH=${SO3LR_KOKKOS_ARCH:-AMPERE80}
CUDA_ARCH=${SO3LR_CUDA_ARCH:-$(sed -E 's/^[A-Z_]+//' <<<"${KOKKOS_ARCH}")}
[[ "${KOKKOS_ARCH}" =~ ^[A-Z_]+[0-9]+$ && "${CUDA_ARCH}" =~ ^[0-9]+$ ]] || \
  fail "invalid SO3LR_KOKKOS_ARCH=${KOKKOS_ARCH} / SO3LR_CUDA_ARCH=${CUDA_ARCH}"

# CPU nodes have no libcuda.so.1; let the linker resolve it from the stub.
STUB_DIR=${SO3LR_INSTALL_ROOT}/cuda-driver-stub
mkdir -p "${STUB_DIR}"
# The CUDA module sets EBROOTCUDA (nvcc on PATH may be NVHPC's copy).
CUDA_STUB=""
for root in "${EBROOTCUDA:-}" "${CUDA_HOME:-}" "${CUDA_PATH:-}" \
            "$(dirname "$(command -v nvcc)")/.."; do
  [[ -n "${root}" && -f "${root}/lib64/stubs/libcuda.so" ]] && \
    { CUDA_STUB=${root}/lib64/stubs/libcuda.so; break; }
done
[[ -n "${CUDA_STUB}" ]] || fail "CUDA driver stub (lib64/stubs/libcuda.so) not found"
ln -sf "$(readlink -f "${CUDA_STUB}")" "${STUB_DIR}/libcuda.so.1"

echo "host=$(hostname)"
echo "kokkos_arch=${KOKKOS_ARCH} cuda_arch=${CUDA_ARCH}"
echo "install_root=${SO3LR_INSTALL_ROOT}"
echo "lammps_source=${LAMMPS_SOURCE}"
echo "build_dir=${BUILD_DIR}"
echo "install_prefix=${INSTALL_PREFIX}"
echo "build_jobs=${BUILD_JOBS}"
echo "log_dir=${LOG_DIR}"
module list

# This is the compiler combination that passed on MeluXina with LAMMPS's
# bundled Kokkos 5.0.2. Do not mix it with another MPI/CUDA toolchain.
export KOKKOS_WRAPPER_CXX=$(command -v mpicxx)
export OMPI_CXX=$(command -v nvc++)

cmake -S "${LAMMPS_SOURCE}/cmake" -B "${BUILD_DIR}" \
  -D CMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_C_COMPILER="$(command -v mpicc)" \
  -D CMAKE_CXX_COMPILER="${LAMMPS_SOURCE}/lib/kokkos/bin/nvcc_wrapper" \
  -D MPI_C_COMPILER="$(command -v mpicc)" \
  -D MPI_CXX_COMPILER="$(command -v mpicxx)" \
  -D CMAKE_CUDA_COMPILER="$(command -v nvcc)" \
  -D BUILD_MPI=ON \
  -D BUILD_OMP=OFF \
  -D BUILD_SHARED_LIBS=ON \
  -D LAMMPS_EXCEPTIONS=ON \
  -D PKG_KOKKOS=ON \
  -D PKG_SO3LR=ON \
  -D PKG_PLUGIN=OFF \
  -D Kokkos_ENABLE_CUDA=ON \
  -D Kokkos_ENABLE_SERIAL=ON \
  -D Kokkos_ARCH_${KOKKOS_ARCH}=ON \
  -D CMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
  -D CMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link,${STUB_DIR}" \
  2>&1 | tee "${LOG_DIR}/cmake.log"

cmake --build "${BUILD_DIR}" --target lmp --parallel "${BUILD_JOBS}" \
  2>&1 | tee "${LOG_DIR}/build.log"
cmake --install "${BUILD_DIR}" \
  2>&1 | tee "${LOG_DIR}/install.log"

LAMMPS_EXE=${INSTALL_PREFIX}/bin/lmp
[[ -x "${LAMMPS_EXE}" ]] || fail "installed LAMMPS executable is missing"
# Both styles must be registered in the installed library (works for a
# resumed build too, where nothing is recompiled).
for style in so3lr/native/mpi so3lr/turbo; do
  grep -qa "${style}" "${INSTALL_PREFIX}"/lib*/liblammps.so* || \
    fail "pair style ${style} is missing from the installed liblammps"
done

# BUILD_SHARED_LIBS installs liblammps.so under lib64 on MeluXina and the
# binary has no usable RPATH, so the environment file exposes that directory.
# It also loads the toolchain modules lmp was built with (the MPI and CUDA
# runtime libraries come from them); MeluXina login nodes have no module
# system, so that part only runs where Lmod exists.
cat > "${SO3LR_INSTALL_ROOT}/use-lammps-so3lr.env" <<ENVFILE
# source this file before running LAMMPS with SO3LR
if [ -f /usr/share/lmod/lmod/init/bash ]; then
  export MODULEPATH=/apps/USE/system/modules:/apps/USE/easybuild/release/2025.1/modules/all
  . /usr/share/lmod/lmod/init/bash
  module load GCCcore/14.2.0 CUDA/12.8.0 NVHPC/25.3-CUDA-12.8.0 OpenMPI/5.0.7-NVHPC-25.3-CUDA-12.8.0
fi
export SO3LR_LAMMPS_ROOT='${INSTALL_PREFIX}'
export SO3LR_MODELS='${SO3LR_RELEASE_ROOT:-}/models'
export PATH='${INSTALL_PREFIX}/bin':"\${PATH}"
export LD_LIBRARY_PATH='${INSTALL_PREFIX}/lib64:${INSTALL_PREFIX}/lib':"\${LD_LIBRARY_PATH:-}"
ENVFILE

sha256sum "${LAMMPS_EXE}" "${PAIR_SOURCE}" | tee "${LOG_DIR}/sha256.txt"
echo "environment_file=${SO3LR_INSTALL_ROOT}/use-lammps-so3lr.env"
echo "lammps_executable=${LAMMPS_EXE}"
echo "kokkos_arch=${KOKKOS_ARCH}"
echo "next: 04_verify_installation.sh checks the installation on a GPU"
echo "SO3LR_MELUXINA_BUILD=PASS"
