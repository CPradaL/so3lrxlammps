#!/bin/bash
# Called by one of the Slurm wrappers; each group owns one GPU and one broker.
set -Eeuo pipefail
umask 0027

EXAMPLE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RELEASE_ROOT=$(cd "${EXAMPLE_DIR}/../.." && pwd)
: "${SO3LR_INSTALL_ROOT:?Set SO3LR_INSTALL_ROOT to the fresh LAMMPS installation}"
: "${SO3LR_TURBO_GROUPS:?Use submit_1gpu.sh or submit_4gpu.sh}"
[[ "${SO3LR_TURBO_GROUPS}" == 1 || "${SO3LR_TURBO_GROUPS}" == 4 ]] || {
  echo 'SO3LR_TURBO_GROUPS must be 1 or 4' >&2
  exit 2
}
IMAGES_PER_GPU=${IMAGES_PER_GPU:-4}
[[ "${IMAGES_PER_GPU}" =~ ^[0-9]+$ && "${IMAGES_PER_GPU}" -ge 2 && "${IMAGES_PER_GPU}" -le 32 ]] || {
  echo 'IMAGES_PER_GPU must be in 2..32 for this example' >&2
  exit 2
}
required_tasks=$((SO3LR_TURBO_GROUPS * IMAGES_PER_GPU))
[[ "${SLURM_NTASKS:-0}" -ge "${required_tasks}" ]] || {
  echo "Need ${required_tasks} allocated tasks; submit with --ntasks=${required_tasks}" >&2
  exit 2
}
RUN_STEPS=${RUN_STEPS:-1000}
[[ "${RUN_STEPS}" =~ ^[0-9]+$ && "${RUN_STEPS}" -ge 1 ]] || {
  echo 'RUN_STEPS must be a positive integer' >&2
  exit 2
}

export MODULEPATH=/apps/USE/system/modules:/apps/USE/easybuild/release/2025.1/modules/all
source /usr/share/lmod/lmod/init/bash
module purge
module load GCCcore/14.2.0
module load CUDA/12.8.0
module load NVHPC/25.3-CUDA-12.8.0
module load OpenMPI/5.0.7-NVHPC-25.3-CUDA-12.8.0
module load Python/3.13.1-GCCcore-14.2.0  # generate_images.py; compute nodes have no python3

LMP=${SO3LR_LMP_BIN:-${SO3LR_INSTALL_ROOT}/install/bin/lmp}
MODEL=${SO3LR_MODEL:-${RELEASE_ROOT}/models/so3lr.so3lr}
DATA=${SO3LR_DATA:-${EXAMPLE_DIR}/water_molecule.data}
RESULT_DIR=${SO3LR_RESULT_DIR:-${SLURM_SUBMIT_DIR}/turbo-example-${SLURM_JOB_ID}}
for file in "${LMP}" "${MODEL}" "${DATA}"; do
  [[ -f "${file}" ]] || { echo "Missing ${file}" >&2; exit 3; }
done
[[ ! -e "${RESULT_DIR}" ]] || { echo "Result directory exists: ${RESULT_DIR}" >&2; exit 3; }
mkdir -p "${RESULT_DIR}"

export OMP_NUM_THREADS=1 OMP_PROC_BIND=false
export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,tcp OMPI_MCA_mtl=^ofi
export UCX_TLS=tcp,self UCX_IB_GPU_DIRECT_RDMA=no UCX_MEMTYPE_CACHE=n
export LD_LIBRARY_PATH="${SO3LR_INSTALL_ROOT}/install/lib64:${SO3LR_INSTALL_ROOT}/install/lib:${LD_LIBRARY_PATH:-}"
"${LMP}" -h > "${RESULT_DIR}/lammps-help.txt"
grep -Eq '(^|[[:space:]])so3lr/native/mpi([[:space:]]|$)' "${RESULT_DIR}/lammps-help.txt"
grep -Eq '(^|[[:space:]])so3lr/turbo([[:space:]]|$)' "${RESULT_DIR}/lammps-help.txt"

pids=()
for ((group=0; group<SO3LR_TURBO_GROUPS; ++group)); do
  printf -v group_name '%02d' "${group}"
  group_dir="${RESULT_DIR}/group_${group_name}"
  mkdir -p "${group_dir}/screens"
  for ((image=0; image<IMAGES_PER_GPU; ++image)); do
    printf -v image_name '%04d' "${image}"
    mkdir -p "${group_dir}/image_${image_name}"
  done
  python3 "${EXAMPLE_DIR}/generate_images.py" \
    --template "${EXAMPLE_DIR}/in.turbo.template" \
    --output "${group_dir}/input.in" \
    --manifest "${group_dir}/images.csv" \
    --images "${IMAGES_PER_GPU}" --group "${group}"
  (
    srun --exact --exclusive --kill-on-bad-exit=1 --nodes=1 \
      --ntasks="${IMAGES_PER_GPU}" --ntasks-per-node="${IMAGES_PER_GPU}" \
      --cpus-per-task=1 --gpus-per-node=1 --gpu-bind=none --cpu-bind=cores \
      "${EXAMPLE_DIR}/bind_gpu.sh" "${group_dir}/gpu-assignment.txt" \
      "${LMP}" -partition "${IMAGES_PER_GPU}x1" -plog none \
      -pscreen "${group_dir}/screens/screen" \
      -var OUT_ROOT "${group_dir}" -var NIMAGES "${IMAGES_PER_GPU}" \
      -var MODEL_FILE "${MODEL}" -var DATA_FILE "${DATA}" \
      -var RUN_STEPS "${RUN_STEPS}" -in "${group_dir}/input.in" \
      > "${group_dir}/universe.stdout" 2>&1
  ) &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then failed=1; fi
done
[[ "${failed}" == 0 ]] || { echo 'At least one turbo group failed' >&2; exit 4; }
for ((group=0; group<SO3LR_TURBO_GROUPS; ++group)); do
  printf -v group_name '%02d' "${group}"
  for ((image=0; image<IMAGES_PER_GPU; ++image)); do
    printf -v image_name '%04d' "${image}"
    grep -q '^SO3LR_TURBO_EXAMPLE_FINAL ' \
      "${RESULT_DIR}/group_${group_name}/image_${image_name}/log.lammps" || {
      echo "Missing final marker for group ${group_name}, image ${image_name}" >&2
      exit 5
    }
  done
done
printf 'status=PASS\ngroups=%s\nimages_per_gpu=%s\nsteps=%s\n' \
  "${SO3LR_TURBO_GROUPS}" "${IMAGES_PER_GPU}" "${RUN_STEPS}" \
  > "${RESULT_DIR}/status.txt"
echo "SO3LR_TURBO_EXAMPLE=PASS result_dir=${RESULT_DIR}"
