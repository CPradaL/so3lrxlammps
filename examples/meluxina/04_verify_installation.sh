#!/bin/bash -l
#SBATCH --job-name=so3lr-verify
#SBATCH --partition=gpu
#SBATCH --qos=test
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=00:20:00
#SBATCH --output=so3lr-verify-%j.out
#SBATCH --error=so3lr-verify-%j.err

# Step 4: check a finished installation on a GPU node.
#
#   sbatch --account=pXXXXXX \
#     --export=ALL,SO3LR_INSTALL_ROOT=/project/home/pXXXXXX/$USER/software/so3lr-lammps-0.4.0 \
#     04_verify_installation.sh
#
# 1. both pair styles are registered in the installed lmp;
# 2. so3lr/native/mpi single points against the JAX references in
#    tests/jaxref/refs: the alanine peptide (neutral) and the acetate anion
#    (charge -1);
# 3. so3lr/turbo: the same two molecules packed on one GPU give the same
#    energies and forces as the ordinary style.
# Writes ${SO3LR_INSTALL_ROOT}/verify-<job>/status.txt and prints
# SO3LR_MELUXINA_VERIFY=PASS on success. Takes a few minutes.
set -Eeuo pipefail
umask 0027

: "${SO3LR_INSTALL_ROOT:?Export SO3LR_INSTALL_ROOT=/project/home/PROJECT/USER/software/so3lr-lammps-0.4.0}"
SO3LR_INSTALL_ROOT=$(readlink -m "${SO3LR_INSTALL_ROOT}")
[[ -f "${SO3LR_INSTALL_ROOT}/installation-paths.env" ]] && \
  source "${SO3LR_INSTALL_ROOT}/installation-paths.env"
ENV_FILE=${SO3LR_INSTALL_ROOT}/use-lammps-so3lr.env

fail() {
  echo "ERROR: $*" >&2
  echo "SO3LR_MELUXINA_VERIFY=FAIL"
  exit 1
}

[[ -f "${ENV_FILE}" ]] || fail "${ENV_FILE} is missing; complete Step 3 first"
: "${SO3LR_RELEASE_ROOT:?SO3LR_RELEASE_ROOT is unknown; export it or complete Step 2}"
source "${ENV_FILE}"
module load Python/3.13.1-GCCcore-14.2.0 SciPy-bundle/2025.06-gfbf-2025a
export OMP_NUM_THREADS=1 OMP_PROC_BIND=false
export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,tcp OMPI_MCA_mtl=^ofi

R=${SO3LR_RELEASE_ROOT}
J=${R}/tests/jaxref
MODEL=so3lr                       # reference set in tests/jaxref/refs
MODEL_FILE=${SO3LR_VERIFY_MODEL:-${R}/models/so3lr.so3lr}
OUT=${SO3LR_INSTALL_ROOT}/verify-${SLURM_JOB_ID:-manual}
mkdir -p "${OUT}/${MODEL}/ordinary" "${OUT}/${MODEL}/turbo"
if [[ -n "${MODEL_FILE}" ]]; then
  MODEL_FILE=$(readlink -f "${MODEL_FILE}")
  [[ -f "${MODEL_FILE}" ]] || fail "SO3LR_VERIFY_MODEL not found: ${SO3LR_VERIFY_MODEL}"
fi

echo "host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "lmp=$(command -v lmp)"
echo "model=${MODEL_FILE:-none (registration check only)}"

echo "=== 1. registered pair styles"
lmp -h > "${OUT}/lmp-help.txt" 2>&1 || fail "lmp -h failed (see ${OUT}/lmp-help.txt)"
for style in so3lr/native/mpi so3lr/turbo; do
  grep -Eq "(^|[[:space:]])${style}([[:space:]]|$)" "${OUT}/lmp-help.txt" || \
    fail "pair style ${style} is not registered"
  echo "  ${style}: ok"
done

if [[ -z "${MODEL_FILE}" ]]; then
  printf 'status=PASS\nchecks=registration only\n' > "${OUT}/status.txt"
  echo "No model file: skipped the numerical checks."
  echo "results=${OUT}"
  echo "SO3LR_MELUXINA_VERIFY=PASS"
  exit 0
fi

echo "=== 2. single points against the JAX reference"
# name  data file  elements  boundary  charge  multiplicity
cases=(
  "alanine|${R}/examples/alanine/alanine.data|C H O N|s s s|0|1"
  "acetate|${J}/molecules/acetate.data|C H O|f f f|-1|1"
)
for c in "${cases[@]}"; do
  IFS='|' read -r name data elements boundary charge mult <<<"${c}"
  srun --ntasks=1 --exact --gpus-per-task=1 lmp -screen none \
    -var DATA_FILE "${data}" -var MODEL_FILE "${MODEL_FILE}" \
    -var ELEMENTS "${elements}" -var BOUNDARY "${boundary}" \
    -var CHARGE "${charge}" -var MULT "${mult}" \
    -var DUMP "${OUT}/${name}.dump" -log "${OUT}/${name}.log" \
    -in "${J}/in.single_point" || fail "LAMMPS failed for ${name} (see ${OUT}/${name}.log)"
  python3 "${J}/compare_native.py" "${J}/refs/${MODEL}_${name}.json" \
    "${OUT}/${name}.dump" "${OUT}/${name}.log" --tol-force 1e-6 \
    --tol-energy-per-atom 1e-7 | grep -E 'energy|max\|dF|COMPARE' || true
  python3 "${J}/compare_native.py" "${J}/refs/${MODEL}_${name}.json" \
    "${OUT}/${name}.dump" "${OUT}/${name}.log" --tol-force 1e-6 \
    --tol-energy-per-atom 1e-7 > /dev/null || fail "${name} disagrees with the JAX reference"
done

echo "=== 3. turbo against the ordinary style"
T=${R}/tests/turbo
TAGS=(alanine acetate)
DATA=("${R}/examples/alanine/alanine.data" "${J}/molecules/acetate.data")
ELEM=("C H O N" "C H O")
CHG=(0 -1)
MLT=(1 1)
for i in 0 1; do
  srun --ntasks=1 --exact --gpus-per-task=1 lmp -screen none \
    -var PAIR so3lr/native/mpi -var MODEL_FILE "${MODEL_FILE}" \
    -var DATA_FILE "${DATA[$i]}" -var ELEMENTS "${ELEM[$i]}" -var TAG "${TAGS[$i]}" \
    -var CHARGE "${CHG[$i]}" -var MULT "${MLT[$i]}" \
    -var OUT "${OUT}/${MODEL}/ordinary" -var RUN_STEPS 0 -in "${T}/in.common" || \
    fail "ordinary run failed for ${TAGS[$i]}"
done
{
  echo "variable TAG world ${TAGS[*]}"
  echo "variable DATA_FILE world ${DATA[*]}"
  printf 'variable ELEMENTS world'; printf ' "%s"' "${ELEM[@]}"; echo
  echo "variable CHARGE world ${CHG[*]}"
  echo "variable MULT world ${MLT[*]}"
  echo "include ${T}/in.common"
} > "${OUT}/in.turbo"
srun --ntasks=2 --exact --gpus-per-node=1 --gpu-bind=none lmp \
  -partition 2x1 -plog none -pscreen none \
  -var PAIR so3lr/turbo -var MODEL_FILE "${MODEL_FILE}" \
  -var OUT "${OUT}/${MODEL}/turbo" -var RUN_STEPS 0 -in "${OUT}/in.turbo" \
  > "${OUT}/${MODEL}/turbo/universe.stdout" 2>&1 || \
  fail "turbo run failed (see ${OUT}/${MODEL}/turbo/universe.stdout)"
python3 "${T}/compare_turbo.py" "${OUT}" || fail "turbo disagrees with the ordinary style"

printf 'status=PASS\nmodel=%s\nlmp=%s\n' "${MODEL}" "$(command -v lmp)" > "${OUT}/status.txt"
echo "results=${OUT}"
echo "SO3LR_MELUXINA_VERIFY=PASS"
