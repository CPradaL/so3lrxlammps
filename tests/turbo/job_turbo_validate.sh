#!/bin/bash -l
#SBATCH --job-name=so3lr-turbo-validate
#SBATCH --partition=gpu
#SBATCH --qos=test
#SBATCH --nodes=1 --ntasks=11 --cpus-per-task=4 --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
# so3lr/turbo against so3lr/native/mpi for every model, on MeluXina.
#
#   sbatch --account=PROJECT --export=ALL,SO3LR_LMP=/path/bin/lmp,\
#          SO3LR_RELEASE_ROOT=/path/so3lr-lammps-0.4.0,\
#          [SO3LR_MODEL_DIR=/path/to/models,MODELS="so3lr other-model"] \
#          job_turbo_validate.sh
#
# Eleven images in one pack on one GPU, neutral, charged and open-shell
# together: the alanine peptide (162 atoms) neutral, again neutral, and as a
# cation doublet; a water molecule, ethanol and a water trimer (neutral);
# acetate (-1), methylammonium (+1), the methyl radical (doublet), O2
# (triplet) and H2O+ (doublet). Each image gets a single point and a 400-step
# NVE run; each molecule is also run alone with the ordinary style.
# compare_turbo.py checks energies and forces per image (the single point
# decides PASS; the NVE numbers are reported).
set -Eeuo pipefail
: "${SO3LR_LMP:?set SO3LR_LMP to the lmp binary}"
: "${SO3LR_RELEASE_ROOT:?set SO3LR_RELEASE_ROOT to the release tree}"
R=$SO3LR_RELEASE_ROOT
T=$R/tests/turbo
OUT=${OUT:-$PWD/turbo-validate-${SLURM_JOB_ID}}
SO3LR_MODEL_DIR=${SO3LR_MODEL_DIR:-$R/models}
MODELS=${MODELS:-"so3lr"}
RUN_STEPS=${RUN_STEPS:-400}

export MODULEPATH=/apps/USE/system/modules:/apps/USE/easybuild/release/2025.1/modules/all
source /usr/share/lmod/lmod/init/bash
module purge
module load GCCcore/14.2.0 CUDA/12.8.0 NVHPC/25.3-CUDA-12.8.0 OpenMPI/5.0.7-NVHPC-25.3-CUDA-12.8.0 \
            Python/3.13.1-GCCcore-14.2.0 SciPy-bundle/2025.06-gfbf-2025a
export OMP_NUM_THREADS=1 OMP_PROC_BIND=false
export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,tcp OMPI_MCA_mtl=^ofi
export UCX_TLS=tcp,self UCX_IB_GPU_DIRECT_RDMA=no UCX_MEMTYPE_CACHE=n
LIBDIR=$(dirname "$(dirname "$SO3LR_LMP")")
export LD_LIBRARY_PATH="$LIBDIR/lib64:$LIBDIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

M=$R/tests/jaxref/molecules
TAGS=(alanine water ethanol trimer alanine_dup alanine_cation acetate methylammonium methyl o2 h2o_cation)
DATA=($R/examples/alanine/alanine.data $T/molecules/water_molecule.data
      $T/molecules/ethanol.data $T/molecules/water_trimer.data
      $R/examples/alanine/alanine.data $R/examples/alanine/alanine.data
      $M/acetate.data $M/methylammonium.data $M/methyl.data $M/o2.data $M/h2o_cation.data)
ELEM=("C H O N" "H O" "C H O" "H O" "C H O N" "C H O N" "C H O" "C H N" "C H" "O" "H O")
CHG=(0 0 0 0 0 1 -1 1 0 0 1)
MLT=(1 1 1 1 1 2 1 1 2 3 2)

mkdir -p "$OUT"
{
  echo "variable TAG world ${TAGS[*]}"
  echo "variable DATA_FILE world ${DATA[*]}"
  printf 'variable ELEMENTS world'; printf ' "%s"' "${ELEM[@]}"; echo
  echo "variable CHARGE world ${CHG[*]}"
  echo "variable MULT world ${MLT[*]}"
  echo "include $T/in.common"
} > "$OUT/in.turbo"

for m in $MODELS; do
  model=$SO3LR_MODEL_DIR/$m.so3lr
  mkdir -p "$OUT/$m/ordinary" "$OUT/$m/turbo"
  echo "=== $m: ordinary, one molecule at a time ==="
  for i in "${!TAGS[@]}"; do
    [[ ${TAGS[$i]} == *_dup ]] && continue
    srun --ntasks=1 --exact --gpus-per-task=1 "$SO3LR_LMP" -screen none \
      -var PAIR so3lr/native/mpi -var MODEL_FILE "$model" -var DATA_FILE "${DATA[$i]}" \
      -var ELEMENTS "${ELEM[$i]}" -var TAG "${TAGS[$i]}" -var OUT "$OUT/$m/ordinary" \
      -var CHARGE "${CHG[$i]}" -var MULT "${MLT[$i]}" \
      -var RUN_STEPS "$RUN_STEPS" -in "$T/in.common" \
      && echo "  ${TAGS[$i]}: ok" || echo "  ${TAGS[$i]}: FAILED"
  done
  echo "=== $m: turbo, ${#TAGS[@]} images in one pack ==="
  srun --ntasks=${#TAGS[@]} --exact --gpus-per-node=1 --gpu-bind=none "$SO3LR_LMP" \
    -partition ${#TAGS[@]}x1 -plog none -pscreen none \
    -var PAIR so3lr/turbo -var MODEL_FILE "$model" -var OUT "$OUT/$m/turbo" \
    -var RUN_STEPS "$RUN_STEPS" -in "$OUT/in.turbo" > "$OUT/$m/turbo/universe.stdout" 2>&1 \
    && echo "  turbo: ok" || { echo "  turbo: FAILED"; tail -5 "$OUT/$m/turbo/universe.stdout"; }
done
python3 "$T/compare_turbo.py" "$OUT" --json "$OUT/turbo_vs_ordinary.json" || true
echo "SO3LR_TURBO_VALIDATE=DONE out=$OUT"
