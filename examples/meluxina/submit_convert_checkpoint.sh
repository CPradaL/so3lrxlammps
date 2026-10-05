#!/bin/bash -l
#SBATCH --job-name=so3lr-convert
#SBATCH --partition=cpu
#SBATCH --qos=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=so3lr-convert-%j.out
#SBATCH --error=so3lr-convert-%j.err

# Required submission variables:
#   CONTAINER, SO3KRATES_SOURCE, SOURCE_CHECKPOINT, OUTPUT_MODEL, SOURCE_COMMIT
# Example:
#   sbatch --account=pXXXXXX \
#     --export=ALL,CONTAINER=/path/runtime.sif,SO3KRATES_SOURCE=/path/So3krates-torch,SOURCE_CHECKPOINT=/path/model,OUTPUT_MODEL=/path/model.so3lr,SOURCE_COMMIT=<git-commit> \
#     submit_convert_checkpoint.sh
set -Eeuo pipefail
umask 0027

export MODULEPATH=/apps/USE/system/modules:/apps/USE/easybuild/release/2025.1/modules/all
source /usr/share/lmod/lmod/init/bash
module purge
module load GCCcore/14.2.0
module load Apptainer/1.4.2-GCCcore-14.2.0

: "${CONTAINER:?Export CONTAINER=/absolute/path/to/the/SO3LR/PyTorch/container}"
: "${SO3KRATES_SOURCE:?Export SO3KRATES_SOURCE=/absolute/path/to/So3krates-torch}"
: "${SOURCE_CHECKPOINT:?Export SOURCE_CHECKPOINT=/absolute/path/to/checkpoint}"
: "${OUTPUT_MODEL:?Export OUTPUT_MODEL=/absolute/path/to/output.so3lr}"
: "${SOURCE_COMMIT:?Export SOURCE_COMMIT=the_So3krates-torch_source_commit}"

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
RELEASE_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
CONVERTER_DIR="${RELEASE_ROOT}/tools/checkpoint_conversion"
SUMMARY=${SUMMARY:-${OUTPUT_MODEL}.conversion.json}
MANIFEST=${MANIFEST:-${OUTPUT_MODEL}.manifest.json}
CONTAINER_PYTHON=${CONTAINER_PYTHON:-/opt/venv/bin/python}

for path in "${CONTAINER}" "${SO3KRATES_SOURCE}/src/so3krates_torch" \
    "${SOURCE_CHECKPOINT}" "${CONVERTER_DIR}/convert_so3lr_checkpoint.py"; do
  [[ -e "${path}" ]] || { echo "ERROR: missing ${path}" >&2; exit 2; }
done
mkdir -p "$(dirname "${OUTPUT_MODEL}")" "$(dirname "${SUMMARY}")" \
  "$(dirname "${MANIFEST}")"

apptainer exec --cleanenv --no-home \
  --bind "${RELEASE_ROOT}:${RELEASE_ROOT}" \
  --bind "${SO3KRATES_SOURCE}:${SO3KRATES_SOURCE}" \
  --bind "$(dirname "${SOURCE_CHECKPOINT}"):$(dirname "${SOURCE_CHECKPOINT}")" \
  --bind "$(dirname "${OUTPUT_MODEL}"):$(dirname "${OUTPUT_MODEL}")" \
  --env PYTHONNOUSERSITE=1 \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env PYTHONPATH="${SO3KRATES_SOURCE}/src:${CONVERTER_DIR}" \
  "${CONTAINER}" "${CONTAINER_PYTHON}" \
    "${CONVERTER_DIR}/convert_so3lr_checkpoint.py" \
      --model "${SOURCE_CHECKPOINT}" \
      --output "${OUTPUT_MODEL}" \
      --source-commit "${SOURCE_COMMIT}" \
      --long-range 12.0 \
      --max-native-z 99 \
      --summary "${SUMMARY}"

apptainer exec --cleanenv --no-home \
  --bind "${RELEASE_ROOT}:${RELEASE_ROOT}" \
  --bind "$(dirname "${OUTPUT_MODEL}"):$(dirname "${OUTPUT_MODEL}")" \
  --env PYTHONNOUSERSITE=1 --env PYTHONDONTWRITEBYTECODE=1 \
  "${CONTAINER}" "${CONTAINER_PYTHON}" \
    "${CONVERTER_DIR}/so3lr_native_format.py" \
      "${OUTPUT_MODEL}" --manifest "${MANIFEST}"

sha256sum "${SOURCE_CHECKPOINT}" "${OUTPUT_MODEL}" \
  "${CONVERTER_DIR}/convert_so3lr_checkpoint.py" \
  "${CONVERTER_DIR}/so3lr_native_format.py" \
  > "${OUTPUT_MODEL}.provenance.sha256"
echo "SO3LR_MELUXINA_CHECKPOINT_CONVERSION=PASS"
