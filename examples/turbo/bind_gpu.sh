#!/bin/bash
# Bind one concurrent Slurm step (one turbo broker) to its allocated GPU.
set -uo pipefail
assignment_file=$1
shift
step_gpu=${SLURM_STEP_GPUS:-}
if [[ ! "${step_gpu}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: expected one numeric SLURM_STEP_GPUS, got ${step_gpu:-unset}" >&2
  exit 91
fi
export CUDA_VISIBLE_DEVICES="${step_gpu}"
if [[ "${SLURM_PROCID:-0}" -eq 0 ]]; then
  printf 'slurm_step_gpus=%s\ncuda_visible_devices=%s\n' "${step_gpu}" "${CUDA_VISIBLE_DEVICES}" > "${assignment_file}"
fi
"$@"
status=$?
if [[ "${SLURM_PROCID:-0}" -eq 0 ]]; then
  printf 'exit_code=%s\n' "${status}" >> "${assignment_file}"
fi
exit "${status}"
