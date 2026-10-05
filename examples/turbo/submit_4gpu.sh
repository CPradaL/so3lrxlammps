#!/bin/bash -l
#SBATCH --job-name=so3lr-turbo-4g
#SBATCH --partition=gpu
#SBATCH --qos=test
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=so3lr-turbo-4g-%j.out
#SBATCH --error=so3lr-turbo-4g-%j.err
set -Eeuo pipefail
export SO3LR_TURBO_GROUPS=4
# Slurm runs a spooled copy of this script, so locate the original through
# the job record rather than BASH_SOURCE.
script=$(scontrol show job "${SLURM_JOB_ID}" | sed -n 's/^ *Command=\([^ ]*\).*/\1/p')
exec "$(dirname "${script}")/run_meluxina.sh"
