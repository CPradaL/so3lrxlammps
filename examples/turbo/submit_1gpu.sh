#!/bin/bash -l
#SBATCH --job-name=so3lr-turbo-1g
#SBATCH --partition=gpu
#SBATCH --qos=test
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=so3lr-turbo-1g-%j.out
#SBATCH --error=so3lr-turbo-1g-%j.err
set -Eeuo pipefail
export SO3LR_TURBO_GROUPS=1
# Slurm runs a spooled copy of this script, so locate the original through
# the job record rather than BASH_SOURCE.
script=$(scontrol show job "${SLURM_JOB_ID}" | sed -n 's/^ *Command=\([^ ]*\).*/\1/p')
exec "$(dirname "${script}")/run_meluxina.sh"
