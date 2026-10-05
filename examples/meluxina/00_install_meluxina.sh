#!/bin/bash -l
# Interactive driver for the MeluXina built-in installation (Steps 2 to 4).
#
# Run it on a MeluXina login node from anywhere inside the unpacked release:
#
#   examples/meluxina/00_install_meluxina.sh
#
# It detects the release root, the Slurm account, and a project directory with
# enough quota, shows them for confirmation, and submits three chained jobs:
#   02_prepare_lammps_so3lr_source.sh  fetch LAMMPS, add SO3LR   (CPU, minutes)
#   03_build_lammps_so3lr.sh           compile and install        (CPU, ~30 min)
#   04_verify_installation.sh          check it on a GPU          (GPU, minutes)
# Every detected value can be overridden:
#
#   --account pXXXXXX        Slurm account to charge (default: detected)
#   --install-root PATH      where LAMMPS is cloned, built, installed
#   --lammps-source PATH     existing/target LAMMPS checkout
#   --kokkos-arch NAME       Kokkos arch, e.g. AMPERE80 (default: AMPERE80, MeluXina A100)
#   --cuda-arch NN           CUDA architecture number, e.g. 80
#   --model PATH             SO3LR .so3lr file for Step 4's numerical checks
#                            (default: models/so3lr.so3lr of this release)
#   --yes                    accept the detected values without prompting
#   --dry-run                print the sbatch commands and exit
#   --step2-only             submit source preparation only
#   --step3-only             submit the build only (Step 2 already done)
#   --no-verify              do not submit Step 4
#   --verify-only            submit Step 4 only (installation already built)
set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RELEASE_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
RELEASE_VERSION=$(cat "${RELEASE_ROOT}/VERSION" 2>/dev/null || echo unknown)

ACCOUNT=${SO3LR_ACCOUNT:-}
INSTALL_ROOT=${SO3LR_INSTALL_ROOT:-}
LAMMPS_SOURCE=${LAMMPS_SOURCE:-}
KOKKOS_ARCH=${SO3LR_KOKKOS_ARCH:-}
CUDA_ARCH=${SO3LR_CUDA_ARCH:-}
VERIFY_MODEL=${SO3LR_VERIFY_MODEL:-}
ASSUME_YES=0
DRY_RUN=0
DO_STEP2=1
DO_STEP3=1
DO_STEP4=1

fail() { echo "ERROR: $*" >&2; exit 1; }

# Absolute, but without resolving symlinks: on MeluXina /project/scratch/<acct>
# points into /mnt/tier1, and the readable form is what users recognise.
abspath() { realpath -m --no-symlinks "$1" 2>/dev/null || readlink -m "$1"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --account)       ACCOUNT=$2; shift 2 ;;
    --install-root)  INSTALL_ROOT=$2; shift 2 ;;
    --lammps-source) LAMMPS_SOURCE=$2; shift 2 ;;
    --kokkos-arch)   KOKKOS_ARCH=$2; shift 2 ;;
    --cuda-arch)     CUDA_ARCH=$2; shift 2 ;;
    --model)         VERIFY_MODEL=$2; shift 2 ;;
    --yes|-y)        ASSUME_YES=1; shift ;;
    --dry-run)       DRY_RUN=1; shift ;;
    --step2-only)    DO_STEP3=0; DO_STEP4=0; shift ;;
    --step3-only)    DO_STEP2=0; DO_STEP4=0; shift ;;
    --no-verify)     DO_STEP4=0; shift ;;
    --verify-only)   DO_STEP2=0; DO_STEP3=0; shift ;;
    -h|--help)       sed -n '2,34p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *)               fail "unknown argument: $1" ;;
  esac
done

[[ -f "${RELEASE_ROOT}/tools/install_into_lammps_source.sh" ]] || \
  fail "not inside a SO3LR release tree: ${RELEASE_ROOT}"
command -v sbatch >/dev/null || fail "sbatch not found; run this on a MeluXina login node"

# --- Slurm account -----------------------------------------------------------
# MeluXina charges every job to a project account. Keep only accounts that can
# actually run work (the 'nocredit' association cannot).
mapfile -t ACCOUNTS < <(
  sacctmgr -nP show assoc user="${USER}" format=Account 2>/dev/null |
    sort -u | grep -Ev '^(nocredit|root|)$'
)

if [[ -z "${ACCOUNT}" ]]; then
  case ${#ACCOUNTS[@]} in
    0) fail "no Slurm account detected; pass --account pXXXXXX" ;;
    1) ACCOUNT=${ACCOUNTS[0]} ;;
    *)
      if [[ ${ASSUME_YES} -eq 1 || ! -t 0 ]]; then
        fail "several accounts available (${ACCOUNTS[*]}); pass --account"
      fi
      echo "Slurm accounts available to ${USER}:"
      select choice in "${ACCOUNTS[@]}"; do
        [[ -n "${choice:-}" ]] && { ACCOUNT=${choice}; break; }
      done
      ;;
  esac
fi
printf '%s\n' "${ACCOUNTS[@]}" | grep -qx "${ACCOUNT}" || \
  echo "WARNING: ${ACCOUNT} is not in your association list (${ACCOUNTS[*]:-none})"

# --- project directory -------------------------------------------------------
# The MeluXina login name (uXXXXXX) is not always the personal directory name,
# so look for a directory owned by this user instead of assuming $USER.
personal_dir() {
  local base=$1 candidate
  [[ -d "${base}" ]] || return 1
  if [[ -d "${base}/${USER}" && -w "${base}/${USER}" ]]; then
    echo "${base}/${USER}"; return 0
  fi
  mapfile -t candidate < <(find "${base}" -mindepth 1 -maxdepth 1 -type d \
    -user "${USER}" 2>/dev/null | sort)
  [[ ${#candidate[@]} -eq 1 ]] || return 1
  [[ -w "${candidate[0]}" ]] || return 1
  echo "${candidate[0]}"
}

# Percentage of the GiB allocation already used on a datapath, via myquota.
quota_used_percent() {
  local datapath=$1
  command -v myquota >/dev/null || return 1
  myquota 2>/dev/null | sed -e 's/\x1b\[[0-9;]*m//g' |
    awk -v path="${datapath}" '$1 == path { for (i = 1; i <= NF; i++)
      if ($i ~ /^[0-9]+%$/) { gsub("%", "", $i); print $i; exit } }'
}

if [[ -z "${INSTALL_ROOT}" ]]; then
  # A LAMMPS source+build tree is large and file-heavy: prefer scratch, and
  # skip any filesystem that is already close to its quota.
  for base in "/project/scratch/${ACCOUNT}" "/project/home/${ACCOUNT}"; do
    home_dir=$(personal_dir "${base}") || continue
    used=$(quota_used_percent "${base}" || true)
    if [[ -n "${used}" && ${used} -ge 90 ]]; then
      echo "skipping ${base} (quota ${used}% used)"
      continue
    fi
    INSTALL_ROOT="${home_dir}/software/so3lr-lammps-${RELEASE_VERSION}"
    break
  done
fi
[[ -n "${INSTALL_ROOT}" ]] || \
  fail "no writable project directory with free quota found; pass --install-root"
INSTALL_ROOT=$(abspath "${INSTALL_ROOT}")
LAMMPS_SOURCE=${LAMMPS_SOURCE:-${INSTALL_ROOT}/lammps-11Feb2026}
LAMMPS_SOURCE=$(abspath "${LAMMPS_SOURCE}")

# --- GPU architecture --------------------------------------------------------
# MeluXina GPU nodes are A100 (AMPERE80), Step 3's default. Step 3 compiles on
# a CPU node, so the architecture cannot be detected there; pass --kokkos-arch
# for other GPUs.
if [[ -n "${KOKKOS_ARCH}" && -z "${CUDA_ARCH}" ]]; then
  CUDA_ARCH=$(sed -E 's/^[A-Z]+//' <<<"${KOKKOS_ARCH}")
fi

# --- confirmation ------------------------------------------------------------
cat <<EOF

SO3LR ${RELEASE_VERSION} - MeluXina built-in installation
  release root   : ${RELEASE_ROOT}
  slurm account  : ${ACCOUNT}
  install root   : ${INSTALL_ROOT}
  lammps source  : ${LAMMPS_SOURCE}
  build dir      : ${INSTALL_ROOT}/build
  install prefix : ${INSTALL_ROOT}/install
  kokkos arch    : ${KOKKOS_ARCH:-AMPERE80 (MeluXina A100)}
  verify model   : ${VERIFY_MODEL:-${RELEASE_ROOT}/models/so3lr.so3lr}
  steps          : $( ((DO_STEP2)) && printf '2 ' )$( ((DO_STEP3)) && printf '3 ' )$( ((DO_STEP4)) && printf '4' )

EOF

if [[ ${ASSUME_YES} -eq 0 && ${DRY_RUN} -eq 0 ]]; then
  [[ -t 0 ]] || fail "not interactive; rerun with --yes"
  read -r -p "Submit these jobs? [y/N] " reply
  [[ "${reply}" =~ ^[Yy] ]] || { echo "aborted"; exit 1; }
fi

EXPORTS="ALL,SO3LR_RELEASE_ROOT=${RELEASE_ROOT},SO3LR_INSTALL_ROOT=${INSTALL_ROOT}"
EXPORTS+=",LAMMPS_SOURCE=${LAMMPS_SOURCE}"
[[ -n "${KOKKOS_ARCH}" ]] && EXPORTS+=",SO3LR_KOKKOS_ARCH=${KOKKOS_ARCH}"
[[ -n "${CUDA_ARCH}" ]] && EXPORTS+=",SO3LR_CUDA_ARCH=${CUDA_ARCH}"
if [[ -n "${VERIFY_MODEL}" ]]; then
  VERIFY_MODEL=$(abspath "${VERIFY_MODEL}")
  [[ -f "${VERIFY_MODEL}" ]] || fail "--model file not found: ${VERIFY_MODEL}"
  EXPORTS+=",SO3LR_VERIFY_MODEL=${VERIFY_MODEL}"
fi

submit() {
  local script=$1; shift
  local cmd=(sbatch --parsable --account="${ACCOUNT}" --export="${EXPORTS}" "$@"
             "${SCRIPT_DIR}/${script}")
  if [[ ${DRY_RUN} -eq 1 ]]; then
    printf '%q ' "${cmd[@]}"; echo; return 0
  fi
  "${cmd[@]}"
}

mkdir -p "${INSTALL_ROOT}"
# Each step waits for the previous one; a failed step cancels the rest.
PREVIOUS=""
chain() {
  local script=$1 label=$2 job
  if [[ -n "${PREVIOUS}" && ${DRY_RUN} -eq 0 ]]; then
    job=$(submit "${script}" --dependency="afterok:${PREVIOUS}" --kill-on-invalid-dep=yes)
  else
    job=$(submit "${script}")
  fi
  echo "${label} job: ${job}"
  PREVIOUS=${job}
}
[[ ${DO_STEP2} -eq 1 ]] && chain 02_prepare_lammps_so3lr_source.sh "step 2 (source preparation)"
[[ ${DO_STEP3} -eq 1 ]] && chain 03_build_lammps_so3lr.sh "step 3 (build, CPU node)"
[[ ${DO_STEP4} -eq 1 ]] && chain 04_verify_installation.sh "step 4 (verification, GPU node)"

[[ ${DRY_RUN} -eq 1 ]] && exit 0

cat <<EOF

Watch progress with:   squeue --me
Job output lands in:   ${PWD}  (so3lr-source-*, so3lr-build-*, so3lr-verify-*)
Success looks like:    SO3LR_MELUXINA_VERIFY=PASS at the end of so3lr-verify-*.out
Then, in a job:        source ${INSTALL_ROOT}/use-lammps-so3lr.env
Installed executable:  ${INSTALL_ROOT}/install/bin/lmp
Model:                 ${RELEASE_ROOT}/models/so3lr.so3lr
EOF
