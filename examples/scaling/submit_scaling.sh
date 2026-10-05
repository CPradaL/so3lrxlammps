#!/bin/bash
# Submit strong- and weak-scaling jobs for pair_style so3lr/native/mpi.
#
# Parameterised over the model. Models
# the build cannot execute are skipped up front by so3lr_capability_probe,
# which is much cheaper than discovering it inside a 64-GPU job.
#
#   MODELS="so3lr" bash submit_scaling.sh        # models/so3lr.so3lr
#
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
LMP="${LMP:?set LMP to the lmp executable}"
PROBE="${PROBE:-}"                 # optional so3lr_capability_probe
# The probe is built on a compute node against GCCcore 14.2; the login node's
# system libstdc++ is too old to even load it. Point at the matching runtime.
PROBE_LIBS="${PROBE_LIBS:-/apps/USE/easybuild/release/2025.1/software/GCCcore/14.2.0/lib64}"
MODEL_DIR="${MODEL_DIR:-${HERE}/../../models}"
ACCOUNT="${ACCOUNT:?set ACCOUNT to your Slurm account, e.g. ACCOUNT=pXXXXXX}"
NSTEPS="${NSTEPS:-200}"
WARMUP="${WARMUP:-50}"
SEEDS=(12345 54321 98765)
REPS="${REPS:-3}"
MODELS="${MODELS:-so3lr}"
# Inter-node MPI transport. "ib" uses Open MPI's defaults (UCX over
# InfiniBand). "tcp" is the conservative host-TCP fallback that the 0.2.0-rc1
# two-node run was validated with (docs/INSTALL_BUILTIN.md). Multi-node timings
# are only meaningful if you know which one produced them, so the transport is
# part of the results path.
TRANSPORT="${TRANSPORT:-ib}"
# The account's submit limit (MaxSubmitJobs) is shared with every other job the
# user runs under it, so the sweep keeps a small footprint: it holds at most
# SWEEP_MAX_QUEUED of its own jobs (pending + running, job names so3lr_*) and
# never leaves fewer than HEADROOM account slots free. Re-running this script
# later submits only the points that are still missing, so it trickles in as
# earlier points finish instead of parking a long pending queue.
SUBMIT_LIMIT="${SUBMIT_LIMIT:-100}"
HEADROOM="${HEADROOM:-30}"
SWEEP_MAX_QUEUED="${SWEEP_MAX_QUEUED:-20}"
STOPPED=0
SUBMITTED=0
REMAINING=0
case "$TRANSPORT" in ib|tcp) ;; *) echo "TRANSPORT must be ib or tcp"; exit 1 ;; esac
# Repeat that also records the pair style's in-pair phase profile (compute vs
# exchange per block). Sampling every PROFILE_EVERY calls keeps the extra
# fences to a handful of steps. PROFILE_REP=0 disables it.
PROFILE_REP=${PROFILE_REP:-1}
PROFILE_EVERY=${PROFILE_EVERY:-50}

# NGPUS NODES NTASKS_PER_NODE GPUS_PER_NODE  (one rank per GPU throughout)
GPU_CONFIGS=(
  "1   1  1 1"
  "2   1  2 2"
  "4   1  4 4"
  "8   2  4 4"
  "16  4  4 4"
  "32  8  4 4"
  "64  16 4 4"
)

# Strong scaling: fixed system, more GPUs. Each series starts at the smallest
# GPU count whose per-GPU load stays under the model's comfortable capacity.
STRONG_SIZES=("24000" "96000" "192000")

# Comfortable atoms per A100-40GB (peak memory <= ~80% of the card), measured
# on one GPU with job_capacity.sh (see README, "Capacity"). Weak scaling uses
# the per-GPU load below. Override with CAP_<model>/WEAK_<model>, dashes as
# underscores (e.g. CAP_so3lr_l=6000).
declare -A CAP_PER_GPU=( [so3lr]=24000 )
declare -A WEAK_LOAD=(   [so3lr]=12000 )

cap_for()  { local v="CAP_${1//-/_}";  echo "${!v:-${CAP_PER_GPU[$1]:-12000}}"; }
weak_for() { local v="WEAK_${1//-/_}"; echo "${!v:-${WEAK_LOAD[$1]:-12000}}"; }

submit() {   # $1 label  $2 model_path  $3 data  $4 ngpus  $5 nodes  $6 ntasks  $7 gpn  $8 rep
  local label=$1 model=$2 data=$3 ngpus=$4 nodes=$5 ntasks=$6 gpn=$7 rep=$8
  local seed=${SEEDS[$((rep-1))]}
  local dir="${HERE}/results/$(basename "$model" .so3lr)/${TRANSPORT}/${label}/gpu_$(printf '%03d' "$ngpus")/rep_${rep}"
  mkdir -p "$dir"
  if [[ -s "$dir/jobid" ]]; then return; fi   # already submitted by an earlier run
  local qos time
  # 64 GPUs exceeds the short QOS cap (40 GPUs per job) and needs default.
  # Measured wall time is 1-3 min per point including startup; 15 min keeps a
  # 5x margin while letting backfill place the jobs.
  if (( ngpus <= 32 )); then qos=short; else qos=default; fi
  time=${JOB_TIME:-00:15:00}

  cat > "$dir/job.sh" <<SLURM
#!/bin/bash -l
#SBATCH --job-name=so3lr_${label}_${ngpus}g_r${rep}
#SBATCH --partition=gpu
#SBATCH --qos=${qos}
#SBATCH --account=${ACCOUNT}
#SBATCH --time=${time}
#SBATCH --nodes=${nodes}
#SBATCH --ntasks-per-node=${ntasks}
#SBATCH --gpus-per-node=${gpn}
#SBATCH --cpus-per-task=32
#SBATCH --output=${dir}/slurm-%j.out
#SBATCH --error=${dir}/slurm-%j.err
set -Eeuo pipefail
export MODULEPATH=/apps/USE/system/modules:/apps/USE/easybuild/release/2025.1/modules/all
source /usr/share/lmod/lmod/init/bash
module purge
module load GCCcore/14.2.0 CUDA/12.8.0 NVHPC/25.3-CUDA-12.8.0 \\
            OpenMPI/5.0.7-NVHPC-25.3-CUDA-12.8.0
export LD_LIBRARY_PATH="\$(dirname ${LMP})/../lib64:\$(dirname ${LMP})/../lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
if [[ "${TRANSPORT}" == tcp ]]; then
  export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,tcp OMPI_MCA_mtl=^ofi
  export UCX_TLS=tcp,self UCX_IB_GPU_DIRECT_RDMA=no UCX_MEMTYPE_CACHE=n
fi
echo "transport=${TRANSPORT} ngpus=${ngpus} nodes=${nodes}"
$( [[ "$rep" == "${PROFILE_REP}" ]] && echo "# In-pair phase profile on this repeat only; it fences only on sampled calls.
export SO3_NATIVE_MPI_PROFILE=1 SO3_NATIVE_MPI_PROFILE_EVERY=${PROFILE_EVERY}" )
cd "${dir}"
# --gpu-bind=single:1 gives each rank exactly one visible GPU, as in the
# validated recipe; without it every rank on a node can land on device 0.
srun --ntasks=${ngpus} --gpus-per-task=1 --gpu-bind=single:1 ${LMP} \\
  -var DATA_FILE ${data} -var MODEL_FILE ${model} \\
  -var NSTEPS ${NSTEPS} -var WARMUP ${WARMUP} -var SEED ${seed} \\
  -in ${HERE}/in.so3lr_scaling
SLURM
  if (( STOPPED )); then REMAINING=$((REMAINING + 1)); return; fi
  local queued own
  queued=$(squeue -h -u "$USER" -A "$ACCOUNT" 2>/dev/null | wc -l)
  own=$(squeue -h -u "$USER" -A "$ACCOUNT" -o "%j" 2>/dev/null | grep -c "^so3lr_" || true)
  if (( queued >= SUBMIT_LIMIT - HEADROOM || own >= SWEEP_MAX_QUEUED )); then
    echo "  account ${ACCOUNT}: ${queued} jobs queued, ${own} of them this sweep (sweep cap ${SWEEP_MAX_QUEUED}, keeping ${HEADROOM} of ${SUBMIT_LIMIT} free); stopping here"
    STOPPED=1; REMAINING=$((REMAINING + 1)); return
  fi
  local out id
  if ! out=$(sbatch "$dir/job.sh" 2>&1); then
    echo "  sbatch refused ${label} ${ngpus}gpu rep${rep}: ${out##*error: }"
    STOPPED=1; REMAINING=$((REMAINING + 1)); return
  fi
  id=$(awk '{print $4}' <<<"$out")
  echo "$id" > "$dir/jobid"
  SUBMITTED=$((SUBMITTED + 1))
  echo "  ${label} ${ngpus}gpu rep${rep} -> job ${id}"
}

for name in $MODELS; do
  model="${MODEL_DIR}/${name}.so3lr"
  if [[ ! -f "$model" ]]; then
    echo "SKIP ${name}: no .so3lr found under ${MODEL_DIR}"; continue
  fi
  # Refuse early and loudly rather than burning GPU hours on a model this
  # build has no kernels for. Two different failures must not be confused: the
  # probe REFUSING the model (it ran, and printed capability_probe=FAIL) versus
  # the probe being unable to RUN here (e.g. a missing libstdc++ symbol). The
  # second says nothing about the model, so it aborts instead of skipping.
  if [[ -n "$PROBE" ]]; then
    probe_out=$(LD_LIBRARY_PATH="${PROBE_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
                "$PROBE" "$model" 2>&1) && probe_rc=0 || probe_rc=$?
    if (( probe_rc != 0 )); then
      if grep -q "^capability_probe=FAIL" <<<"$probe_out"; then
        echo "SKIP ${name}: this build cannot execute it --"
        sed 's/^/    /' <<<"$probe_out" | head -20
        continue
      fi
      echo "ERROR: the capability probe could not run on this node (rc=${probe_rc}):"
      sed 's/^/    /' <<<"$probe_out" | head -5
      echo "Set PROBE_LIBS to the GCC runtime it was built against, or unset PROBE."
      exit 1
    fi
  fi
  echo "=== ${name} ==="

  cap=$(cap_for "$name"); weak_per_gpu=$(weak_for "$name")
  echo "  capacity ${cap} atoms/GPU; weak load ${weak_per_gpu} atoms/GPU"
  for natoms in "${STRONG_SIZES[@]}"; do
    data="${HERE}/data/water_${natoms}.data"
    [[ -f "$data" ]] || { echo "  missing $data (run generate_water_boxes.py)"; continue; }
    for config in "${GPU_CONFIGS[@]}"; do
      read -r n nodes ntasks gpn <<< "$config"
      (( natoms > cap * n )) && continue
      for rep in $(seq 1 "$REPS"); do
        submit "strong_$((natoms/1000))k" "$model" "$data" "$n" "$nodes" "$ntasks" "$gpn" "$rep"
      done
    done
  done

  for config in "${GPU_CONFIGS[@]}"; do
    read -r n nodes ntasks gpn <<< "$config"
    data="${HERE}/data/water_$((weak_per_gpu * n)).data"
    [[ -f "$data" ]] || { echo "  weak ${n}gpu: missing $(basename "$data")"; continue; }
    for rep in $(seq 1 "$REPS"); do
      submit "weak_$((weak_per_gpu / 1000))k" "$model" "$data" "$n" "$nodes" "$ntasks" "$gpn" "$rep"
    done
  done
done

echo
echo "submitted ${SUBMITTED} job(s) this run; ${REMAINING} still to submit."
if (( REMAINING > 0 )); then
  echo "Re-run the same command later -- already-submitted points are skipped."
fi
