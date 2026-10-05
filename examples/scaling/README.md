# Scaling benchmark

Strong- and weak-scaling sweeps of `pair_style so3lr/native/mpi` on water
boxes, as Slurm jobs (written for MeluXina, easy to adapt).

## Running

```bash
# 1. water boxes: 12k atoms per GPU for weak scaling, fixed sizes for strong
python generate_water_boxes.py --output-dir data/ --skip-existing \
       --sizes 12000 24000 48000 96000 192000 384000 768000

# 2. submit the sweep for MODEL_DIR/<name>.so3lr (default ../../models/so3lr.so3lr)
ACCOUNT=pXXXXXX LMP=/path/to/install/bin/lmp MODELS="so3lr" bash submit_scaling.sh

# 3. tables and plots from the logs
python analyze_scaling.py --results-dir results/so3lr/ib --plots-dir plots/so3lr/ib
```

- **Job submission.** `submit_scaling.sh` submits one job per point and keeps
  a bounded number of its jobs in the queue (`SWEEP_MAX_QUEUED`, default 20).
  Rerun it to submit the points still missing.
- **Probe.** `PROBE=/path/to/so3lr_capability_probe` (built with the native
  library) makes it skip a model the build cannot run before anything is
  submitted.
- **Transport.** `TRANSPORT=ib` (default, Open MPI over InfiniBand) or `tcp`
  (the conservative host-TCP fallback). It is part of the results path.
- **Timing.** Each point runs `WARMUP` (50) steps that are discarded, then
  `NSTEPS` (200) timed NVE steps, three times with different velocity seeds.
  `analyze_scaling.py` reads the timed run.
- **Launch.** Inputs set `comm_modify cutoff 13.0` and are launched with one
  MPI rank per GPU, without LAMMPS's global Kokkos switches.

## Configurations

- **strong:** 24k, 96k and 192k atoms, up to 64 GPUs, starting from the
  smallest GPU count within the model's comfortable capacity (below);
- **weak:** 12,000 atoms per GPU, 1 to 64 GPUs.

For another model, set its per-GPU loads with `CAP_<name>=` and
`WEAK_<name>=` (dashes in the name become underscores).

## Capacity: atoms per GPU

One A100-40GB, peak device memory and throughput (atom-steps/s):

| model | 3k | 6k | 12k | 24k | 30k | 36k | comfortable (≤ 80% memory) |
|---|---|---|---|---|---|---|---|
| so3lr | 4.5 GB / 66k | 8.5 / 68k | 15.4 / 75k | 29.6 / 73k | 37.2 / 72k | OOM | ~24k |

One node with 4 GPUs holds 96k atoms (30.1 GB per GPU).
Throughput per GPU is flat from 3k atoms up to the memory limit, so memory,
not occupancy, bounds the per-GPU load. Most of the memory is per-edge float64
buffers kept for the backward pass, about 1.2 MB per atom.

## Reference results (SO3LR, MeluXina A100-40GB, InfiniBand)

| configuration | atom-steps/s |
|---|---|
| 24k atoms, 1 GPU | 69,706 |
| 24k atoms, 4 GPUs | 253,065 |
| 24k atoms, 16 GPUs | 536,166 |
| 192k atoms, 16 GPUs | 893,853 |
| 192k atoms, 32 GPUs | 1,471,400 |
| weak, 12k/GPU, 32 GPUs | 1,290,763 |

Strong-scaling efficiency:
- 24k atoms: 91% on 4 GPUs (one node), 72% on 8, 48% on 16;
- 192k atoms: 82% from 16 to 32 GPUs.

Strong scaling stays useful down to about 3,000 local atoms per GPU. Below
that, the model's halo exchanges dominate. They run inside the pair style once
per interaction block, forward and backward.

## Reading the timing breakdown

LAMMPS's `Comm` row covers only its own atom halo exchange. The pair style's
feature and adjoint exchanges (`MPI_Alltoallv`) happen inside the pair compute
and are booked as `Pair`, so a low `Comm(%)` does not mean low communication
cost. To split the pair time into compute and exchange, run with
`SO3_NATIVE_MPI_PROFILE=1` (sampled every `SO3_NATIVE_MPI_PROFILE_EVERY`
calls). It prints `SO3LR_NATIVE_MPI_SR_REVERSE_PROFILE` lines with per-block
`block<b>_ms` and `exchange<b>_ms`. Profiling adds synchronization, so leave it
off for timed runs.
