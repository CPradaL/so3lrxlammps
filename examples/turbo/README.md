# Turbo example: independent molecules

This is a small, nonperiodic NVE smoke test, not a water property simulation.
It packs four three-atom water molecules per GPU as *four independent LAMMPS
images*, not as one 12-atom system. The default run length is 1,000 steps at
0.1 fs with a distinct velocity seed per image.

First build this release *into* a fresh LAMMPS checkout with
`PKG_SO3LR=ON`; see `../../docs/TURBO.md`. The old ordinary-style-only
executable will not recognize `so3lr/turbo`.

Set the installation root (the directory that contains `install/bin/lmp`,
as created by `examples/meluxina/00_install_meluxina.sh`), then submit either
wrapper:

```bash
export SO3LR_INSTALL_ROOT=/project/scratch/PROJECT/USER/software/so3lr-lammps-0.4.0
sbatch --account=PROJECT submit_1gpu.sh
sbatch --account=PROJECT submit_4gpu.sh
```

The model is `models/so3lr.so3lr` unless `SO3LR_MODEL` names another file.

Set `RUN_STEPS=2000` before submission to override the default. To change
`IMAGES_PER_GPU`, also request that many Slurm tasks on one GPU, or four times
as many on four GPUs, for example `IMAGES_PER_GPU=8 sbatch --ntasks=32
--account=PROJECT submit_4gpu.sh` (2–32 images/GPU for this example).
`SO3LR_LMP_BIN`, `SO3LR_MODEL`,
`SO3LR_DATA`, and `SO3LR_RESULT_DIR` are optional absolute-path overrides.
Each GPU has its own independent broker group. Each image has its own log and
trajectory under `group_XX/image_XXXX/`; images never share a thermostat,
velocity seed, or evolving atomic state. This example assumes MeluXina's
Slurm GPU-step semantics and the compiler/MPI/CUDA module stack used by the
validated ordinary installation.

The last line of the Slurm output should read `SO3LR_TURBO_EXAMPLE=PASS`.
That means the run completed and every image wrote its final marker. For a
numerical comparison against `so3lr/native/mpi`, including charged and
open-shell molecules, use `tests/turbo/job_turbo_validate.sh`.
