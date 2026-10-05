# Examples

All inputs take file paths through LAMMPS `-var` arguments. Run them on a GPU
node with one MPI rank per GPU. Do not add LAMMPS's own Kokkos switches
(`-k on`, `-sf kk`, `-pk kokkos`): the pair styles run their own Kokkos/CUDA
runtime. On MeluXina, `source <install root>/use-lammps-so3lr.env` first.

## First runs

Periodic water, 3,000 atoms (single point plus a short run):

```bash
mpirun -np 1 lmp \
  -var DATA_FILE "$PWD/water/water_n10_3000atoms_rho1.data" \
  -var MODEL_FILE ../models/so3lr.so3lr \
  -in water/in.so3lr_builtin_smoke
```

An isolated alanine peptide (162 atoms, elements C H O N), 1,000 NVE steps:

```bash
mpirun -np 1 lmp \
  -var DATA_FILE "$PWD/alanine/alanine.data" \
  -var MODEL_FILE ../models/so3lr.so3lr \
  -in alanine/in.so3lr_builtin_nve
```

For several GPUs, launch more ranks and bind one GPU to each, for example
`srun --ntasks=4 --gpus-per-task=1 --gpu-bind=single:1 lmp ...`.

The `*_plugin_*` and `*_rc_smoke` inputs are the same runs for the external
plugin build. They also need `-var PLUGIN_FILE /path/to/so3lrnativeplugin.so`.

Every input sets `comm_modify cutoff 13.0`. The pair style requests a 12 Å
long-range neighbour list, which LAMMPS refuses (`Custom neighbor list cutoff
too large for communication cutoff`) unless the communication cutoff is at
least that large.

To run a charged or open-shell system, give the total charge and spin
multiplicity after the model: `pair_style so3lr/native/mpi MODEL -1 2`.

## Directories

| directory | content |
|---|---|
| `water/`, `alanine/` | the first runs above |
| `turbo/` | `so3lr/turbo`: independent molecules packed on 1 or 4 GPUs (MeluXina job scripts) |
| `scaling/` | strong- and weak-scaling benchmark, and per-model GPU capacity |
| `meluxina/` | installation jobs and checkpoint conversion on MeluXina |

These are software checks and starting points, not production protocols. A
short alanine NVE run, for example, says nothing about the model's accuracy
for peptides.
