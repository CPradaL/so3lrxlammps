# Compatibility matrix

| Component | Ordinary style status |
|---|---|
| LAMMPS `patch_11Feb2026` | validated host version |
| Later LAMMPS releases | unsupported until rebuilt and validated |
| NVIDIA A100 / Kokkos AMPERE80 | validated development target |
| Bundled Kokkos 5.0.2 in LAMMPS 11Feb2026 | clean build validated |
| Other NVIDIA architectures | source-port path exists; not yet validated |
| AMD/SYCL GPUs | not implemented |
| MPI, one rank per GPU | validated for 1, 2, and 4 GPUs on one node and 8 GPUs on two nodes |
| Pair runtime | internal Kokkos/CUDA; LAMMPS global Kokkos activation unsupported |
| Neighbor/Newton mode | `newton on`; full SR and half LR lists requested internally |
| Multi-node execution | validated on 2 MeluXina nodes with conservative host MPI transport |
| Element mapping Z=1--99 | supported by symbols or atomic numbers |
| Total charge, spin multiplicity | `pair_style so3lr/native/mpi MODEL Q M`; uses the model's charge and spin embeddings |
| `pair_style hybrid` with `NULL` | not implemented |
| Python/JAX/PyTorch during MD | not required |
| Original checkpoint conversion | requires Python, PyTorch, and `so3krates_torch` |

Loading a model checks its architecture against the runtime's
declared capabilities (`so3lr_capabilities.cpp`) and refuses anything outside
them; a checkpoint that fails this check must not be forced into the runtime
format.

Compiling LAMMPS with `PKG_KOKKOS=ON` is required because SO3LR uses the
bundled Kokkos implementation internally. This build requirement does not mean
that LAMMPS global Kokkos mode should be enabled at runtime. In this release,
passing global Kokkos launch or input switches changes LAMMPS dispatch outside
the validated SO3LR adapter contract and is unsupported.

## Turbo mode (`so3lr/turbo`)

| Component | 0.4.0 status |
|---|---|
| `so3lr/turbo`, built into LAMMPS | supported; validated against `so3lr/native/mpi` (VALIDATION.md) |
| Models | every model the ordinary style supports (same pipeline) |
| External `so3lr/turbo` plugin | unsupported (separate Kokkos runtime crashed in development) |
| Images | nonperiodic, one rank per independent image, two or more per GPU |
| GPUs | one and four A100s (examples/turbo) |
| Ensembles | NVE validated; NVT should be checked per setup; NPT excluded |
| Global/per-atom virial | not computed by turbo; pressure and per-atom stress excluded |
| Charge and spin multiplicity | per image (`pair_style so3lr/turbo MODEL Q M`) |
| Periodic images | excluded |
