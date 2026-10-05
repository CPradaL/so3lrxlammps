# so3lr-lammps 0.4.0

Native GPU implementation of the **SO3LR** machine-learned force field for
[LAMMPS](https://www.lammps.org). Forces come from hand-written C++/CUDA/Kokkos
kernels: no Python, JAX or PyTorch runs during a simulation.

Two pair styles are provided:

| pair style | for | parallelism |
|---|---|---|
| `so3lr/native/mpi` | one system of any size: molecules, liquids, membranes, materials | spatial decomposition, one MPI rank per GPU, multi-node |
| `so3lr/turbo` | many small, independent molecules at once (conformer or replica sampling) | one LAMMPS partition per molecule, many molecules packed on one GPU |

## Features

- **SO3LR**, read from a native `.so3lr` model file. Widths, depth and
  degree layout come from the model's architecture record, not the code.
- **Total charge and spin multiplicity**, through SO3LR's trained charge and
  spin embeddings, for both pair styles.
- **Full SO3LR physics**: learned short-range energy, repulsion, and
  long-range electrostatics and dispersion from the predicted charges and
  ratios.
- **Multi-GPU, multi-node** MD with periodic or non-periodic boundaries, NVE,
  NVT and NPT (global virial).
- **Validated against the JAX reference** in float64: forces agree to
  ~1e-10 eV/Å when the reference runs fully in float64, and 1 GPU and 4 GPUs
  give bit-identical energies ([VALIDATION.md](VALIDATION.md)).

## Requirements

- LAMMPS `patch_11Feb2026` (commit `b1ef9f45`), the validated host version;
- an NVIDIA GPU with CUDA and cuBLAS (validated on A100, Kokkos `AMPERE80`);
- MPI, CMake ≥ 3.16, a C++20 compiler supported by LAMMPS's bundled Kokkos;
- LAMMPS built with `PKG_KOKKOS=ON` and `PKG_SO3LR=ON`. The pair styles use
  Kokkos internally; LAMMPS's own Kokkos mode is **not** switched on at run
  time (no `-k on`, `-sf kk`, `-pk kokkos`).

## Installation

### On MeluXina

From a login node, inside this directory:

```bash
examples/meluxina/00_install_meluxina.sh
```

It detects your Slurm account and a project directory, asks for confirmation,
and submits three chained jobs:
1. fetch LAMMPS and add SO3LR to it;
2. compile on a CPU node;
3. check the result on a GPU, including single points against the JAX
   reference and a short turbo run.

When they finish:

```bash
source <install root>/use-lammps-so3lr.env     # modules, PATH, LD_LIBRARY_PATH
lmp -h | grep so3lr
```

Options and manual steps: [examples/meluxina/README.md](examples/meluxina/README.md).

### Anywhere else

```bash
git clone --branch patch_11Feb2026 --depth 1 https://github.com/lammps/lammps.git lammps-11Feb2026
tools/install_into_lammps_source.sh lammps-11Feb2026
cmake -S lammps-11Feb2026/cmake -B build \
  -D CMAKE_BUILD_TYPE=Release -D BUILD_MPI=ON -D BUILD_SHARED_LIBS=ON \
  -D PKG_KOKKOS=ON -D PKG_SO3LR=ON \
  -D Kokkos_ENABLE_CUDA=ON -D Kokkos_ENABLE_SERIAL=ON -D Kokkos_ARCH_AMPERE80=ON \
  -D CMAKE_CXX_COMPILER=$PWD/lammps-11Feb2026/lib/kokkos/bin/nvcc_wrapper
cmake --build build --parallel 16 && cmake --install build
```

Set `Kokkos_ARCH_*` for your GPU. Compiler, MPI and CUDA details are in
[docs/INSTALL_BUILTIN.md](docs/INSTALL_BUILTIN.md). `so3lr/native/mpi` can
also be built as a LAMMPS plugin without rebuilding LAMMPS; turbo cannot. See
[docs/INSTALL_PLUGIN.md](docs/INSTALL_PLUGIN.md).

## Usage

```lammps
units           metal
atom_style      atomic
newton          on
read_data       system.data
pair_style      so3lr/native/mpi models/so3lr.so3lr
pair_coeff      * * C H O N            # element of each atom type, in type order
neighbor        1.0 bin
neigh_modify    every 1 delay 0 check yes
comm_modify     cutoff 13.0            # required: the long-range list is 12 Å
```

Launch with one MPI rank per GPU:

```bash
mpirun -np 1 lmp -in in.lammps                                              # 1 GPU
srun --ntasks=8 --ntasks-per-node=4 --gpus-per-task=1 --gpu-bind=single:1 lmp -in in.lammps   # 2 nodes
```

Element symbols are case-insensitive (H to Es, Z = 1–99); atomic numbers are
also accepted.

### Charge and spin

Two optional numbers after the model give the system's total charge and spin
multiplicity. The defaults are 0 and 1, a neutral singlet:

```lammps
pair_style      so3lr/native/mpi models/so3lr.so3lr -1 2    # anion, doublet
```

The run stops if either value is not an integer, or if the multiplicity does
not fit the electron count (for example, an odd number of electrons with
multiplicity 1). A net charge in a periodic cell gives a warning: SO3LR's
long-range Coulomb is truncated in real space, so the energy is defined, but
there is no neutralizing background.

### Many molecules on one GPU: `so3lr/turbo`

Each molecule is its own LAMMPS partition (one MPI rank) with its own data,
fixes and output. The style batches all partitions on a GPU into one
evaluation per step. Inputs use world variables:

```lammps
variable        DATA world mol_a.data mol_b.data mol_c.data mol_d.data
variable        Q    world 0 -1 1 0
variable        M    world 1 1 1 2
boundary        f f f
read_data       ${DATA}
pair_style      so3lr/turbo models/so3lr.so3lr ${Q} ${M}
pair_coeff      * * C H O N
```

```bash
mpirun -np 4 lmp -partition 4x1 -in in.turbo
```

Turbo requires non-periodic boundaries, at least two partitions per GPU, and
the same number of force calls in every partition (no NPT). See
[docs/TURBO.md](docs/TURBO.md) and [examples/turbo](examples/turbo).

## Models

The SO3LR model is included: `models/so3lr.so3lr` (4.5 Å short-range and
12 Å long-range cutoffs, elements Z = 1–99). One A100-40GB comfortably holds
about 24,000 atoms; one node of four holds about 96,000. Memory, not GPU
occupancy, limits the load per GPU: throughput per GPU is flat from ~3,000
atoms up to the memory limit.

Other or retrained SO3LR checkpoints are converted once with the tools in
[tools/checkpoint_conversion](tools/checkpoint_conversion)
([docs/MODEL_CONVERSION.md](docs/MODEL_CONVERSION.md)). Loading a model checks
its architecture against what this build supports, and refuses anything
unsupported with a list of the offending features.

## Performance

On MeluXina A100-40GB with SO3LR:
- **1 GPU:** about 70k atom-steps/s.
- **24k atoms:** 253k atom-steps/s on 4 GPUs, 91% strong-scaling efficiency.
- **192k atoms:** 1.47M atom-steps/s on 32 GPUs.
- **Strong scaling:** stays useful down to ~3,000 atoms per GPU.

Benchmark scripts and the capacity table are in [examples/scaling](examples/scaling).

## Examples and tests

| directory | content |
|---|---|
| `examples/water`, `examples/alanine` | first runs: periodic water, an isolated peptide |
| `examples/turbo` | turbo on 1 and 4 GPUs (MeluXina job scripts) |
| `examples/scaling` | strong/weak scaling benchmark |
| `examples/meluxina` | installation and checkpoint conversion jobs |
| `tests/jaxref` | single points against the JAX reference (references included) |
| `tests/turbo` | turbo against `so3lr/native/mpi` on neutral, charged and open-shell molecules |

`tools/validate_release.sh` runs the static checks (no GPU needed).

## Documentation

- [docs/INSTALL_BUILTIN.md](docs/INSTALL_BUILTIN.md): building into LAMMPS
- [docs/INSTALL_PLUGIN.md](docs/INSTALL_PLUGIN.md): plugin build
- [docs/TURBO.md](docs/TURBO.md): turbo mode
- [docs/MODEL_CONVERSION.md](docs/MODEL_CONVERSION.md): checkpoints to `.so3lr`
- [docs/COMPATIBILITY.md](docs/COMPATIBILITY.md): what is supported and validated
- [VALIDATION.md](VALIDATION.md): validation record
- [CHANGELOG.md](CHANGELOG.md), [PROVENANCE.md](PROVENANCE.md)

## Status

0.4.0 is a development release for the SO3LR collaboration. Its source
licence is pending ([LICENSE_STATUS.md](LICENSE_STATUS.md)).
