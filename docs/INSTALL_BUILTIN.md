# Built directly into LAMMPS

This is the recommended release installation. SO3LR is compiled into the same
LAMMPS library and executable as the other pair styles. There is no `plugin
load` command and no separately installed `.so` module.

The same build registers both `so3lr/native/mpi` and `so3lr/turbo` (see
[turbo usage](TURBO.md)). Use a fresh source/install path for each version.
On MeluXina, `examples/meluxina/00_install_meluxina.sh` runs all of the steps
below as jobs; see [its README](../examples/meluxina/README.md).

## 1. Prepare a clean host tree

Use the exact `patch_11Feb2026` source for the first validation:

```bash
git clone --branch patch_11Feb2026 --depth 1 \
  https://github.com/lammps/lammps.git lammps-11Feb2026
```

## 2. Install the SO3LR source package

```bash
/path/to/so3lr-lammps-0.4.0/tools/install_into_lammps_source.sh \
  /path/to/lammps-11Feb2026
```

The helper performs three transparent operations:

- copies the numerical headers and sources to `src/SO3LR`;
- copies `SO3LR.cmake` to `cmake/Modules/Packages`;
- applies the small versioned patch that adds the `PKG_SO3LR` CMake option.

It refuses a second installation into the same tree. Use a fresh LAMMPS source
tree instead of trying to merge two generated copies.

## 3. Build

Load one internally consistent compiler, CUDA, MPI, and CMake stack. When using
the Kokkos `nvcc_wrapper`, set its underlying compiler before configuration:

```bash
export KOKKOS_WRAPPER_CXX=$(command -v mpicxx)
export OMPI_CXX=$(command -v nvc++)

cmake -S /path/to/lammps-11Feb2026/cmake -B /path/to/build \
  -D CMAKE_INSTALL_PREFIX=/path/to/install \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_CXX_COMPILER=/path/to/lammps-11Feb2026/lib/kokkos/bin/nvcc_wrapper \
  -D MPI_CXX_COMPILER=$(command -v mpicxx) \
  -D BUILD_MPI=ON \
  -D BUILD_OMP=OFF \
  -D BUILD_SHARED_LIBS=ON \
  -D PKG_KOKKOS=ON \
  -D PKG_SO3LR=ON \
  -D Kokkos_ENABLE_CUDA=ON \
  -D Kokkos_ENABLE_SERIAL=ON \
  -D Kokkos_ARCH_AMPERE80=ON \
  -D CMAKE_CUDA_ARCHITECTURES=80

cmake --build /path/to/build --parallel 16
cmake --install /path/to/build
```

Other LAMMPS packages can be enabled in the same command. `PKG_PLUGIN` and
`PKG_ML-IAP` are not dependencies of the native SO3LR pair style.

## 4. Verify registration

On a GPU compute node:

```bash
/path/to/install/bin/lmp -h | grep -E 'so3lr/(native/mpi|turbo)'
```

`examples/meluxina/04_verify_installation.sh` shows a complete check: single
points against the bundled JAX references and turbo against the ordinary
style. It is written for MeluXina, but easy to adapt.

Then run the built-in water or alanine example. Do not pass `PLUGIN_FILE` and
do not use `plugin load`. The native MPI pair style requires `newton on` in the
input and requests its own full short-range and half long-range neighbor lists.
Although LAMMPS is compiled with `PKG_KOKKOS=ON`, the pair style owns the
Kokkos/CUDA runtime internally. Do not pass `-k`, `-sf kk`, `-pk kokkos`,
`package kokkos`, or `suffix kk` for this release. A one-GPU invocation is:

```bash
mpirun -np 1 /path/to/install/bin/lmp \
  -var DATA_FILE /path/to/water.data \
  -var MODEL_FILE /path/to/model.so3lr \
  -in /path/to/in.so3lr_builtin_smoke
```

For multiple GPUs, use one MPI rank per GPU. On one four-GPU Slurm node:

```bash
srun --nodes=1 --ntasks=4 --gpus-per-task=1 --gpu-bind=single:1 \
  /path/to/install/bin/lmp \
  -var DATA_FILE /path/to/water.data \
  -var MODEL_FILE /path/to/model.so3lr \
  -in /path/to/in.so3lr_builtin_smoke
```

The validated two-node/eight-GPU MeluXina configuration used conservative host
MPI transport:

```bash
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,tcp
export OMPI_MCA_mtl=^ofi
export UCX_TLS=tcp,self
export UCX_IB_GPU_DIRECT_RDMA=no
export UCX_MEMTYPE_CACHE=n

srun --nodes=2 --ntasks=8 --ntasks-per-node=4 \
  --gpus-per-task=1 --gpu-bind=single:1 \
  /path/to/install/bin/lmp -in /path/to/input.lmp
```

These MPI transport settings are a validated conservative MeluXina fallback,
not a universal requirement for other clusters.

## Compatibility policy

The installer rejects other LAMMPS versions by default. For a deliberate port
test only, set `SO3LR_ALLOW_UNTESTED_LAMMPS=1`; success of source installation
does not establish numerical or ABI compatibility. A small version-specific
CMake patch should be reviewed and validated for each later LAMMPS release.
