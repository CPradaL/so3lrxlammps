# External LAMMPS plugin

This route builds only `so3lr/native/mpi`. Turbo currently requires the
[built-in installation](TURBO.md); its earlier separate plugin prototype
crashed due to a second Kokkos runtime.

This route builds `so3lrnativeplugin.so` from the same numerical source used by
the built-in package. It is convenient during development but couples the
module to the target LAMMPS headers, compiler ABI, MPI, CUDA, and Kokkos build.

The target LAMMPS executable must have been built with `PKG_PLUGIN=ON`.

```bash
export SO3LR_KOKKOS_ARCH=AMPERE80
export SO3LR_MPI_CXX=$(command -v mpicxx)
export SO3LR_CUDA_HOST_CXX=$(command -v nvc++)
export SO3LR_BUILD_JOBS=16

tools/build_external_plugin.sh \
  /path/to/exact-lammps-source \
  /path/to/so3lr-plugin-build \
  /path/to/so3lr-plugin-install
```

At runtime:

```lammps
plugin load /path/to/so3lr-plugin-install/lib/so3lrnativeplugin.so
pair_style so3lr/native/mpi /path/to/model.so3lr
pair_coeff * * H O
```

Launch without LAMMPS global Kokkos arguments. The plugin uses the same
internal Kokkos/CUDA runtime contract as the built-in pair style:

```bash
mpirun -np 1 /path/to/lmp \
  -var PLUGIN_FILE /path/to/so3lrnativeplugin.so \
  -var DATA_FILE /path/to/water.data \
  -var MODEL_FILE /path/to/model.so3lr \
  -in /path/to/in.so3lr_rc_smoke
```

Do not add `-k`, `-sf kk`, `-pk kokkos`, `package kokkos`, or `suffix kk` for
this release.

Compile the plugin locally on each target platform. Do not copy a MeluXina
A100 binary to an unrelated machine and assume compatibility.
