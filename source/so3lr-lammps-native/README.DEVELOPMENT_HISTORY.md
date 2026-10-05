# SO3LR native LAMMPS plugin

This repository is the implementation home for the external
`pair_style so3lr/kk` LAMMPS plugin and reusable `libso3lr` runtime.

Current development status:

- Stage 3.0: PyTorch reference and operation profile complete.
- Stage 3.1: versioned framework-neutral model format and exporter.
- Native forward/reverse kernels and the LAMMPS pair-style adapter are not yet
  implemented.

The initial user-facing target is:

```lammps
plugin load /path/to/liblammps_so3lr.so
pair_style so3lr/kk
pair_coeff * * model.so3lr H O
```

Repository layout will grow as follows:

```text
docs/       model format and design documentation
exporter/   checkpoint-to-native conversion and independent validation
libso3lr/   native model loader and CPU/Kokkos kernels
lammps/     thin pair-style and plugin-registration adapter
tests/      tensor, force, virial, MPI and Kokkos regression tests
examples/   reproducible LAMMPS inputs
```

The trained model mathematics remain unchanged. The native implementation must
match the frozen PyTorch reference before any performance comparison is valid.

Licensing and public release metadata will be finalized with the SO3LR authors
before the first external release.
