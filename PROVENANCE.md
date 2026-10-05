# Source provenance

## 0.4.0

Two parents, both from the `0.2.0-rc1` package below:

- `so3lr-lammps-0.3.0-general`: the numerical `src`/`include` trees, the
  ordinary pair style, the flax exporter, the JAX validation and the
  scaling study. Its shared numerical kernels are carried over unchanged.
- `so3lr-lammps-turbo-0.3.0-dev14`: `pair_so3lr_turbo.{cpp,h}`, image
  segmentation, replica bridge and MPI broker, the turbo example and docs.
  Turbo source was reconstructed from the passed stage-4 dev_1–dev_8
  integration templates.

New in 0.4.0: `kokkos_packed_so3lr_evaluator.{hpp,cpp}` rewritten on the
generalized pipeline; `so3lr_repulsion_setup.{hpp,cpp}` factored out of
`pair_so3lr_native_mpi.cpp`; the charge/spin embedding
(`so3lr_charge_spin.{hpp,cpp}`); `tests/turbo/`.

## Base ordinary-release provenance

The numerical `source/so3lr-lammps-native/src` and
`source/so3lr-lammps-native/include` trees are byte-identical to the dev_12
release snapshot at commit in the **base `0.2.0-rc1` package**:

```text
99674789160d0bda7c92439fe76cbcb6ecf42580
```

Relative to the preceding `0.1.0-rc1` package, changes in the LAMMPS-facing
files are limited to:

- element-symbol parsing in `pair_coeff`, while preserving numeric mappings;
- standard `PairStyle(...)` registration guards needed by a built-in LAMMPS
  compilation;
- a release-neutral internal namespace name;
- package version and registration descriptions.

The built-in integration was authored against the official LAMMPS
`patch_11Feb2026` tag at commit:

```text
b1ef9f45934826495efe9b442b39e9c74e20095d
```

The generic converter is the version validated by the foundational-model
C/H/O/N alanine campaign, with only its release identifier and diagnostic
wording updated.
