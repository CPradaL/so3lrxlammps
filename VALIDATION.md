# Validation

Everything below ran on MeluXina: NVIDIA A100-40GB, GCCcore 14.2.0,
CUDA 12.8.0, NVHPC 25.3, OpenMPI 5.0.7, LAMMPS `patch_11Feb2026` with its
bundled Kokkos 5.0.2. The model is SO3LR, exported from the JAX checkpoint
with `tools/checkpoint_conversion/export_flax_so3lr.py`.

## Against the JAX reference

`tests/jaxref`: a LAMMPS single point (`in.single_point`) compared atom by atom
with the float64 JAX implementation (`so3lr_reference.py`). Besides energies
and forces, the debug dump compares every term: network energy, repulsion,
partial charges, Hirshfeld ratios and the long-range energy.

**Neutral systems** (alanine peptide, 162 atoms, non-periodic; liquid water,
3,000 atoms, periodic):

| system | max\|dF\| (eV/Å) | network energy | partial charges |
|---|---|---|---|
| alanine | 5.4e-7 | 2.0e-14 | 2.2e-15 |
| water | 1.3e-6 | 3.3e-14 | 2.9e-15 |

Every network term agrees to ~1e-14; the ~5e-7 eV/Å floor in the total force
is most likely the reference's own precision. The JAX checkpoint is stored in
float32, and the reference evaluates it without converting. Dense layers
promote the weights against float64 activations, but some operations use the
float32 parameters directly. `so3lr_reference.py --float64-params` casts the
checkpoint to float64 before use. With it, the runtime agrees to ~1e-10 eV/Å
on every system in the next table. These two neutral references were made
without the flag.

**Charged and open-shell systems**, references with `--float64-params`:

| system | charge, multiplicity | max\|dF\| (eV/Å) |
|---|---|---|
| alanine peptide cation | +1, doublet | 1.3e-10 |
| acetate | −1, singlet | 4.4e-11 |
| methylammonium | +1, singlet | 5.9e-11 |
| methyl radical | 0, doublet | 1.0e-11 |
| O2 | 0, triplet | 6.7e-12 |
| H2O+ | +1, doublet | 1.9e-11 |

Partial charges agree to ≤ 2e-15 e. The charge and spin embedding table
alone, computed on the CPU (`so3lr_charge_spin_table`), matches the
reference's `ChargeSpinEmbedSparse` to ≤ 3e-15. The input checks refuse a
non-integer charge, a multiplicity below 1, a multiplicity whose parity does
not match the electron count, and more unpaired electrons than electrons.

## Multi-GPU consistency

The same single point on 1 and on 4 GPUs (feature and adjoint exchanges
between ranks; for a charged system, the composition reduced over ranks):

| system | energy | max\|dF\| (eV/Å) |
|---|---|---|
| water, 3,000 atoms, periodic | identical | 6.6e-15 |
| alanine cation doublet | identical | 5.6e-15 |

## `so3lr/turbo` against `so3lr/native/mpi`

`tests/turbo/job_turbo_validate.sh`. Eleven images in one pack on one GPU:
- the alanine peptide twice (neutral) and once as a cation doublet;
- a water molecule, ethanol and a cyclic water trimer;
- acetate, methylammonium, the methyl radical, O2 (triplet) and H2O+.

Each image has its own charge and multiplicity. Every image was compared with
the same molecule run alone through the ordinary style, first as a single
point and then after 400 NVE steps (0.5 fs, same seed). The worst image:

| quantity | max difference |
|---|---|
| single-point energy | 3.6e-12 eV |
| single-point force | 6.9e-15 eV/Å |
| positions after 400 steps | 9.4e-14 Å |
| total energy after 400 steps | 1.8e-11 eV |

Native unit tests for turbo:
- **packed evaluator:** four water clusters in one pack against the
  single-rank evaluator, one cluster at a time. Forces agree to 3.2e-15 eV/Å,
  atomic energies to 2.8e-14 eV, and the per-image charge sum is zero to
  4.7e-15.
- **replica bridge:** trajectory difference 3e-22 Å.
- **MPI broker:** 4 ranks; forces agree to 5.6e-15 eV/Å.

The turbo examples (`examples/turbo`) completed on 1 GPU (4 images) and on
4 GPUs (16 images, one GPU per group).

## Regression between versions

The same ten single points with the 0.3.0 and the 0.4.0 builds agree to
≤ 1.1e-14 eV/Å. That is the run-to-run level of the order-dependent atomic
additions in the reductions.

## Installation

`examples/meluxina/00_install_meluxina.sh` was run end to end from a copy of
exactly the distributed files, into an empty directory:
- **step 2:** cloned LAMMPS from GitHub and installed the sources;
- **step 3:** built on a CPU node in 9 minutes;
- **step 4:** passed all of its checks on a GPU in under a minute.

`tools/validate_release.sh` (shell syntax, Python compilation, the element-map
unit test, turbo source integration, the manifest) passes without a GPU.

## Earlier release gates (0.2.0)

- compiled-in LAMMPS build with bundled Kokkos 5.0.2; `so3lr/native/mpi`
  registered without an external plugin;
- the built-in installer applies cleanly to an unmodified LAMMPS
  `patch_11Feb2026` checkout and refuses a second installation;
- 3,000-atom water on one and two GPUs, energies and forces agreeing to
  numerical precision;
- 24,000-atom, 10,000-step NPT runs on 1, 2 and 4 GPUs, and on 8 GPUs across
  two nodes: 888.094 s loop time, 301.318 K final temperature,
  1.054965 g cm⁻³ final density, no dangerous neighbour-list builds;
- the C/H/O/N alanine peptide with element symbols in `pair_coeff`.

The 0.2.0 alanine inputs lacked `comm_modify cutoff 13.0` and could not run
as shipped (`Custom neighbor list cutoff too large for communication
cutoff`). Every input sets it since 0.3.0.
