# Changelog

## 0.4.0

Consolidates the two branches that grew from 0.2.0-rc1: the runtime
generalization (0.3.0) and turbo mode (turbo-0.3.0-dev14).

- `so3lr/turbo` (independent images packed on one GPU), with its image
  segmentation, replica bridge and MPI broker, built into LAMMPS next to
  `so3lr/native/mpi`.
- The packed turbo evaluator now runs the same generalized pipeline as
  `so3lr/native/mpi`, with charge correction and its adjoint applied per image,
  so turbo supports every model the ordinary style supports.
- **Total charge and spin multiplicity**:
  - `pair_style so3lr/native/mpi MODEL [charge [multiplicity]]`, and the same
    per image for `so3lr/turbo` (for example through world variables);
  - they feed the model's trained charge and spin embeddings and the
    partial-charge conservation;
  - the embedding is a per-element table built once per run (per image in
    turbo), so it adds no force term and no per-step cost;
  - invalid combinations stop the run; a charged periodic cell gives a
    warning.
- The repulsion setup is shared by both pair styles
  (`so3lr_repulsion_setup.{hpp,cpp}`).
- `tests/turbo/`: turbo against the ordinary style with neutral, charged and
  open-shell molecules. `tests/jaxref` gained `--charge`, `--multiplicity` and
  `--float64-params`, plus references for charged and open-shell molecules.
- MeluXina installer:
  - step 3 builds on a CPU node, linking against the CUDA driver stub (GPU
    nodes load the real driver);
  - it honours `SO3LR_KOKKOS_ARCH`/`SO3LR_CUDA_ARCH`, which were previously
    ignored in favour of a hard-coded AMPERE80;
  - its environment file also loads the runtime modules;
  - new step 4 verifies the installation on a GPU; `00_install_meluxina.sh`
    chains steps 2–4.
- `examples/turbo`: the submit wrappers find `run_meluxina.sh` under Slurm, and
  the runner loads Python for `generate_images.py`.
- Packaging:
  - user-facing README and documentation;
  - `.gitignore` for development material and model files;
  - `tools/update_manifest.sh` builds the manifest over the distributed files.
- The only distributed model is SO3LR, `models/so3lr.so3lr`, re-exported
  from the JAX checkpoint (it matches the JAX reference more closely than the
  0.2.0 conversion).

## 0.3.0

- The runtime reads its widths, interaction depth, attention heads and degree
  layout (including duplicated degrees) from the model's architecture record
  instead of compile-time constants, in every forward and backward kernel and
  in the MPI feature exchanges.
- Capability negotiation: loading a model checks it against a declared table
  of supported architecture features and names every unsupported one, instead
  of failing with `unsupported architecture contract`.
  `so3lr_capability_probe` answers the same question without LAMMPS.
- `export_flax_so3lr.py` exports JAX/flax checkpoints directly to `.so3lr`.
- `tests/jaxref`: single points against the float64 JAX reference, term by
  term.
- `examples/scaling`: strong/weak scaling benchmark and per-GPU capacity.
- Fixed the alanine examples, which lacked `comm_modify cutoff 13.0` and
  aborted at the first run.

## turbo-0.3.0-dev14 (parallel branch, merged in 0.4.0)

- Added the independent-image `so3lr/turbo` style and its packed evaluator to
  the compiled-in LAMMPS installation route.
- Preserved `so3lr/native/mpi`, the ordinary external plugin route, the model
  converter, model data, and their existing examples.
- Added neutral, nonperiodic one- and four-GPU MeluXina turbo examples.
- External turbo plugin loading is **not supported**; it previously failed
  because the plugin and LAMMPS had independent Kokkos runtimes.

## 0.2.0-rc1

- freeze the optimized native numerical implementation from stage-3
  optimization dev_12;
- package one source tree for external-plugin and built-in LAMMPS builds;
- add versioned `PKG_SO3LR` integration for LAMMPS 11Feb2026;
- accept element symbols as well as atomic numbers in `pair_coeff`;
- include the validated general SO3LR checkpoint converter and native-format
  verifier;
- add water and C/H/O/N alanine examples;
- accept Kokkos 5 view labels in the compiled-in LAMMPS adapter;
- correct the runtime contract: the pair style uses internal Kokkos/CUDA and
  must not be launched with LAMMPS global Kokkos switches;
- validate 24,000-atom NPT dynamics on eight GPUs across two MeluXina nodes
  using one MPI rank per GPU and conservative host MPI transport;
- include a converted model for authorized collaboration (no longer
  distributed since 0.4.0).
