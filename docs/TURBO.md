# Independent-image turbo mode

`pair_style so3lr/turbo MODEL.so3lr` batches the SO3LR evaluation for several
independent, single-rank LAMMPS partitions on one GPU. Every partition retains
its own atoms, velocities, fixes, integration, log, and trajectory. The broker
gathers each partition's short- and long-range graphs, evaluates a segmented
batch on one GPU, and sends each image's energy and forces back. It synchronizes
once per force evaluation; images cannot advance asynchronously.

Turbo supports every model `so3lr/native/mpi` supports: since 0.4.0 its
packed evaluator runs the same
generalized pipeline as the ordinary style, with the charge constraint applied
per image. Each image can have its own total charge and spin multiplicity
(see below). Turbo is for **nonperiodic systems only**. It requires at least
two one-rank partitions per GPU. It rejects periodic boundaries and per-atom
virial requests. Do not use it for NPT, for one spatially decomposed system,
or for runs with different force-call schedules among images. Use
`so3lr/native/mpi` for ordinary MD instead.

## Build and load

Turbo is included by `tools/install_into_lammps_source.sh` and built when
LAMMPS is compiled with `PKG_SO3LR=ON`. Both native styles appear in `lmp -h`.
The external `so3lrnativeplugin.so` route remains available for the ordinary
style only. **Do not attempt to build or load an external turbo plugin** from
these sources: an earlier prototype had two Kokkos runtimes and crashed at its
first packed evaluation.

The built-in installation steps in `examples/meluxina` still apply. Use a
*fresh* LAMMPS source and installation directory for this candidate; do not
overlay it on an earlier installation. Before running MD,
check that both `so3lr/native/mpi` and `so3lr/turbo` appear in `lmp -h`.

## LAMMPS input

One image uses ordinary LAMMPS commands, with the turbo style substituted:

```lammps
units           metal
atom_style      atomic
boundary        f f f
read_data       molecule.data
pair_style      so3lr/turbo model.so3lr
pair_coeff      * * C H O N
neighbor        1.0 bin
neigh_modify    every 1 delay 0 check yes
fix             integrator all nve
run             1000
```

Two optional numbers after the model give the image's total charge and spin
multiplicity, as for `so3lr/native/mpi`; the defaults are 0 and 1. With world
variables, every image in a pack can differ:

```lammps
variable        Q world 0 -1 1 0
variable        M world 1 1 1 2
pair_style      so3lr/turbo model.so3lr ${Q} ${M}
```

Each image is checked when the run starts: the charge and multiplicity must
be integers and must match the image's electron count.

`pair_coeff` element names follow LAMMPS atom-type order, just as for the
ordinary style. The example under `examples/turbo` creates image-specific
world variables for image names and velocity seeds. A single-GPU run of four
images uses `lmp -partition 4x1`; it does **not** use four ranks to decompose
one molecule. Four-GPU execution starts four independent groups, each with
its own GPU coordinator and image batch.

## MeluXina example

After a new built-in installation, from any directory:

```bash
export SO3LR_INSTALL_ROOT=/project/home/PROJECT/USER/software/so3lr-lammps-0.4.0
sbatch --account=PROJECT /path/to/release/examples/turbo/submit_1gpu.sh
sbatch --account=PROJECT /path/to/release/examples/turbo/submit_4gpu.sh
```

The default example runs four independent, isolated water-molecule images
per GPU for 1,000 NVE steps. Set `RUN_STEPS`, `SO3LR_MODEL`, or `SO3LR_DATA`
in the submitted environment to change the case. If changing
`IMAGES_PER_GPU`, also pass the matching `sbatch --ntasks` allocation; read
`examples/turbo/README.md` first. The files under each `group_XX/image_XXXX`
are normal LAMMPS outputs for that image. `status.txt` is written only after
every group finishes and every image prints its final marker.

For a production use case, validate the model and chosen integration setup
against separate ordinary `so3lr/native/mpi` runs first.
`tests/turbo/job_turbo_validate.sh` does this for any set of models: eleven
different molecules packed on one GPU, compared image by image with ordinary
single-molecule runs (single point and a short NVE run). The results for this
version are in `VALIDATION.md`.
