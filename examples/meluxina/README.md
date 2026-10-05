# Installing on MeluXina

These scripts build LAMMPS with both SO3LR pair styles compiled in. They
contain no project number or user name. Keep the release directory intact; the
scripts find the rest of the package relative to themselves.

## Automatic: one command

On a login node, from anywhere inside the release:

```bash
examples/meluxina/00_install_meluxina.sh
```

It shows what it detected and asks before submitting anything:
- **Slurm account:** detected from your associations.
- **Install root:** your directory under `/project/scratch/<account>`, or
  `/project/home/<account>` if scratch is close to quota.
- **GPU architecture:** AMPERE80, for the MeluXina A100.

It then submits three chained jobs. Each waits for the previous one, and a
failure cancels the rest.

| step | script | node | time | does |
|---|---|---|---|---|
| 2 | `02_prepare_lammps_so3lr_source.sh` | CPU | minutes | clones LAMMPS `patch_11Feb2026` and installs the SO3LR sources into it |
| 3 | `03_build_lammps_so3lr.sh` | CPU | ~30 min | configures, compiles and installs; writes `use-lammps-so3lr.env` |
| 4 | `04_verify_installation.sh` | GPU (`test` QOS) | minutes | runs the checks listed below |

Step 4's checks:
- both pair styles are registered;
- single points match the bundled JAX references for a neutral peptide and a
  charged anion;
- turbo matches the ordinary style.

These use `models/so3lr.so3lr`; `--model PATH` checks another SO3LR file
instead.

Options:

```text
--account pXXXXXX     Slurm account (default: detected)
--install-root PATH   where LAMMPS is cloned, built and installed
--lammps-source PATH  LAMMPS checkout to use or create
--kokkos-arch NAME    Kokkos GPU architecture, e.g. AMPERE80, HOPPER90
--model PATH          .so3lr file for Step 4 (default: models/so3lr.so3lr)
--yes                 do not ask for confirmation
--dry-run             print the sbatch commands only
--step2-only | --step3-only | --verify-only | --no-verify
```

Success is `SO3LR_MELUXINA_VERIFY=PASS` at the end of
`so3lr-verify-<job>.out` (in the directory you ran the driver from). The
installation then contains:

```text
<install root>/install/bin/lmp           LAMMPS with so3lr/native/mpi and so3lr/turbo
<install root>/use-lammps-so3lr.env      source this in every job that runs lmp
<install root>/logs/build-<job>/         cmake, build and install logs
<install root>/verify-<job>/             Step 4 inputs and results
```

## Running

In a job script:

```bash
source /path/to/install-root/use-lammps-so3lr.env   # modules, PATH, LD_LIBRARY_PATH
srun --ntasks=4 --gpus-per-task=1 --gpu-bind=single:1 lmp -in in.lammps
```

One MPI rank per GPU. For runs across nodes, the validated MPI settings are:

```bash
export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,tcp OMPI_MCA_mtl=^ofi
export UCX_TLS=tcp,self UCX_IB_GPU_DIRECT_RDMA=no UCX_MEMTYPE_CACHE=n
```

`lmp` is not usable on login nodes: they have no GPU, no CUDA driver and no
module system.

## Manual: the same steps by hand

```bash
cd examples/meluxina
export SO3LR_INSTALL_ROOT=/project/scratch/pXXXXXX/$USER/software/so3lr-lammps-0.4.0
EXP=ALL,SO3LR_INSTALL_ROOT=$SO3LR_INSTALL_ROOT

j2=$(sbatch --parsable --account=pXXXXXX --export=$EXP 02_prepare_lammps_so3lr_source.sh)
j3=$(sbatch --parsable --account=pXXXXXX --export=$EXP --dependency=afterok:$j2 03_build_lammps_so3lr.sh)
sbatch --account=pXXXXXX --export=$EXP --dependency=afterok:$j3 04_verify_installation.sh
```

Step 3 keeps its build directory. If it is interrupted by a time limit or a
node failure, submit it again with the same `SO3LR_INSTALL_ROOT` and CMake
resumes where it stopped. Use a new install root for a new release rather than
overlaying an old one.

## Build details

- **Toolchain:** GCCcore 14.2.0, CUDA 12.8.0, NVHPC 25.3 (`nvc++` as the
  host compiler behind Kokkos's `nvcc_wrapper`), OpenMPI 5.0.7, CMake 3.31.3,
  and LAMMPS's bundled Kokkos 5.0.2.
- **Shared library:** LAMMPS is built with `BUILD_SHARED_LIBS=ON`, and
  `liblammps.so` is installed under `install/lib64`. The environment file puts
  it on `LD_LIBRARY_PATH`.
- **Kokkos at run time:** the build enables LAMMPS's Kokkos package because
  SO3LR uses Kokkos internally. Do not switch on LAMMPS's own Kokkos mode at
  run time (`-k on`, `-sf kk`, `-pk kokkos`, `package kokkos`, `suffix kk`).
- **Linking on a CPU node:** Step 3 compiles on a CPU node, which has no CUDA
  driver library, so `lmp` is linked against the CUDA toolkit's driver stub.
  GPU nodes load the real driver. Submitting Step 3 to a GPU node also works
  (`sbatch --partition=gpu --gpus-per-task=1 ...`); it just usually waits
  longer for a node.
- **Validated LAMMPS version:** `patch_11Feb2026`, commit `b1ef9f45`. Other
  versions need a compatibility check (`tools/install_into_lammps_source.sh`
  refuses them unless `SO3LR_ALLOW_UNTESTED_LAMMPS=1`).

## Converting a checkpoint

`submit_convert_checkpoint.sh` converts a PyTorch SO3LR checkpoint inside a
container; see [../../docs/MODEL_CONVERSION.md](../../docs/MODEL_CONVERSION.md).
