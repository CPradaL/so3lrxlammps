#include "lammpsplugin.h"
#include "version.h"

#include "pair_so3lr_native.h"
#include "pair_so3lr_native_mpi.h"

using namespace LAMMPS_NS;

static Pair *so3lr_native_creator(LAMMPS *lmp) {
  return new PairSO3LRNative(lmp);
}

static Pair *so3lr_native_mpi_creator(LAMMPS *lmp) {
  return new PairSO3LRNativeMPI(lmp);
}

extern "C" void lammpsplugin_init(void *lmp, void *handle, void *regfunc) {
  auto register_plugin = reinterpret_cast<lammpsplugin_regfunc>(regfunc);
  lammpsplugin_t plugin;
  plugin.version = LAMMPS_VERSION;
  plugin.style = "pair";
  plugin.author = "SO3LR LAMMPS collaborators";
  plugin.handle = handle;

  plugin.name = "so3lr/native";
  plugin.info = "SO3LR validated one-rank native reference v1.9";
  plugin.creator.v1 =
      reinterpret_cast<lammpsplugin_factory1 *>(&so3lr_native_creator);
  (*register_plugin)(&plugin, lmp);

  plugin.name = "so3lr/native/mpi";
  plugin.info = "SO3LR native ownership-aware MPI pair style 0.4.0";
  plugin.creator.v1 =
      reinterpret_cast<lammpsplugin_factory1 *>(&so3lr_native_mpi_creator);
  (*register_plugin)(&plugin, lmp);
}
