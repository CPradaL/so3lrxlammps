#include "lammpsplugin.h"
#include "version.h"

#include "pair_so3lr_adapter.h"

using namespace LAMMPS_NS;

static Pair *so3lr_adapter_creator(LAMMPS *lmp)
{
  return new PairSO3LRAdapter(lmp);
}

extern "C" void lammpsplugin_init(void *lmp, void *handle, void *regfunc)
{
  auto register_plugin = reinterpret_cast<lammpsplugin_regfunc>(regfunc);
  lammpsplugin_t plugin;
  plugin.version = LAMMPS_VERSION;
  plugin.style = "pair";
  plugin.name = "so3lr/adapter";
  plugin.info = "SO3LR live MPI ownership and persistent-topology preflight v0.2";
  plugin.author = "SO3LR LAMMPS development";
  plugin.creator.v1 =
      reinterpret_cast<lammpsplugin_factory1 *>(&so3lr_adapter_creator);
  plugin.handle = handle;
  (*register_plugin)(&plugin, lmp);
}
