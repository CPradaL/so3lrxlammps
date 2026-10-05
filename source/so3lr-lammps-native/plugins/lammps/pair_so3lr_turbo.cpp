#include "pair_so3lr_turbo.h"
#include "so3lr_element_map.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "universe.h"
#include "update.h"
#include "utils.h"

#include "so3lr/lammps_neighbor_adapter.hpp"
#include "so3lr/so3lr_charge_spin.hpp"

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using namespace LAMMPS_NS;

namespace {

so3lr::LammpsNeighborSnapshot snapshot_neighbor_list(NeighList *list)
{
  if (list == nullptr) throw std::runtime_error("missing LAMMPS neighbor list");
  so3lr::LammpsNeighborSnapshot result;
  result.ilist.reserve(static_cast<std::size_t>(list->inum));
  result.offsets.reserve(static_cast<std::size_t>(list->inum) + 1);
  result.offsets.push_back(0);
  for (int ii = 0; ii < list->inum; ++ii) {
    const int i = list->ilist[ii];
    result.ilist.push_back(static_cast<std::size_t>(i));
    const int count = list->numneigh[i];
    int *neighbors = list->firstneigh[i];
    for (int jj = 0; jj < count; ++jj)
      result.neighbors.push_back(
          static_cast<std::size_t>(neighbors[jj] & NEIGHMASK));
    result.offsets.push_back(result.neighbors.size());
  }
  return result;
}

void require_isolated_owned_graph(const so3lr::LammpsCompactGraph &graph,
                                  std::size_t nlocal, const char *label)
{
  if (graph.nodes() != nlocal || graph.owned_local.size() != nlocal ||
      !graph.ghost_local.empty())
    throw std::runtime_error(std::string(label) +
                             " graph is not an isolated owned graph");
  for (std::size_t node = 0; node < nlocal; ++node)
    if (graph.source_atom_rows[node] != node ||
        graph.owned_local[node] != node)
      throw std::runtime_error(std::string(label) +
                               " graph changed the owned atom ordering");
  for (const auto sender : graph.senders)
    if (sender >= nlocal)
      throw std::runtime_error(std::string(label) +
                               " sender is outside the owned graph");
  for (const auto receiver : graph.receivers)
    if (receiver >= nlocal)
      throw std::runtime_error(std::string(label) +
                               " receiver is outside the owned graph");
}

}  // namespace

PairSO3LRTurbo::PairSO3LRTurbo(LAMMPS *lmp) : Pair(lmp)
{
  manybody_flag = 1;
  one_coeff = 1;
  restartinfo = 0;
  single_enable = 0;
  MPI_Comm_rank(universe->uworld, &universe_rank_);
  MPI_Comm_size(universe->uworld, &universe_size_);
  if (universe_rank_ == 0 && !Kokkos::is_initialized()) {
    Kokkos::initialize();
    initialized_kokkos_here_ = true;
  }
}

PairSO3LRTurbo::~PairSO3LRTurbo()
{
  broker_.reset();
  model_.reset();
  if (initialized_kokkos_here_ && Kokkos::is_initialized()) Kokkos::finalize();
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
}

void PairSO3LRTurbo::allocate()
{
  allocated = 1;
  const int n = atom->ntypes;
  memory->create(setflag, n + 1, n + 1, "so3lr/turbo:setflag");
  memory->create(cutsq, n + 1, n + 1, "so3lr/turbo:cutsq");
  for (int i = 1; i <= n; ++i)
    for (int j = i; j <= n; ++j) setflag[i][j] = 0;
}

void PairSO3LRTurbo::settings(int narg, char **arg)
{
  if (narg < 1 || narg > 3)
    error->universe_all(FLERR,
                        "Illegal pair_style so3lr/turbo command: expected "
                        "'pair_style so3lr/turbo MODEL [charge [multiplicity]]'");
  model_path_ = arg[0];
  // Per image: each partition reads its own values (e.g. world variables).
  total_charge_ = narg > 1 ? utils::numeric(FLERR, arg[1], false, lmp) : 0.0;
  multiplicity_ = narg > 2 ? utils::numeric(FLERR, arg[2], false, lmp) : 1.0;
  int model_ok = 1;
  if (universe_rank_ == 0) {
    try {
      model_ = std::make_unique<so3lr::NativeModel>(
          so3lr::NativeModel::load(model_path_));
      short_range_cutoff_ =
          model_->architecture_number("short_range_cutoff_angstrom");
      long_range_cutoff_ =
          model_->architecture_number("long_range_cutoff_angstrom");
      if (!(short_range_cutoff_ > 0.0) ||
          !(long_range_cutoff_ >= short_range_cutoff_))
        throw std::runtime_error("invalid cutoffs in native model");
    } catch (const std::exception &exc) {
      std::fprintf(stderr, "SO3LR turbo coordinator model error: %s\n",
                   exc.what());
      model_ok = 0;
    }
  }
  MPI_Bcast(&model_ok, 1, MPI_INT, 0, universe->uworld);
  if (!model_ok)
    error->universe_all(FLERR,
                        "Cannot initialize the coordinator SO3LR model");
  MPI_Bcast(&short_range_cutoff_, 1, MPI_DOUBLE, 0, universe->uworld);
  MPI_Bcast(&long_range_cutoff_, 1, MPI_DOUBLE, 0, universe->uworld);
  try {
    broker_ = std::make_unique<so3lr::MpiTurboReplicaBroker>(
        universe->uworld, 0, universe_rank_ == 0 ? model_.get() : nullptr);
  } catch (const std::exception &exc) {
    error->universe_all(
        FLERR,
        std::string("Cannot initialize SO3LR turbo broker: ") + exc.what());
  }
}

void PairSO3LRTurbo::coeff(int narg, char **arg)
{
  if (!allocated) allocate();
  if (narg != atom->ntypes + 2 || std::strcmp(arg[0], "*") != 0 ||
      std::strcmp(arg[1], "*") != 0)
    error->all(FLERR,
               "Pair coeff for so3lr/turbo must be '* * E1 ... Entypes'");
  type_to_atomic_number_.assign(static_cast<std::size_t>(atom->ntypes) + 1,
                                0);
  for (int type = 1; type <= atom->ntypes; ++type) {
    const int z = so3lr_lammps::atomic_number_from_token(arg[type + 1]);
    if (z == 0)
      error->all(FLERR, "Unknown SO3LR element '{}'", arg[type + 1]);
    type_to_atomic_number_[static_cast<std::size_t>(type)] = z;
  }
  for (int i = 1; i <= atom->ntypes; ++i)
    for (int j = i; j <= atom->ntypes; ++j) setflag[i][j] = 1;
}

double PairSO3LRTurbo::init_one(int i, int j)
{
  if (!setflag[i][j])
    error->all(FLERR, "All so3lr/turbo coefficients are not set");
  return long_range_cutoff_;
}

void PairSO3LRTurbo::init_style()
{
  if (comm->nprocs != 1)
    error->universe_all(
        FLERR, "so3lr/turbo requires one MPI rank per LAMMPS partition");
  if (universe->nworlds != universe_size_ || universe_size_ < 2)
    error->universe_all(
        FLERR, "so3lr/turbo requires multiple one-rank partitions");
  if (domain->xperiodic || domain->yperiodic || domain->zperiodic)
    error->universe_all(
        FLERR, "so3lr/turbo requires non-periodic boundaries");
  if (!force->newton_pair)
    error->all(FLERR, "Pair style so3lr/turbo requires newton pair on");
  if (!broker_)
    error->universe_all(FLERR, "SO3LR turbo broker was not initialized");
  if (type_to_atomic_number_.size() !=
      static_cast<std::size_t>(atom->ntypes) + 1)
    error->all(FLERR, "Pair style so3lr/turbo coefficients are missing");

  auto *sr = neighbor->add_request(this, NeighConst::REQ_FULL);
  sr->set_id(0);
  sr->set_cutoff(short_range_cutoff_);
  auto *lr = neighbor->add_request(this);
  lr->set_id(1);
  lr->set_cutoff(long_range_cutoff_);

  so3lr::ElementCounts counts{};
  for (int i = 0; i < atom->nlocal; ++i) {
    const std::int64_t z =
        type_to_atomic_number_[static_cast<std::size_t>(atom->type[i])];
    if (z > 0 && z <= 118) ++counts[static_cast<std::size_t>(z)];
  }
  const std::string problem = so3lr::charge_multiplicity_problem(
      counts, total_charge_, multiplicity_);
  if (!problem.empty())
    error->universe_one(FLERR, fmt::format(
        "Pair style so3lr/turbo image {}: charge {} with multiplicity {}: {}",
        universe->iworld, total_charge_, multiplicity_, problem));
  utils::logmesg(lmp, "SO3LR turbo image {}: total charge {:g}, spin "
                 "multiplicity {:g}\n", universe->iworld, total_charge_,
                 multiplicity_);
}

void PairSO3LRTurbo::init_list(int which, NeighList *ptr)
{
  if (which == 0)
    short_range_list_ = ptr;
  else if (which == 1)
    long_range_list_ = ptr;
  else
    error->all(FLERR, "Unexpected so3lr/turbo neighbor-list ID");
}

void PairSO3LRTurbo::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);
  if (vflag_atom)
    error->universe_all(
        FLERR, "so3lr/turbo does not provide per-atom virial");
  if (short_range_list_ == nullptr || long_range_list_ == nullptr)
    error->all(FLERR, "SO3LR turbo neighbor lists were not initialized");

  try {
    so3lr::LammpsAtomSnapshot atoms;
    atoms.nlocal = static_cast<std::size_t>(atom->nlocal);
    const std::size_t rows =
        static_cast<std::size_t>(atom->nlocal + atom->nghost);
    atoms.tags.reserve(rows);
    atoms.types.reserve(rows);
    atoms.positions.reserve(rows * 3);
    for (std::size_t row = 0; row < rows; ++row) {
      atoms.tags.push_back(static_cast<std::int64_t>(atom->tag[row]));
      atoms.types.push_back(atom->type[row]);
      atoms.positions.push_back(atom->x[row][0]);
      atoms.positions.push_back(atom->x[row][1]);
      atoms.positions.push_back(atom->x[row][2]);
    }
    auto graphs = so3lr::build_lammps_native_graphs(
        atoms, snapshot_neighbor_list(short_range_list_),
        snapshot_neighbor_list(long_range_list_), type_to_atomic_number_,
        short_range_cutoff_, long_range_cutoff_, force->newton_pair != 0);
    const std::size_t nodes = static_cast<std::size_t>(atom->nlocal);
    require_isolated_owned_graph(graphs.short_range, nodes, "SR");
    require_isolated_owned_graph(graphs.long_range, nodes, "LR");
    if (graphs.short_range.atomic_numbers !=
        graphs.long_range.atomic_numbers)
      throw std::runtime_error("SR/LR atomic-number ordering differs");

    so3lr::TurboReplicaGraph local_graph;
    local_graph.atomic_numbers = graphs.short_range.atomic_numbers;
    local_graph.sr_edge_vectors = graphs.short_range.vectors;
    local_graph.sr_senders = graphs.short_range.senders;
    local_graph.sr_receivers = graphs.short_range.receivers;
    local_graph.lr_pair_vectors = graphs.long_range.vectors;
    local_graph.lr_senders = graphs.long_range.senders;
    local_graph.lr_receivers = graphs.long_range.receivers;
    local_graph.total_charge = total_charge_;
    local_graph.unpaired_electrons = multiplicity_ - 1.0;

    const double broker_start = MPI_Wtime();
    const auto result = broker_->evaluate(local_graph);
    const double broker_ms = (MPI_Wtime() - broker_start) * 1000.0;
    if (result.atomic_energies.size() != nodes ||
        result.atomic_forces.size() != nodes * 3 ||
        result.partial_charges.size() != nodes ||
        result.hirshfeld_ratios.size() != nodes)
      throw std::runtime_error("turbo result has invalid dimensions");

    for (std::size_t node = 0; node < nodes; ++node) {
      for (std::size_t component = 0; component < 3; ++component)
        atom->f[node][component] += result.atomic_forces[node * 3 + component];
      if (eflag_atom) eatom[node] += result.atomic_energies[node];
    }
    if (eflag_global) eng_vdwl += result.total_energy;
    if (vflag_fdotr) virial_fdotr_compute();
    ++evaluation_count_;

    if (!reported_) {
      const auto report = [&](FILE *stream) {
        if (stream != nullptr)
          std::fprintf(
              stream,
              "SO3LR_TURBO_LIVE image=%d universe_rank=%d coordinator=%d "
              "images=%d atoms=%zu sr_edges=%zu lr_pairs=%zu broker_ms=%.9g\n",
              universe->iworld, universe_rank_, universe_rank_ == 0 ? 1 : 0,
              universe_size_, nodes, local_graph.sr_senders.size(),
              local_graph.lr_senders.size(), broker_ms);
      };
      report(screen);
      if (logfile != screen) report(logfile);
      reported_ = true;
    }
  } catch (const std::exception &exc) {
    error->universe_all(
        FLERR,
        std::string("SO3LR turbo LAMMPS evaluation failed: ") + exc.what());
  }
}
