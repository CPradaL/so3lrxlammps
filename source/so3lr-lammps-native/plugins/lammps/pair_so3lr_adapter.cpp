#include "pair_so3lr_adapter.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "utils.h"

#include "so3lr/lammps_neighbor_adapter.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace LAMMPS_NS;

namespace {

constexpr std::uint64_t fnv_offset = UINT64_C(1469598103934665603);
constexpr std::uint64_t fnv_prime = UINT64_C(1099511628211);

void hash_value(std::uint64_t &hash, std::uint64_t value)
{
  for (int byte = 0; byte < 8; ++byte) {
    hash ^= (value >> (byte * 8)) & UINT64_C(0xff);
    hash *= fnv_prime;
  }
}

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

std::uint64_t owned_layout_signature(const so3lr::LammpsAtomSnapshot &atoms)
{
  std::uint64_t hash = fnv_offset;
  hash_value(hash, atoms.nlocal);
  for (std::size_t row = 0; row < atoms.nlocal; ++row) {
    hash_value(hash, static_cast<std::uint64_t>(atoms.tags[row]));
    hash_value(hash, static_cast<std::uint64_t>(atoms.types[row]));
  }
  return hash;
}

std::uint64_t neighbor_signature(const so3lr::LammpsNeighborSnapshot &list,
                                 const so3lr::LammpsAtomSnapshot &atoms)
{
  std::uint64_t hash = fnv_offset;
  hash_value(hash, list.ilist.size());
  hash_value(hash, list.neighbors.size());
  for (const auto row : list.ilist) {
    hash_value(hash, row);
    hash_value(hash, static_cast<std::uint64_t>(atoms.tags.at(row)));
  }
  for (const auto offset : list.offsets) hash_value(hash, offset);
  for (const auto row : list.neighbors) {
    hash_value(hash, row);
    hash_value(hash, static_cast<std::uint64_t>(atoms.tags.at(row)));
  }
  return hash;
}

std::uint64_t graph_signature(const so3lr::LammpsCompactGraph &graph)
{
  std::uint64_t hash = fnv_offset;
  hash_value(hash, graph.nodes());
  hash_value(hash, graph.interactions());
  for (std::size_t node = 0; node < graph.nodes(); ++node) {
    hash_value(hash, graph.source_atom_rows[node]);
    hash_value(hash, static_cast<std::uint64_t>(graph.tags[node]));
    hash_value(hash, static_cast<std::uint64_t>(graph.atomic_numbers[node]));
  }
  for (std::size_t edge = 0; edge < graph.interactions(); ++edge) {
    hash_value(hash, graph.senders[edge]);
    hash_value(hash, graph.receivers[edge]);
  }
  return hash;
}

struct OwnershipAudit {
  long long global_owned = 0;
  long long global_atom_ghost_rows = 0;
  long long sr_remote_ghost_rows = 0;
  long long sr_periodic_image_rows = 0;
  long long lr_remote_ghost_rows = 0;
  long long lr_periodic_image_rows = 0;
  long long sr_owner_peers = 0;
  long long lr_owner_peers = 0;
};

OwnershipAudit audit_ownership(
    const so3lr::LammpsAtomSnapshot &atoms,
    const so3lr::LammpsCompactGraph &sr,
    const so3lr::LammpsCompactGraph &lr,
    const std::vector<std::int64_t> &type_to_z, int rank, int ranks,
    MPI_Comm world)
{
  if (atoms.nlocal > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::runtime_error("owned atom count exceeds MPI int count");
  const int local_count = static_cast<int>(atoms.nlocal);
  std::vector<int> counts(static_cast<std::size_t>(ranks));
  MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, world);
  std::vector<int> displacements(static_cast<std::size_t>(ranks));
  int total = 0;
  for (int peer = 0; peer < ranks; ++peer) {
    displacements[static_cast<std::size_t>(peer)] = total;
    if (counts[static_cast<std::size_t>(peer)] < 0 ||
        total > std::numeric_limits<int>::max() -
                    counts[static_cast<std::size_t>(peer)])
      throw std::runtime_error("global owned atom count exceeds MPI int range");
    total += counts[static_cast<std::size_t>(peer)];
  }
  std::vector<long long> local_tags(atoms.nlocal);
  std::vector<long long> local_z(atoms.nlocal);
  for (std::size_t row = 0; row < atoms.nlocal; ++row) {
    local_tags[row] = static_cast<long long>(atoms.tags[row]);
    local_z[row] = static_cast<long long>(
        type_to_z.at(static_cast<std::size_t>(atoms.types[row])));
  }
  std::vector<long long> all_tags(static_cast<std::size_t>(total));
  std::vector<long long> all_z(static_cast<std::size_t>(total));
  MPI_Allgatherv(local_tags.data(), local_count, MPI_LONG_LONG,
                 all_tags.data(), counts.data(), displacements.data(),
                 MPI_LONG_LONG, world);
  MPI_Allgatherv(local_z.data(), local_count, MPI_LONG_LONG,
                 all_z.data(), counts.data(), displacements.data(),
                 MPI_LONG_LONG, world);

  std::unordered_map<long long, std::pair<int, long long>> owner;
  owner.reserve(all_tags.size());
  for (int peer = 0; peer < ranks; ++peer) {
    const int begin = displacements[static_cast<std::size_t>(peer)];
    const int end = begin + counts[static_cast<std::size_t>(peer)];
    for (int item = begin; item < end; ++item) {
      if (!owner.emplace(all_tags[static_cast<std::size_t>(item)],
                         std::make_pair(peer, all_z[static_cast<std::size_t>(item)])).second)
        throw std::runtime_error("an atom tag is owned by more than one MPI rank");
    }
  }
  if (owner.size() != all_tags.size())
    throw std::runtime_error("global owned-tag table is not unique");

  for (std::size_t row = atoms.nlocal; row < atoms.tags.size(); ++row) {
    const auto found = owner.find(static_cast<long long>(atoms.tags[row]));
    if (found == owner.end())
      throw std::runtime_error("LAMMPS ghost tag has no global owner");
    const auto z = type_to_z.at(static_cast<std::size_t>(atoms.types[row]));
    if (found->second.second != z)
      throw std::runtime_error("LAMMPS ghost element differs from its owner");
  }

  auto count_graph = [&](const so3lr::LammpsCompactGraph &graph,
                         long long &remote, long long &periodic,
                         long long &peer_count) {
    std::unordered_set<int> peers;
    for (const auto node : graph.ghost_local) {
      const auto found = owner.find(static_cast<long long>(graph.tags.at(node)));
      if (found == owner.end())
        throw std::runtime_error("compact graph ghost has no global owner");
      if (found->second.second != graph.atomic_numbers.at(node))
        throw std::runtime_error("compact graph ghost element mismatch");
      if (found->second.first == rank)
        ++periodic;
      else {
        ++remote;
        peers.insert(found->second.first);
      }
    }
    peer_count = static_cast<long long>(peers.size());
  };

  OwnershipAudit result;
  result.global_owned = total;
  long long local_atom_ghosts =
      static_cast<long long>(atoms.tags.size() - atoms.nlocal);
  MPI_Allreduce(&local_atom_ghosts, &result.global_atom_ghost_rows, 1,
                MPI_LONG_LONG, MPI_SUM, world);
  count_graph(sr, result.sr_remote_ghost_rows,
              result.sr_periodic_image_rows, result.sr_owner_peers);
  count_graph(lr, result.lr_remote_ghost_rows,
              result.lr_periodic_image_rows, result.lr_owner_peers);
  return result;
}

}  // namespace

PairSO3LRAdapter::PairSO3LRAdapter(LAMMPS *lmp) : Pair(lmp)
{
  manybody_flag = 1;
  one_coeff = 1;
  restartinfo = 0;
  single_enable = 0;
  no_virial_fdotr_compute = 1;
}

PairSO3LRAdapter::~PairSO3LRAdapter()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
}

void PairSO3LRAdapter::allocate()
{
  allocated = 1;
  const int n = atom->ntypes;
  memory->create(setflag, n + 1, n + 1, "so3lr/adapter:setflag");
  memory->create(cutsq, n + 1, n + 1, "so3lr/adapter:cutsq");
  for (int i = 1; i <= n; ++i)
    for (int j = i; j <= n; ++j) setflag[i][j] = 0;
}

void PairSO3LRAdapter::settings(int narg, char **arg)
{
  if (narg != 2)
    error->all(FLERR, "Pair style so3lr/adapter requires SR and LR cutoffs");
  short_range_cutoff_ = utils::numeric(FLERR, arg[0], false, lmp);
  long_range_cutoff_ = utils::numeric(FLERR, arg[1], false, lmp);
  if (short_range_cutoff_ <= 0.0 ||
      long_range_cutoff_ < short_range_cutoff_)
    error->all(FLERR, "Invalid so3lr/adapter SR/LR cutoffs");
}

void PairSO3LRAdapter::coeff(int narg, char **arg)
{
  if (!allocated) allocate();
  if (narg != atom->ntypes + 2 || std::strcmp(arg[0], "*") != 0 ||
      std::strcmp(arg[1], "*") != 0)
    error->all(FLERR,
               "Pair coeff for so3lr/adapter must be '* * Z1 ... Zntypes'");
  type_to_atomic_number_.assign(static_cast<std::size_t>(atom->ntypes) + 1, 0);
  for (int type = 1; type <= atom->ntypes; ++type) {
    const int z = utils::inumeric(FLERR, arg[type + 1], false, lmp);
    if (z <= 0) error->all(FLERR, "SO3LR atomic numbers must be positive");
    type_to_atomic_number_[static_cast<std::size_t>(type)] = z;
  }
  for (int i = 1; i <= atom->ntypes; ++i)
    for (int j = i; j <= atom->ntypes; ++j) setflag[i][j] = 1;
}

double PairSO3LRAdapter::init_one(int i, int j)
{
  if (!setflag[i][j])
    error->all(FLERR, "All so3lr/adapter coefficients are not set");
  return long_range_cutoff_;
}

void PairSO3LRAdapter::init_style()
{
  if (!force->newton_pair)
    error->all(FLERR, "Pair style so3lr/adapter requires newton pair on");
  if (type_to_atomic_number_.size() !=
      static_cast<std::size_t>(atom->ntypes) + 1)
    error->all(FLERR, "Pair style so3lr/adapter coefficients are missing");
  auto *sr = neighbor->add_request(this, NeighConst::REQ_FULL);
  sr->set_id(0);
  sr->set_cutoff(short_range_cutoff_);
  auto *lr = neighbor->add_request(this);
  lr->set_id(1);
  lr->set_cutoff(long_range_cutoff_);
}

void PairSO3LRAdapter::init_list(int which, NeighList *ptr)
{
  if (which == 0)
    short_range_list_ = ptr;
  else if (which == 1)
    long_range_list_ = ptr;
  else
    error->all(FLERR, "Unexpected so3lr/adapter neighbor-list ID");
}

void PairSO3LRAdapter::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);
  if (short_range_list_ == nullptr || long_range_list_ == nullptr)
    error->all(FLERR, "SO3LR live neighbor lists were not initialized");
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
    const auto sr_list = snapshot_neighbor_list(short_range_list_);
    const auto lr_list = snapshot_neighbor_list(long_range_list_);
    const auto graphs = so3lr::build_lammps_native_graphs(
        atoms, sr_list, lr_list, type_to_atomic_number_, short_range_cutoff_,
        long_range_cutoff_, force->newton_pair != 0);
    const auto ownership = audit_ownership(
        atoms, graphs.short_range, graphs.long_range,
        type_to_atomic_number_, comm->me, comm->nprocs, world);

    const auto owned_hash = owned_layout_signature(atoms);
    const auto sr_list_hash = neighbor_signature(sr_list, atoms);
    const auto lr_list_hash = neighbor_signature(lr_list, atoms);
    const auto sr_graph_hash = graph_signature(graphs.short_range);
    const auto lr_graph_hash = graph_signature(graphs.long_range);
    const int local_reuse[5] = {
        calls_ > 0 && owned_hash == previous_owned_layout_,
        calls_ > 0 && sr_list_hash == previous_sr_list_topology_,
        calls_ > 0 && lr_list_hash == previous_lr_list_topology_,
        calls_ > 0 && sr_graph_hash == previous_sr_active_topology_,
        calls_ > 0 && lr_graph_hash == previous_lr_active_topology_};
    int global_reuse[5] = {};
    MPI_Allreduce(local_reuse, global_reuse, 5, MPI_INT, MPI_MIN, world);

    long long local[10] = {
        static_cast<long long>(atom->nlocal),
        static_cast<long long>(atom->nghost),
        static_cast<long long>(graphs.short_range.interactions()),
        static_cast<long long>(graphs.long_range.interactions()),
        static_cast<long long>(graphs.short_range.ghost_local.size()),
        static_cast<long long>(graphs.long_range.ghost_local.size()),
        ownership.sr_remote_ghost_rows,
        ownership.sr_periodic_image_rows,
        ownership.lr_remote_ghost_rows,
        ownership.lr_periodic_image_rows};
    long long global[10] = {};
    MPI_Allreduce(local, global, 10, MPI_LONG_LONG, MPI_SUM, world);

    ++calls_;
    const auto report_rank = [&](FILE *stream) {
      if (stream == nullptr) return;
      std::fprintf(
          stream,
          "SO3LR_MPI_PREFLIGHT_RANK call=%zu rank=%d ranks=%d nlocal=%lld "
          "nghost=%lld sr_edges=%lld lr_pairs=%lld sr_ghost_rows=%lld "
          "lr_ghost_rows=%lld sr_remote_ghost_rows=%lld "
          "sr_periodic_image_rows=%lld lr_remote_ghost_rows=%lld "
          "lr_periodic_image_rows=%lld sr_owner_peers=%lld lr_owner_peers=%lld\n",
          calls_, comm->me, comm->nprocs, local[0], local[1], local[2],
          local[3], local[4], local[5], local[6], local[7], local[8], local[9],
          ownership.sr_owner_peers, ownership.lr_owner_peers);
    };
    report_rank(screen);
    if (logfile != screen) report_rank(logfile);
    if (comm->me == 0) {
      const auto report_global = [&](FILE *stream) {
        if (stream == nullptr) return;
        std::fprintf(
            stream,
            "SO3LR_MPI_PREFLIGHT_GLOBAL call=%zu ranks=%d owned_atoms=%lld "
            "atom_ghost_rows=%lld sr_edges=%lld lr_pairs=%lld "
            "sr_compact_ghost_rows=%lld lr_compact_ghost_rows=%lld "
            "sr_remote_ghost_rows=%lld sr_periodic_image_rows=%lld "
            "lr_remote_ghost_rows=%lld lr_periodic_image_rows=%lld "
            "owned_layout_reused=%d sr_list_topology_reused=%d "
            "lr_list_topology_reused=%d sr_active_topology_reused=%d "
            "lr_active_topology_reused=%d\n",
            calls_, comm->nprocs, global[0], ownership.global_atom_ghost_rows,
            global[2], global[3], global[4], global[5], global[6], global[7],
            global[8], global[9], global_reuse[0], global_reuse[1],
            global_reuse[2], global_reuse[3], global_reuse[4]);
        if (calls_ == 2 && global_reuse[0] && global_reuse[1] &&
            global_reuse[2] && global_reuse[3] && global_reuse[4]) {
          std::fprintf(stream, "SO3LR_PERSISTENT_GRAPH_CANDIDATE=PASS\n");
          std::fprintf(stream, "SO3LR_MULTI_GPU_PREFLIGHT=PASS\n");
        }
      };
      report_global(screen);
      if (logfile != screen) report_global(logfile);
    }

    previous_owned_layout_ = owned_hash;
    previous_sr_list_topology_ = sr_list_hash;
    previous_lr_list_topology_ = lr_list_hash;
    previous_sr_active_topology_ = sr_graph_hash;
    previous_lr_active_topology_ = lr_graph_hash;
  } catch (const std::exception &exc) {
    error->all(FLERR, "SO3LR live MPI preflight failed: {}", exc.what());
  }
}
