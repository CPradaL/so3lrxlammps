#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace so3lr {

// Header-independent snapshots of the LAMMPS atom and neighbour-list data
// consumed by the future pair style.  Atom rows [0,nlocal) are owned; later
// rows are LAMMPS ghosts and may include repeated tags for periodic images.
struct LammpsAtomSnapshot {
  std::size_t nlocal = 0;
  std::vector<std::int64_t> tags;
  std::vector<int> types;
  std::vector<double> positions;
};

// CSR form of a LAMMPS neighbour list. ilist entries are receiver rows;
// offsets delimit their neighbour rows in neighbors. Entries must already be
// stripped of LAMMPS neighbour-bit flags by the caller.
struct LammpsNeighborSnapshot {
  std::vector<std::size_t> ilist;
  std::vector<std::size_t> offsets;
  std::vector<std::size_t> neighbors;
};

struct LammpsCompactGraph {
  std::vector<std::size_t> source_atom_rows;
  std::vector<std::int64_t> tags;
  std::vector<std::int64_t> atomic_numbers;
  std::vector<std::size_t> owned_local;
  std::vector<std::size_t> ghost_local;
  std::vector<double> vectors;
  std::vector<std::size_t> senders;
  std::vector<std::size_t> receivers;
  std::vector<std::size_t> original_neighbor_entries;

  std::size_t nodes() const { return source_atom_rows.size(); }
  std::size_t interactions() const { return senders.size(); }
};

struct LammpsNativeGraphs {
  LammpsCompactGraph short_range;
  LammpsCompactGraph long_range;
};

LammpsNativeGraphs build_lammps_native_graphs(
    const LammpsAtomSnapshot &atoms,
    const LammpsNeighborSnapshot &full_short_range,
    const LammpsNeighborSnapshot &half_long_range,
    const std::vector<std::int64_t> &type_to_atomic_number,
    double short_range_cutoff, double long_range_cutoff,
    bool newton_pair);

// SO3LR_STAGE3_OPTIM_DEV7_TOPOLOGY_ONLY_FASTPATH
// The caller guarantees that the two snapshots came directly from explicit
// LAMMPS skin-expanded full-SR and half-LR requests.  Candidate topology only
// needs compact node/end-point maps; exact-cutoff vectors are rebuilt on the
// device every force call.
LammpsNativeGraphs build_lammps_native_candidate_topologies(
    const LammpsAtomSnapshot &atoms,
    const LammpsNeighborSnapshot &full_short_range,
    const LammpsNeighborSnapshot &half_long_range,
    const std::vector<std::int64_t> &type_to_atomic_number,
    bool newton_pair);

}  // namespace so3lr
