#include "so3lr/lammps_neighbor_adapter.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace so3lr {
namespace {

void validate(const LammpsAtomSnapshot &atoms,
              const LammpsNeighborSnapshot &list,
              const std::vector<std::int64_t> &type_to_z) {
  const std::size_t rows = atoms.tags.size();
  if (atoms.nlocal == 0 || atoms.nlocal > rows || atoms.types.size() != rows ||
      atoms.positions.size() != rows * 3 || type_to_z.size() < 2)
    throw std::runtime_error("SO3LR invalid LAMMPS atom snapshot");
  if (list.offsets.size() != list.ilist.size() + 1 ||
      list.offsets.empty() || list.offsets.front() != 0 ||
      list.offsets.back() != list.neighbors.size())
    throw std::runtime_error("SO3LR invalid LAMMPS neighbour CSR");
  for (std::size_t i = 1; i < list.offsets.size(); ++i)
    if (list.offsets[i] < list.offsets[i - 1])
      throw std::runtime_error("SO3LR decreasing neighbour offset");
  for (const auto row : list.ilist)
    if (row >= atoms.nlocal)
      throw std::runtime_error("SO3LR neighbour receiver is not owned");
  for (const auto row : list.neighbors)
    if (row >= rows)
      throw std::runtime_error("SO3LR neighbour row is out of range");
  for (const auto type : atoms.types)
    if (type <= 0 || static_cast<std::size_t>(type) >= type_to_z.size() ||
        type_to_z[static_cast<std::size_t>(type)] <= 0)
      throw std::runtime_error("SO3LR LAMMPS type-to-element map is invalid");
}

LammpsCompactGraph compact_graph(
    const LammpsAtomSnapshot &atoms, const LammpsNeighborSnapshot &list,
    const std::vector<std::int64_t> &type_to_z, double cutoff,
    bool require_unique_half_pairs) {
  if (!std::isfinite(cutoff) || cutoff <= 0.0)
    throw std::runtime_error("SO3LR invalid neighbour cutoff");
  LammpsCompactGraph graph;
  const std::size_t rows = atoms.tags.size();
  const std::size_t entries = list.neighbors.size();
  graph.source_atom_rows.reserve(rows);
  graph.owned_local.reserve(atoms.nlocal);
  graph.senders.reserve(entries);
  graph.receivers.reserve(entries);
  graph.vectors.reserve(entries * 3);
  graph.original_neighbor_entries.reserve(entries);
  const std::size_t missing = std::numeric_limits<std::size_t>::max();
  // Atom rows are already a dense LAMMPS index space.  A dense row-to-node
  // table avoids millions of unordered_map probes and node allocations.
  std::vector<std::size_t> compact(rows, missing);
  for (std::size_t row = 0; row < atoms.nlocal; ++row) {
    compact[row] = row;
    graph.source_atom_rows.push_back(row);
    graph.owned_local.push_back(row);
  }

  std::unordered_set<std::uint64_t> seen;
  if (require_unique_half_pairs) seen.reserve(entries);
  const double cutoff2 = cutoff * cutoff;
  for (std::size_t ii = 0; ii < list.ilist.size(); ++ii) {
    const auto receiver_row = list.ilist[ii];
    for (std::size_t entry = list.offsets[ii]; entry < list.offsets[ii + 1];
         ++entry) {
      const auto sender_row = list.neighbors[entry];
      if (sender_row == receiver_row) continue;
      const double dx = atoms.positions[receiver_row * 3] -
                        atoms.positions[sender_row * 3];
      const double dy = atoms.positions[receiver_row * 3 + 1] -
                        atoms.positions[sender_row * 3 + 1];
      const double dz = atoms.positions[receiver_row * 3 + 2] -
                        atoms.positions[sender_row * 3 + 2];
      const double r2 = dx * dx + dy * dy + dz * dz;
      if (!(r2 > 0.0) || r2 >= cutoff2) continue;
      if (require_unique_half_pairs) {
        const auto low = std::min(receiver_row, sender_row);
        const auto high = std::max(receiver_row, sender_row);
        const std::uint64_t key =
            (static_cast<std::uint64_t>(low) << 32) |
            static_cast<std::uint64_t>(high);
        if (!seen.insert(key).second)
          throw std::runtime_error("SO3LR duplicate LR half-pair entry");
      }
      std::size_t sender = compact[sender_row];
      if (sender == missing) {
        sender = graph.source_atom_rows.size();
        compact[sender_row] = sender;
        graph.source_atom_rows.push_back(sender_row);
      }
      graph.senders.push_back(sender);
      graph.receivers.push_back(receiver_row);
      graph.vectors.push_back(dx);
      graph.vectors.push_back(dy);
      graph.vectors.push_back(dz);
      graph.original_neighbor_entries.push_back(entry);
    }
  }
  if (graph.interactions() == 0)
    throw std::runtime_error("SO3LR compact neighbour graph is empty");
  graph.tags.reserve(graph.nodes());
  graph.atomic_numbers.reserve(graph.nodes());
  for (const auto row : graph.source_atom_rows) {
    graph.tags.push_back(atoms.tags[row]);
    graph.atomic_numbers.push_back(
        type_to_z[static_cast<std::size_t>(atoms.types[row])]);
  }
  for (std::size_t local = atoms.nlocal; local < graph.nodes(); ++local)
    graph.ghost_local.push_back(local);
  return graph;
}

// SO3LR_STAGE3_OPTIM_DEV7_TOPOLOGY_ONLY_FASTPATH
// LAMMPS has already applied the requested cutoff plus neighbor skin and has
// already imposed full/half list ownership.  The candidate cache therefore
// does not need to recompute distances, hash every LR pair for uniqueness, or
// retain host vectors that the device active-graph path immediately replaces.
LammpsCompactGraph compact_candidate_topology(
    const LammpsAtomSnapshot &atoms, const LammpsNeighborSnapshot &list,
    const std::vector<std::int64_t> &type_to_z) {
  LammpsCompactGraph graph;
  const std::size_t rows = atoms.tags.size();
  const std::size_t entries = list.neighbors.size();
  graph.source_atom_rows.reserve(rows);
  graph.owned_local.reserve(atoms.nlocal);
  graph.senders.reserve(entries);
  graph.receivers.reserve(entries);
  const std::size_t missing = std::numeric_limits<std::size_t>::max();
  std::vector<std::size_t> compact(rows, missing);
  for (std::size_t row = 0; row < atoms.nlocal; ++row) {
    compact[row] = row;
    graph.source_atom_rows.push_back(row);
    graph.owned_local.push_back(row);
  }

  for (std::size_t ii = 0; ii < list.ilist.size(); ++ii) {
    const std::size_t receiver_row = list.ilist[ii];
    for (std::size_t entry = list.offsets[ii]; entry < list.offsets[ii + 1];
         ++entry) {
      const std::size_t sender_row = list.neighbors[entry];
      if (sender_row == receiver_row) continue;
      std::size_t sender = compact[sender_row];
      if (sender == missing) {
        sender = graph.source_atom_rows.size();
        compact[sender_row] = sender;
        graph.source_atom_rows.push_back(sender_row);
      }
      graph.senders.push_back(sender);
      // Owned LAMMPS rows are inserted first and retain their row index.
      graph.receivers.push_back(receiver_row);
    }
  }
  if (graph.interactions() == 0)
    throw std::runtime_error("SO3LR candidate topology is empty");
  graph.tags.reserve(graph.nodes());
  graph.atomic_numbers.reserve(graph.nodes());
  for (const auto row : graph.source_atom_rows) {
    graph.tags.push_back(atoms.tags[row]);
    graph.atomic_numbers.push_back(
        type_to_z[static_cast<std::size_t>(atoms.types[row])]);
  }
  for (std::size_t local = atoms.nlocal; local < graph.nodes(); ++local)
    graph.ghost_local.push_back(local);
  return graph;
}

}  // namespace

LammpsNativeGraphs build_lammps_native_graphs(
    const LammpsAtomSnapshot &atoms,
    const LammpsNeighborSnapshot &full_short_range,
    const LammpsNeighborSnapshot &half_long_range,
    const std::vector<std::int64_t> &type_to_atomic_number,
    double short_range_cutoff, double long_range_cutoff,
    bool newton_pair) {
  validate(atoms, full_short_range, type_to_atomic_number);
  validate(atoms, half_long_range, type_to_atomic_number);
  if (!newton_pair)
    throw std::runtime_error(
        "SO3LR native LR requires LAMMPS newton pair on");
  if (!(long_range_cutoff >= short_range_cutoff))
    throw std::runtime_error("SO3LR LR cutoff is smaller than SR cutoff");
  return {
      compact_graph(atoms, full_short_range, type_to_atomic_number,
                    short_range_cutoff, false),
      compact_graph(atoms, half_long_range, type_to_atomic_number,
                    long_range_cutoff, true),
  };
}

LammpsNativeGraphs build_lammps_native_candidate_topologies(
    const LammpsAtomSnapshot &atoms,
    const LammpsNeighborSnapshot &full_short_range,
    const LammpsNeighborSnapshot &half_long_range,
    const std::vector<std::int64_t> &type_to_atomic_number,
    bool newton_pair) {
  validate(atoms, full_short_range, type_to_atomic_number);
  validate(atoms, half_long_range, type_to_atomic_number);
  if (!newton_pair)
    throw std::runtime_error(
        "SO3LR native LR requires LAMMPS newton pair on");
  return {
      compact_candidate_topology(atoms, full_short_range,
                                 type_to_atomic_number),
      compact_candidate_topology(atoms, half_long_range,
                                 type_to_atomic_number),
  };
}

}  // namespace so3lr
