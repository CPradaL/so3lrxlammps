#include "pair_so3lr_native_mpi.h"
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
#include "update.h"
#include "utils.h"

#include "so3lr/kokkos_node_feature_exchange.hpp"
#include "so3lr/kokkos_physical_long_range_exchange.hpp"
#include "so3lr/so3lr_repulsion_setup.hpp"
#include "so3lr/lammps_neighbor_adapter.hpp"

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace LAMMPS_NS;

namespace {

using DoubleView = Kokkos::View<double *>;
using Int64View = Kokkos::View<std::int64_t *>;
using IndexView = Kokkos::View<std::size_t *>;
using HostDoubleView = Kokkos::View<double *, Kokkos::HostSpace>;
using HostInt64View =
    Kokkos::View<std::int64_t *, Kokkos::HostSpace>;
using HostIndexView =
    Kokkos::View<std::size_t *, Kokkos::HostSpace>;

bool environment_enabled(const char *name) {
  const char *value = std::getenv(name);
  return value != nullptr && value[0] != '\0' &&
         std::strcmp(value, "0") != 0;
}

std::size_t environment_interval(const char *name) {
  const char *value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') return 1;
  const auto parsed =
      static_cast<std::size_t>(std::strtoull(value, nullptr, 10));
  return parsed == 0 ? 1 : parsed;
}

void mpi_require(int status, const char *operation) {
  if (status == MPI_SUCCESS) return;
  char message[MPI_MAX_ERROR_STRING]{};
  int length = 0;
  MPI_Error_string(status, message, &length);
  throw std::runtime_error(std::string(operation) + ": " +
                           std::string(message, static_cast<std::size_t>(length)));
}

so3lr::LammpsNeighborSnapshot snapshot_neighbor_list(NeighList *list) {
  if (list == nullptr) throw std::runtime_error("missing LAMMPS neighbor list");
  so3lr::LammpsNeighborSnapshot result;
  const std::size_t receivers = static_cast<std::size_t>(list->inum);
  std::size_t entries = 0;
  for (int ii = 0; ii < list->inum; ++ii) {
    const int count = list->numneigh[list->ilist[ii]];
    if (count < 0 || entries > std::numeric_limits<std::size_t>::max() -
                                   static_cast<std::size_t>(count))
      throw std::runtime_error("SO3LR neighbor-list size overflow");
    entries += static_cast<std::size_t>(count);
  }
  // dev_5: size the CSR snapshot once.  The previous push-back growth copied
  // millions of LR entries during every LAMMPS reneighbor event.
  result.ilist.reserve(receivers);
  result.offsets.reserve(receivers + 1);
  result.neighbors.reserve(entries);
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

template <class View, class Values>
void copy_to_device(const View &device, const Values &values) {
  if (device.extent(0) != values.size())
    throw std::runtime_error("SO3LR host/device size mismatch");
  auto host = Kokkos::create_mirror_view(device);
  for (std::size_t i = 0; i < values.size(); ++i) host(i) = values[i];
  Kokkos::deep_copy(device, host);
}


std::size_t retained_capacity(std::size_t current, std::size_t required,
                              std::size_t minimum_slack) {
  if (current >= required) return current;
  if (required > std::numeric_limits<std::size_t>::max() - minimum_slack)
    throw std::runtime_error("SO3LR workspace capacity overflow");
  if (current == 0) {
    const std::size_t proportional = required / 100;
    const std::size_t slack = std::max(minimum_slack, proportional);
    if (required > std::numeric_limits<std::size_t>::max() - slack)
      throw std::runtime_error("SO3LR workspace capacity overflow");
    return required + slack;
  }
  std::size_t capacity = current;
  while (capacity < required) {
    const std::size_t increment =
        std::max(minimum_slack, std::max<std::size_t>(capacity / 4, 1));
    if (capacity > std::numeric_limits<std::size_t>::max() - increment)
      throw std::runtime_error("SO3LR workspace capacity overflow");
    capacity += increment;
  }
  return capacity;
}

void upload_padded_graph(
    const Int64View &device_z, const DoubleView &device_vectors,
    const IndexView &device_senders, const IndexView &device_receivers,
    const HostInt64View &host_z, const HostDoubleView &host_vectors,
    const HostIndexView &host_senders, const HostIndexView &host_receivers,
    const std::vector<std::int64_t> &atomic_numbers,
    const std::vector<double> &vectors,
    const std::vector<std::size_t> &senders,
    const std::vector<std::size_t> &receivers, double cutoff) {
  const std::size_t node_capacity = device_z.extent(0);
  const std::size_t edge_capacity = device_senders.extent(0);
  const std::size_t nodes = atomic_numbers.size();
  const std::size_t edges = senders.size();
  if (nodes == 0 || nodes >= node_capacity || edges > edge_capacity ||
      vectors.size() != edges * 3 || receivers.size() != edges ||
      device_vectors.extent(0) != edge_capacity * 3 ||
      device_receivers.extent(0) != edge_capacity ||
      host_z.extent(0) != node_capacity ||
      host_vectors.extent(0) != edge_capacity * 3 ||
      host_senders.extent(0) != edge_capacity ||
      host_receivers.extent(0) != edge_capacity || !(cutoff > 0.0))
    throw std::runtime_error("SO3LR padded graph capacity mismatch");

  // Capacity-only rows form a disconnected H-only dummy tail.  Their
  // zero-contribution self topology is represented by vectors beyond the
  // model cutoff, so every cutoff-weighted SR/LR contribution is zero.
  for (std::size_t node = 0; node < node_capacity; ++node) host_z(node) = 1;
  for (std::size_t node = 0; node < nodes; ++node)
    host_z(node) = atomic_numbers[node];
  const std::size_t dummy_nodes = node_capacity - nodes;
  for (std::size_t edge = 0; edge < edge_capacity; ++edge) {
    const std::size_t dummy = nodes + ((edge - std::min(edge, edges)) % dummy_nodes);
    host_senders(edge) = dummy;
    host_receivers(edge) = dummy;
    host_vectors(edge * 3) = cutoff + 1.0;
    host_vectors(edge * 3 + 1) = 0.0;
    host_vectors(edge * 3 + 2) = 0.0;
  }
  for (std::size_t edge = 0; edge < edges; ++edge) {
    if (senders[edge] >= nodes || receivers[edge] >= nodes)
      throw std::runtime_error("SO3LR logical graph index exceeds node count");
    host_senders(edge) = senders[edge];
    host_receivers(edge) = receivers[edge];
    host_vectors(edge * 3) = vectors[edge * 3];
    host_vectors(edge * 3 + 1) = vectors[edge * 3 + 1];
    host_vectors(edge * 3 + 2) = vectors[edge * 3 + 2];
  }
  Kokkos::deep_copy(device_z, host_z);
  Kokkos::deep_copy(device_vectors, host_vectors);
  Kokkos::deep_copy(device_senders, host_senders);
  Kokkos::deep_copy(device_receivers, host_receivers);
}

// SO3LR_STAGE3_OPTIM_DEV8_PERSISTENT_DIRECT_CANDIDATE_UPLOAD
// Candidate topology changes only on LAMMPS neighbor rebuilds.  Retain
// capacity-managed device arrays across rebuilds and deep-copy directly from
// the contiguous std::vector storage, avoiding six fresh device allocations,
// six temporary host mirrors, and their element-wise staging loops.
struct DeviceCandidateGraph {
  IndexView source_atom_rows;
  IndexView senders;
  IndexView receivers;
  std::size_t source_rows = 0;
  std::size_t edges = 0;
};

void upload_device_candidate_graph(
    DeviceCandidateGraph &result,
    const so3lr::LammpsCompactGraph &candidate, const char *prefix) {
  if (candidate.nodes() == 0 || candidate.interactions() == 0 ||
      candidate.source_atom_rows.size() != candidate.nodes() ||
      candidate.senders.size() != candidate.receivers.size())
    throw std::runtime_error("SO3LR invalid persistent candidate graph");
  for (std::size_t edge = 0; edge < candidate.interactions(); ++edge)
    if (candidate.senders[edge] >= candidate.nodes() ||
        candidate.receivers[edge] >= candidate.nodes())
      throw std::runtime_error("SO3LR persistent candidate endpoint invalid");

  const auto ensure_capacity = [prefix](IndexView &view,
                                        std::size_t required,
                                        const char *suffix) {
    if (view.extent(0) >= required) return;
    const std::size_t capacity =
        retained_capacity(view.extent(0), required, 1024);
    view = IndexView(std::string(prefix) + suffix, capacity);
  };
  ensure_capacity(result.source_atom_rows, candidate.nodes(), "_source_rows");
  ensure_capacity(result.senders, candidate.interactions(), "_senders");
  ensure_capacity(result.receivers, candidate.interactions(), "_receivers");

  using UnmanagedHostIndices = Kokkos::View<
      const std::size_t *, Kokkos::HostSpace,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  const UnmanagedHostIndices host_source_rows(
      candidate.source_atom_rows.data(), candidate.nodes());
  const UnmanagedHostIndices host_senders(
      candidate.senders.data(), candidate.interactions());
  const UnmanagedHostIndices host_receivers(
      candidate.receivers.data(), candidate.interactions());
  Kokkos::deep_copy(
      Kokkos::subview(result.source_atom_rows,
                      std::pair<std::size_t, std::size_t>(0,
                                                         candidate.nodes())),
      host_source_rows);
  Kokkos::deep_copy(
      Kokkos::subview(result.senders,
                      std::pair<std::size_t, std::size_t>(
                          0, candidate.interactions())),
      host_senders);
  Kokkos::deep_copy(
      Kokkos::subview(result.receivers,
                      std::pair<std::size_t, std::size_t>(
                          0, candidate.interactions())),
      host_receivers);
  result.source_rows = candidate.nodes();
  result.edges = candidate.interactions();
}

std::size_t count_active_device_edges(
    const DeviceCandidateGraph &candidate, const DoubleView &positions,
    double cutoff, const char *label) {
  if (!(cutoff > 0.0) || candidate.edges == 0 ||
      candidate.source_rows == 0 ||
      candidate.source_atom_rows.extent(0) < candidate.source_rows ||
      candidate.senders.extent(0) < candidate.edges ||
      candidate.receivers.extent(0) < candidate.edges)
    throw std::runtime_error("SO3LR invalid persistent active-edge input");
  const auto source_rows = candidate.source_atom_rows;
  const auto senders = candidate.senders;
  const auto receivers = candidate.receivers;
  const std::size_t edges = candidate.edges;
  const double cutoff2 = cutoff * cutoff;
  std::size_t active = 0;
  Kokkos::parallel_reduce(
      label, Kokkos::RangePolicy<>(0, edges),
      KOKKOS_LAMBDA(const std::size_t edge, std::size_t &count) {
        const std::size_t sender_row = source_rows(senders(edge));
        const std::size_t receiver_row = source_rows(receivers(edge));
        const double dx = positions(receiver_row * 3) -
                          positions(sender_row * 3);
        const double dy = positions(receiver_row * 3 + 1) -
                          positions(sender_row * 3 + 1);
        const double dz = positions(receiver_row * 3 + 2) -
                          positions(sender_row * 3 + 2);
        const double r2 = dx * dx + dy * dy + dz * dz;
        if (r2 > 0.0 && r2 < cutoff2) ++count;
      },
      active);
  if (active == 0)
    throw std::runtime_error("SO3LR persistent device active graph is empty");
  return active;
}

void upload_padded_atomic_numbers(
    const Int64View &device_z, const HostInt64View &host_z,
    const std::vector<std::int64_t> &atomic_numbers) {
  if (atomic_numbers.empty() || atomic_numbers.size() >= device_z.extent(0) ||
      host_z.extent(0) != device_z.extent(0))
    throw std::runtime_error("SO3LR padded atomic-number capacity mismatch");
  for (std::size_t node = 0; node < host_z.extent(0); ++node)
    host_z(node) = 1;
  for (std::size_t node = 0; node < atomic_numbers.size(); ++node)
    host_z(node) = atomic_numbers[node];
  Kokkos::deep_copy(device_z, host_z);
}

void compact_active_device_graph(
    const DeviceCandidateGraph &candidate, const DoubleView &positions,
    double cutoff, std::size_t active_edges, std::size_t logical_nodes,
    const DoubleView &vectors, const IndexView &senders,
    const IndexView &receivers, const char *prefix) {
  const std::size_t node_capacity =
      logical_nodes + 1;  // caller guarantees at least one dummy node
  const std::size_t edge_capacity = senders.extent(0);
  if (!(cutoff > 0.0) || active_edges == 0 ||
      logical_nodes == 0 || active_edges > edge_capacity ||
      vectors.extent(0) != edge_capacity * 3 ||
      receivers.extent(0) != edge_capacity)
    throw std::runtime_error("SO3LR device compact graph capacity mismatch");

  const std::size_t dummy = logical_nodes;
  const std::string fill_label = std::string(prefix) + "_fill_dummy_tail";
  Kokkos::parallel_for(
      fill_label, Kokkos::RangePolicy<>(0, edge_capacity),
      KOKKOS_LAMBDA(const std::size_t edge) {
        senders(edge) = dummy;
        receivers(edge) = dummy;
        vectors(edge * 3) = cutoff + 1.0;
        vectors(edge * 3 + 1) = 0.0;
        vectors(edge * 3 + 2) = 0.0;
      });

  if (logical_nodes != candidate.source_rows ||
      candidate.source_atom_rows.extent(0) < candidate.source_rows ||
      candidate.senders.extent(0) < candidate.edges ||
      candidate.receivers.extent(0) < candidate.edges)
    throw std::runtime_error("SO3LR persistent compact topology mismatch");
  const auto source_rows = candidate.source_atom_rows;
  const auto candidate_senders = candidate.senders;
  const auto candidate_receivers = candidate.receivers;
  const double cutoff2 = cutoff * cutoff;
  const std::string compact_label = std::string(prefix) + "_compact_active";
  std::size_t compacted = 0;
  Kokkos::parallel_scan(
      compact_label, Kokkos::RangePolicy<>(0, candidate.edges),
      KOKKOS_LAMBDA(const std::size_t edge, std::size_t &offset,
                    const bool final) {
        const std::size_t sender = candidate_senders(edge);
        const std::size_t receiver = candidate_receivers(edge);
        const std::size_t sender_row = source_rows(sender);
        const std::size_t receiver_row = source_rows(receiver);
        const double dx = positions(receiver_row * 3) -
                          positions(sender_row * 3);
        const double dy = positions(receiver_row * 3 + 1) -
                          positions(sender_row * 3 + 1);
        const double dz = positions(receiver_row * 3 + 2) -
                          positions(sender_row * 3 + 2);
        const double r2 = dx * dx + dy * dy + dz * dz;
        if (r2 > 0.0 && r2 < cutoff2) {
          if (final) {
            senders(offset) = sender;
            receivers(offset) = receiver;
            vectors(offset * 3) = dx;
            vectors(offset * 3 + 1) = dy;
            vectors(offset * 3 + 2) = dz;
          }
          ++offset;
        }
      },
      compacted);
  if (compacted != active_edges)
    throw std::runtime_error("SO3LR device active-edge count changed");
  (void)node_capacity;
}

void initialize_active_graph_nodes(
    const so3lr::LammpsCompactGraph &candidate,
    so3lr::LammpsCompactGraph &active) {
  active.source_atom_rows = candidate.source_atom_rows;
  active.tags = candidate.tags;
  active.atomic_numbers = candidate.atomic_numbers;
  active.owned_local = candidate.owned_local;
  active.ghost_local = candidate.ghost_local;
  active.vectors.clear();
  active.senders.clear();
  active.receivers.clear();
  active.original_neighbor_entries.clear();
  active.vectors.reserve(candidate.interactions() * 3);
  active.senders.reserve(candidate.interactions());
  active.receivers.reserve(candidate.interactions());
  active.original_neighbor_entries.reserve(candidate.interactions());
}

void refresh_active_graph(
    const so3lr::LammpsCompactGraph &candidate,
    const so3lr::LammpsAtomSnapshot &atoms, double cutoff,
    so3lr::LammpsCompactGraph &active) {
  if (!(cutoff > 0.0) || candidate.nodes() == 0 ||
      candidate.interactions() == 0 ||
      candidate.senders.size() != candidate.receivers.size() ||
      candidate.senders.size() != candidate.original_neighbor_entries.size() ||
      active.source_atom_rows != candidate.source_atom_rows ||
      active.tags != candidate.tags ||
      active.atomic_numbers != candidate.atomic_numbers)
    throw std::runtime_error("SO3LR cached topology contract changed");
  active.vectors.clear();
  active.senders.clear();
  active.receivers.clear();
  active.original_neighbor_entries.clear();
  const double cutoff2 = cutoff * cutoff;
  for (std::size_t edge = 0; edge < candidate.interactions(); ++edge) {
    const std::size_t sender = candidate.senders[edge];
    const std::size_t receiver = candidate.receivers[edge];
    if (sender >= candidate.nodes() || receiver >= candidate.nodes())
      throw std::runtime_error("SO3LR cached topology endpoint is invalid");
    const std::size_t sender_row = candidate.source_atom_rows[sender];
    const std::size_t receiver_row = candidate.source_atom_rows[receiver];
    if (sender_row >= atoms.tags.size() || receiver_row >= atoms.tags.size())
      throw std::runtime_error("SO3LR cached atom row is no longer valid");
    const double dx = atoms.positions[receiver_row * 3] -
                      atoms.positions[sender_row * 3];
    const double dy = atoms.positions[receiver_row * 3 + 1] -
                      atoms.positions[sender_row * 3 + 1];
    const double dz = atoms.positions[receiver_row * 3 + 2] -
                      atoms.positions[sender_row * 3 + 2];
    const double r2 = dx * dx + dy * dy + dz * dz;
    if (!(r2 > 0.0) || r2 >= cutoff2) continue;
    active.senders.push_back(sender);
    active.receivers.push_back(receiver);
    active.vectors.insert(active.vectors.end(), {dx, dy, dz});
    active.original_neighbor_entries.push_back(
        candidate.original_neighbor_entries[edge]);
  }
  if (active.interactions() == 0)
    throw std::runtime_error("SO3LR active cached graph is empty");
}

IndexView device_indices(const std::vector<std::size_t> &values,
                         const char *label) {
  IndexView result(std::string(label), values.size());
  copy_to_device(result, values);
  return result;
}

struct OwnerRecord {
  int rank = -1;
  std::int64_t atomic_number = 0;
};

struct GlobalOwnership {
  std::unordered_map<std::int64_t, OwnerRecord> owner_by_tag;
  std::unordered_map<std::int64_t, std::size_t> local_by_tag;
  std::size_t global_nodes = 0;
};

GlobalOwnership gather_global_ownership(
    const so3lr::LammpsCompactGraph &sr_graph, std::size_t nlocal,
    MPI_Comm world) {
  int ranks = 0;
  int rank = -1;
  mpi_require(MPI_Comm_size(world, &ranks), "MPI_Comm_size");
  mpi_require(MPI_Comm_rank(world, &rank), "MPI_Comm_rank");
  if (sr_graph.nodes() < nlocal || sr_graph.owned_local.size() != nlocal)
    throw std::runtime_error("invalid SR owned-node prefix");
  if (nlocal > static_cast<std::size_t>(INT_MAX))
    throw std::runtime_error("local atom count exceeds MPI int range");

  const int local_count = static_cast<int>(nlocal);
  std::vector<int> counts(static_cast<std::size_t>(ranks));
  mpi_require(MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1,
                            MPI_INT, world),
              "MPI_Allgather owned counts");
  std::vector<int> displacements(static_cast<std::size_t>(ranks), 0);
  for (int peer = 1; peer < ranks; ++peer)
    displacements[peer] = displacements[peer - 1] + counts[peer - 1];
  const int total = displacements.back() + counts.back();
  std::vector<long long> local_tags(nlocal);
  std::vector<long long> local_z(nlocal);
  GlobalOwnership ownership;
  ownership.global_nodes = static_cast<std::size_t>(total);
  ownership.local_by_tag.reserve(nlocal);
  for (std::size_t node = 0; node < nlocal; ++node) {
    if (sr_graph.source_atom_rows[node] != node ||
        sr_graph.owned_local[node] != node)
      throw std::runtime_error("SR graph changed owned atom ordering");
    local_tags[node] = static_cast<long long>(sr_graph.tags[node]);
    local_z[node] = static_cast<long long>(sr_graph.atomic_numbers[node]);
    if (!ownership.local_by_tag.emplace(sr_graph.tags[node], node).second)
      throw std::runtime_error("duplicate locally owned atom tag");
  }
  std::vector<long long> global_tags(static_cast<std::size_t>(total));
  std::vector<long long> global_z(static_cast<std::size_t>(total));
  mpi_require(MPI_Allgatherv(local_tags.data(), local_count, MPI_LONG_LONG,
                             global_tags.data(), counts.data(),
                             displacements.data(), MPI_LONG_LONG, world),
              "MPI_Allgatherv owned tags");
  mpi_require(MPI_Allgatherv(local_z.data(), local_count, MPI_LONG_LONG,
                             global_z.data(), counts.data(),
                             displacements.data(), MPI_LONG_LONG, world),
              "MPI_Allgatherv owned elements");
  ownership.owner_by_tag.reserve(static_cast<std::size_t>(total));
  for (int peer = 0; peer < ranks; ++peer) {
    for (int offset = 0; offset < counts[peer]; ++offset) {
      const int flat = displacements[peer] + offset;
      const auto tag = static_cast<std::int64_t>(global_tags[flat]);
      if (!ownership.owner_by_tag
               .emplace(tag, OwnerRecord{peer, global_z[flat]})
               .second)
        throw std::runtime_error("owned atom tag is not globally unique");
    }
  }
  return ownership;
}

struct ExchangePlan {
  IndexView publish;
  IndexView ghosts;
  std::vector<int> publish_counts;
  std::vector<int> publish_displacements;
  std::vector<int> ghost_counts;
  std::vector<int> ghost_displacements;
  std::size_t remote_or_image_nodes = 0;
};


struct ExchangePayloadArena {
  DoubleView send;
  DoubleView receive;
  std::size_t send_capacity = 0;
  std::size_t receive_capacity = 0;
  std::size_t growths = 0;
};

std::size_t payload_capacity(std::size_t current, std::size_t required) {
  if (current != 0 && required <= current) return current;
  std::size_t capacity = std::max<std::size_t>(current, 1024);
  while (capacity < required) {
    const std::size_t increment = std::max<std::size_t>(capacity / 2, 1024);
    if (capacity > std::numeric_limits<std::size_t>::max() - increment)
      throw std::runtime_error("SO3LR MPI payload capacity overflow");
    capacity += increment;
  }
  return capacity;
}

std::pair<DoubleView, DoubleView> acquire_payload_views(
    ExchangePayloadArena &arena, std::size_t send_values,
    std::size_t receive_values) {
  const bool grow_send =
      arena.send_capacity == 0 || send_values > arena.send_capacity;
  const bool grow_receive =
      arena.receive_capacity == 0 ||
      receive_values > arena.receive_capacity;
  // An unpack kernel from the previous sequential exchange may still refer
  // to the old arena.  Fence only on the uncommon growth path before a View
  // reassignment can release that allocation.
  if (grow_send || grow_receive) Kokkos::fence();
  if (grow_send) {
    arena.send_capacity = payload_capacity(arena.send_capacity, send_values);
    arena.send = DoubleView("so3lr_mpi_persistent_send",
                            arena.send_capacity);
    ++arena.growths;
  }
  if (grow_receive) {
    arena.receive_capacity =
        payload_capacity(arena.receive_capacity, receive_values);
    arena.receive = DoubleView("so3lr_mpi_persistent_receive",
                               arena.receive_capacity);
    ++arena.growths;
  }
  auto send_prefix = Kokkos::subview(
      arena.send, std::pair<std::size_t, std::size_t>{0, send_values});
  auto receive_prefix = Kokkos::subview(
      arena.receive,
      std::pair<std::size_t, std::size_t>{0, receive_values});
  DoubleView send = send_prefix;
  DoubleView receive = receive_prefix;
  return {send, receive};
}

std::vector<int> displacements(const std::vector<int> &counts) {
  std::vector<int> result(counts.size(), 0);
  for (std::size_t i = 1; i < counts.size(); ++i)
    result[i] = result[i - 1] + counts[i - 1];
  return result;
}

ExchangePlan build_exchange_plan(
    const so3lr::LammpsCompactGraph &target_graph, std::size_t nlocal,
    const GlobalOwnership &ownership, MPI_Comm world, const char *label) {
  int ranks = 0;
  mpi_require(MPI_Comm_size(world, &ranks), "MPI_Comm_size exchange plan");
  if (target_graph.nodes() < nlocal)
    throw std::runtime_error(std::string(label) + " graph is smaller than nlocal");
  for (const auto receiver : target_graph.receivers)
    if (receiver >= nlocal)
      throw std::runtime_error(std::string(label) +
                               " graph contains a non-owned receiver");

  std::vector<std::vector<long long>> tags_by_owner(
      static_cast<std::size_t>(ranks));
  std::vector<std::vector<std::size_t>> ghosts_by_owner(
      static_cast<std::size_t>(ranks));
  for (const auto ghost : target_graph.ghost_local) {
    if (ghost < nlocal || ghost >= target_graph.nodes())
      throw std::runtime_error(std::string(label) + " ghost index is invalid");
    const auto found = ownership.owner_by_tag.find(target_graph.tags[ghost]);
    if (found == ownership.owner_by_tag.end())
      throw std::runtime_error(std::string(label) + " ghost tag has no owner");
    if (found->second.atomic_number != target_graph.atomic_numbers[ghost])
      throw std::runtime_error(std::string(label) +
                               " ghost element differs from its owner");
    tags_by_owner[static_cast<std::size_t>(found->second.rank)].push_back(
        static_cast<long long>(target_graph.tags[ghost]));
    ghosts_by_owner[static_cast<std::size_t>(found->second.rank)].push_back(
        ghost);
  }

  ExchangePlan plan;
  plan.ghost_counts.resize(static_cast<std::size_t>(ranks));
  for (int peer = 0; peer < ranks; ++peer) {
    const auto count = tags_by_owner[static_cast<std::size_t>(peer)].size();
    if (count > static_cast<std::size_t>(INT_MAX))
      throw std::runtime_error("SO3LR ghost request exceeds MPI int range");
    plan.ghost_counts[static_cast<std::size_t>(peer)] =
        static_cast<int>(count);
  }
  plan.ghost_displacements = displacements(plan.ghost_counts);
  plan.publish_counts.resize(static_cast<std::size_t>(ranks));
  mpi_require(MPI_Alltoall(plan.ghost_counts.data(), 1, MPI_INT,
                           plan.publish_counts.data(), 1, MPI_INT, world),
              "MPI_Alltoall ghost request counts");
  plan.publish_displacements = displacements(plan.publish_counts);
  const int total_ghosts = plan.ghost_displacements.back() +
                           plan.ghost_counts.back();
  const int total_publish = plan.publish_displacements.back() +
                            plan.publish_counts.back();
  std::vector<long long> request_tags(static_cast<std::size_t>(total_ghosts));
  std::vector<std::size_t> ghost_indices(static_cast<std::size_t>(total_ghosts));
  for (int peer = 0; peer < ranks; ++peer) {
    const int begin = plan.ghost_displacements[static_cast<std::size_t>(peer)];
    const auto &peer_tags = tags_by_owner[static_cast<std::size_t>(peer)];
    const auto &peer_ghosts = ghosts_by_owner[static_cast<std::size_t>(peer)];
    for (std::size_t i = 0; i < peer_tags.size(); ++i) {
      request_tags[static_cast<std::size_t>(begin) + i] = peer_tags[i];
      ghost_indices[static_cast<std::size_t>(begin) + i] = peer_ghosts[i];
    }
  }
  std::vector<long long> published_tags(static_cast<std::size_t>(total_publish));
  mpi_require(MPI_Alltoallv(
                  request_tags.data(), plan.ghost_counts.data(),
                  plan.ghost_displacements.data(), MPI_LONG_LONG,
                  published_tags.data(), plan.publish_counts.data(),
                  plan.publish_displacements.data(), MPI_LONG_LONG, world),
              "MPI_Alltoallv ghost tag requests");
  std::vector<std::size_t> publish_indices(
      static_cast<std::size_t>(total_publish));
  for (std::size_t i = 0; i < publish_indices.size(); ++i) {
    const auto found = ownership.local_by_tag.find(
        static_cast<std::int64_t>(published_tags[i]));
    if (found == ownership.local_by_tag.end())
      throw std::runtime_error(std::string(label) +
                               " publish request is not locally owned");
    publish_indices[i] = found->second;
  }
  plan.publish = device_indices(publish_indices, "so3lr_mpi_publish_nodes");
  plan.ghosts = device_indices(ghost_indices, "so3lr_mpi_ghost_nodes");
  plan.remote_or_image_nodes = ghost_indices.size();
  return plan;
}

std::vector<int> scaled(const std::vector<int> &values, std::size_t width) {
  std::vector<int> result(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    const auto product = static_cast<long long>(values[i]) *
                         static_cast<long long>(width);
    if (product > INT_MAX) throw std::runtime_error("MPI payload is too large");
    result[i] = static_cast<int>(product);
  }
  return result;
}

void device_alltoallv(const DoubleView &send, const DoubleView &receive,
                      const std::vector<int> &send_counts_items,
                      const std::vector<int> &send_displacements_items,
                      const std::vector<int> &receive_counts_items,
                      const std::vector<int> &receive_displacements_items,
                      std::size_t width, MPI_Comm world,
                      const char *operation) {
  const auto send_counts = scaled(send_counts_items, width);
  const auto send_displacements = scaled(send_displacements_items, width);
  const auto receive_counts = scaled(receive_counts_items, width);
  const auto receive_displacements =
      scaled(receive_displacements_items, width);
  Kokkos::fence();
  mpi_require(MPI_Alltoallv(send.data(), send_counts.data(),
                            send_displacements.data(), MPI_DOUBLE,
                            receive.data(), receive_counts.data(),
                            receive_displacements.data(), MPI_DOUBLE, world),
              operation);
  Kokkos::fence();
}

void exchange_features_forward(
    const ExchangePlan &plan, const so3lr::ArchDims &dims,
    const DoubleView &invariant, const DoubleView &equivariant,
    MPI_Comm world, ExchangePayloadArena &arena) {
  const so3lr::KokkosNodeFeatureExchange exchange(dims.invariant_width,
                                                  dims.equivariant_width);
  const std::size_t width = exchange.row_width();
  auto [send, receive] = acquire_payload_views(
      arena, plan.publish.extent(0) * width,
      plan.ghosts.extent(0) * width);
  exchange.pack_device(invariant, equivariant, plan.publish, send);
  device_alltoallv(send, receive, plan.publish_counts,
                   plan.publish_displacements, plan.ghost_counts,
                   plan.ghost_displacements, width, world,
                   "MPI_Alltoallv owner-to-ghost features");
  exchange.unpack_overwrite_device(receive, plan.ghosts, invariant,
                                   equivariant);
}

void exchange_adjoints_reverse(
    const ExchangePlan &plan, const so3lr::KokkosLocalEnergyReverse &reverse,
    const DoubleView &grad_inv, const DoubleView &grad_ev, MPI_Comm world,
    ExchangePayloadArena &arena) {
  const so3lr::KokkosNodeFeatureExchange exchange(
      reverse.dims().invariant_width, reverse.dims().equivariant_width);
  const std::size_t width = exchange.row_width();
  auto [send, receive] = acquire_payload_views(
      arena, plan.ghosts.extent(0) * width,
      plan.publish.extent(0) * width);
  exchange.pack_device(grad_inv, grad_ev, plan.ghosts, send);
  device_alltoallv(send, receive, plan.ghost_counts,
                   plan.ghost_displacements, plan.publish_counts,
                   plan.publish_displacements, width, world,
                   "MPI_Alltoallv ghost-to-owner adjoints");
  reverse.zero_selected_adjoint_device(plan.ghosts, grad_inv, grad_ev);
  exchange.unpack_accumulate_device(receive, plan.publish, grad_inv, grad_ev);
}

void copy_first_rows2(const DoubleView &source_a, const DoubleView &source_b,
                      const DoubleView &target_a, const DoubleView &target_b,
                      std::size_t rows) {
  Kokkos::parallel_for(
      "so3lr_mpi_copy_owned_lr_outputs", Kokkos::RangePolicy<>(0, rows),
      KOKKOS_LAMBDA(const std::size_t row) {
        target_a(row) = source_a(row);
        target_b(row) = source_b(row);
      });
}

void exchange_lr_outputs_forward(
    const ExchangePlan &plan, const DoubleView &source_charges,
    const DoubleView &source_hirshfeld, const DoubleView &target_charges,
    const DoubleView &target_hirshfeld, MPI_Comm world,
    ExchangePayloadArena &arena) {
  constexpr std::size_t width = 2;
  auto [send, receive] = acquire_payload_views(
      arena, plan.publish.extent(0) * width,
      plan.ghosts.extent(0) * width);
  const so3lr::KokkosPhysicalLongRangeExchange exchange;
  exchange.pack_charge_hirshfeld_device(source_charges, source_hirshfeld,
                                        plan.publish, send);
  device_alltoallv(send, receive, plan.publish_counts,
                   plan.publish_displacements, plan.ghost_counts,
                   plan.ghost_displacements, width, world,
                   "MPI_Alltoallv LR charge/Hirshfeld halo");
  exchange.unpack_charge_hirshfeld_overwrite_device(
      receive, plan.ghosts, target_charges, target_hirshfeld);
}

// One scalar per atom, owner -> ghost copies (overwrite). Used for the SO3LR
// C6 ratio (models with a C6 head), which rides alongside the (q, h) halo when the model has a C6
// head; v1 models never call it.
void exchange_scalar_forward(const ExchangePlan &plan, const DoubleView &source,
                             const DoubleView &target, MPI_Comm world,
                             ExchangePayloadArena &arena, const char *label) {
  auto [send, receive] = acquire_payload_views(
      arena, plan.publish.extent(0), plan.ghosts.extent(0));
  const auto publish = plan.publish;
  const auto ghosts = plan.ghosts;
  Kokkos::parallel_for(
      "so3lr_mpi_pack_scalar", Kokkos::RangePolicy<>(0, publish.extent(0)),
      KOKKOS_LAMBDA(const std::size_t i) { send(i) = source(publish(i)); });
  device_alltoallv(send, receive, plan.publish_counts,
                   plan.publish_displacements, plan.ghost_counts,
                   plan.ghost_displacements, 1, world, label);
  Kokkos::parallel_for(
      "so3lr_mpi_unpack_scalar", Kokkos::RangePolicy<>(0, ghosts.extent(0)),
      KOKKOS_LAMBDA(const std::size_t i) { target(ghosts(i)) = receive(i); });
}

// One scalar per atom, ghost copies -> owner (accumulate). The ghost entries
// of `source` are zeroed once packed, matching exchange_lr_reverse_fields.
void exchange_scalar_reverse(const ExchangePlan &plan, const DoubleView &source,
                             const DoubleView &target, MPI_Comm world,
                             ExchangePayloadArena &arena, const char *label) {
  auto [send, receive] = acquire_payload_views(
      arena, plan.ghosts.extent(0), plan.publish.extent(0));
  const auto publish = plan.publish;
  const auto ghosts = plan.ghosts;
  Kokkos::parallel_for(
      "so3lr_mpi_pack_scalar_reverse", Kokkos::RangePolicy<>(0, ghosts.extent(0)),
      KOKKOS_LAMBDA(const std::size_t i) {
        send(i) = source(ghosts(i));
        source(ghosts(i)) = 0.0;
      });
  device_alltoallv(send, receive, plan.ghost_counts, plan.ghost_displacements,
                   plan.publish_counts, plan.publish_displacements, 1, world,
                   label);
  Kokkos::parallel_for(
      "so3lr_mpi_unpack_scalar_reverse",
      Kokkos::RangePolicy<>(0, publish.extent(0)),
      KOKKOS_LAMBDA(const std::size_t i) {
        Kokkos::atomic_add(&target(publish(i)), receive(i));
      });
}

void copy_lr_owned_reverse_fields(
    const so3lr::PhysicalLongRangeWorkspace &lr_workspace,
    const DoubleView &energy, const DoubleView &charge,
    const DoubleView &hirshfeld, std::size_t nlocal) {
  const auto lr_energy = lr_workspace.atomic_energy;
  const auto lr_charge = lr_workspace.charge_gradient;
  const auto lr_hirshfeld = lr_workspace.hirshfeld_gradient;
  Kokkos::parallel_for(
      "so3lr_mpi_copy_owned_lr_reverse_fields",
      Kokkos::RangePolicy<>(0, nlocal),
      KOKKOS_LAMBDA(const std::size_t node) {
        energy(node) += lr_energy(node);
        charge(node) += lr_charge(node);
        hirshfeld(node) += lr_hirshfeld(node);
      });
}

void exchange_lr_reverse_fields(
    const ExchangePlan &plan,
    const so3lr::PhysicalLongRangeWorkspace &lr_workspace,
    const DoubleView &energy, const DoubleView &charge,
    const DoubleView &hirshfeld, MPI_Comm world,
    ExchangePayloadArena &arena) {
  constexpr std::size_t width = 3;
  auto [send, receive] = acquire_payload_views(
      arena, plan.ghosts.extent(0) * width,
      plan.publish.extent(0) * width);
  const so3lr::KokkosPhysicalLongRangeExchange exchange;
  exchange.pack_reverse_fields_device(lr_workspace, plan.ghosts, send);
  device_alltoallv(send, receive, plan.ghost_counts,
                   plan.ghost_displacements, plan.publish_counts,
                   plan.publish_displacements, width, world,
                   "MPI_Alltoallv LR reverse fields");
  exchange.zero_reverse_fields_device(lr_workspace, plan.ghosts);
  exchange.unpack_reverse_fields_accumulate_device(
      receive, plan.publish, energy, charge, hirshfeld);
}

void pack_vectors(const DoubleView &vectors, const IndexView &selected,
                  const DoubleView &packed) {
  if (packed.extent(0) != selected.extent(0) * 3)
    throw std::runtime_error("SO3LR vector pack shape mismatch");
  Kokkos::parallel_for(
      "so3lr_mpi_pack_vectors", Kokkos::RangePolicy<>(0, packed.extent(0)),
      KOKKOS_LAMBDA(const std::size_t flat) {
        const std::size_t item = flat / 3;
        const std::size_t component = flat % 3;
        packed(flat) = vectors(selected(item) * 3 + component);
      });
}

void unpack_vectors_accumulate(const DoubleView &packed,
                               const IndexView &selected,
                               const DoubleView &vectors) {
  if (packed.extent(0) != selected.extent(0) * 3)
    throw std::runtime_error("SO3LR vector unpack shape mismatch");
  Kokkos::parallel_for(
      "so3lr_mpi_accumulate_vectors",
      Kokkos::RangePolicy<>(0, packed.extent(0)),
      KOKKOS_LAMBDA(const std::size_t flat) {
        const std::size_t item = flat / 3;
        const std::size_t component = flat % 3;
        Kokkos::atomic_add(&vectors(selected(item) * 3 + component),
                           packed(flat));
      });
}

void copy_lr_owned_forces(
    const so3lr::PhysicalLongRangeWorkspace &lr_workspace,
    const DoubleView &target, std::size_t nlocal) {
  const auto source = lr_workspace.atomic_forces;
  Kokkos::parallel_for(
      "so3lr_mpi_copy_owned_lr_forces",
      Kokkos::RangePolicy<>(0, nlocal * 3),
      KOKKOS_LAMBDA(const std::size_t flat) { target(flat) += source(flat); });
}

void exchange_lr_forces_reverse(
    const ExchangePlan &plan,
    const so3lr::PhysicalLongRangeWorkspace &lr_workspace,
    const DoubleView &target, MPI_Comm world,
    ExchangePayloadArena &arena) {
  constexpr std::size_t width = 3;
  auto [send, receive] = acquire_payload_views(
      arena, plan.ghosts.extent(0) * width,
      plan.publish.extent(0) * width);
  pack_vectors(lr_workspace.atomic_forces, plan.ghosts, send);
  device_alltoallv(send, receive, plan.ghost_counts,
                   plan.ghost_displacements, plan.publish_counts,
                   plan.publish_displacements, width, world,
                   "MPI_Alltoallv LR direct forces");
  unpack_vectors_accumulate(receive, plan.publish, target);
}

void exchange_sr_forces_reverse(
    const ExchangePlan &plan,
    const so3lr::KokkosLocalCartesianForces &force_model,
    const so3lr::LocalCartesianForcesWorkspace &workspace,
    MPI_Comm world, ExchangePayloadArena &arena) {
  constexpr std::size_t width = 3;
  auto [send, receive] = acquire_payload_views(
      arena, plan.ghosts.extent(0) * width,
      plan.publish.extent(0) * width);
  force_model.pack_selected_forces_device(plan.ghosts, send, workspace);
  device_alltoallv(send, receive, plan.ghost_counts,
                   plan.ghost_displacements, plan.publish_counts,
                   plan.publish_displacements, width, world,
                   "MPI_Alltoallv SR implicit forces");
  force_model.zero_selected_forces_device(plan.ghosts, workspace);
  force_model.accumulate_packed_forces_device(receive, plan.publish,
                                              workspace);
}

double sum_prefix(const DoubleView &values, std::size_t count,
                  const char *label) {
  if (values.extent(0) < count)
    throw std::runtime_error("SO3LR prefix reduction shape mismatch");
  double result = 0.0;
  Kokkos::parallel_reduce(
      label, Kokkos::RangePolicy<>(0, count),
      KOKKOS_LAMBDA(const std::size_t i, double &sum) { sum += values(i); },
      result);
  return result;
}

struct VirialReductionValue {
  double component[6];
};

struct VirialReduction {
  using value_type = VirialReductionValue;
  DoubleView vectors;
  DoubleView gradients;

  KOKKOS_INLINE_FUNCTION void init(value_type &value) const {
    for (int i = 0; i < 6; ++i) value.component[i] = 0.0;
  }
  KOKKOS_INLINE_FUNCTION void join(value_type &destination,
                                   const value_type &source) const {
    for (int i = 0; i < 6; ++i)
      destination.component[i] += source.component[i];
  }
  KOKKOS_INLINE_FUNCTION void operator()(std::size_t edge,
                                         value_type &value) const {
    const double dx = vectors(edge * 3);
    const double dy = vectors(edge * 3 + 1);
    const double dz = vectors(edge * 3 + 2);
    const double gx = gradients(edge * 3);
    const double gy = gradients(edge * 3 + 1);
    const double gz = gradients(edge * 3 + 2);
    value.component[0] -= dx * gx;
    value.component[1] -= dy * gy;
    value.component[2] -= dz * gz;
    value.component[3] -= 0.5 * (dx * gy + dy * gx);
    value.component[4] -= 0.5 * (dx * gz + dz * gx);
    value.component[5] -= 0.5 * (dy * gz + dz * gy);
  }
};

std::array<double, 6> local_virial(const DoubleView &vectors,
                                   const DoubleView &gradients,
                                   const char *label) {
  if (vectors.extent(0) != gradients.extent(0) ||
      vectors.extent(0) % 3 != 0)
    throw std::runtime_error("SO3LR virial shape mismatch");
  VirialReductionValue reduced{};
  Kokkos::parallel_reduce(label,
                          Kokkos::RangePolicy<>(0, vectors.extent(0) / 3),
                          VirialReduction{vectors, gradients}, reduced);
  return {reduced.component[0], reduced.component[1], reduced.component[2],
          reduced.component[3], reduced.component[4], reduced.component[5]};
}

}  // namespace

namespace so3lr_lammps_native_detail {

struct PersistentCache {
  bool topology_valid = false;
  std::size_t topology_nlocal = 0;
  std::size_t topology_source_rows = 0;
  so3lr::LammpsNativeGraphs candidate_graphs;
  so3lr::LammpsNativeGraphs active_graphs;
  DeviceCandidateGraph device_sr_candidate;
  DeviceCandidateGraph device_lr_candidate;
  DoubleView device_atom_positions;

  bool plan_valid = false;
  std::size_t plan_nlocal = 0;
  std::size_t global_nodes = 0;
  std::vector<std::int64_t> sr_tags;
  std::vector<std::int64_t> lr_tags;
  std::vector<std::int64_t> sr_atomic_numbers;
  std::vector<std::int64_t> lr_atomic_numbers;
  ExchangePlan sr_plan;
  ExchangePlan lr_plan;
  ExchangePayloadArena payload_arena;

  bool workspace_valid = false;
  std::size_t nlocal_capacity = 0;
  std::size_t sr_node_capacity = 0;
  std::size_t sr_edge_capacity = 0;
  std::size_t lr_node_capacity = 0;
  std::size_t lr_pair_capacity = 0;
  Int64View sr_z;
  DoubleView sr_vectors;
  IndexView sr_senders;
  IndexView sr_receivers;
  HostInt64View host_sr_z;
  HostDoubleView host_sr_vectors;
  HostIndexView host_sr_senders;
  HostIndexView host_sr_receivers;
  Int64View lr_z;
  DoubleView lr_vectors;
  IndexView lr_senders;
  IndexView lr_receivers;
  HostInt64View host_lr_z;
  HostDoubleView host_lr_vectors;
  HostIndexView host_lr_senders;
  HostIndexView host_lr_receivers;
  IndexView owned_mask;
  DoubleView energy_seeds;
  DoubleView charge_seeds;
  DoubleView hirshfeld_seeds;
  DoubleView c6_seeds;
  DoubleView physical_energy;
  DoubleView direct_lr_forces;
  DoubleView lr_charges;
  DoubleView lr_hirshfeld;
  DoubleView lr_c6;
  DoubleView owned_energy;
  DoubleView owned_forces;
  std::unique_ptr<so3lr::LocalCartesianForcesWorkspace> sr_workspace;
  std::unique_ptr<so3lr::PhysicalZblWorkspace> zbl_workspace;
  std::unique_ptr<so3lr::PhysicalLongRangeWorkspace> lr_workspace;
};

}  // namespace so3lr_lammps_native_detail

PairSO3LRNativeMPI::PairSO3LRNativeMPI(LAMMPS *lmp) : Pair(lmp) {
  manybody_flag = 1;
  one_coeff = 1;
  restartinfo = 0;
  single_enable = 0;
  no_virial_fdotr_compute = 1;
  if (!Kokkos::is_initialized()) {
    Kokkos::initialize();
    initialized_kokkos_here_ = true;
  }
}

PairSO3LRNativeMPI::~PairSO3LRNativeMPI() {
  persistent_.reset();
  long_range_model_.reset();
  force_model_.reset();
  model_.reset();
  // Every Kokkos allocation this style owns must be released before it
  // finalizes the runtime it started; members destroyed after this body would
  // be freed after Kokkos::finalize() and abort the process.
  nlh_a_ = Kokkos::View<double *>();
  nlh_b_ = Kokkos::View<double *>();
  charge_spin_table_ = Kokkos::View<double *>();
  if (initialized_kokkos_here_ && Kokkos::is_initialized()) Kokkos::finalize();
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
}

void PairSO3LRNativeMPI::allocate() {
  allocated = 1;
  const int n = atom->ntypes;
  memory->create(setflag, n + 1, n + 1, "so3lr/native/mpi:setflag");
  memory->create(cutsq, n + 1, n + 1, "so3lr/native/mpi:cutsq");
  for (int i = 1; i <= n; ++i)
    for (int j = i; j <= n; ++j) setflag[i][j] = 0;
}

void PairSO3LRNativeMPI::settings(int narg, char **arg) {
  if (narg < 1 || narg > 3)
    error->all(FLERR,
               "Illegal pair_style so3lr/native/mpi command: expected "
               "'pair_style so3lr/native/mpi MODEL [charge [multiplicity]]'");
  model_path_ = arg[0];
  total_charge_ = narg > 1 ? utils::numeric(FLERR, arg[1], false, lmp) : 0.0;
  multiplicity_ = narg > 2 ? utils::numeric(FLERR, arg[2], false, lmp) : 1.0;
  persistent_.reset();
  try {
    model_ = std::make_unique<so3lr::NativeModel>(
        so3lr::NativeModel::load(model_path_));
    short_range_cutoff_ =
        model_->architecture_number("short_range_cutoff_angstrom");
    long_range_cutoff_ =
        model_->architecture_number("long_range_cutoff_angstrom");
    if (!(short_range_cutoff_ > 0.0) ||
        !(long_range_cutoff_ >= short_range_cutoff_))
      throw std::runtime_error("invalid native model cutoffs");
    force_model_ =
        std::make_unique<so3lr::KokkosLocalCartesianForces>(*model_);
    long_range_model_ =
        std::make_unique<so3lr::KokkosPhysicalLongRangeModel>(*model_);
    charge_spin_ = so3lr::load_charge_spin_embedding(*model_);
    {
      auto repulsion = so3lr::load_so3lr_repulsion(*model_);
      zbl_parameters_ = repulsion.parameters;
      nlh_a_ = repulsion.nlh_a;
      nlh_b_ = repulsion.nlh_b;
    }
    if (!force_model_->distributed_cartesian_force_contract() ||
        !force_model_->receiver_owned_edge_contract())
      throw std::runtime_error("distributed SR force contract failed");
    if (std::abs(long_range_model_->parameters().cutoff -
                 long_range_cutoff_) > 1.0e-12)
      throw std::runtime_error("distributed LR cutoff contract failed");
  } catch (const std::exception &exc) {
    error->all(FLERR, "Cannot initialize MPI-native SO3LR model: {}",
               exc.what());
  }
}

void PairSO3LRNativeMPI::coeff(int narg, char **arg) {
  if (!allocated) allocate();
  if (narg != atom->ntypes + 2 || std::strcmp(arg[0], "*") != 0 ||
      std::strcmp(arg[1], "*") != 0)
    error->all(
        FLERR,
        "Pair coeff for so3lr/native/mpi must be '* * E1 ... Entypes' "
        "where each E is an element symbol or atomic number");
  type_to_atomic_number_.assign(static_cast<std::size_t>(atom->ntypes) + 1,
                                0);
  for (int type = 1; type <= atom->ntypes; ++type) {
    const int z = so3lr_lammps::atomic_number_from_token(arg[type + 1]);
    if (z == 0)
      error->all(FLERR,
                 "Unknown SO3LR element '{}'; use a symbol or atomic number "
                 "from H/1 through Es/99",
                 arg[type + 1]);
    type_to_atomic_number_[static_cast<std::size_t>(type)] = z;
  }
  for (int i = 1; i <= atom->ntypes; ++i)
    for (int j = i; j <= atom->ntypes; ++j) setflag[i][j] = 1;
}

double PairSO3LRNativeMPI::init_one(int i, int j) {
  if (!setflag[i][j])
    error->all(FLERR, "All so3lr/native/mpi coefficients are not set");

  // SO3LR_STAGE3_OPTIM_DEV6_SPLIT_NEIGHBOR_CONTRACT
  // The default full list belongs to the SR GNN.  The independently
  // requested LR half list below carries the larger model cutoff.  This is
  // a neighbor-list contract change only; compute() still evaluates the
  // unchanged SR and LR mathematical cutoffs stored in the model.
  return short_range_cutoff_;
}

void PairSO3LRNativeMPI::init_style() {
  if (!force->newton_pair)
    error->all(FLERR, "Pair style so3lr/native/mpi requires newton pair on");
  if (!model_ || !force_model_ || !long_range_model_)
    error->all(FLERR,
               "Pair style so3lr/native/mpi model was not initialized");
  if (type_to_atomic_number_.size() !=
      static_cast<std::size_t>(atom->ntypes) + 1)
    error->all(FLERR,
               "Pair style so3lr/native/mpi coefficients are missing");
  setup_charge_spin();

  auto *sr = neighbor->add_request(this, NeighConst::REQ_FULL);
  sr->set_id(0);
  sr->set_cutoff(short_range_cutoff_);
  auto *lr = neighbor->add_request(this);
  lr->set_id(1);
  lr->set_cutoff(long_range_cutoff_);
}

// The charge/spin embedding depends on the whole system's composition, so
// the element counts are reduced over all ranks once per run.
void PairSO3LRNativeMPI::setup_charge_spin() {
  so3lr::ElementCounts local{}, counts{};
  for (int i = 0; i < atom->nlocal; ++i) {
    const std::int64_t z =
        type_to_atomic_number_[static_cast<std::size_t>(atom->type[i])];
    if (z > 0 && z <= 118) ++local[static_cast<std::size_t>(z)];
  }
  MPI_Allreduce(local.data(), counts.data(), static_cast<int>(counts.size()),
                MPI_INT64_T, MPI_SUM, world);
  const std::string problem = so3lr::charge_multiplicity_problem(
      counts, total_charge_, multiplicity_);
  if (!problem.empty())
    error->all(FLERR, "Pair style so3lr/native/mpi: charge {} with "
               "multiplicity {}: {}", total_charge_, multiplicity_, problem);
  charge_spin_table_ = Kokkos::View<double *>();
  if (total_charge_ != 0.0 || multiplicity_ != 1.0) {
    std::vector<double> table;
    try {
      table = so3lr::charge_spin_offset_table(charge_spin_, counts,
                                              total_charge_, multiplicity_ - 1.0);
    } catch (const std::exception &exc) {
      error->all(FLERR, exc.what());
    }
    charge_spin_table_ =
        Kokkos::View<double *>("so3lr_mpi_charge_spin_table", table.size());
    auto host = Kokkos::create_mirror_view(charge_spin_table_);
    for (std::size_t i = 0; i < table.size(); ++i) host(i) = table[i];
    Kokkos::deep_copy(charge_spin_table_, host);
  }
  if (comm->me == 0) {
    utils::logmesg(lmp, "SO3LR: total charge {:g}, spin multiplicity {:g}\n",
                   total_charge_, multiplicity_);
    if (total_charge_ != 0.0 &&
        (domain->xperiodic || domain->yperiodic || domain->zperiodic))
      error->warning(FLERR, "SO3LR: net charge {:g} in a periodic cell. SO3LR's "
                     "long range is truncated real-space Coulomb, so the energy "
                     "is defined, but there is no neutralizing background",
                     total_charge_);
  }
}

void PairSO3LRNativeMPI::init_list(int which, NeighList *ptr) {
  if (which == 0)
    short_range_list_ = ptr;
  else if (which == 1)
    long_range_list_ = ptr;
  else
    error->all(FLERR, "Unexpected so3lr/native/mpi neighbor-list ID");
}

void PairSO3LRNativeMPI::compute(int eflag, int vflag) {
  ev_init(eflag, vflag);
  if (vflag_atom)
    error->all(
        FLERR,
        "Dev_59 so3lr/native/mpi does not provide per-atom virial/stress");
  if (short_range_list_ == nullptr || long_range_list_ == nullptr)
    error->all(FLERR, "SO3LR MPI neighbor lists were not initialized");

  try {
    using Clock = std::chrono::steady_clock;
    const auto start = Clock::now();
    const std::size_t call_index = evaluation_count_ + 1;
    static const bool phase_profile_requested =
        environment_enabled("SO3_NATIVE_MPI_PROFILE");
    static const std::size_t phase_profile_every =
        environment_interval("SO3_NATIVE_MPI_PROFILE_EVERY");
    static const bool topology_profile_requested =
        environment_enabled("SO3_NATIVE_TOPOLOGY_PROFILE");
    const bool phase_profile = phase_profile_requested &&
        ((call_index - 1) % phase_profile_every == 0);
    so3lr::LammpsAtomSnapshot atoms;
    atoms.nlocal = static_cast<std::size_t>(atom->nlocal);
    const std::size_t source_rows =
        static_cast<std::size_t>(atom->nlocal + atom->nghost);
    atoms.tags.reserve(source_rows);
    atoms.types.reserve(source_rows);
    atoms.positions.reserve(source_rows * 3);
    for (std::size_t row = 0; row < source_rows; ++row) {
      atoms.tags.push_back(static_cast<std::int64_t>(atom->tag[row]));
      atoms.types.push_back(atom->type[row]);
      atoms.positions.push_back(atom->x[row][0]);
      atoms.positions.push_back(atom->x[row][1]);
      atoms.positions.push_back(atom->x[row][2]);
    }
    const auto atom_snapshot_done = Clock::now();
    const std::size_t nlocal = atoms.nlocal;
    if (!persistent_)
      persistent_ =
          std::make_unique<so3lr_lammps_native_detail::PersistentCache>();
    const int neighbor_ago = neighbor->ago;
    const bool local_topology_match =
        persistent_->topology_valid && neighbor_ago > 0 &&
        persistent_->topology_nlocal == nlocal &&
        persistent_->topology_source_rows == source_rows;
    int local_topology_hit = local_topology_match ? 1 : 0;
    int global_topology_hit = 0;
    mpi_require(MPI_Allreduce(&local_topology_hit, &global_topology_hit, 1,
                              MPI_INT, MPI_MIN, world),
                "MPI_Allreduce topology-cache state");
    const bool topology_cache_hit = global_topology_hit != 0;
    if (!topology_cache_hit) {
      // Snapshot the complete LAMMPS lists, then retain independent
      // skin-expanded candidate radii so SR does not inherit the LR halo.
      // Active interactions are re-filtered at the exact physical cutoffs
      // every call, so cutoff crossings remain safe until reneighboring.
      const double neighbor_skin = neighbor->skin;
      if (!(neighbor_skin >= 0.0) || !std::isfinite(neighbor_skin))
        throw std::runtime_error("SO3LR invalid LAMMPS neighbor skin");
      const double sr_candidate_cutoff =
          short_range_cutoff_ + neighbor_skin;
      const double lr_candidate_cutoff =
          long_range_cutoff_ + neighbor_skin;
      // dev_5 records the rare rebuild path separately from cached calls.
      // Profiling is opt-in; its fences do not affect normal production.
      const auto topology_rebuild_start = Clock::now();
      const auto short_snapshot = snapshot_neighbor_list(short_range_list_);
      const auto short_snapshot_done = Clock::now();
      const auto long_snapshot = snapshot_neighbor_list(long_range_list_);
      const auto long_snapshot_done = Clock::now();
      // SO3LR_STAGE3_OPTIM_DEV7_TOPOLOGY_ONLY_FASTPATH
      // Both snapshots are explicit LAMMPS skin-expanded requests.  Retain
      // endpoint topology only; exact physical vectors/cutoffs are refreshed
      // by device active-graph compaction below on every evaluation.
      persistent_->candidate_graphs =
          so3lr::build_lammps_native_candidate_topologies(
              atoms, short_snapshot, long_snapshot,
              type_to_atomic_number_, force->newton_pair != 0);
      const auto compact_done = Clock::now();
      // SO3LR_STAGE3_OPTIM_DEV8_PERSISTENT_DIRECT_CANDIDATE_UPLOAD
      upload_device_candidate_graph(
          persistent_->device_sr_candidate,
          persistent_->candidate_graphs.short_range,
          "so3lr_mpi_sr_candidate");
      const auto sr_upload_done = Clock::now();
      upload_device_candidate_graph(
          persistent_->device_lr_candidate,
          persistent_->candidate_graphs.long_range,
          "so3lr_mpi_lr_candidate");
      const auto lr_upload_done = Clock::now();
      persistent_->device_atom_positions = DoubleView(
          "so3lr_mpi_atom_positions", atoms.positions.size());
      const auto allocation_done = Clock::now();
      persistent_->topology_nlocal = nlocal;
      persistent_->topology_source_rows = source_rows;
      persistent_->topology_valid = true;
      ++topology_cache_rebuilds_;
      if (topology_profile_requested) {
        // The unqualified host-to-device deep_copy operations above are
        // synchronous; do not add a diagnostic-only fence to the timed run.
        const auto profile_done = Clock::now();
        const auto elapsed_ms = [](auto begin, auto end) {
          return std::chrono::duration<double, std::milli>(end - begin).count();
        };
        std::fprintf(
            stdout,
            "SO3LR_TOPOLOGY_REBUILD_PROFILE rank=%d call=%zu step=%lld "
            "skin=%.9g sr_candidate_cutoff=%.9g lr_candidate_cutoff=%.9g "
            "sr_list_entries=%zu lr_list_entries=%zu "
            "sr_candidate_edges=%zu lr_candidate_pairs=%zu "
            "sr_snapshot_ms=%.9g lr_snapshot_ms=%.9g compact_ms=%.9g "
            "sr_upload_ms=%.9g lr_upload_ms=%.9g allocation_ms=%.9g "
            "total_ms=%.9g\n",
            comm->me, call_index, static_cast<long long>(update->ntimestep),
            neighbor_skin, sr_candidate_cutoff, lr_candidate_cutoff,
            short_snapshot.neighbors.size(), long_snapshot.neighbors.size(),
            persistent_->candidate_graphs.short_range.interactions(),
            persistent_->candidate_graphs.long_range.interactions(),
            elapsed_ms(topology_rebuild_start, short_snapshot_done),
            elapsed_ms(short_snapshot_done, long_snapshot_done),
            elapsed_ms(long_snapshot_done, compact_done),
            elapsed_ms(compact_done, sr_upload_done),
            elapsed_ms(sr_upload_done, lr_upload_done),
            elapsed_ms(lr_upload_done, allocation_done),
            elapsed_ms(topology_rebuild_start, profile_done));
        std::fflush(stdout);
      }
    } else {
      ++topology_cache_hits_;
    }
    const auto topology_done = Clock::now();
    // Node/tag ownership is identical for candidate and active graphs. Edge
    // vectors and exact-cutoff interaction lists are now refreshed on-device.
    const auto &graphs = persistent_->candidate_graphs;
    const bool local_plan_match =
        persistent_->plan_valid && persistent_->plan_nlocal == nlocal &&
        persistent_->sr_tags == graphs.short_range.tags &&
        persistent_->lr_tags == graphs.long_range.tags &&
        persistent_->sr_atomic_numbers == graphs.short_range.atomic_numbers &&
        persistent_->lr_atomic_numbers == graphs.long_range.atomic_numbers;
    int local_plan_hit = local_plan_match ? 1 : 0;
    int global_plan_hit = 0;
    mpi_require(MPI_Allreduce(&local_plan_hit, &global_plan_hit, 1, MPI_INT,
                              MPI_MIN, world),
                "MPI_Allreduce exchange-plan cache state");
    const bool plan_cache_hit = global_plan_hit != 0;
    if (!plan_cache_hit) {
      const auto ownership =
          gather_global_ownership(graphs.short_range, nlocal, world);
      persistent_->sr_plan = build_exchange_plan(
          graphs.short_range, nlocal, ownership, world, "SR");
      persistent_->lr_plan = build_exchange_plan(
          graphs.long_range, nlocal, ownership, world, "LR");
      persistent_->plan_nlocal = nlocal;
      persistent_->global_nodes = ownership.global_nodes;
      persistent_->sr_tags = graphs.short_range.tags;
      persistent_->lr_tags = graphs.long_range.tags;
      persistent_->sr_atomic_numbers = graphs.short_range.atomic_numbers;
      persistent_->lr_atomic_numbers = graphs.long_range.atomic_numbers;
      persistent_->plan_valid = true;
      ++plan_cache_rebuilds_;
    } else {
      ++plan_cache_hits_;
    }
    const auto &sr_plan = persistent_->sr_plan;
    const auto &lr_plan = persistent_->lr_plan;
    const std::size_t global_nodes = persistent_->global_nodes;
    const std::size_t payload_growths_before =
        persistent_->payload_arena.growths;
    if (graphs.long_range.nodes() < nlocal)
      throw std::runtime_error("LR graph changed owned prefix");
    for (std::size_t node = 0; node < nlocal; ++node) {
      if (graphs.short_range.atomic_numbers[node] !=
              graphs.long_range.atomic_numbers[node] ||
          graphs.short_range.tags[node] != graphs.long_range.tags[node])
        throw std::runtime_error("SR/LR owned-node order differs");
    }
    const auto graph_done = Clock::now();

    if (persistent_->device_atom_positions.extent(0) != atoms.positions.size())
      throw std::runtime_error("SO3LR cached device position shape changed");
    using UnmanagedHostPositions = Kokkos::View<
        const double *, Kokkos::HostSpace,
        Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    const UnmanagedHostPositions host_positions(
        atoms.positions.data(), atoms.positions.size());
    Kokkos::deep_copy(persistent_->device_atom_positions, host_positions);

    const std::size_t sr_nodes = graphs.short_range.nodes();
    const std::size_t sr_edges = count_active_device_edges(
        persistent_->device_sr_candidate,
        persistent_->device_atom_positions, short_range_cutoff_,
        "so3lr_mpi_count_active_sr");
    const std::size_t lr_nodes = graphs.long_range.nodes();
    const std::size_t lr_pairs = count_active_device_edges(
        persistent_->device_lr_candidate,
        persistent_->device_atom_positions, long_range_cutoff_,
        "so3lr_mpi_count_active_lr");
    const bool workspace_cache_hit =
        persistent_->workspace_valid &&
        nlocal <= persistent_->nlocal_capacity &&
        sr_nodes < persistent_->sr_node_capacity &&
        sr_edges <= persistent_->sr_edge_capacity &&
        lr_nodes < persistent_->lr_node_capacity &&
        lr_pairs <= persistent_->lr_pair_capacity;
    if (!workspace_cache_hit) {
      Kokkos::fence();
      persistent_->sr_workspace.reset();
      persistent_->zbl_workspace.reset();
      persistent_->lr_workspace.reset();
      persistent_->nlocal_capacity = retained_capacity(
          persistent_->nlocal_capacity, nlocal, 32);
      persistent_->sr_node_capacity = retained_capacity(
          persistent_->sr_node_capacity, sr_nodes + 1, 64);
      persistent_->sr_edge_capacity = retained_capacity(
          persistent_->sr_edge_capacity, sr_edges, 1024);
      persistent_->lr_node_capacity = retained_capacity(
          persistent_->lr_node_capacity, lr_nodes + 1, 64);
      persistent_->lr_pair_capacity = retained_capacity(
          persistent_->lr_pair_capacity, lr_pairs, 4096);
      const std::size_t sr_node_capacity = persistent_->sr_node_capacity;
      const std::size_t sr_edge_capacity = persistent_->sr_edge_capacity;
      const std::size_t lr_node_capacity = persistent_->lr_node_capacity;
      const std::size_t lr_pair_capacity = persistent_->lr_pair_capacity;
      const std::size_t nlocal_capacity = persistent_->nlocal_capacity;
      persistent_->sr_z = Int64View("so3lr_mpi_sr_z", sr_node_capacity);
      persistent_->sr_vectors =
          DoubleView("so3lr_mpi_sr_vectors", sr_edge_capacity * 3);
      persistent_->sr_senders =
          IndexView("so3lr_mpi_sr_senders", sr_edge_capacity);
      persistent_->sr_receivers =
          IndexView("so3lr_mpi_sr_receivers", sr_edge_capacity);
      persistent_->host_sr_z =
          HostInt64View("so3lr_mpi_host_sr_z", sr_node_capacity);
      persistent_->lr_z = Int64View("so3lr_mpi_lr_z", lr_node_capacity);
      persistent_->lr_vectors =
          DoubleView("so3lr_mpi_lr_vectors", lr_pair_capacity * 3);
      persistent_->lr_senders =
          IndexView("so3lr_mpi_lr_senders", lr_pair_capacity);
      persistent_->lr_receivers =
          IndexView("so3lr_mpi_lr_receivers", lr_pair_capacity);
      persistent_->host_lr_z =
          HostInt64View("so3lr_mpi_host_lr_z", lr_node_capacity);
      persistent_->owned_mask =
          IndexView("so3lr_mpi_owned_mask", sr_node_capacity);
      persistent_->energy_seeds =
          DoubleView("so3lr_mpi_energy_seeds", sr_node_capacity);
      persistent_->charge_seeds =
          DoubleView("so3lr_mpi_charge_seeds", sr_node_capacity);
      persistent_->hirshfeld_seeds =
          DoubleView("so3lr_mpi_hirshfeld_seeds", sr_node_capacity);
      persistent_->c6_seeds =
          DoubleView("so3lr_mpi_c6_seeds", sr_node_capacity);
      persistent_->physical_energy =
          DoubleView("so3lr_mpi_physical_energy", sr_node_capacity);
      persistent_->direct_lr_forces =
          DoubleView("so3lr_mpi_direct_lr_forces", sr_node_capacity * 3);
      persistent_->lr_charges =
          DoubleView("so3lr_mpi_lr_charges", lr_node_capacity);
      persistent_->lr_hirshfeld =
          DoubleView("so3lr_mpi_lr_hirshfeld", lr_node_capacity);
      persistent_->lr_c6 = DoubleView("so3lr_mpi_lr_c6", lr_node_capacity);
      persistent_->owned_energy =
          DoubleView("so3lr_mpi_owned_energy", nlocal_capacity);
      persistent_->owned_forces =
          DoubleView("so3lr_mpi_owned_forces", nlocal_capacity * 3);
    }
    auto &sr_z = persistent_->sr_z;
    auto &sr_vectors = persistent_->sr_vectors;
    auto &sr_senders = persistent_->sr_senders;
    auto &sr_receivers = persistent_->sr_receivers;
    auto &lr_z = persistent_->lr_z;
    auto &lr_vectors = persistent_->lr_vectors;
    auto &lr_senders = persistent_->lr_senders;
    auto &lr_receivers = persistent_->lr_receivers;
    auto &owned_mask = persistent_->owned_mask;
    auto &energy_seeds = persistent_->energy_seeds;
    auto &charge_seeds = persistent_->charge_seeds;
    auto &hirshfeld_seeds = persistent_->hirshfeld_seeds;
    auto &c6_seeds = persistent_->c6_seeds;
    auto &physical_energy = persistent_->physical_energy;
    auto &direct_lr_forces = persistent_->direct_lr_forces;
    auto &lr_charges = persistent_->lr_charges;
    auto &lr_hirshfeld = persistent_->lr_hirshfeld;
    auto &lr_c6 = persistent_->lr_c6;
    auto &owned_energy = persistent_->owned_energy;
    auto &owned_forces = persistent_->owned_forces;
    if (!topology_cache_hit || !workspace_cache_hit) {
      upload_padded_atomic_numbers(
          sr_z, persistent_->host_sr_z,
          graphs.short_range.atomic_numbers);
      upload_padded_atomic_numbers(
          lr_z, persistent_->host_lr_z,
          graphs.long_range.atomic_numbers);
    }
    compact_active_device_graph(
        persistent_->device_sr_candidate,
        persistent_->device_atom_positions, short_range_cutoff_, sr_edges,
        sr_nodes, sr_vectors, sr_senders, sr_receivers,
        "so3lr_mpi_sr");
    compact_active_device_graph(
        persistent_->device_lr_candidate,
        persistent_->device_atom_positions, long_range_cutoff_, lr_pairs,
        lr_nodes, lr_vectors, lr_senders, lr_receivers,
        "so3lr_mpi_lr");
    Kokkos::deep_copy(owned_mask, static_cast<std::size_t>(0));
    Kokkos::deep_copy(energy_seeds, 0.0);
    Kokkos::parallel_for(
        "so3lr_mpi_mark_owned", Kokkos::RangePolicy<>(0, nlocal),
        KOKKOS_LAMBDA(const std::size_t node) {
          owned_mask(node) = 1;
          energy_seeds(node) = 1.0;
        });
    Kokkos::deep_copy(charge_seeds, 0.0);
    Kokkos::deep_copy(hirshfeld_seeds, 0.0);
    Kokkos::deep_copy(c6_seeds, 0.0);
    Kokkos::deep_copy(physical_energy, 0.0);
    Kokkos::deep_copy(direct_lr_forces, 0.0);
    Kokkos::deep_copy(lr_charges, 0.0);
    Kokkos::deep_copy(lr_hirshfeld, 0.0);
    Kokkos::deep_copy(lr_c6, 0.0);
    const auto upload_done = Clock::now();
    if (!workspace_cache_hit) {
      persistent_->sr_workspace =
          std::make_unique<so3lr::LocalCartesianForcesWorkspace>(
              persistent_->sr_node_capacity,
              persistent_->sr_edge_capacity, model_->arch());
      persistent_->zbl_workspace =
          std::make_unique<so3lr::PhysicalZblWorkspace>(
              persistent_->nlocal_capacity,
              persistent_->sr_edge_capacity);
      persistent_->lr_workspace =
          std::make_unique<so3lr::PhysicalLongRangeWorkspace>(
              persistent_->lr_node_capacity,
              persistent_->lr_pair_capacity);
      persistent_->workspace_valid = true;
      ++workspace_cache_rebuilds_;
    } else {
      ++workspace_cache_hits_;
    }
    auto &sr_workspace = *persistent_->sr_workspace;
    auto zbl_workspace = *persistent_->zbl_workspace;
    auto zbl_owned_prefix = Kokkos::subview(
        zbl_workspace.atomic_energy,
        std::pair<std::size_t, std::size_t>{0, nlocal});
    zbl_workspace.atomic_energy = zbl_owned_prefix;
    auto &lr_workspace = *persistent_->lr_workspace;
    const auto workspace_done = Clock::now();

    const auto &reverse = force_model_->reverse();
    if (charge_spin_table_.extent(0) != 0)
      so3lr::fill_charge_spin_offset_device(
          sr_z, charge_spin_table_, reverse.dims().invariant_width,
          sr_workspace.reverse.embedding_offset);
    force_model_->launch_geometry_forward_device(sr_vectors, sr_workspace);
    so3lr::launch_so3lr_zbl_device(
        sr_z, nlocal, force_model_->distances(sr_workspace), sr_senders,
        sr_receivers, zbl_parameters_, zbl_workspace, nlh_a_, nlh_b_);

    // Interaction blocks, with the ghost rows of each block's output made
    // consistent before the next block reads them. No exchange follows the
    // last block: the output heads only read owned rows.
    const std::size_t layers = reverse.layers();
    for (std::size_t b = 0; b < layers; ++b) {
      reverse.launch_block_forward_device(
          b, sr_z, force_model_->distances(sr_workspace),
          force_model_->sh_vectors(sr_workspace), sr_senders, sr_receivers,
          sr_workspace.reverse);
      if (b + 1 < layers)
        exchange_features_forward(
            sr_plan, reverse.dims(),
            reverse.block_final_inv(sr_workspace.reverse, b),
            reverse.block_final_ev(sr_workspace.reverse, b), world,
            persistent_->payload_arena);
    }
    reverse.launch_output_heads_device(sr_z, sr_workspace.reverse);
    Kokkos::fence();

    const double local_raw_charge =
        sum_prefix(reverse.raw_charges(sr_workspace.reverse), nlocal,
                   "so3lr_mpi_raw_charge_sum");
    double global_raw_charge = 0.0;
    mpi_require(MPI_Allreduce(&local_raw_charge, &global_raw_charge, 1,
                              MPI_DOUBLE, MPI_SUM, world),
                "MPI_Allreduce raw charge");
    const double charge_correction =
        (total_charge_ - global_raw_charge) / static_cast<double>(global_nodes);
    reverse.launch_distributed_charge_correction_device(charge_correction,
                                                        sr_workspace.reverse);
    const auto sr_forward_done = Clock::now();

    copy_first_rows2(reverse.partial_charges(sr_workspace.reverse),
                     reverse.hirshfeld_ratios(sr_workspace.reverse),
                     lr_charges, lr_hirshfeld, nlocal);
    exchange_lr_outputs_forward(
        lr_plan, reverse.partial_charges(sr_workspace.reverse),
        reverse.hirshfeld_ratios(sr_workspace.reverse), lr_charges,
        lr_hirshfeld, world, persistent_->payload_arena);
    // With a C6 head, the C6 ratio travels as a third field.
    const bool has_c6 = reverse.has_c6_head();
    if (has_c6) {
      const auto head_c6 = reverse.c6_ratios(sr_workspace.reverse);
      const auto target = lr_c6;
      Kokkos::parallel_for(
          "so3lr_mpi_copy_owned_c6", Kokkos::RangePolicy<>(0, nlocal),
          KOKKOS_LAMBDA(const std::size_t node) { target(node) = head_c6(node); });
      exchange_scalar_forward(lr_plan, head_c6, lr_c6, world,
                              persistent_->payload_arena,
                              "MPI_Alltoallv LR C6-ratio halo");
    }
    long_range_model_->launch_device(lr_z, lr_charges, lr_hirshfeld,
                                     lr_vectors, lr_senders, lr_receivers,
                                     lr_workspace, has_c6 ? lr_c6 : DoubleView());
    copy_lr_owned_reverse_fields(lr_workspace, physical_energy, charge_seeds,
                                 hirshfeld_seeds, nlocal);
    exchange_lr_reverse_fields(lr_plan, lr_workspace, physical_energy,
                               charge_seeds, hirshfeld_seeds, world,
                               persistent_->payload_arena);
    if (has_c6) {
      const auto lr_c6_gradient = lr_workspace.c6_gradient;
      const auto seeds = c6_seeds;
      Kokkos::parallel_for(
          "so3lr_mpi_copy_owned_c6_gradient", Kokkos::RangePolicy<>(0, nlocal),
          KOKKOS_LAMBDA(const std::size_t node) { seeds(node) += lr_c6_gradient(node); });
      exchange_scalar_reverse(lr_plan, lr_workspace.c6_gradient, c6_seeds, world,
                              persistent_->payload_arena,
                              "MPI_Alltoallv LR C6-ratio reverse");
    }
    copy_lr_owned_forces(lr_workspace, direct_lr_forces, nlocal);
    exchange_lr_forces_reverse(lr_plan, lr_workspace, direct_lr_forces,
                               world, persistent_->payload_arena);
    const auto lr_done = Clock::now();

    // Diagnostic-only sampled fences around the actual distributed SR
    // reverse path.  Rank-critical maxima are computed during postprocessing.
    const auto reverse_profile_mark = [&]() {
      if (phase_profile) Kokkos::fence();
      return Clock::now();
    };
    const auto reverse_milliseconds = [](auto begin, auto end) {
      return std::chrono::duration<double, std::milli>(end - begin).count();
    };
    const auto reverse_profile_start = reverse_profile_mark();

    const double local_charge_seed =
        sum_prefix(charge_seeds, nlocal, "so3lr_mpi_charge_seed_sum");
    double global_charge_seed = 0.0;
    mpi_require(MPI_Allreduce(&local_charge_seed, &global_charge_seed, 1,
                              MPI_DOUBLE, MPI_SUM, world),
                "MPI_Allreduce charge seed");
    const double global_charge_seed_mean =
        global_charge_seed / static_cast<double>(global_nodes);
    reverse.launch_owned_multihead_reverse_device(
        sr_z, owned_mask, energy_seeds, charge_seeds, hirshfeld_seeds,
        global_charge_seed_mean, sr_workspace.reverse,
        has_c6 ? c6_seeds : DoubleView());
    const auto reverse_seed_head_done = reverse_profile_mark();
    // Reverse through the blocks, last to first. After block b's reverse, its
    // input adjoint on ghost rows is sent to the owners and accumulated there
    // before block b-1 consumes it; nothing follows block 0.
    std::vector<double> block_ms(layers, 0.0), exchange_ms(layers, 0.0);
    auto previous_mark = reverse_seed_head_done;
    for (std::size_t b = layers; b-- > 0;) {
      reverse.launch_block_reverse_device(
          b, force_model_->distances(sr_workspace),
          force_model_->sh_vectors(sr_workspace), sr_senders, sr_receivers,
          sr_workspace.reverse, true);
      const auto block_done = reverse_profile_mark();
      block_ms[b] = reverse_milliseconds(previous_mark, block_done);
      previous_mark = block_done;
      if (b > 0) {
        exchange_adjoints_reverse(
            sr_plan, reverse, reverse.block_grad_inv(sr_workspace.reverse, b),
            reverse.block_grad_ev(sr_workspace.reverse, b), world,
            persistent_->payload_arena);
        const auto exchange_done = reverse_profile_mark();
        exchange_ms[b] = reverse_milliseconds(previous_mark, exchange_done);
        previous_mark = exchange_done;
      }
    }
    const auto reverse_block0_done = previous_mark;
    force_model_->launch_geometry_reverse_with_extra_radial_device(
        sr_vectors, sr_senders, sr_receivers,
        zbl_workspace.edge_radial_gradient, sr_workspace);
    const auto reverse_geometry_done = reverse_profile_mark();
    exchange_sr_forces_reverse(sr_plan, *force_model_, sr_workspace, world,
                               persistent_->payload_arena);
    const auto reverse_force_exchange_done = reverse_profile_mark();
    const auto sr_reverse_done = Clock::now();
    if (phase_profile) {
      // Token names match the fixed three-block format (block2_ms=
      // exchange2_ms= ... block0_ms=), so existing log parsers keep working.
      std::string blocks_text;
      char token[64];
      for (std::size_t b = layers; b-- > 0;) {
        std::snprintf(token, sizeof(token), " block%zu_ms=%.9g", b, block_ms[b]);
        blocks_text += token;
        if (b > 0) {
          std::snprintf(token, sizeof(token), " exchange%zu_ms=%.9g", b, exchange_ms[b]);
          blocks_text += token;
        }
      }
      std::fprintf(
          stdout,
          "SO3LR_NATIVE_MPI_SR_REVERSE_PROFILE rank=%d call=%zu owned=%zu "
          "nodes=%zu edges=%zu seed_head_ms=%.9g%s geometry_ms=%.9g "
          "force_exchange_ms=%.9g total_ms=%.9g\n",
          comm->me, call_index, nlocal, sr_nodes, sr_edges,
          reverse_milliseconds(reverse_profile_start, reverse_seed_head_done),
          blocks_text.c_str(),
          reverse_milliseconds(reverse_block0_done, reverse_geometry_done),
          reverse_milliseconds(reverse_geometry_done,
                               reverse_force_exchange_done),
          reverse_milliseconds(reverse_profile_start,
                               reverse_force_exchange_done));
      std::fflush(stdout);
    }

    const auto sr_virial = local_virial(
        sr_vectors, force_model_->edge_energy_gradients(sr_workspace),
        "so3lr_mpi_sr_virial");
    const auto lr_virial = local_virial(
        lr_vectors, lr_workspace.pair_force_vectors, "so3lr_mpi_lr_virial");
    std::array<double, 6> local_total_virial{};
    for (std::size_t component = 0; component < 6; ++component)
      local_total_virial[component] =
          sr_virial[component] + lr_virial[component];

    const auto learned_energy = reverse.atomic_energies(sr_workspace.reverse);
    const auto zbl_energy = zbl_workspace.atomic_energy;
    const auto implicit_forces = force_model_->atomic_forces(sr_workspace);
    Kokkos::parallel_for(
        "so3lr_mpi_assemble_owned_energy", Kokkos::RangePolicy<>(0, nlocal),
        KOKKOS_LAMBDA(const std::size_t node) {
          owned_energy(node) = learned_energy(node) + zbl_energy(node) +
                               physical_energy(node);
        });
    Kokkos::parallel_for(
        "so3lr_mpi_assemble_owned_forces",
        Kokkos::RangePolicy<>(0, nlocal * 3),
        KOKKOS_LAMBDA(const std::size_t flat) {
          owned_forces(flat) = implicit_forces(flat) + direct_lr_forces(flat);
        });
    const double local_energy =
        sum_prefix(owned_energy, nlocal, "so3lr_mpi_total_energy");
    const auto host_energy = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), owned_energy);
    const auto host_forces = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), owned_forces);
    double global_energy_diagnostic = local_energy;
    double global_max_force = 0.0;
    if (phase_profile || !reported_) {
      double local_max_force = 0.0;
      for (std::size_t flat = 0; flat < nlocal * 3; ++flat)
        local_max_force =
            std::max(local_max_force, std::abs(host_forces(flat)));
      mpi_require(MPI_Allreduce(&local_energy, &global_energy_diagnostic, 1,
                                MPI_DOUBLE, MPI_SUM, world),
                  "MPI_Allreduce diagnostic energy");
      mpi_require(MPI_Allreduce(&local_max_force, &global_max_force, 1,
                                MPI_DOUBLE, MPI_MAX, world),
                  "MPI_Allreduce diagnostic maximum force");
    }
    const auto download_done = Clock::now();

    // Opt-in per-atom decomposition for validation against the JAX reference
    // (tests/jaxref/so3lr_reference.py). Single rank only: a per-atom split of
    // the long-range energy is not meaningful across an owned/ghost boundary,
    // and this is a debugging aid, not an output. The production path does not
    // pay for it: nothing below runs unless the variable is set.
    const char *debug_path = std::getenv("SO3LR_NATIVE_DEBUG_DUMP");
    if (debug_path != nullptr && debug_path[0] != '\0') {
      if (comm->nprocs != 1)
        error->all(FLERR, "SO3LR_NATIVE_DEBUG_DUMP requires a single MPI rank");
      const auto to_host = [](const auto &view) {
        return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view);
      };
      const auto h_learned = to_host(learned_energy);
      const auto h_zbl = to_host(zbl_energy);
      const auto h_physical = to_host(physical_energy);
      const auto h_charges = to_host(reverse.partial_charges(sr_workspace.reverse));
      const auto h_ratios = to_host(reverse.hirshfeld_ratios(sr_workspace.reverse));
      // The LR workspace is zeroed on every launch, so summing the whole
      // capacity is exact even when fewer pairs are active.
      const auto h_elec = to_host(lr_workspace.electrostatic_pair_energy);
      const auto h_disp = to_host(lr_workspace.dispersion_pair_energy);
      double electrostatic_total = 0.0, dispersion_total = 0.0;
      for (std::size_t pair = 0; pair < h_elec.extent(0); ++pair)
        electrostatic_total += h_elec(pair);
      for (std::size_t pair = 0; pair < h_disp.extent(0); ++pair)
        dispersion_total += h_disp(pair);

      FILE *out = std::fopen(debug_path, "w");
      if (out == nullptr)
        error->one(FLERR, std::string("cannot open SO3LR_NATIVE_DEBUG_DUMP ") + debug_path);
      const auto write_array = [&](const char *name, auto value, bool last) {
        std::fprintf(out, "\"%s\":[", name);
        for (std::size_t node = 0; node < nlocal; ++node)
          std::fprintf(out, node ? ",%.17g" : "%.17g", static_cast<double>(value(node)));
        std::fprintf(out, "]%s", last ? "" : ",");
      };
      std::fprintf(out, "{\"schema\":\"so3lr-native-debug-v1\",\"natoms\":%zu,",
                   static_cast<std::size_t>(nlocal));
      std::fprintf(out, "\"electrostatic_total\":%.17g,\"dispersion_total\":%.17g,",
                   electrostatic_total, dispersion_total);
      write_array("tag", [&](std::size_t i) { return static_cast<double>(atom->tag[i]); }, false);
      write_array("type", [&](std::size_t i) { return static_cast<double>(atom->type[i]); }, false);
      write_array("learned_energy", [&](std::size_t i) { return h_learned(i); }, false);
      write_array("zbl_energy", [&](std::size_t i) { return h_zbl(i); }, false);
      write_array("physical_energy", [&](std::size_t i) { return h_physical(i); }, false);
      write_array("partial_charges", [&](std::size_t i) { return h_charges(i); }, false);
      const bool dump_c6 = reverse.has_c6_head();
      write_array("hirshfeld_ratios", [&](std::size_t i) { return h_ratios(i); }, !dump_c6);
      if (dump_c6) {
        const auto h_c6 = to_host(reverse.c6_ratios(sr_workspace.reverse));
        write_array("c6_ratios", [&](std::size_t i) { return h_c6(i); }, true);
      }
      std::fprintf(out, "}\n");
      std::fclose(out);
    }

    for (std::size_t node = 0; node < nlocal; ++node) {
      for (std::size_t component = 0; component < 3; ++component)
        atom->f[node][component] += host_forces(node * 3 + component);
      if (eflag_atom) eatom[node] += host_energy(node);
    }
    // LAMMPS performs the inter-rank reduction of these local contributions.
    // Replicating the already-global values here would multiply energy/virial
    // by the number of MPI ranks.
    if (eflag_global) eng_vdwl += local_energy;
    if (vflag_global)
      for (std::size_t component = 0; component < 6; ++component)
        virial[component] += local_total_virial[component];
    const auto update_done = Clock::now();

    const bool payload_buffer_hit =
        persistent_->payload_arena.growths == payload_growths_before;
    if (payload_buffer_hit)
      ++payload_cache_hits_;
    payload_cache_growths_ = persistent_->payload_arena.growths;
    ++evaluation_count_;
    const auto milliseconds = [](auto begin, auto end) {
      return std::chrono::duration<double, std::milli>(end - begin).count();
    };
    const auto report = [&](FILE *stream) {
      if (stream == nullptr) return;
      std::fprintf(
          stream,
          "SO3LR_MPI_PHASE_PROFILE rank=%d call=%zu step=%lld owned=%zu "
          "sr_nodes=%zu sr_edges=%zu lr_nodes=%zu lr_pairs=%zu "
          "sr_ghosts=%zu lr_ghosts=%zu neighbor_ago=%d "
          "topology_cache_hit=%d device_active_graph=1 "
          "sr_candidate_cutoff=%.9g "
          "lr_candidate_cutoff=%.9g sr_candidate_edges=%zu "
          "lr_candidate_pairs=%zu atom_snapshot_ms=%.9g "
          "topology_refresh_ms=%.9g exchange_plan_ms=%.9g "
          "graph_plan_ms=%.9g upload_ms=%.9g "
          "workspace_ms=%.9g sr_forward_ms=%.9g native_lr_ms=%.9g "
          "sr_reverse_ms=%.9g download_ms=%.9g lammps_update_ms=%.9g "
          "total_ms=%.9g local_energy=%.17g global_energy=%.17g "
          "global_max_force=%.17g plan_cache_hit=%d "
          "workspace_cache_hit=%d topology_cache_hits=%zu "
          "topology_cache_rebuilds=%zu plan_cache_hits=%zu "
          "plan_cache_rebuilds=%zu workspace_cache_hits=%zu "
          "workspace_cache_rebuilds=%zu workspace_nlocal_capacity=%zu "
          "workspace_sr_node_capacity=%zu "
          "workspace_sr_edge_capacity=%zu "
          "workspace_lr_node_capacity=%zu "
          "workspace_lr_pair_capacity=%zu payload_buffer_hit=%d "
          "payload_send_capacity=%zu payload_receive_capacity=%zu "
          "payload_cache_hits=%zu payload_cache_growths=%zu\n",
          comm->me, evaluation_count_,
          static_cast<long long>(update->ntimestep), nlocal, sr_nodes,
          sr_edges, lr_nodes, lr_pairs, sr_plan.remote_or_image_nodes,
          lr_plan.remote_or_image_nodes, neighbor_ago,
          topology_cache_hit ? 1 : 0,
          short_range_cutoff_ + neighbor->skin,
          long_range_cutoff_ + neighbor->skin,
          persistent_->candidate_graphs.short_range.interactions(),
          persistent_->candidate_graphs.long_range.interactions(),
          milliseconds(start, atom_snapshot_done),
          milliseconds(atom_snapshot_done, topology_done),
          milliseconds(topology_done, graph_done),
          milliseconds(start, graph_done),
          milliseconds(graph_done, upload_done),
          milliseconds(upload_done, workspace_done),
          milliseconds(workspace_done, sr_forward_done),
          milliseconds(sr_forward_done, lr_done),
          milliseconds(lr_done, sr_reverse_done),
          milliseconds(sr_reverse_done, download_done),
          milliseconds(download_done, update_done),
          milliseconds(start, update_done), local_energy,
          global_energy_diagnostic, global_max_force,
          plan_cache_hit ? 1 : 0, workspace_cache_hit ? 1 : 0,
          topology_cache_hits_, topology_cache_rebuilds_,
          plan_cache_hits_, plan_cache_rebuilds_,
          workspace_cache_hits_, workspace_cache_rebuilds_,
          persistent_->nlocal_capacity, persistent_->sr_node_capacity,
          persistent_->sr_edge_capacity, persistent_->lr_node_capacity,
          persistent_->lr_pair_capacity, payload_buffer_hit ? 1 : 0,
          persistent_->payload_arena.send_capacity,
          persistent_->payload_arena.receive_capacity,
          payload_cache_hits_, payload_cache_growths_);
    };
    // Profiling is opt-in and may be periodically sampled. This avoids
    // diagnostic reductions, formatting and I/O on normal production calls.
    if (phase_profile) {
      report(stdout);
      std::fflush(stdout);
      if (logfile != nullptr && logfile != stdout) report(logfile);
    }
    if (!reported_) {
      if (comm->me == 0) {
        if (screen)
          std::fprintf(screen,
                       "SO3LR_NATIVE_MPI_FORCE_PATH=PASS ranks=%d "
                       "global_nodes=%zu energy=%.17g max_force=%.17g\n",
                       comm->nprocs, global_nodes,
                       global_energy_diagnostic, global_max_force);
        if (logfile && logfile != screen)
          std::fprintf(logfile,
                       "SO3LR_NATIVE_MPI_FORCE_PATH=PASS ranks=%d "
                       "global_nodes=%zu energy=%.17g max_force=%.17g\n",
                       comm->nprocs, global_nodes,
                       global_energy_diagnostic, global_max_force);
      }
      reported_ = true;
    }
  } catch (const std::exception &exc) {
    error->all(FLERR, "SO3LR MPI-native LAMMPS evaluation failed: {}",
               exc.what());
  }
}
