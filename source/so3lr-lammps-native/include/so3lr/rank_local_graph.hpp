#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace so3lr {

// Framework-neutral host metadata for one receiver-owned MPI subgraph.
// Owned nodes are stored first, followed by the unique sender ghosts required
// by this rank's edges. Edge indices are remapped into that compact table.
struct RankLocalGraph {
  std::size_t global_nodes = 0;
  std::size_t global_edges = 0;
  int rank = -1;
  std::vector<std::size_t> local_to_global;
  std::vector<std::int64_t> atomic_numbers;
  std::vector<std::size_t> owned_local;
  std::vector<std::size_t> ghost_local;
  std::vector<double> edge_vectors;
  std::vector<std::size_t> senders;
  std::vector<std::size_t> receivers;
  std::vector<std::size_t> original_edges;

  std::size_t local_nodes() const { return local_to_global.size(); }
  std::size_t local_edges() const { return senders.size(); }
};

RankLocalGraph build_receiver_owned_rank_local_graph(
    int rank, const std::vector<int> &node_owner,
    const std::vector<std::int64_t> &global_atomic_numbers,
    const std::vector<double> &global_edge_vectors,
    const std::vector<std::size_t> &global_senders,
    const std::vector<std::size_t> &global_receivers);

// Convert an ordered list of requested global IDs into local row indices.
// Every requested node must be present and owned by this rank when
// require_owned is true. The ordering is preserved for MPI pack/unpack.
std::vector<std::size_t> rank_local_indices_for_global_ids(
    const RankLocalGraph &graph,
    const std::vector<std::size_t> &global_ids, bool require_owned);

std::vector<std::size_t> rank_local_global_ids(
    const RankLocalGraph &graph,
    const std::vector<std::size_t> &local_indices);

}  // namespace so3lr
