#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace so3lr {

// Compact ownership-aware graph for a unique physical long-range half-pair
// list.  This graph is deliberately independent of RankLocalGraph: the GNN
// retains only its short-range receiver halo, while the inexpensive physical
// kernel owns a separate long-range halo.
struct RankLocalHalfPairGraph {
  std::size_t global_nodes = 0;
  std::size_t global_pairs = 0;
  int rank = -1;
  std::vector<std::size_t> local_to_global;
  std::vector<std::int64_t> atomic_numbers;
  std::vector<std::size_t> owned_local;
  std::vector<std::size_t> ghost_local;
  std::vector<double> pair_vectors;
  std::vector<std::size_t> senders;
  std::vector<std::size_t> receivers;
  std::vector<std::size_t> original_pairs;

  std::size_t local_nodes() const { return local_to_global.size(); }
  std::size_t local_pairs() const { return senders.size(); }
};

RankLocalHalfPairGraph build_owned_rank_local_half_pair_graph(
    int rank, const std::vector<int> &node_owner,
    const std::vector<std::int64_t> &global_atomic_numbers,
    const std::vector<double> &global_pair_vectors,
    const std::vector<std::size_t> &global_senders,
    const std::vector<std::size_t> &global_receivers);

std::vector<std::size_t> half_pair_local_indices_for_global_ids(
    const RankLocalHalfPairGraph &graph,
    const std::vector<std::size_t> &global_ids, bool require_owned);

std::vector<std::size_t> half_pair_global_ids(
    const RankLocalHalfPairGraph &graph,
    const std::vector<std::size_t> &local_indices);

}  // namespace so3lr
