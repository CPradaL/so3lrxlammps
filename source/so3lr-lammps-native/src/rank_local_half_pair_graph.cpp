#include "so3lr/rank_local_half_pair_graph.hpp"

#include <algorithm>
#include <stdexcept>
#include <unordered_map>

namespace so3lr {
namespace {
constexpr std::size_t components = 3;

int physical_pair_owner(std::size_t sender, std::size_t receiver,
                        const std::vector<int> &node_owner) {
  const int sender_owner = node_owner[sender];
  const int receiver_owner = node_owner[receiver];
  if (sender_owner == receiver_owner) return sender_owner;
  const std::size_t low = std::min(sender, receiver);
  const std::size_t high = std::max(sender, receiver);
  return ((low + high) % 2 == 0) ? node_owner[low] : node_owner[high];
}
}  // namespace

RankLocalHalfPairGraph build_owned_rank_local_half_pair_graph(
    int rank, const std::vector<int> &node_owner,
    const std::vector<std::int64_t> &global_atomic_numbers,
    const std::vector<double> &global_pair_vectors,
    const std::vector<std::size_t> &global_senders,
    const std::vector<std::size_t> &global_receivers) {
  const std::size_t nodes = global_atomic_numbers.size();
  const std::size_t pairs = global_senders.size();
  if (rank < 0 || nodes == 0 || pairs == 0 || node_owner.size() != nodes ||
      global_receivers.size() != pairs ||
      global_pair_vectors.size() != pairs * components)
    throw std::runtime_error("SO3LR half-pair graph input shape mismatch");

  RankLocalHalfPairGraph graph;
  graph.global_nodes = nodes;
  graph.global_pairs = pairs;
  graph.rank = rank;
  std::vector<std::size_t> owned_global;
  for (std::size_t node = 0; node < nodes; ++node)
    if (node_owner[node] == rank) owned_global.push_back(node);
  if (owned_global.empty())
    throw std::runtime_error("SO3LR half-pair rank owns no atoms");

  std::vector<std::size_t> ghost_global;
  for (std::size_t pair = 0; pair < pairs; ++pair) {
    const auto sender = global_senders[pair];
    const auto receiver = global_receivers[pair];
    if (sender >= nodes || receiver >= nodes || sender == receiver)
      throw std::runtime_error("SO3LR invalid physical half-pair endpoint");
    if (physical_pair_owner(sender, receiver, node_owner) != rank) continue;
    graph.original_pairs.push_back(pair);
    if (node_owner[sender] != rank) ghost_global.push_back(sender);
    if (node_owner[receiver] != rank) ghost_global.push_back(receiver);
  }
  if (graph.original_pairs.empty())
    throw std::runtime_error("SO3LR rank-local physical graph has no pairs");
  std::sort(ghost_global.begin(), ghost_global.end());
  ghost_global.erase(std::unique(ghost_global.begin(), ghost_global.end()),
                     ghost_global.end());
  graph.local_to_global = owned_global;
  graph.local_to_global.insert(graph.local_to_global.end(), ghost_global.begin(),
                               ghost_global.end());
  graph.atomic_numbers.reserve(graph.local_to_global.size());
  for (const auto global : graph.local_to_global)
    graph.atomic_numbers.push_back(global_atomic_numbers[global]);
  graph.owned_local.resize(owned_global.size());
  for (std::size_t i = 0; i < owned_global.size(); ++i) graph.owned_local[i] = i;
  graph.ghost_local.resize(ghost_global.size());
  for (std::size_t i = 0; i < ghost_global.size(); ++i)
    graph.ghost_local[i] = owned_global.size() + i;

  std::unordered_map<std::size_t, std::size_t> local;
  for (std::size_t i = 0; i < graph.local_to_global.size(); ++i)
    if (!local.emplace(graph.local_to_global[i], i).second)
      throw std::runtime_error("SO3LR duplicate LR local row");
  graph.senders.reserve(graph.original_pairs.size());
  graph.receivers.reserve(graph.original_pairs.size());
  graph.pair_vectors.reserve(graph.original_pairs.size() * components);
  for (const auto pair : graph.original_pairs) {
    graph.senders.push_back(local.at(global_senders[pair]));
    graph.receivers.push_back(local.at(global_receivers[pair]));
    for (std::size_t c = 0; c < components; ++c)
      graph.pair_vectors.push_back(global_pair_vectors[pair * components + c]);
  }
  return graph;
}

std::vector<std::size_t> half_pair_local_indices_for_global_ids(
    const RankLocalHalfPairGraph &graph,
    const std::vector<std::size_t> &global_ids, bool require_owned) {
  std::unordered_map<std::size_t, std::size_t> lookup;
  for (std::size_t local = 0; local < graph.local_to_global.size(); ++local)
    lookup.emplace(graph.local_to_global[local], local);
  std::vector<std::size_t> result;
  result.reserve(global_ids.size());
  for (const auto global : global_ids) {
    const auto found = lookup.find(global);
    if (found == lookup.end())
      throw std::runtime_error("SO3LR requested LR node is not local");
    if (require_owned && found->second >= graph.owned_local.size())
      throw std::runtime_error("SO3LR requested LR publish node is not owned");
    result.push_back(found->second);
  }
  return result;
}

std::vector<std::size_t> half_pair_global_ids(
    const RankLocalHalfPairGraph &graph,
    const std::vector<std::size_t> &local_indices) {
  std::vector<std::size_t> result;
  result.reserve(local_indices.size());
  for (const auto local : local_indices) {
    if (local >= graph.local_to_global.size())
      throw std::runtime_error("SO3LR LR local node index out of range");
    result.push_back(graph.local_to_global[local]);
  }
  return result;
}

}  // namespace so3lr
