#include "so3lr/rank_local_graph.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace so3lr {
namespace {

constexpr std::size_t components = 3;

}  // namespace

RankLocalGraph build_receiver_owned_rank_local_graph(
    int rank, const std::vector<int> &node_owner,
    const std::vector<std::int64_t> &global_atomic_numbers,
    const std::vector<double> &global_edge_vectors,
    const std::vector<std::size_t> &global_senders,
    const std::vector<std::size_t> &global_receivers) {
  const std::size_t nodes = global_atomic_numbers.size();
  const std::size_t edges = global_senders.size();
  if (rank < 0 || nodes == 0 || edges == 0 || node_owner.size() != nodes ||
      global_receivers.size() != edges ||
      global_edge_vectors.size() != edges * components)
    throw std::runtime_error("SO3LR rank-local graph input shape mismatch");

  RankLocalGraph graph;
  graph.global_nodes = nodes;
  graph.global_edges = edges;
  graph.rank = rank;

  std::vector<std::size_t> owned_global;
  for (std::size_t node = 0; node < nodes; ++node)
    if (node_owner[node] == rank) owned_global.push_back(node);
  if (owned_global.empty())
    throw std::runtime_error("SO3LR rank owns no nodes");

  std::vector<std::size_t> selected_edges;
  std::vector<std::size_t> ghost_global;
  for (std::size_t edge = 0; edge < edges; ++edge) {
    const std::size_t sender = global_senders[edge];
    const std::size_t receiver = global_receivers[edge];
    if (sender >= nodes || receiver >= nodes || sender == receiver)
      throw std::runtime_error("SO3LR invalid global edge endpoint");
    if (node_owner[receiver] != rank) continue;
    selected_edges.push_back(edge);
    if (node_owner[sender] != rank) ghost_global.push_back(sender);
  }
  if (selected_edges.empty())
    throw std::runtime_error("SO3LR rank-local graph has no edges");
  std::sort(ghost_global.begin(), ghost_global.end());
  ghost_global.erase(std::unique(ghost_global.begin(), ghost_global.end()),
                     ghost_global.end());

  graph.local_to_global = owned_global;
  graph.local_to_global.insert(graph.local_to_global.end(),
                               ghost_global.begin(), ghost_global.end());
  graph.atomic_numbers.reserve(graph.local_to_global.size());
  for (const auto global : graph.local_to_global)
    graph.atomic_numbers.push_back(global_atomic_numbers[global]);
  graph.owned_local.resize(owned_global.size());
  for (std::size_t i = 0; i < graph.owned_local.size(); ++i)
    graph.owned_local[i] = i;
  graph.ghost_local.resize(ghost_global.size());
  for (std::size_t i = 0; i < graph.ghost_local.size(); ++i)
    graph.ghost_local[i] = owned_global.size() + i;

  std::unordered_map<std::size_t, std::size_t> global_to_local;
  global_to_local.reserve(graph.local_to_global.size());
  for (std::size_t local = 0; local < graph.local_to_global.size(); ++local)
    if (!global_to_local.emplace(graph.local_to_global[local], local).second)
      throw std::runtime_error("SO3LR duplicate local node row");

  graph.senders.reserve(selected_edges.size());
  graph.receivers.reserve(selected_edges.size());
  graph.original_edges = selected_edges;
  graph.edge_vectors.reserve(selected_edges.size() * components);
  for (const auto edge : selected_edges) {
    const auto sender = global_to_local.find(global_senders[edge]);
    const auto receiver = global_to_local.find(global_receivers[edge]);
    if (sender == global_to_local.end() || receiver == global_to_local.end())
      throw std::runtime_error("SO3LR global-to-local edge remap failed");
    if (receiver->second >= owned_global.size())
      throw std::runtime_error("SO3LR local edge receiver is not owned");
    graph.senders.push_back(sender->second);
    graph.receivers.push_back(receiver->second);
    for (std::size_t component = 0; component < components; ++component)
      graph.edge_vectors.push_back(
          global_edge_vectors[edge * components + component]);
  }
  return graph;
}

std::vector<std::size_t> rank_local_indices_for_global_ids(
    const RankLocalGraph &graph,
    const std::vector<std::size_t> &global_ids, bool require_owned) {
  std::unordered_map<std::size_t, std::size_t> lookup;
  lookup.reserve(graph.local_to_global.size());
  for (std::size_t local = 0; local < graph.local_to_global.size(); ++local)
    lookup.emplace(graph.local_to_global[local], local);
  std::vector<std::size_t> result;
  result.reserve(global_ids.size());
  for (const auto global : global_ids) {
    const auto found = lookup.find(global);
    if (found == lookup.end())
      throw std::runtime_error("SO3LR requested global node is not local");
    if (require_owned && found->second >= graph.owned_local.size())
      throw std::runtime_error("SO3LR requested publish node is not owned");
    result.push_back(found->second);
  }
  return result;
}

std::vector<std::size_t> rank_local_global_ids(
    const RankLocalGraph &graph,
    const std::vector<std::size_t> &local_indices) {
  std::vector<std::size_t> result;
  result.reserve(local_indices.size());
  for (const auto local : local_indices) {
    if (local >= graph.local_to_global.size())
      throw std::runtime_error("SO3LR local node index out of range");
    result.push_back(graph.local_to_global[local]);
  }
  return result;
}

}  // namespace so3lr
