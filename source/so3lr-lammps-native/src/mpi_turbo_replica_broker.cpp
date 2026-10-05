#include "so3lr/mpi_turbo_replica_broker.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace so3lr {
namespace {

void mpi_check(int code, const char *operation) {
  if (code != MPI_SUCCESS)
    throw std::runtime_error(std::string("SO3LR MPI failure in ") + operation);
}

std::vector<int> checked_counts(const std::vector<std::uint64_t> &values,
                                std::uint64_t multiplier) {
  std::vector<int> result(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    const std::uint64_t value = values[i] * multiplier;
    if (value > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
      throw std::runtime_error("SO3LR MPI count exceeds INT_MAX");
    result[i] = static_cast<int>(value);
  }
  return result;
}

std::vector<int> displacements(const std::vector<int> &counts) {
  std::vector<int> result(counts.size(), 0);
  for (std::size_t i = 1; i < counts.size(); ++i) {
    if (result[i - 1] > std::numeric_limits<int>::max() - counts[i - 1])
      throw std::runtime_error("SO3LR MPI displacement overflow");
    result[i] = result[i - 1] + counts[i - 1];
  }
  return result;
}

bool valid_local_graph(const TurboReplicaGraph &graph) {
  const std::size_t nodes = graph.atomic_numbers.size();
  if (graph.sr_senders.size() != graph.sr_receivers.size() ||
      graph.sr_edge_vectors.size() != graph.sr_senders.size() * 3 ||
      graph.lr_senders.size() != graph.lr_receivers.size() ||
      graph.lr_pair_vectors.size() != graph.lr_senders.size() * 3 ||
      !std::isfinite(graph.total_charge) ||
      !std::isfinite(graph.unpaired_electrons))
    return false;
  for (std::size_t i = 0; i < graph.sr_senders.size(); ++i)
    if (graph.sr_senders[i] >= nodes || graph.sr_receivers[i] >= nodes)
      return false;
  for (std::size_t i = 0; i < graph.lr_senders.size(); ++i)
    if (graph.lr_senders[i] >= nodes || graph.lr_receivers[i] >= nodes)
      return false;
  return true;
}

template <class T>
std::vector<T> gather_vector(const std::vector<T> &local,
                             const std::vector<int> &counts,
                             const std::vector<int> &offsets,
                             MPI_Datatype datatype, int coordinator,
                             int rank, MPI_Comm communicator) {
  if (local.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::runtime_error("SO3LR local MPI vector exceeds INT_MAX");
  std::vector<T> gathered;
  if (rank == coordinator) {
    const int total = counts.empty() ? 0 : offsets.back() + counts.back();
    gathered.resize(static_cast<std::size_t>(total));
  }
  mpi_check(MPI_Gatherv(local.empty() ? nullptr : local.data(),
                        static_cast<int>(local.size()), datatype,
                        rank == coordinator ? gathered.data() : nullptr,
                        rank == coordinator ? counts.data() : nullptr,
                        rank == coordinator ? offsets.data() : nullptr,
                        datatype, coordinator, communicator),
            "MPI_Gatherv");
  return gathered;
}

std::vector<double> flatten(
    const std::vector<TurboReplicaResult> &replicas,
    const std::vector<double> TurboReplicaResult::*member) {
  std::vector<double> result;
  for (const auto &replica : replicas) {
    const auto &values = replica.*member;
    result.insert(result.end(), values.begin(), values.end());
  }
  return result;
}

std::vector<double> scatter_vector(const std::vector<double> &root_values,
                                   const std::vector<int> &counts,
                                   const std::vector<int> &offsets,
                                   int local_count, int coordinator, int rank,
                                   MPI_Comm communicator) {
  std::vector<double> local(static_cast<std::size_t>(local_count));
  mpi_check(MPI_Scatterv(
                rank == coordinator ? root_values.data() : nullptr,
                rank == coordinator ? counts.data() : nullptr,
                rank == coordinator ? offsets.data() : nullptr, MPI_DOUBLE,
                local.data(), local_count, MPI_DOUBLE, coordinator,
                communicator),
            "MPI_Scatterv");
  return local;
}

}  // namespace

MpiTurboReplicaBroker::MpiTurboReplicaBroker(
    MPI_Comm communicator, int coordinator,
    const NativeModel *coordinator_model)
    : communicator_(communicator), coordinator_(coordinator) {
  if (communicator_ == MPI_COMM_NULL)
    throw std::runtime_error("SO3LR turbo broker communicator is null");
  mpi_check(MPI_Comm_rank(communicator_, &rank_), "MPI_Comm_rank");
  mpi_check(MPI_Comm_size(communicator_, &size_), "MPI_Comm_size");
  if (coordinator_ < 0 || coordinator_ >= size_)
    throw std::runtime_error("SO3LR turbo coordinator rank is invalid");
  if (rank_ == coordinator_) {
    if (coordinator_model == nullptr)
      throw std::runtime_error("SO3LR coordinator model is missing");
    bridge_ = std::make_unique<KokkosTurboReplicaBridge>(*coordinator_model);
  } else if (coordinator_model != nullptr) {
    throw std::runtime_error("SO3LR worker unexpectedly received a model");
  }
}

TurboReplicaResult MpiTurboReplicaBroker::evaluate(
    const TurboReplicaGraph &local_graph) {
  const int local_graph_ok = valid_local_graph(local_graph) ? 1 : 0;
  int all_graphs_ok = 0;
  mpi_check(MPI_Allreduce(&local_graph_ok, &all_graphs_ok, 1, MPI_INT, MPI_MIN,
                          communicator_),
            "MPI_Allreduce graph validation");
  if (!all_graphs_ok)
    throw std::runtime_error("SO3LR turbo broker received an invalid graph");

  const std::uint64_t local_counts[3] = {
      static_cast<std::uint64_t>(local_graph.atomic_numbers.size()),
      static_cast<std::uint64_t>(local_graph.sr_senders.size()),
      static_cast<std::uint64_t>(local_graph.lr_senders.size())};
  std::vector<std::uint64_t> all_counts(static_cast<std::size_t>(size_) * 3);
  mpi_check(MPI_Gather(local_counts, 3, MPI_UINT64_T,
                       is_coordinator() ? all_counts.data() : nullptr, 3,
                       MPI_UINT64_T, coordinator_, communicator_),
            "MPI_Gather counts");
  mpi_check(MPI_Bcast(all_counts.data(), size_ * 3, MPI_UINT64_T,
                      coordinator_, communicator_),
            "MPI_Bcast counts");

  std::vector<std::uint64_t> nodes(size_), sr_edges(size_), lr_pairs(size_);
  for (int image = 0; image < size_; ++image) {
    nodes[image] = all_counts[static_cast<std::size_t>(image) * 3];
    sr_edges[image] = all_counts[static_cast<std::size_t>(image) * 3 + 1];
    lr_pairs[image] = all_counts[static_cast<std::size_t>(image) * 3 + 2];
  }
  const auto node_counts = checked_counts(nodes, 1);
  const auto node_offsets = displacements(node_counts);
  const auto node3_counts = checked_counts(nodes, 3);
  const auto node3_offsets = displacements(node3_counts);
  const auto sr_counts = checked_counts(sr_edges, 1);
  const auto sr_offsets = displacements(sr_counts);
  const auto sr3_counts = checked_counts(sr_edges, 3);
  const auto sr3_offsets = displacements(sr3_counts);
  const auto lr_counts = checked_counts(lr_pairs, 1);
  const auto lr_offsets = displacements(lr_counts);
  const auto lr3_counts = checked_counts(lr_pairs, 3);
  const auto lr3_offsets = displacements(lr3_counts);

  std::vector<std::uint64_t> local_sr_senders(local_graph.sr_senders.begin(),
                                               local_graph.sr_senders.end());
  std::vector<std::uint64_t> local_sr_receivers(
      local_graph.sr_receivers.begin(), local_graph.sr_receivers.end());
  std::vector<std::uint64_t> local_lr_senders(local_graph.lr_senders.begin(),
                                               local_graph.lr_senders.end());
  std::vector<std::uint64_t> local_lr_receivers(
      local_graph.lr_receivers.begin(), local_graph.lr_receivers.end());

  const auto gathered_z = gather_vector(
      local_graph.atomic_numbers, node_counts, node_offsets, MPI_INT64_T,
      coordinator_, rank_, communicator_);
  const auto gathered_sr_vectors = gather_vector(
      local_graph.sr_edge_vectors, sr3_counts, sr3_offsets, MPI_DOUBLE,
      coordinator_, rank_, communicator_);
  const auto gathered_sr_senders = gather_vector(
      local_sr_senders, sr_counts, sr_offsets, MPI_UINT64_T, coordinator_,
      rank_, communicator_);
  const auto gathered_sr_receivers = gather_vector(
      local_sr_receivers, sr_counts, sr_offsets, MPI_UINT64_T, coordinator_,
      rank_, communicator_);
  const auto gathered_lr_vectors = gather_vector(
      local_graph.lr_pair_vectors, lr3_counts, lr3_offsets, MPI_DOUBLE,
      coordinator_, rank_, communicator_);
  const auto gathered_lr_senders = gather_vector(
      local_lr_senders, lr_counts, lr_offsets, MPI_UINT64_T, coordinator_,
      rank_, communicator_);
  const auto gathered_lr_receivers = gather_vector(
      local_lr_receivers, lr_counts, lr_offsets, MPI_UINT64_T, coordinator_,
      rank_, communicator_);
  // Per image: total charge and number of unpaired electrons.
  const double local_state[2] = {local_graph.total_charge,
                                 local_graph.unpaired_electrons};
  std::vector<double> gathered_state;
  if (is_coordinator()) gathered_state.resize(2 * static_cast<std::size_t>(size_));
  mpi_check(MPI_Gather(local_state, 2, MPI_DOUBLE,
                       is_coordinator() ? gathered_state.data() : nullptr, 2,
                       MPI_DOUBLE, coordinator_, communicator_),
            "MPI_Gather charge and spin");

  int evaluation_ok = 1;
  std::string evaluation_error;
  if (is_coordinator()) {
    try {
      std::vector<TurboReplicaGraph> graphs(static_cast<std::size_t>(size_));
      for (int image = 0; image < size_; ++image) {
        auto &graph = graphs[static_cast<std::size_t>(image)];
        const int nb = node_offsets[image], ne = nb + node_counts[image];
        graph.atomic_numbers.assign(gathered_z.begin() + nb,
                                    gathered_z.begin() + ne);
        const int svb = sr3_offsets[image], sve = svb + sr3_counts[image];
        graph.sr_edge_vectors.assign(gathered_sr_vectors.begin() + svb,
                                     gathered_sr_vectors.begin() + sve);
        const int seb = sr_offsets[image], see = seb + sr_counts[image];
        for (int i = seb; i < see; ++i) {
          graph.sr_senders.push_back(
              static_cast<std::size_t>(gathered_sr_senders[i]));
          graph.sr_receivers.push_back(
              static_cast<std::size_t>(gathered_sr_receivers[i]));
        }
        const int lvb = lr3_offsets[image], lve = lvb + lr3_counts[image];
        graph.lr_pair_vectors.assign(gathered_lr_vectors.begin() + lvb,
                                     gathered_lr_vectors.begin() + lve);
        const int lpb = lr_offsets[image], lpe = lpb + lr_counts[image];
        for (int i = lpb; i < lpe; ++i) {
          graph.lr_senders.push_back(
              static_cast<std::size_t>(gathered_lr_senders[i]));
          graph.lr_receivers.push_back(
              static_cast<std::size_t>(gathered_lr_receivers[i]));
        }
        graph.total_charge = gathered_state[2 * static_cast<std::size_t>(image)];
        graph.unpaired_electrons =
            gathered_state[2 * static_cast<std::size_t>(image) + 1];
      }
      coordinator_batch_ = bridge_->evaluate(graphs);
    } catch (const std::exception &error) {
      evaluation_ok = 0;
      evaluation_error = error.what();
    }
  }
  mpi_check(MPI_Bcast(&evaluation_ok, 1, MPI_INT, coordinator_, communicator_),
            "MPI_Bcast evaluation status");
  if (!evaluation_ok) {
    int length = is_coordinator() ? static_cast<int>(evaluation_error.size()) : 0;
    mpi_check(MPI_Bcast(&length, 1, MPI_INT, coordinator_, communicator_),
              "MPI_Bcast error length");
    std::vector<char> message(static_cast<std::size_t>(length));
    if (is_coordinator())
      std::copy(evaluation_error.begin(), evaluation_error.end(),
                message.begin());
    mpi_check(MPI_Bcast(message.data(), length, MPI_CHAR, coordinator_,
                        communicator_),
              "MPI_Bcast error message");
    throw std::runtime_error("SO3LR coordinator evaluation failed: " +
                             std::string(message.begin(), message.end()));
  }

  std::vector<double> root_energies, root_charges, root_hirshfeld, root_forces;
  std::vector<double> root_total_energies;
  if (is_coordinator()) {
    root_energies = flatten(coordinator_batch_.replicas,
                            &TurboReplicaResult::atomic_energies);
    root_charges = flatten(coordinator_batch_.replicas,
                           &TurboReplicaResult::partial_charges);
    root_hirshfeld = flatten(coordinator_batch_.replicas,
                             &TurboReplicaResult::hirshfeld_ratios);
    root_forces = flatten(coordinator_batch_.replicas,
                          &TurboReplicaResult::atomic_forces);
    for (const auto &replica : coordinator_batch_.replicas)
      root_total_energies.push_back(replica.total_energy);
  }
  TurboReplicaResult local_result;
  local_result.atomic_energies = scatter_vector(
      root_energies, node_counts, node_offsets,
      static_cast<int>(local_graph.atomic_numbers.size()), coordinator_, rank_,
      communicator_);
  local_result.partial_charges = scatter_vector(
      root_charges, node_counts, node_offsets,
      static_cast<int>(local_graph.atomic_numbers.size()), coordinator_, rank_,
      communicator_);
  local_result.hirshfeld_ratios = scatter_vector(
      root_hirshfeld, node_counts, node_offsets,
      static_cast<int>(local_graph.atomic_numbers.size()), coordinator_, rank_,
      communicator_);
  local_result.atomic_forces = scatter_vector(
      root_forces, node3_counts, node3_offsets,
      static_cast<int>(local_graph.atomic_numbers.size() * 3), coordinator_,
      rank_, communicator_);
  mpi_check(MPI_Scatter(is_coordinator() ? root_total_energies.data() : nullptr,
                        1, MPI_DOUBLE, &local_result.total_energy, 1,
                        MPI_DOUBLE, coordinator_, communicator_),
            "MPI_Scatter total energy");
  return local_result;
}

}  // namespace so3lr
