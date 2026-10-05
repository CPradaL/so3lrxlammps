#include "so3lr/kokkos_local_transformer_forward.hpp"
#include "so3lr/kokkos_node_feature_exchange.hpp"

#include <Kokkos_Core.hpp>
#include <cuda_runtime_api.h>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using DoubleView = Kokkos::View<double *>;
using Int64View = Kokkos::View<std::int64_t *>;
using IndexView = Kokkos::View<std::size_t *>;
using Exchange = so3lr::KokkosNodeFeatureExchange;

void mpi_check(int code, const char *operation) {
  if (code == MPI_SUCCESS) return;
  char text[MPI_MAX_ERROR_STRING] = {};
  int length = 0;
  MPI_Error_string(code, text, &length);
  throw std::runtime_error(std::string(operation) + ": " +
                           std::string(text, static_cast<std::size_t>(length)));
}

void cuda_check(cudaError_t code, const char *operation) {
  if (code == cudaSuccess) return;
  throw std::runtime_error(std::string(operation) + ": " +
                           cudaGetErrorString(code));
}

std::string read_text(const std::string &path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open fixture " + path);
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

std::vector<double> numbers(const so3lr::Json &value) {
  std::vector<double> result;
  result.reserve(value.array().size());
  for (const auto &item : value.array()) result.push_back(item.number());
  return result;
}

std::vector<std::int64_t> integers(const so3lr::Json &value) {
  std::vector<std::int64_t> result;
  result.reserve(value.array().size());
  for (const auto &item : value.array())
    result.push_back(static_cast<std::int64_t>(item.unsigned_integer()));
  return result;
}

std::vector<std::size_t> indices(const so3lr::Json &value) {
  std::vector<std::size_t> result;
  result.reserve(value.array().size());
  for (const auto &item : value.array())
    result.push_back(static_cast<std::size_t>(item.unsigned_integer()));
  return result;
}

template <class View, class Values>
void fill(const View &device, const Values &values) {
  if (device.extent(0) != values.size())
    throw std::runtime_error("host-to-device size mismatch");
  auto host = Kokkos::create_mirror_view(device);
  for (std::size_t i = 0; i < values.size(); ++i) host(i) = values[i];
  Kokkos::deep_copy(device, host);
}

std::vector<double> copy(const DoubleView &device) {
  const auto host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), device);
  std::vector<double> result(device.extent(0));
  for (std::size_t i = 0; i < result.size(); ++i) result[i] = host(i);
  return result;
}

std::vector<std::size_t> owned_nodes(int rank, std::size_t nodes) {
  const std::size_t split = (nodes + 1) / 2;
  const std::size_t begin = rank == 0 ? 0 : split;
  const std::size_t end = rank == 0 ? split : nodes;
  std::vector<std::size_t> result;
  for (std::size_t node = begin; node < end; ++node) result.push_back(node);
  return result;
}

std::vector<std::size_t> ghost_nodes(int rank, std::size_t nodes) {
  return owned_nodes(1 - rank, nodes);
}

bool is_owned(int rank, std::size_t node, std::size_t nodes) {
  const std::size_t split = (nodes + 1) / 2;
  return rank == 0 ? node < split : node >= split;
}

struct LocalGraph {
  std::vector<double> distances;
  std::vector<double> sh;
  std::vector<std::size_t> senders;
  std::vector<std::size_t> receivers;
  std::vector<int> membership;
};

LocalGraph receiver_owned_graph(int rank, std::size_t nodes,
                                const std::vector<double> &distances,
                                const std::vector<double> &sh,
                                const std::vector<std::size_t> &senders,
                                const std::vector<std::size_t> &receivers) {
  if (distances.size() != senders.size() ||
      receivers.size() != senders.size() ||
      sh.size() != senders.size() * Exchange::equivariant_width)
    throw std::runtime_error("full SR graph shape mismatch");
  LocalGraph local;
  local.membership.assign(senders.size(), 0);
  for (std::size_t edge = 0; edge < senders.size(); ++edge) {
    if (!is_owned(rank, receivers[edge], nodes)) continue;
    local.membership[edge] = 1;
    local.distances.push_back(distances[edge]);
    local.senders.push_back(senders[edge]);
    local.receivers.push_back(receivers[edge]);
    for (std::size_t channel = 0; channel < Exchange::equivariant_width;
         ++channel)
      local.sh.push_back(
          sh[edge * Exchange::equivariant_width + channel]);
  }
  if (local.distances.empty())
    throw std::runtime_error("rank-local SR graph is empty");
  return local;
}

double selected_error(const DoubleView &actual,
                      const std::vector<double> &expected,
                      const std::vector<std::size_t> &nodes,
                      std::size_t width, const std::string &label) {
  const auto host = copy(actual);
  if (host.size() != expected.size())
    throw std::runtime_error(label + " size mismatch");
  double maximum = 0.0;
  for (const auto node : nodes) {
    for (std::size_t channel = 0; channel < width; ++channel) {
      const std::size_t i = node * width + channel;
      const double error = std::abs(host[i] - expected[i]);
      maximum = std::max(maximum, error);
      if (error > 5.0e-10 + 5.0e-10 * std::abs(expected[i]))
        throw std::runtime_error(label + " mismatch at " +
                                 std::to_string(i));
    }
  }
  return maximum;
}

double selected_difference(const DoubleView &actual,
                           const std::vector<double> &reference,
                           const std::vector<std::size_t> &nodes,
                           std::size_t width) {
  const auto host = copy(actual);
  double maximum = 0.0;
  for (const auto node : nodes)
    for (std::size_t channel = 0; channel < width; ++channel) {
      const std::size_t i = node * width + channel;
      maximum = std::max(maximum, std::abs(host[i] - reference[i]));
    }
  return maximum;
}

double all_feature_error(const DoubleView &inv, const DoubleView &ev,
                         const std::vector<double> &expected_inv,
                         const std::vector<double> &expected_ev,
                         const std::string &label) {
  const std::vector<std::size_t> all_nodes = [&] {
    std::vector<std::size_t> result(expected_inv.size() /
                                    Exchange::invariant_width);
    for (std::size_t i = 0; i < result.size(); ++i) result[i] = i;
    return result;
  }();
  return std::max(selected_error(inv, expected_inv, all_nodes,
                                 Exchange::invariant_width, label + " inv"),
                  selected_error(ev, expected_ev, all_nodes,
                                 Exchange::equivariant_width,
                                 label + " ev"));
}

void sendrecv(const DoubleView &send, const DoubleView &receive, int peer,
              int tag) {
  if (send.extent(0) > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      receive.extent(0) >
          static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::runtime_error("MPI feature message exceeds INT_MAX");
  Kokkos::fence();
  mpi_check(MPI_Sendrecv(send.data(), static_cast<int>(send.extent(0)),
                         MPI_DOUBLE, peer, tag, receive.data(),
                         static_cast<int>(receive.extent(0)), MPI_DOUBLE, peer,
                         tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE),
            "MPI_Sendrecv local transformer features");
}

void publish_owner_features(
    const Exchange &exchange, const IndexView &owners, const IndexView &ghosts,
    const DoubleView &inv, const DoubleView &ev,
    so3lr::NodeFeatureExchangeWorkspace &workspace, int peer, int tag) {
  exchange.pack_device(inv, ev, owners, workspace.send_buffer);
  sendrecv(workspace.send_buffer, workspace.receive_buffer, peer, tag);
  exchange.unpack_overwrite_device(workspace.receive_buffer, ghosts, inv, ev);
  Kokkos::fence();
}

std::string uuid_string(const std::array<unsigned char, 16> &uuid) {
  static constexpr char digits[] = "0123456789abcdef";
  std::string result;
  result.reserve(32);
  for (const auto value : uuid) {
    result.push_back(digits[value >> 4]);
    result.push_back(digits[value & 0x0f]);
  }
  return result;
}

}  // namespace

int main(int argc, char **argv) {
  mpi_check(MPI_Init(&argc, &argv), "MPI_Init");
  int rank = -1;
  int ranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
  int visible_devices = 0;
  cuda_check(cudaGetDeviceCount(&visible_devices), "cudaGetDeviceCount");
  if (visible_devices <= 0) {
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
  int local_rank = rank;
  if (const char *value = std::getenv("SLURM_LOCALID"))
    local_rank = std::atoi(value);
  cuda_check(cudaSetDevice(local_rank % visible_devices), "cudaSetDevice");
  Kokkos::initialize(argc, argv);
  int result_code = 0;
  try {
#ifndef KOKKOS_ENABLE_CUDA
    throw std::runtime_error("test was not compiled for CUDA");
#endif
    if (ranks != 2)
      throw std::runtime_error("dev_30 requires exactly two MPI ranks");
    if (argc != 3)
      throw std::runtime_error("usage: test MODEL STACK012_FIXTURE");
    const int peer = 1 - rank;

    int cuda_device = -1;
    cudaDeviceProp properties{};
    cuda_check(cudaGetDevice(&cuda_device), "cudaGetDevice");
    cuda_check(cudaGetDeviceProperties(&properties, cuda_device),
               "cudaGetDeviceProperties");
    std::array<unsigned char, 16> local_uuid{};
    std::copy(std::begin(properties.uuid.bytes), std::end(properties.uuid.bytes),
              local_uuid.begin());
    std::array<unsigned char, 32> gathered_uuids{};
    mpi_check(MPI_Allgather(local_uuid.data(), 16, MPI_UNSIGNED_CHAR,
                            gathered_uuids.data(), 16, MPI_UNSIGNED_CHAR,
                            MPI_COMM_WORLD),
              "MPI_Allgather GPU UUIDs");
    bool distinct_gpus = false;
    for (std::size_t i = 0; i < 16; ++i)
      distinct_gpus = distinct_gpus || gathered_uuids[i] != gathered_uuids[16 + i];
    if (!distinct_gpus)
      throw std::runtime_error("MPI ranks selected the same physical GPU");

    const auto model = so3lr::NativeModel::load(argv[1]);
    const auto fixture = so3lr::Json::parse(read_text(argv[2]));
    if (fixture.at("schema").string() !=
        "so3lr-native-transformer-stack012-fixture-v1")
      throw std::runtime_error("unexpected transformer fixture contract");
    const auto host_z = integers(fixture.at("atomic_numbers"));
    const auto full_distances = numbers(fixture.at("distances"));
    const auto full_sh = numbers(fixture.at("sh_vectors"));
    const auto full_senders = indices(fixture.at("senders"));
    const auto full_receivers = indices(fixture.at("receivers"));
    const std::size_t nodes = host_z.size();
    const std::size_t full_edges = full_senders.size();
    const auto local = receiver_owned_graph(rank, nodes, full_distances,
                                            full_sh, full_senders,
                                            full_receivers);
    const auto host_owners = owned_nodes(rank, nodes);
    const auto host_ghosts = ghost_nodes(rank, nodes);

    std::vector<int> global_membership(full_edges, 0);
    mpi_check(MPI_Allreduce(local.membership.data(), global_membership.data(),
                            static_cast<int>(full_edges), MPI_INT, MPI_SUM,
                            MPI_COMM_WORLD),
              "MPI_Allreduce edge membership");
    if (std::any_of(global_membership.begin(), global_membership.end(),
                    [](int count) { return count != 1; }))
      throw std::runtime_error("local SR edge sets are not disjoint/exhaustive");
    unsigned long long local_edge_count = local.senders.size();
    unsigned long long total_edge_count = 0;
    std::array<unsigned long long, 2> rank_edge_counts{};
    mpi_check(MPI_Allreduce(&local_edge_count, &total_edge_count, 1,
                            MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD),
              "MPI_Allreduce SR edge count");
    mpi_check(MPI_Allgather(&local_edge_count, 1, MPI_UNSIGNED_LONG_LONG,
                            rank_edge_counts.data(), 1,
                            MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD),
              "MPI_Allgather SR edge counts");
    if (total_edge_count != full_edges)
      throw std::runtime_error("global SR edge count changed");

    Int64View z("so3lr_dev30_z", nodes);
    DoubleView distances("so3lr_dev30_local_distances", local.distances.size());
    DoubleView sh("so3lr_dev30_local_sh", local.sh.size());
    IndexView senders("so3lr_dev30_local_senders", local.senders.size());
    IndexView receivers("so3lr_dev30_local_receivers", local.receivers.size());
    IndexView owners("so3lr_dev30_owners", host_owners.size());
    IndexView ghosts("so3lr_dev30_ghosts", host_ghosts.size());
    fill(z, host_z);
    fill(distances, local.distances);
    fill(sh, local.sh);
    fill(senders, local.senders);
    fill(receivers, local.receivers);
    fill(owners, host_owners);
    fill(ghosts, host_ghosts);

    const auto expected0_inv = numbers(fixture.at("block0_final_inv"));
    const auto expected0_ev = numbers(fixture.at("block0_final_ev"));
    const auto expected1_inv = numbers(fixture.at("block1_final_inv"));
    const auto expected1_ev = numbers(fixture.at("block1_final_ev"));
    const auto expected2_inv = numbers(fixture.at("block2_final_inv"));
    const auto expected2_ev = numbers(fixture.at("block2_final_ev"));

    const so3lr::KokkosLocalTransformerForward forward(model);
    if (!forward.device_contract_verified() ||
        !forward.explicit_block_boundaries() ||
        !forward.receiver_owned_edge_contract())
      throw std::runtime_error("local transformer API contract failed");
    so3lr::LocalTransformerForwardWorkspace workspace(nodes,
                                                       local.senders.size());
    const Exchange exchange;
    so3lr::NodeFeatureExchangeWorkspace boundary(host_owners.size(),
                                                 host_ghosts.size());

    forward.launch_block0_device(z, distances, sh, senders, receivers,
                                 workspace);
    Kokkos::fence();
    const double block0_owned_error = std::max(
        selected_error(forward.block0_final_inv(workspace), expected0_inv,
                       host_owners, Exchange::invariant_width,
                       "block0 owned inv"),
        selected_error(forward.block0_final_ev(workspace), expected0_ev,
                       host_owners, Exchange::equivariant_width,
                       "block0 owned ev"));
    const double block0_ghost_pre_exchange_difference = std::max(
        selected_difference(forward.block0_final_inv(workspace), expected0_inv,
                            host_ghosts, Exchange::invariant_width),
        selected_difference(forward.block0_final_ev(workspace), expected0_ev,
                            host_ghosts, Exchange::equivariant_width));
    if (block0_ghost_pre_exchange_difference <= 1.0e-8)
      throw std::runtime_error("block0 ghost outputs were unexpectedly complete");
    publish_owner_features(exchange, owners, ghosts,
                           forward.block0_final_inv(workspace),
                           forward.block0_final_ev(workspace), boundary, peer,
                           3000);
    const double block0_post_exchange_error = all_feature_error(
        forward.block0_final_inv(workspace), forward.block0_final_ev(workspace),
        expected0_inv, expected0_ev, "block0 post exchange");

    forward.launch_block1_device(distances, sh, senders, receivers, workspace);
    Kokkos::fence();
    const double block1_owned_error = std::max(
        selected_error(forward.block1_final_inv(workspace), expected1_inv,
                       host_owners, Exchange::invariant_width,
                       "block1 owned inv"),
        selected_error(forward.block1_final_ev(workspace), expected1_ev,
                       host_owners, Exchange::equivariant_width,
                       "block1 owned ev"));
    const double block1_ghost_pre_exchange_difference = std::max(
        selected_difference(forward.block1_final_inv(workspace), expected1_inv,
                            host_ghosts, Exchange::invariant_width),
        selected_difference(forward.block1_final_ev(workspace), expected1_ev,
                            host_ghosts, Exchange::equivariant_width));
    if (block1_ghost_pre_exchange_difference <= 1.0e-8)
      throw std::runtime_error("block1 ghost outputs were unexpectedly complete");
    publish_owner_features(exchange, owners, ghosts,
                           forward.block1_final_inv(workspace),
                           forward.block1_final_ev(workspace), boundary, peer,
                           3001);
    const double block1_post_exchange_error = all_feature_error(
        forward.block1_final_inv(workspace), forward.block1_final_ev(workspace),
        expected1_inv, expected1_ev, "block1 post exchange");

    forward.launch_block2_device(distances, sh, senders, receivers, workspace);
    Kokkos::fence();
    const double block2_owned_error = std::max(
        selected_error(forward.block2_final_inv(workspace), expected2_inv,
                       host_owners, Exchange::invariant_width,
                       "block2 owned inv"),
        selected_error(forward.block2_final_ev(workspace), expected2_ev,
                       host_owners, Exchange::equivariant_width,
                       "block2 owned ev"));
    // Publish final owner outputs only for validation.  Production output
    // heads operate on owned nodes and do not require this third exchange.
    publish_owner_features(exchange, owners, ghosts,
                           forward.block2_final_inv(workspace),
                           forward.block2_final_ev(workspace), boundary, peer,
                           3002);
    const double block2_global_error = all_feature_error(
        forward.block2_final_inv(workspace), forward.block2_final_ev(workspace),
        expected2_inv, expected2_ev, "block2 global owner assembly");

    double local_max_error =
        std::max({block0_owned_error, block0_post_exchange_error,
                  block1_owned_error, block1_post_exchange_error,
                  block2_owned_error, block2_global_error});
    double global_max_error = 0.0;
    mpi_check(MPI_Reduce(&local_max_error, &global_max_error, 1, MPI_DOUBLE,
                         MPI_MAX, 0, MPI_COMM_WORLD),
              "MPI_Reduce distributed forward error");

    if (rank == 0) {
      std::array<unsigned char, 16> rank0_uuid{};
      std::array<unsigned char, 16> rank1_uuid{};
      std::copy_n(gathered_uuids.begin(), 16, rank0_uuid.begin());
      std::copy_n(gathered_uuids.begin() + 16, 16, rank1_uuid.begin());
      const auto maximum_rank_edges =
          std::max(rank_edge_counts[0], rank_edge_counts[1]);
      const double maximum_rank_fraction =
          static_cast<double>(maximum_rank_edges) /
          static_cast<double>(full_edges);
      const double maximum_rank_reduction = 100.0 * (1.0 - maximum_rank_fraction);
      std::cout << std::setprecision(17)
                << "mpi_ranks=2\n"
                << "rank0_gpu_uuid=" << uuid_string(rank0_uuid) << '\n'
                << "rank1_gpu_uuid=" << uuid_string(rank1_uuid) << '\n'
                << "distinct_physical_gpus=1\n"
                << "global_sr_edges=" << full_edges << '\n'
                << "rank0_local_sr_edges=" << rank_edge_counts[0] << '\n'
                << "rank1_local_sr_edges=" << rank_edge_counts[1] << '\n'
                << "sum_rank_local_sr_edges=" << total_edge_count << '\n'
                << "sr_edge_replication_factor=1\n"
                << "previous_sr_edge_replication_factor=2\n"
                << "maximum_rank_edge_fraction=" << maximum_rank_fraction
                << '\n'
                << "maximum_rank_edge_storage_reduction_percent="
                << maximum_rank_reduction << '\n'
                << "forward_feature_exchange_boundaries=2\n"
                << "final_validation_publish_exchanges=1\n"
                << "block0_owned_max_abs_error=" << block0_owned_error << '\n'
                << "block0_ghost_pre_exchange_difference="
                << block0_ghost_pre_exchange_difference << '\n'
                << "block0_post_exchange_max_abs_error="
                << block0_post_exchange_error << '\n'
                << "block1_owned_max_abs_error=" << block1_owned_error << '\n'
                << "block1_ghost_pre_exchange_difference="
                << block1_ghost_pre_exchange_difference << '\n'
                << "block1_post_exchange_max_abs_error="
                << block1_post_exchange_error << '\n'
                << "block2_owned_max_abs_error=" << block2_owned_error << '\n'
                << "block2_global_max_abs_error=" << block2_global_error
                << '\n'
                << "distributed_forward_global_max_abs_error="
                << global_max_error << '\n'
                << "application_host_staging_bytes=0\n"
                << "receiver_owned_edge_partition=PASS\n"
                << "disjoint_exhaustive_sr_edge_sets=PASS\n"
                << "nonreplicated_local_sr_edge_graph=PASS\n"
                << "layerwise_owner_ghost_forward=PASS\n"
                << "distributed_transformer_forward_equivalence=PASS\n"
                << "pytorch_transformer_stack012_reference=PASS\n"
                << "distinct_rank_gpu_binding=PASS\n"
                << "SO3LR_NATIVE_DEV30_LOCAL_FORWARD_TEST=PASS\n";
    }
  } catch (const std::exception &error) {
    std::cerr << "rank=" << rank
              << " SO3LR_NATIVE_DEV30_LOCAL_FORWARD_TEST=FAIL: "
              << error.what() << '\n';
    result_code = 1;
  }
  if (result_code != 0) {
    Kokkos::finalize();
    MPI_Abort(MPI_COMM_WORLD, result_code);
    return result_code;
  }
  Kokkos::finalize();
  MPI_Finalize();
  return 0;
}
