#include "so3lr/kokkos_local_energy_reverse.hpp"
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
#include <numeric>
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
  std::vector<std::size_t> original_edges;
};

LocalGraph receiver_owned_graph(int rank, std::size_t nodes,
                                const std::vector<double> &distances,
                                const std::vector<double> &sh,
                                const std::vector<std::size_t> &senders,
                                const std::vector<std::size_t> &receivers) {
  LocalGraph local;
  for (std::size_t edge = 0; edge < senders.size(); ++edge) {
    if (!is_owned(rank, receivers[edge], nodes)) continue;
    local.original_edges.push_back(edge);
    local.distances.push_back(distances[edge]);
    local.senders.push_back(senders[edge]);
    local.receivers.push_back(receivers[edge]);
    for (std::size_t channel = 0; channel < Exchange::equivariant_width;
         ++channel)
      local.sh.push_back(
          sh[edge * Exchange::equivariant_width + channel]);
  }
  if (local.distances.empty())
    throw std::runtime_error("rank-local reverse graph is empty");
  return local;
}

double compare(const std::vector<double> &actual,
               const std::vector<double> &expected,
               const std::string &label) {
  if (actual.size() != expected.size())
    throw std::runtime_error(label + " size mismatch");
  double maximum = 0.0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const double error = std::abs(actual[i] - expected[i]);
    maximum = std::max(maximum, error);
    if (error > 5.0e-8 + 5.0e-8 * std::abs(expected[i]))
      throw std::runtime_error(label + " mismatch at " +
                               std::to_string(i));
  }
  return maximum;
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
            "MPI_Sendrecv distributed reverse");
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

double selected_magnitude(const DoubleView &inv, const DoubleView &ev,
                          const std::vector<std::size_t> &nodes) {
  const auto host_inv = copy(inv);
  const auto host_ev = copy(ev);
  double maximum = 0.0;
  for (const auto node : nodes) {
    for (std::size_t channel = 0; channel < Exchange::invariant_width;
         ++channel)
      maximum = std::max(
          maximum,
          std::abs(host_inv[node * Exchange::invariant_width + channel]));
    for (std::size_t channel = 0; channel < Exchange::equivariant_width;
         ++channel)
      maximum = std::max(
          maximum,
          std::abs(host_ev[node * Exchange::equivariant_width + channel]));
  }
  return maximum;
}

double accumulate_ghost_adjoints(
    const Exchange &exchange, const so3lr::KokkosLocalEnergyReverse &reverse,
    const IndexView &owners, const IndexView &ghosts,
    const std::vector<std::size_t> &host_ghosts, const DoubleView &grad_inv,
    const DoubleView &grad_ev,
    so3lr::NodeFeatureExchangeWorkspace &workspace, int peer, int tag) {
  const double ghost_magnitude =
      selected_magnitude(grad_inv, grad_ev, host_ghosts);
  if (ghost_magnitude <= 1.0e-14)
    throw std::runtime_error("local reverse produced no ghost adjoint");
  exchange.pack_device(grad_inv, grad_ev, ghosts, workspace.send_buffer);
  sendrecv(workspace.send_buffer, workspace.receive_buffer, peer, tag);
  reverse.zero_selected_adjoint_device(ghosts, grad_inv, grad_ev);
  exchange.unpack_accumulate_device(workspace.receive_buffer, owners, grad_inv,
                                    grad_ev);
  Kokkos::fence();
  if (selected_magnitude(grad_inv, grad_ev, host_ghosts) > 1.0e-14)
    throw std::runtime_error("ghost adjoints were not cleared after exchange");
  return ghost_magnitude;
}

std::vector<double> assemble_owned_nodes(
    const DoubleView &local, std::size_t width,
    const std::vector<std::size_t> &owners, int rank) {
  const auto host = copy(local);
  std::vector<double> contribution(host.size(), 0.0);
  for (const auto node : owners)
    for (std::size_t channel = 0; channel < width; ++channel)
      contribution[node * width + channel] = host[node * width + channel];
  std::vector<double> global(host.size(), 0.0);
  mpi_check(MPI_Allreduce(contribution.data(), global.data(),
                          static_cast<int>(global.size()), MPI_DOUBLE, MPI_SUM,
                          MPI_COMM_WORLD),
            "MPI_Allreduce owned node values");
  static_cast<void>(rank);
  return global;
}

std::vector<double> assemble_local_edges(
    const DoubleView &local, std::size_t width,
    const std::vector<std::size_t> &original_edges, std::size_t full_edges) {
  const auto host = copy(local);
  if (host.size() != original_edges.size() * width)
    throw std::runtime_error("local edge gradient shape mismatch");
  std::vector<double> contribution(full_edges * width, 0.0);
  for (std::size_t local_edge = 0; local_edge < original_edges.size();
       ++local_edge)
    for (std::size_t channel = 0; channel < width; ++channel)
      contribution[original_edges[local_edge] * width + channel] =
          host[local_edge * width + channel];
  std::vector<double> global(contribution.size(), 0.0);
  mpi_check(MPI_Allreduce(contribution.data(), global.data(),
                          static_cast<int>(global.size()), MPI_DOUBLE, MPI_SUM,
                          MPI_COMM_WORLD),
            "MPI_Allreduce local edge gradients");
  return global;
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
      throw std::runtime_error("dev_31 requires exactly two MPI ranks");
    if (argc != 3)
      throw std::runtime_error("usage: test MODEL ENERGY_REVERSE_FIXTURE");
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
            "so3lr-native-learned-energy-reverse-chain-fixture-v1" ||
        fixture.at("reverse_scope").string() !=
            "energy_head_through_transformer_blocks_2_1_0")
      throw std::runtime_error("unexpected learned-energy reverse fixture");
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
    if (local.senders.size() * 2 != full_edges)
      throw std::runtime_error("dev_31 reverse fixture is not evenly partitioned");

    Int64View z("so3lr_dev31_z", nodes);
    DoubleView distances("so3lr_dev31_local_distances", local.distances.size());
    DoubleView sh("so3lr_dev31_local_sh", local.sh.size());
    IndexView senders("so3lr_dev31_local_senders", local.senders.size());
    IndexView receivers("so3lr_dev31_local_receivers", local.receivers.size());
    IndexView owners("so3lr_dev31_owners", host_owners.size());
    IndexView ghosts("so3lr_dev31_ghosts", host_ghosts.size());
    IndexView owned_mask("so3lr_dev31_owned_mask", nodes);
    fill(z, host_z);
    fill(distances, local.distances);
    fill(sh, local.sh);
    fill(senders, local.senders);
    fill(receivers, local.receivers);
    fill(owners, host_owners);
    fill(ghosts, host_ghosts);
    std::vector<std::size_t> host_owned_mask(nodes, 0);
    for (const auto node : host_owners) host_owned_mask[node] = 1;
    fill(owned_mask, host_owned_mask);

    const so3lr::KokkosLocalEnergyReverse reverse(model);
    if (!reverse.staged_reverse_contract_verified() ||
        !reverse.receiver_owned_edge_contract() ||
        !reverse.owned_output_seed_contract())
      throw std::runtime_error("local energy reverse API contract failed");
    so3lr::LocalEnergyReverseWorkspace workspace(nodes,
                                                  local.senders.size());
    const Exchange exchange;
    so3lr::NodeFeatureExchangeWorkspace forward_boundary(host_owners.size(),
                                                         host_ghosts.size());
    so3lr::NodeFeatureExchangeWorkspace reverse_boundary(host_ghosts.size(),
                                                         host_owners.size());

    reverse.launch_block0_forward_device(z, distances, sh, senders, receivers,
                                         workspace);
    publish_owner_features(exchange, owners, ghosts,
                           reverse.block_final_inv(workspace, 0),
                           reverse.block_final_ev(workspace, 0),
                           forward_boundary, peer, 3100);
    reverse.launch_block1_forward_device(distances, sh, senders, receivers,
                                         workspace);
    publish_owner_features(exchange, owners, ghosts,
                           reverse.block_final_inv(workspace, 1),
                           reverse.block_final_ev(workspace, 1),
                           forward_boundary, peer, 3101);
    reverse.launch_block2_forward_device(distances, sh, senders, receivers,
                                         workspace);
    reverse.launch_owned_energy_head_reverse_device(z, owned_mask, workspace);
    Kokkos::fence();

    const auto local_atomic_energy = copy(reverse.atomic_energies(workspace));
    std::vector<double> energy_contribution(nodes, 0.0);
    for (const auto node : host_owners)
      energy_contribution[node] = local_atomic_energy[node];
    std::vector<double> distributed_atomic_energy(nodes, 0.0);
    mpi_check(MPI_Allreduce(energy_contribution.data(),
                            distributed_atomic_energy.data(),
                            static_cast<int>(nodes), MPI_DOUBLE, MPI_SUM,
                            MPI_COMM_WORLD),
              "MPI_Allreduce atomic energies");
    const double atomic_energy_error = compare(
        distributed_atomic_energy, numbers(fixture.at("atomic_energies")),
        "distributed atomic energies");
    const double distributed_total_energy = std::accumulate(
        distributed_atomic_energy.begin(), distributed_atomic_energy.end(),
        0.0);
    const double total_energy_error = std::abs(
        distributed_total_energy - fixture.at("total_energy").number());
    if (total_energy_error >
        5.0e-8 + 5.0e-8 * std::abs(fixture.at("total_energy").number()))
      throw std::runtime_error("distributed total energy mismatch");

    reverse.launch_block2_reverse_device(distances, sh, senders, receivers,
                                         workspace);
    const double block2_ghost_adjoint = accumulate_ghost_adjoints(
        exchange, reverse, owners, ghosts, host_ghosts,
        reverse.block_grad_inv(workspace, 2),
        reverse.block_grad_ev(workspace, 2), reverse_boundary, peer, 3102);
    reverse.launch_block1_reverse_device(distances, sh, senders, receivers,
                                         workspace);
    const double block1_ghost_adjoint = accumulate_ghost_adjoints(
        exchange, reverse, owners, ghosts, host_ghosts,
        reverse.block_grad_inv(workspace, 1),
        reverse.block_grad_ev(workspace, 1), reverse_boundary, peer, 3103);
    reverse.launch_block0_reverse_device(distances, sh, senders, receivers,
                                         workspace);
    const double block0_ghost_adjoint = accumulate_ghost_adjoints(
        exchange, reverse, owners, ghosts, host_ghosts,
        reverse.embedding_gradient(workspace),
        reverse.initial_ev_gradient(workspace), reverse_boundary, peer, 3104);

    const double embedding_error = compare(
        assemble_owned_nodes(reverse.embedding_gradient(workspace),
                             Exchange::invariant_width, host_owners, rank),
        numbers(fixture.at("grad_embedding")),
        "distributed embedding gradient");
    const double initial_ev_error = compare(
        assemble_owned_nodes(reverse.initial_ev_gradient(workspace),
                             Exchange::equivariant_width, host_owners, rank),
        numbers(fixture.at("grad_initial_ev")),
        "distributed initial EV gradient");

    std::array<double, 3> block_distance_error{};
    std::array<double, 3> block_sh_error{};
    std::vector<double> accumulated_distance(full_edges, 0.0);
    std::vector<double> accumulated_sh(
        full_edges * Exchange::equivariant_width, 0.0);
    for (std::size_t block = 0; block < 3; ++block) {
      const auto global_distance = assemble_local_edges(
          reverse.block_grad_distances(workspace, block), 1,
          local.original_edges, full_edges);
      const auto global_sh = assemble_local_edges(
          reverse.block_grad_sh(workspace, block), Exchange::equivariant_width,
          local.original_edges, full_edges);
      block_distance_error[block] = compare(
          global_distance,
          numbers(fixture.at("grad_distances_block" + std::to_string(block))),
          "distributed block distance gradient");
      block_sh_error[block] = compare(
          global_sh,
          numbers(fixture.at("grad_sh_vectors_block" + std::to_string(block))),
          "distributed block SH gradient");
      for (std::size_t i = 0; i < accumulated_distance.size(); ++i)
        accumulated_distance[i] += global_distance[i];
      for (std::size_t i = 0; i < accumulated_sh.size(); ++i)
        accumulated_sh[i] += global_sh[i];
    }
    const double distance_error = compare(
        accumulated_distance, numbers(fixture.at("grad_distances")),
        "distributed accumulated distance gradient");
    const double sh_error = compare(
        accumulated_sh, numbers(fixture.at("grad_sh_vectors")),
        "distributed accumulated SH gradient");

    double local_max_error =
        std::max({atomic_energy_error, total_energy_error, embedding_error,
                  initial_ev_error, distance_error, sh_error,
                  *std::max_element(block_distance_error.begin(),
                                    block_distance_error.end()),
                  *std::max_element(block_sh_error.begin(),
                                    block_sh_error.end())});
    double global_max_error = 0.0;
    mpi_check(MPI_Reduce(&local_max_error, &global_max_error, 1, MPI_DOUBLE,
                         MPI_MAX, 0, MPI_COMM_WORLD),
              "MPI_Reduce reverse maximum error");

    if (rank == 0) {
      std::array<unsigned char, 16> rank0_uuid{};
      std::array<unsigned char, 16> rank1_uuid{};
      std::copy_n(gathered_uuids.begin(), 16, rank0_uuid.begin());
      std::copy_n(gathered_uuids.begin() + 16, 16, rank1_uuid.begin());
      std::cout << std::setprecision(17)
                << "mpi_ranks=2\n"
                << "rank0_gpu_uuid=" << uuid_string(rank0_uuid) << '\n'
                << "rank1_gpu_uuid=" << uuid_string(rank1_uuid) << '\n'
                << "distinct_physical_gpus=1\n"
                << "global_sr_edges=" << full_edges << '\n'
                << "local_sr_edges_per_rank=" << local.senders.size() << '\n'
                << "sr_edge_replication_factor=1\n"
                << "forward_feature_exchange_boundaries=2\n"
                << "reverse_adjoint_exchange_boundaries=3\n"
                << "production_reverse_interblock_boundaries=2\n"
                << "final_embedding_owner_assembly_exchanges=1\n"
                << "distributed_total_energy=" << distributed_total_energy
                << '\n'
                << "atomic_energy_max_abs_error=" << atomic_energy_error
                << '\n'
                << "total_energy_abs_error=" << total_energy_error << '\n'
                << "embedding_gradient_max_abs_error=" << embedding_error
                << '\n'
                << "initial_ev_gradient_max_abs_error=" << initial_ev_error
                << '\n'
                << "distance_gradient_max_abs_error=" << distance_error
                << '\n'
                << "sh_gradient_max_abs_error=" << sh_error << '\n';
      for (std::size_t block = 0; block < 3; ++block)
        std::cout << "block" << block
                  << "_distance_gradient_max_abs_error="
                  << block_distance_error[block] << '\n'
                  << "block" << block << "_sh_gradient_max_abs_error="
                  << block_sh_error[block] << '\n';
      std::cout << "block2_ghost_adjoint_pre_exchange_max_abs="
                << block2_ghost_adjoint << '\n'
                << "block1_ghost_adjoint_pre_exchange_max_abs="
                << block1_ghost_adjoint << '\n'
                << "block0_ghost_adjoint_pre_exchange_max_abs="
                << block0_ghost_adjoint << '\n'
                << "distributed_reverse_global_max_abs_error="
                << global_max_error << '\n'
                << "application_host_staging_bytes=0\n"
                << "owned_energy_head_seeding=PASS\n"
                << "ghost_adjoint_zero_after_send=PASS\n"
                << "reverse_ghost_to_owner_accumulation=PASS\n"
                << "nonreplicated_local_sr_reverse_graph=PASS\n"
                << "distributed_embedding_gradient=PASS\n"
                << "distributed_per_edge_geometry_gradients=PASS\n"
                << "distributed_energy_reverse_equivalence=PASS\n"
                << "pytorch_learned_energy_reverse_chain_reference=PASS\n"
                << "distinct_rank_gpu_binding=PASS\n"
                << "SO3LR_NATIVE_DEV31_LOCAL_REVERSE_TEST=PASS\n";
    }
  } catch (const std::exception &error) {
    std::cerr << "rank=" << rank
              << " SO3LR_NATIVE_DEV31_LOCAL_REVERSE_TEST=FAIL: "
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
