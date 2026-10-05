#include "so3lr/kokkos_local_cartesian_forces.hpp"
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
  std::vector<double> vectors;
  std::vector<std::size_t> senders;
  std::vector<std::size_t> receivers;
  std::vector<std::size_t> original_edges;
};

LocalGraph receiver_owned_graph(int rank, std::size_t nodes,
                                const std::vector<double> &vectors,
                                const std::vector<std::size_t> &senders,
                                const std::vector<std::size_t> &receivers) {
  LocalGraph local;
  for (std::size_t edge = 0; edge < senders.size(); ++edge) {
    if (!is_owned(rank, receivers[edge], nodes)) continue;
    local.original_edges.push_back(edge);
    local.senders.push_back(senders[edge]);
    local.receivers.push_back(receivers[edge]);
    for (std::size_t component = 0; component < 3; ++component)
      local.vectors.push_back(vectors[edge * 3 + component]);
  }
  if (local.senders.empty())
    throw std::runtime_error("rank-local Cartesian graph is empty");
  return local;
}

double compare(const std::vector<double> &actual,
               const std::vector<double> &expected,
               const std::string &label, double absolute = 5.0e-7,
               double relative = 5.0e-7) {
  if (actual.size() != expected.size())
    throw std::runtime_error(label + " size mismatch");
  double maximum = 0.0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const double error = std::abs(actual[i] - expected[i]);
    maximum = std::max(maximum, error);
    if (error > absolute + relative * std::abs(expected[i]))
      throw std::runtime_error(label + " mismatch at " +
                               std::to_string(i));
  }
  return maximum;
}

void sendrecv(const DoubleView &send, const DoubleView &receive, int peer,
              int tag, const char *label) {
  if (send.extent(0) > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      receive.extent(0) >
          static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::runtime_error("MPI device message exceeds INT_MAX");
  Kokkos::fence();
  mpi_check(MPI_Sendrecv(send.data(), static_cast<int>(send.extent(0)),
                         MPI_DOUBLE, peer, tag, receive.data(),
                         static_cast<int>(receive.extent(0)), MPI_DOUBLE, peer,
                         tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE),
            label);
}

void publish_owner_features(
    const Exchange &exchange, const IndexView &owners, const IndexView &ghosts,
    const DoubleView &inv, const DoubleView &ev,
    so3lr::NodeFeatureExchangeWorkspace &workspace, int peer, int tag) {
  exchange.pack_device(inv, ev, owners, workspace.send_buffer);
  sendrecv(workspace.send_buffer, workspace.receive_buffer, peer, tag,
           "MPI_Sendrecv forward features");
  exchange.unpack_overwrite_device(workspace.receive_buffer, ghosts, inv, ev);
  Kokkos::fence();
}

double selected_feature_magnitude(const DoubleView &inv, const DoubleView &ev,
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
  const double magnitude =
      selected_feature_magnitude(grad_inv, grad_ev, host_ghosts);
  if (magnitude <= 1.0e-14)
    throw std::runtime_error("local reverse produced no ghost adjoint");
  exchange.pack_device(grad_inv, grad_ev, ghosts, workspace.send_buffer);
  sendrecv(workspace.send_buffer, workspace.receive_buffer, peer, tag,
           "MPI_Sendrecv reverse adjoints");
  reverse.zero_selected_adjoint_device(ghosts, grad_inv, grad_ev);
  exchange.unpack_accumulate_device(workspace.receive_buffer, owners, grad_inv,
                                    grad_ev);
  Kokkos::fence();
  if (selected_feature_magnitude(grad_inv, grad_ev, host_ghosts) > 1.0e-14)
    throw std::runtime_error("ghost adjoints not cleared after exchange");
  return magnitude;
}

std::vector<double> assemble_local_edges(
    const DoubleView &local, std::size_t width,
    const std::vector<std::size_t> &original_edges, std::size_t full_edges) {
  const auto host = copy(local);
  if (host.size() != original_edges.size() * width)
    throw std::runtime_error("local edge value shape mismatch");
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
            "MPI_Allreduce local edges");
  return global;
}

std::vector<double> assemble_owned_forces(
    const DoubleView &forces, const std::vector<std::size_t> &owners,
    std::size_t nodes) {
  const auto host = copy(forces);
  std::vector<double> contribution(nodes * 3, 0.0);
  for (const auto node : owners)
    for (std::size_t component = 0; component < 3; ++component)
      contribution[node * 3 + component] = host[node * 3 + component];
  std::vector<double> global(nodes * 3, 0.0);
  mpi_check(MPI_Allreduce(contribution.data(), global.data(),
                          static_cast<int>(global.size()), MPI_DOUBLE, MPI_SUM,
                          MPI_COMM_WORLD),
            "MPI_Allreduce owned forces");
  return global;
}

double selected_force_magnitude(const DoubleView &forces,
                                const std::vector<std::size_t> &nodes) {
  const auto host = copy(forces);
  double maximum = 0.0;
  for (const auto node : nodes)
    for (std::size_t component = 0; component < 3; ++component)
      maximum = std::max(maximum,
                         std::abs(host[node * 3 + component]));
  return maximum;
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
      throw std::runtime_error("dev_32 requires exactly two MPI ranks");
    if (argc != 3)
      throw std::runtime_error("usage: test MODEL CARTESIAN_FORCE_FIXTURE");
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
            "so3lr-native-learned-energy-cartesian-force-fixture-v1" ||
        fixture.at("vector_convention").string() != "receiver_minus_sender")
      throw std::runtime_error("unexpected Cartesian-force fixture");
    const auto host_z = integers(fixture.at("atomic_numbers"));
    const auto full_vectors = numbers(fixture.at("edge_vectors"));
    const auto full_senders = indices(fixture.at("senders"));
    const auto full_receivers = indices(fixture.at("receivers"));
    const auto positions = numbers(fixture.at("positions"));
    const std::size_t nodes = host_z.size();
    const std::size_t full_edges = full_senders.size();
    const auto local = receiver_owned_graph(rank, nodes, full_vectors,
                                            full_senders, full_receivers);
    const auto host_owners = owned_nodes(rank, nodes);
    const auto host_ghosts = ghost_nodes(rank, nodes);
    if (local.senders.size() * 2 != full_edges)
      throw std::runtime_error("Cartesian fixture not evenly partitioned");

    Int64View z("so3lr_dev32_z", nodes);
    DoubleView vectors("so3lr_dev32_local_vectors", local.vectors.size());
    IndexView senders("so3lr_dev32_local_senders", local.senders.size());
    IndexView receivers("so3lr_dev32_local_receivers", local.receivers.size());
    IndexView owners("so3lr_dev32_owners", host_owners.size());
    IndexView ghosts("so3lr_dev32_ghosts", host_ghosts.size());
    IndexView owned_mask("so3lr_dev32_owned_mask", nodes);
    fill(z, host_z);
    fill(vectors, local.vectors);
    fill(senders, local.senders);
    fill(receivers, local.receivers);
    fill(owners, host_owners);
    fill(ghosts, host_ghosts);
    std::vector<std::size_t> host_owned_mask(nodes, 0);
    for (const auto node : host_owners) host_owned_mask[node] = 1;
    fill(owned_mask, host_owned_mask);

    const so3lr::KokkosLocalCartesianForces force_model(model);
    if (!force_model.distributed_cartesian_force_contract() ||
        !force_model.receiver_owned_edge_contract())
      throw std::runtime_error("local Cartesian-force API contract failed");
    so3lr::LocalCartesianForcesWorkspace workspace(nodes,
                                                    local.senders.size());
    const auto &reverse = force_model.reverse();
    const Exchange exchange;
    so3lr::NodeFeatureExchangeWorkspace forward_boundary(host_owners.size(),
                                                         host_ghosts.size());
    so3lr::NodeFeatureExchangeWorkspace reverse_boundary(host_ghosts.size(),
                                                         host_owners.size());

    force_model.launch_geometry_forward_device(vectors, workspace);
    reverse.launch_block0_forward_device(
        z, force_model.distances(workspace), force_model.sh_vectors(workspace),
        senders, receivers, workspace.reverse);
    publish_owner_features(exchange, owners, ghosts,
                           reverse.block_final_inv(workspace.reverse, 0),
                           reverse.block_final_ev(workspace.reverse, 0),
                           forward_boundary, peer, 3200);
    reverse.launch_block1_forward_device(
        force_model.distances(workspace), force_model.sh_vectors(workspace),
        senders, receivers, workspace.reverse);
    publish_owner_features(exchange, owners, ghosts,
                           reverse.block_final_inv(workspace.reverse, 1),
                           reverse.block_final_ev(workspace.reverse, 1),
                           forward_boundary, peer, 3201);
    reverse.launch_block2_forward_device(
        force_model.distances(workspace), force_model.sh_vectors(workspace),
        senders, receivers, workspace.reverse);
    reverse.launch_owned_energy_head_reverse_device(z, owned_mask,
                                                     workspace.reverse);

    reverse.launch_block2_reverse_device(
        force_model.distances(workspace), force_model.sh_vectors(workspace),
        senders, receivers, workspace.reverse);
    const double block2_ghost_adjoint = accumulate_ghost_adjoints(
        exchange, reverse, owners, ghosts, host_ghosts,
        reverse.block_grad_inv(workspace.reverse, 2),
        reverse.block_grad_ev(workspace.reverse, 2), reverse_boundary, peer,
        3202);
    reverse.launch_block1_reverse_device(
        force_model.distances(workspace), force_model.sh_vectors(workspace),
        senders, receivers, workspace.reverse);
    const double block1_ghost_adjoint = accumulate_ghost_adjoints(
        exchange, reverse, owners, ghosts, host_ghosts,
        reverse.block_grad_inv(workspace.reverse, 1),
        reverse.block_grad_ev(workspace.reverse, 1), reverse_boundary, peer,
        3203);
    reverse.launch_block0_reverse_device(
        force_model.distances(workspace), force_model.sh_vectors(workspace),
        senders, receivers, workspace.reverse);
    force_model.launch_geometry_reverse_device(vectors, senders, receivers,
                                               workspace);
    Kokkos::fence();

    const auto global_edge_gradients = assemble_local_edges(
        force_model.edge_energy_gradients(workspace), 3, local.original_edges,
        full_edges);
    const double edge_gradient_error = compare(
        global_edge_gradients, numbers(fixture.at("edge_energy_gradients")),
        "distributed Cartesian edge gradients");

    const double ghost_force_before = selected_force_magnitude(
        force_model.atomic_forces(workspace), host_ghosts);
    if (ghost_force_before <= 1.0e-14)
      throw std::runtime_error("local geometry produced no ghost force");
    DoubleView force_send("so3lr_dev32_force_send", host_ghosts.size() * 3);
    DoubleView force_receive("so3lr_dev32_force_receive",
                             host_owners.size() * 3);
    force_model.pack_selected_forces_device(ghosts, force_send, workspace);
    sendrecv(force_send, force_receive, peer, 3204,
             "MPI_Sendrecv reverse forces");
    force_model.zero_selected_forces_device(ghosts, workspace);
    force_model.accumulate_packed_forces_device(force_receive, owners,
                                                workspace);
    Kokkos::fence();
    const double ghost_force_after = selected_force_magnitude(
        force_model.atomic_forces(workspace), host_ghosts);
    if (ghost_force_after > 1.0e-14)
      throw std::runtime_error("ghost forces not cleared after reverse exchange");

    const auto distributed_forces = assemble_owned_forces(
        force_model.atomic_forces(workspace), host_owners, nodes);
    const double atomic_force_error = compare(
        distributed_forces, numbers(fixture.at("atomic_forces")),
        "distributed atomic forces");

    std::array<double, 3> net_force{};
    std::array<double, 3> net_torque{};
    for (std::size_t node = 0; node < nodes; ++node) {
      const double fx = distributed_forces[node * 3];
      const double fy = distributed_forces[node * 3 + 1];
      const double fz = distributed_forces[node * 3 + 2];
      const double x = positions[node * 3];
      const double y = positions[node * 3 + 1];
      const double z_position = positions[node * 3 + 2];
      net_force[0] += fx;
      net_force[1] += fy;
      net_force[2] += fz;
      net_torque[0] += y * fz - z_position * fy;
      net_torque[1] += z_position * fx - x * fz;
      net_torque[2] += x * fy - y * fx;
    }
    const double maximum_net_force = *std::max_element(
        net_force.begin(), net_force.end(),
        [](double a, double b) { return std::abs(a) < std::abs(b); });
    const double maximum_net_torque = *std::max_element(
        net_torque.begin(), net_torque.end(),
        [](double a, double b) { return std::abs(a) < std::abs(b); });
    if (std::abs(maximum_net_force) > 2.0e-8 ||
        std::abs(maximum_net_torque) > 2.0e-8)
      throw std::runtime_error("distributed force invariance failed");

    double local_max_error = std::max(edge_gradient_error, atomic_force_error);
    double global_max_error = 0.0;
    mpi_check(MPI_Reduce(&local_max_error, &global_max_error, 1, MPI_DOUBLE,
                         MPI_MAX, 0, MPI_COMM_WORLD),
              "MPI_Reduce force maximum error");

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
                << "reverse_adjoint_exchange_boundaries=2\n"
                << "reverse_force_exchange_boundaries=1\n"
                << "block2_ghost_adjoint_pre_exchange_max_abs="
                << block2_ghost_adjoint << '\n'
                << "block1_ghost_adjoint_pre_exchange_max_abs="
                << block1_ghost_adjoint << '\n'
                << "ghost_force_pre_exchange_max_abs=" << ghost_force_before
                << '\n'
                << "ghost_force_post_exchange_max_abs=" << ghost_force_after
                << '\n'
                << "edge_gradient_max_abs_error=" << edge_gradient_error
                << '\n'
                << "atomic_force_max_abs_error=" << atomic_force_error << '\n'
                << "maximum_abs_net_force=" << std::abs(maximum_net_force)
                << '\n'
                << "maximum_abs_net_torque=" << std::abs(maximum_net_torque)
                << '\n'
                << "distributed_cartesian_global_max_abs_error="
                << global_max_error << '\n'
                << "application_host_staging_bytes=0\n"
                << "blockwise_geometry_gradient_reduction=PASS\n"
                << "rank_local_cartesian_edge_vjp=PASS\n"
                << "ghost_force_zero_after_send=PASS\n"
                << "reverse_ghost_to_owner_force_accumulation=PASS\n"
                << "distributed_atomic_force_equivalence=PASS\n"
                << "translation_rotation_invariance=PASS\n"
                << "pytorch_learned_energy_cartesian_force_reference=PASS\n"
                << "nonreplicated_local_sr_force_graph=PASS\n"
                << "distinct_rank_gpu_binding=PASS\n"
                << "SO3LR_NATIVE_DEV32_LOCAL_CARTESIAN_FORCE_TEST=PASS\n";
    }
  } catch (const std::exception &error) {
    std::cerr << "rank=" << rank
              << " SO3LR_NATIVE_DEV32_LOCAL_CARTESIAN_FORCE_TEST=FAIL: "
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
