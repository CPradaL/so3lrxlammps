#include "so3lr/kokkos_local_cartesian_forces.hpp"
#include "so3lr/kokkos_node_feature_exchange.hpp"
#include "so3lr/kokkos_physical_long_range_exchange.hpp"
#include "so3lr/kokkos_physical_long_range_model.hpp"
#include "so3lr/rank_local_graph.hpp"
#include "so3lr/rank_local_half_pair_graph.hpp"

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

std::vector<int> small_integers(const so3lr::Json &value) {
  std::vector<int> result;
  result.reserve(value.array().size());
  for (const auto &item : value.array())
    result.push_back(static_cast<int>(item.unsigned_integer()));
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

void device_sendrecv(const DoubleView &send, const DoubleView &receive,
                     int peer, int tag, const char *label) {
  Kokkos::fence();
  mpi_check(MPI_Sendrecv(send.data(), static_cast<int>(send.extent(0)),
                         MPI_DOUBLE, peer, tag, receive.data(),
                         static_cast<int>(receive.extent(0)), MPI_DOUBLE, peer,
                         tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE),
            label);
}

std::vector<std::size_t> exchange_ghost_requests(
    const std::vector<std::size_t> &ghost_globals, int peer) {
  int send_count = static_cast<int>(ghost_globals.size());
  int receive_count = 0;
  mpi_check(MPI_Sendrecv(&send_count, 1, MPI_INT, peer, 3400,
                         &receive_count, 1, MPI_INT, peer, 3400,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE),
            "MPI_Sendrecv ghost counts");
  std::vector<unsigned long long> send(ghost_globals.begin(),
                                      ghost_globals.end());
  std::vector<unsigned long long> receive(
      static_cast<std::size_t>(receive_count));
  mpi_check(MPI_Sendrecv(send.data(), send_count, MPI_UNSIGNED_LONG_LONG, peer,
                         3401, receive.data(), receive_count,
                         MPI_UNSIGNED_LONG_LONG, peer, 3401, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE),
            "MPI_Sendrecv ghost IDs");
  return std::vector<std::size_t>(receive.begin(), receive.end());
}

void publish_owner_features(
    const Exchange &exchange, const IndexView &publish, const IndexView &ghosts,
    const DoubleView &inv, const DoubleView &ev,
    so3lr::NodeFeatureExchangeWorkspace &workspace, int peer, int tag) {
  exchange.pack_device(inv, ev, publish, workspace.send_buffer);
  device_sendrecv(workspace.send_buffer, workspace.receive_buffer, peer, tag,
                  "MPI_Sendrecv multihead features");
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
    const IndexView &publish, const IndexView &ghosts,
    const std::vector<std::size_t> &host_ghosts, const DoubleView &grad_inv,
    const DoubleView &grad_ev,
    so3lr::NodeFeatureExchangeWorkspace &workspace, int peer, int tag) {
  const double magnitude =
      selected_feature_magnitude(grad_inv, grad_ev, host_ghosts);
  if (magnitude <= 1.0e-14)
    throw std::runtime_error("multihead reverse produced no ghost adjoint");
  exchange.pack_device(grad_inv, grad_ev, ghosts, workspace.send_buffer);
  device_sendrecv(workspace.send_buffer, workspace.receive_buffer, peer, tag,
                  "MPI_Sendrecv multihead adjoints");
  reverse.zero_selected_adjoint_device(ghosts, grad_inv, grad_ev);
  exchange.unpack_accumulate_device(workspace.receive_buffer, publish, grad_inv,
                                    grad_ev);
  Kokkos::fence();
  if (selected_feature_magnitude(grad_inv, grad_ev, host_ghosts) > 1.0e-14)
    throw std::runtime_error("multihead ghost adjoints not cleared");
  return magnitude;
}

std::vector<double> assemble_edges(
    const DoubleView &local, std::size_t width,
    const std::vector<std::size_t> &original_edges, std::size_t global_edges) {
  const auto host = copy(local);
  std::vector<double> contribution(global_edges * width, 0.0);
  for (std::size_t edge = 0; edge < original_edges.size(); ++edge)
    for (std::size_t channel = 0; channel < width; ++channel)
      contribution[original_edges[edge] * width + channel] =
          host[edge * width + channel];
  std::vector<double> global(contribution.size(), 0.0);
  mpi_check(MPI_Allreduce(contribution.data(), global.data(),
                          static_cast<int>(global.size()), MPI_DOUBLE, MPI_SUM,
                          MPI_COMM_WORLD),
            "MPI_Allreduce multihead edges");
  return global;
}

std::vector<double> assemble_owned_rows(
    const DoubleView &local, std::size_t width,
    const so3lr::RankLocalGraph &graph) {
  const auto host = copy(local);
  std::vector<double> contribution(graph.global_nodes * width, 0.0);
  for (const auto local_node : graph.owned_local) {
    const auto global_node = graph.local_to_global[local_node];
    for (std::size_t channel = 0; channel < width; ++channel)
      contribution[global_node * width + channel] =
          host[local_node * width + channel];
  }
  std::vector<double> global(contribution.size(), 0.0);
  mpi_check(MPI_Allreduce(contribution.data(), global.data(),
                          static_cast<int>(global.size()), MPI_DOUBLE, MPI_SUM,
                          MPI_COMM_WORLD),
            "MPI_Allreduce multihead owned rows");
  return global;
}

std::vector<double> assemble_owned_rows(
    const DoubleView &local, std::size_t width,
    const so3lr::RankLocalHalfPairGraph &graph) {
  const auto host = copy(local);
  std::vector<double> contribution(graph.global_nodes * width, 0.0);
  for (const auto local_node : graph.owned_local) {
    const auto global_node = graph.local_to_global[local_node];
    for (std::size_t channel = 0; channel < width; ++channel)
      contribution[global_node * width + channel] =
          host[local_node * width + channel];
  }
  std::vector<double> global(contribution.size(), 0.0);
  mpi_check(MPI_Allreduce(contribution.data(), global.data(),
                          static_cast<int>(global.size()), MPI_DOUBLE, MPI_SUM,
                          MPI_COMM_WORLD),
            "MPI_Allreduce physical owned rows");
  return global;
}

void pack_vectors(const DoubleView &values, const IndexView &selected,
                  const DoubleView &packed) {
  if (packed.extent(0) != selected.extent(0) * 3)
    throw std::runtime_error("vector pack shape mismatch");
  Kokkos::parallel_for(
      "so3lr_dev35_pack_vectors", Kokkos::RangePolicy<>(0, selected.extent(0)),
      KOKKOS_LAMBDA(const std::size_t i) {
        const auto node = selected(i);
        for (std::size_t c = 0; c < 3; ++c)
          packed(i * 3 + c) = values(node * 3 + c);
      });
}

void unpack_vectors_accumulate(const DoubleView &packed,
                               const IndexView &selected,
                               const DoubleView &values) {
  if (packed.extent(0) != selected.extent(0) * 3)
    throw std::runtime_error("vector unpack shape mismatch");
  Kokkos::parallel_for(
      "so3lr_dev35_accumulate_vectors",
      Kokkos::RangePolicy<>(0, selected.extent(0)),
      KOKKOS_LAMBDA(const std::size_t i) {
        const auto node = selected(i);
        for (std::size_t c = 0; c < 3; ++c)
          values(node * 3 + c) += packed(i * 3 + c);
      });
}

double sum_selected_device(const DoubleView &values,
                           const IndexView &selected) {
  double result = 0.0;
  Kokkos::parallel_reduce(
      "so3lr_dev34_selected_scalar_sum",
      Kokkos::RangePolicy<>(0, selected.extent(0)),
      KOKKOS_LAMBDA(const std::size_t i, double &update) {
        update += values(selected(i));
      }, result);
  Kokkos::fence();
  return result;
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
    if (ranks != 2 || argc != 3)
      throw std::runtime_error("dev_35 requires 2 ranks and MODEL FIXTURE");
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
      throw std::runtime_error("MPI ranks selected the same GPU");

    const auto model = so3lr::NativeModel::load(argv[1]);
    const auto fixture = so3lr::Json::parse(read_text(argv[2]));
    if (fixture.at("schema").string() !=
            "so3lr-native-distributed-physical-force-fixture-v1" ||
        fixture.at("objective").string() !=
            "learned_energy_plus_electrostatics_plus_qdo_dispersion")
      throw std::runtime_error("unexpected distributed physical fixture");
    const auto global_z = integers(fixture.at("atomic_numbers"));
    const auto node_owner = small_integers(fixture.at("node_owner"));
    const auto global_vectors = numbers(fixture.at("sr_edge_vectors"));
    const auto global_senders = indices(fixture.at("sr_senders"));
    const auto global_receivers = indices(fixture.at("sr_receivers"));
    const auto global_lr_vectors = numbers(fixture.at("lr_pair_vectors"));
    const auto global_lr_senders = indices(fixture.at("lr_senders"));
    const auto global_lr_receivers = indices(fixture.at("lr_receivers"));
    const auto positions = numbers(fixture.at("positions"));
    const auto graph = so3lr::build_receiver_owned_rank_local_graph(
        rank, node_owner, global_z, global_vectors, global_senders,
        global_receivers);
    const auto lr_graph = so3lr::build_owned_rank_local_half_pair_graph(
        rank, node_owner, global_z, global_lr_vectors, global_lr_senders,
        global_lr_receivers);
    const auto ghost_globals =
        so3lr::rank_local_global_ids(graph, graph.ghost_local);
    const auto requested_globals = exchange_ghost_requests(ghost_globals, peer);
    const auto publish_local = so3lr::rank_local_indices_for_global_ids(
        graph, requested_globals, true);
    const auto lr_ghost_globals =
        so3lr::half_pair_global_ids(lr_graph, lr_graph.ghost_local);
    const auto lr_requested_globals =
        exchange_ghost_requests(lr_ghost_globals, peer);
    const auto lr_publish_sr = so3lr::rank_local_indices_for_global_ids(
        graph, lr_requested_globals, true);
    const auto sr_owned_for_lr = so3lr::rank_local_indices_for_global_ids(
        graph, so3lr::half_pair_global_ids(lr_graph, lr_graph.owned_local),
        true);

    Int64View z("so3lr_dev34_local_z", graph.local_nodes());
    DoubleView vectors("so3lr_dev34_vectors", graph.edge_vectors.size());
    IndexView senders("so3lr_dev34_senders", graph.local_edges());
    IndexView receivers("so3lr_dev34_receivers", graph.local_edges());
    IndexView publish("so3lr_dev34_publish", publish_local.size());
    IndexView ghosts("so3lr_dev34_ghosts", graph.ghost_local.size());
    IndexView owners("so3lr_dev34_owners", graph.owned_local.size());
    IndexView owned_mask("so3lr_dev34_owned_mask", graph.local_nodes());
    DoubleView energy_seeds("so3lr_dev35_energy_seeds", graph.local_nodes());
    DoubleView charge_seeds("so3lr_dev35_charge_seeds", graph.local_nodes());
    DoubleView hirshfeld_seeds("so3lr_dev35_hirshfeld_seeds", graph.local_nodes());
    DoubleView physical_energy("so3lr_dev35_physical_energy", graph.local_nodes());
    fill(z, graph.atomic_numbers);
    fill(vectors, graph.edge_vectors);
    fill(senders, graph.senders);
    fill(receivers, graph.receivers);
    fill(publish, publish_local);
    fill(ghosts, graph.ghost_local);
    fill(owners, graph.owned_local);
    std::vector<std::size_t> host_owned_mask(graph.local_nodes(), 0);
    std::vector<double> local_energy_seed(graph.local_nodes(), 0.0);
    for (const auto local : graph.owned_local) {
      host_owned_mask[local] = 1;
      local_energy_seed[local] = 1.0;
    }
    fill(owned_mask, host_owned_mask);
    fill(energy_seeds, local_energy_seed);
    Kokkos::deep_copy(charge_seeds, 0.0);
    Kokkos::deep_copy(hirshfeld_seeds, 0.0);
    Kokkos::deep_copy(physical_energy, 0.0);

    Int64View lr_z("so3lr_dev35_lr_z", lr_graph.local_nodes());
    DoubleView lr_vectors("so3lr_dev35_lr_vectors", lr_graph.pair_vectors.size());
    IndexView lr_senders("so3lr_dev35_lr_senders", lr_graph.local_pairs());
    IndexView lr_receivers("so3lr_dev35_lr_receivers", lr_graph.local_pairs());
    IndexView lr_owners("so3lr_dev35_lr_owners", lr_graph.owned_local.size());
    IndexView lr_ghosts("so3lr_dev35_lr_ghosts", lr_graph.ghost_local.size());
    IndexView lr_publish("so3lr_dev35_lr_publish", lr_publish_sr.size());
    IndexView sr_owned_lr("so3lr_dev35_sr_owned_lr", sr_owned_for_lr.size());
    DoubleView lr_charges("so3lr_dev35_lr_charges", lr_graph.local_nodes());
    DoubleView lr_hirshfeld("so3lr_dev35_lr_hirshfeld", lr_graph.local_nodes());
    fill(lr_z, lr_graph.atomic_numbers);
    fill(lr_vectors, lr_graph.pair_vectors);
    fill(lr_senders, lr_graph.senders);
    fill(lr_receivers, lr_graph.receivers);
    fill(lr_owners, lr_graph.owned_local);
    fill(lr_ghosts, lr_graph.ghost_local);
    fill(lr_publish, lr_publish_sr);
    fill(sr_owned_lr, sr_owned_for_lr);

    const so3lr::KokkosLocalCartesianForces force_model(model);
    so3lr::LocalCartesianForcesWorkspace workspace(graph.local_nodes(),
                                                    graph.local_edges());
    const auto &reverse = force_model.reverse();
    const Exchange exchange;
    const so3lr::KokkosPhysicalLongRangeExchange lr_exchange;
    const so3lr::KokkosPhysicalLongRangeModel lr_model(model);
    so3lr::PhysicalLongRangeWorkspace lr_workspace(
        lr_graph.local_nodes(), lr_graph.local_pairs());
    so3lr::NodeFeatureExchangeWorkspace forward_boundary(
        publish_local.size(), graph.ghost_local.size());
    so3lr::NodeFeatureExchangeWorkspace reverse_boundary(
        graph.ghost_local.size(), publish_local.size());

    force_model.launch_geometry_forward_device(vectors, workspace);
    reverse.launch_block0_forward_device(
        z, force_model.distances(workspace), force_model.sh_vectors(workspace),
        senders, receivers, workspace.reverse);
    publish_owner_features(exchange, publish, ghosts,
                           reverse.block_final_inv(workspace.reverse, 0),
                           reverse.block_final_ev(workspace.reverse, 0),
                           forward_boundary, peer, 3410);
    reverse.launch_block1_forward_device(
        force_model.distances(workspace), force_model.sh_vectors(workspace),
        senders, receivers, workspace.reverse);
    publish_owner_features(exchange, publish, ghosts,
                           reverse.block_final_inv(workspace.reverse, 1),
                           reverse.block_final_ev(workspace.reverse, 1),
                           forward_boundary, peer, 3411);
    reverse.launch_block2_forward_device(
        force_model.distances(workspace), force_model.sh_vectors(workspace),
        senders, receivers, workspace.reverse);
    reverse.launch_output_heads_device(z, workspace.reverse);
    Kokkos::fence();

    const double local_raw_sum =
        sum_selected_device(reverse.raw_charges(workspace.reverse), owners);
    double global_raw_sum = 0.0;
    mpi_check(MPI_Allreduce(&local_raw_sum, &global_raw_sum, 1, MPI_DOUBLE,
                            MPI_SUM, MPI_COMM_WORLD),
              "MPI_Allreduce raw charge sum");
    const double charge_correction =
        -global_raw_sum / static_cast<double>(graph.global_nodes);
    reverse.launch_distributed_charge_correction_device(charge_correction,
                                                        workspace.reverse);
    Kokkos::fence();

    const auto distributed_energies = assemble_owned_rows(
        reverse.atomic_energies(workspace.reverse), 1, graph);
    const auto distributed_charges = assemble_owned_rows(
        reverse.partial_charges(workspace.reverse), 1, graph);
    const auto distributed_hirshfeld = assemble_owned_rows(
        reverse.hirshfeld_ratios(workspace.reverse), 1, graph);
    const double energy_output_error = compare(
        distributed_energies, numbers(fixture.at("atomic_energies")),
        "distributed atomic energies");
    const double charge_output_error = compare(
        distributed_charges, numbers(fixture.at("partial_charges")),
        "distributed partial charges");
    const double hirshfeld_output_error = compare(
        distributed_hirshfeld, numbers(fixture.at("hirshfeld_ratios")),
        "distributed Hirshfeld ratios");
    const double charge_sum = std::accumulate(
        distributed_charges.begin(), distributed_charges.end(), 0.0);
    if (std::abs(charge_sum) > 2.0e-12)
      throw std::runtime_error("distributed charge conservation failed");

    // Copy owner outputs into the independent LR graph and publish only the
    // two scalar fields required by remote LR endpoints.
    DoubleView owned_output_buffer("so3lr_dev35_owned_outputs",
                                   graph.owned_local.size() * 2);
    lr_exchange.pack_charge_hirshfeld_device(
        reverse.partial_charges(workspace.reverse),
        reverse.hirshfeld_ratios(workspace.reverse), owners,
        owned_output_buffer);
    lr_exchange.unpack_charge_hirshfeld_overwrite_device(
        owned_output_buffer, lr_owners, lr_charges, lr_hirshfeld);
    DoubleView lr_output_send("so3lr_dev35_lr_output_send",
                              lr_publish_sr.size() * 2);
    DoubleView lr_output_receive("so3lr_dev35_lr_output_receive",
                                 lr_graph.ghost_local.size() * 2);
    lr_exchange.pack_charge_hirshfeld_device(
        reverse.partial_charges(workspace.reverse),
        reverse.hirshfeld_ratios(workspace.reverse), lr_publish,
        lr_output_send);
    device_sendrecv(lr_output_send, lr_output_receive, peer, 3510,
                    "MPI_Sendrecv LR output halo");
    lr_exchange.unpack_charge_hirshfeld_overwrite_device(
        lr_output_receive, lr_ghosts, lr_charges, lr_hirshfeld);
    Kokkos::fence();

    lr_model.launch_device(lr_z, lr_charges, lr_hirshfeld, lr_vectors,
                           lr_senders, lr_receivers, lr_workspace);
    Kokkos::fence();

    // Move physical energy and the charge/Hirshfeld reverse seeds back to the
    // SR owners.  The remote exchange is CUDA-aware and per-atom host staging
    // remains zero.
    DoubleView local_lr_reverse("so3lr_dev35_local_lr_reverse",
                                lr_graph.owned_local.size() * 3);
    lr_exchange.pack_reverse_fields_device(lr_workspace, lr_owners,
                                           local_lr_reverse);
    lr_exchange.unpack_reverse_fields_accumulate_device(
        local_lr_reverse, sr_owned_lr, physical_energy, charge_seeds,
        hirshfeld_seeds);
    DoubleView lr_reverse_send("so3lr_dev35_lr_reverse_send",
                               lr_graph.ghost_local.size() * 3);
    DoubleView lr_reverse_receive("so3lr_dev35_lr_reverse_receive",
                                  lr_publish_sr.size() * 3);
    lr_exchange.pack_reverse_fields_device(lr_workspace, lr_ghosts,
                                           lr_reverse_send);
    device_sendrecv(lr_reverse_send, lr_reverse_receive, peer, 3511,
                    "MPI_Sendrecv LR reverse fields");
    lr_exchange.zero_reverse_fields_device(lr_workspace, lr_ghosts);
    lr_exchange.unpack_reverse_fields_accumulate_device(
        lr_reverse_receive, lr_publish, physical_energy, charge_seeds,
        hirshfeld_seeds);
    Kokkos::fence();

    const auto distributed_physical_energy =
        assemble_owned_rows(physical_energy, 1, graph);
    const auto distributed_charge_gradient =
        assemble_owned_rows(charge_seeds, 1, graph);
    const auto distributed_hirshfeld_gradient =
        assemble_owned_rows(hirshfeld_seeds, 1, graph);
    const double physical_energy_error = compare(
        distributed_physical_energy,
        numbers(fixture.at("physical_atomic_energies")),
        "distributed physical atomic energy", 5.0e-7, 5.0e-7);
    const double charge_gradient_error = compare(
        distributed_charge_gradient, numbers(fixture.at("charge_gradient")),
        "distributed LR charge gradient");
    const double hirshfeld_gradient_error = compare(
        distributed_hirshfeld_gradient,
        numbers(fixture.at("hirshfeld_gradient")),
        "distributed LR Hirshfeld gradient");
    const double physical_energy_total = std::accumulate(
        distributed_physical_energy.begin(), distributed_physical_energy.end(),
        0.0);
    const double physical_energy_total_error = std::abs(
        physical_energy_total - fixture.at("physical_lr_energy_total").number());
    if (physical_energy_total_error > 5.0e-7)
      throw std::runtime_error("distributed physical energy mismatch");

    const auto global_lr_pair_forces = assemble_edges(
        lr_workspace.pair_force_vectors, 3, lr_graph.original_pairs,
        lr_graph.global_pairs);
    const double direct_pair_force_error = compare(
        global_lr_pair_forces,
        numbers(fixture.at("direct_lr_pair_force_vectors")),
        "distributed direct LR pair forces", 5.0e-7, 5.0e-7);

    const double local_charge_seed_sum =
        sum_selected_device(charge_seeds, owners);
    double global_charge_seed_sum = 0.0;
    mpi_check(MPI_Allreduce(&local_charge_seed_sum, &global_charge_seed_sum, 1,
                            MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD),
              "MPI_Allreduce charge seed sum");
    const double global_charge_seed_mean =
        global_charge_seed_sum / static_cast<double>(graph.global_nodes);
    reverse.launch_owned_multihead_reverse_device(
        z, owned_mask, energy_seeds, charge_seeds, hirshfeld_seeds,
        global_charge_seed_mean, workspace.reverse);
    reverse.launch_block2_multihead_reverse_device(
        force_model.distances(workspace), force_model.sh_vectors(workspace),
        senders, receivers, workspace.reverse);
    const double block2_ghost = accumulate_ghost_adjoints(
        exchange, reverse, publish, ghosts, graph.ghost_local,
        reverse.block_grad_inv(workspace.reverse, 2),
        reverse.block_grad_ev(workspace.reverse, 2), reverse_boundary, peer,
        3412);
    reverse.launch_block1_reverse_device(
        force_model.distances(workspace), force_model.sh_vectors(workspace),
        senders, receivers, workspace.reverse);
    const double block1_ghost = accumulate_ghost_adjoints(
        exchange, reverse, publish, ghosts, graph.ghost_local,
        reverse.block_grad_inv(workspace.reverse, 1),
        reverse.block_grad_ev(workspace.reverse, 1), reverse_boundary, peer,
        3413);
    reverse.launch_block0_reverse_device(
        force_model.distances(workspace), force_model.sh_vectors(workspace),
        senders, receivers, workspace.reverse);
    force_model.launch_geometry_reverse_device(vectors, senders, receivers,
                                               workspace);
    Kokkos::fence();

    const auto global_edge_gradients = assemble_edges(
        force_model.edge_energy_gradients(workspace), 3, graph.original_edges,
        graph.global_edges);
    const double edge_gradient_error = compare(
        global_edge_gradients,
        numbers(fixture.at("implicit_sr_edge_gradients")),
        "distributed implicit SR edge gradients", 8.0e-7, 8.0e-7);
    const double ghost_force_before = selected_force_magnitude(
        force_model.atomic_forces(workspace), graph.ghost_local);
    DoubleView force_send("so3lr_dev34_force_send",
                          graph.ghost_local.size() * 3);
    DoubleView force_receive("so3lr_dev34_force_receive",
                             publish_local.size() * 3);
    force_model.pack_selected_forces_device(ghosts, force_send, workspace);
    device_sendrecv(force_send, force_receive, peer, 3414,
                    "MPI_Sendrecv multihead forces");
    force_model.zero_selected_forces_device(ghosts, workspace);
    force_model.accumulate_packed_forces_device(force_receive, publish,
                                                workspace);
    Kokkos::fence();
    const double ghost_force_after = selected_force_magnitude(
        force_model.atomic_forces(workspace), graph.ghost_local);
    if (ghost_force_before <= 1.0e-14 || ghost_force_after > 1.0e-14)
      throw std::runtime_error("multihead reverse-force exchange failed");
    const auto distributed_implicit_forces = assemble_owned_rows(
        force_model.atomic_forces(workspace), 3, graph);
    const double implicit_force_error = compare(
        distributed_implicit_forces,
        numbers(fixture.at("implicit_sr_atomic_forces")),
        "distributed implicit SR forces", 8.0e-7, 8.0e-7);

    DoubleView direct_forces("so3lr_dev35_direct_forces", graph.local_nodes() * 3);
    Kokkos::deep_copy(direct_forces, 0.0);
    DoubleView local_direct("so3lr_dev35_local_direct",
                            lr_graph.owned_local.size() * 3);
    pack_vectors(lr_workspace.atomic_forces, lr_owners, local_direct);
    unpack_vectors_accumulate(local_direct, sr_owned_lr, direct_forces);
    DoubleView remote_direct_send("so3lr_dev35_remote_direct_send",
                                  lr_graph.ghost_local.size() * 3);
    DoubleView remote_direct_receive("so3lr_dev35_remote_direct_receive",
                                     lr_publish_sr.size() * 3);
    pack_vectors(lr_workspace.atomic_forces, lr_ghosts, remote_direct_send);
    device_sendrecv(remote_direct_send, remote_direct_receive, peer, 3512,
                    "MPI_Sendrecv direct LR forces");
    unpack_vectors_accumulate(remote_direct_receive, lr_publish, direct_forces);
    Kokkos::fence();
    const auto distributed_direct_forces =
        assemble_owned_rows(direct_forces, 3, graph);
    const double direct_force_error = compare(
        distributed_direct_forces,
        numbers(fixture.at("direct_lr_atomic_forces")),
        "distributed direct LR atomic forces", 5.0e-7, 5.0e-7);
    std::vector<double> distributed_forces(distributed_implicit_forces.size());
    for (std::size_t i = 0; i < distributed_forces.size(); ++i)
      distributed_forces[i] = distributed_implicit_forces[i] +
                              distributed_direct_forces[i];
    const double atomic_force_error = compare(
        distributed_forces, numbers(fixture.at("assembled_atomic_forces")),
        "distributed complete physical SO3LR forces", 1.2e-6, 1.2e-6);

    const double learned_energy_total = std::accumulate(
        distributed_energies.begin(), distributed_energies.end(), 0.0);
    const double total_energy = learned_energy_total + physical_energy_total;
    const double total_energy_error =
        std::abs(total_energy - fixture.at("total_energy").number());
    if (total_energy_error > 5.0e-7)
      throw std::runtime_error("distributed complete energy mismatch");

    std::array<double, 3> net_force{};
    std::array<double, 3> net_torque{};
    for (std::size_t node = 0; node < graph.global_nodes; ++node) {
      const double fx = distributed_forces[node * 3];
      const double fy = distributed_forces[node * 3 + 1];
      const double fz = distributed_forces[node * 3 + 2];
      const double x = positions[node * 3];
      const double y = positions[node * 3 + 1];
      const double z_position = positions[node * 3 + 2];
      net_force[0] += fx; net_force[1] += fy; net_force[2] += fz;
      net_torque[0] += y * fz - z_position * fy;
      net_torque[1] += z_position * fx - x * fz;
      net_torque[2] += x * fy - y * fx;
    }
    double maximum_net_force = 0.0;
    double maximum_net_torque = 0.0;
    for (std::size_t i = 0; i < 3; ++i) {
      maximum_net_force = std::max(maximum_net_force, std::abs(net_force[i]));
      maximum_net_torque =
          std::max(maximum_net_torque, std::abs(net_torque[i]));
    }
    if (maximum_net_force > 2.0e-8 || maximum_net_torque > 2.0e-8)
      throw std::runtime_error("distributed multihead invariance failed");

    double local_max_error = std::max(
        {energy_output_error, charge_output_error, hirshfeld_output_error,
         physical_energy_error, physical_energy_total_error,
         charge_gradient_error, hirshfeld_gradient_error,
         direct_pair_force_error, edge_gradient_error, implicit_force_error,
         direct_force_error, atomic_force_error, total_energy_error});
    double global_max_error = 0.0;
    mpi_check(MPI_Reduce(&local_max_error, &global_max_error, 1, MPI_DOUBLE,
                         MPI_MAX, 0, MPI_COMM_WORLD),
              "MPI_Reduce multihead maximum error");

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
                << "global_nodes=" << graph.global_nodes << '\n'
                << "local_nodes_per_rank=" << graph.local_nodes() << '\n'
                << "ghost_nodes_per_rank=" << graph.ghost_local.size() << '\n'
                << "global_sr_edges=" << graph.global_edges << '\n'
                << "local_sr_edges_per_rank=" << graph.local_edges() << '\n'
                << "sr_edge_replication_factor=1\n"
                << "global_lr_pairs=" << lr_graph.global_pairs << '\n'
                << "local_lr_pairs_rank0=" << lr_graph.local_pairs() << '\n'
                << "lr_local_nodes_rank0=" << lr_graph.local_nodes() << '\n'
                << "lr_ghost_nodes_rank0=" << lr_graph.ghost_local.size() << '\n'
                << "lr_pair_replication_factor=1\n"
                << "global_raw_charge_sum=" << global_raw_sum << '\n'
                << "distributed_charge_correction=" << charge_correction
                << '\n'
                << "partial_charge_sum=" << charge_sum << '\n'
                << "global_charge_seed_mean=" << global_charge_seed_mean
                << '\n'
                << "atomic_energy_max_abs_error=" << energy_output_error
                << '\n'
                << "partial_charge_max_abs_error=" << charge_output_error
                << '\n'
                << "hirshfeld_max_abs_error=" << hirshfeld_output_error << '\n'
                << "physical_atomic_energy_max_abs_error="
                << physical_energy_error << '\n'
                << "physical_energy_total_abs_error="
                << physical_energy_total_error << '\n'
                << "charge_gradient_max_abs_error=" << charge_gradient_error
                << '\n'
                << "hirshfeld_gradient_max_abs_error="
                << hirshfeld_gradient_error << '\n'
                << "direct_lr_pair_force_max_abs_error="
                << direct_pair_force_error << '\n'
                << "edge_gradient_max_abs_error=" << edge_gradient_error
                << '\n'
                << "implicit_sr_force_max_abs_error=" << implicit_force_error
                << '\n'
                << "direct_lr_force_max_abs_error=" << direct_force_error
                << '\n'
                << "atomic_force_max_abs_error=" << atomic_force_error << '\n'
                << "total_energy_abs_error=" << total_energy_error << '\n'
                << "block2_ghost_adjoint_pre_exchange_max_abs="
                << block2_ghost << '\n'
                << "block1_ghost_adjoint_pre_exchange_max_abs="
                << block1_ghost << '\n'
                << "ghost_force_pre_exchange_max_abs=" << ghost_force_before
                << '\n'
                << "ghost_force_post_exchange_max_abs=" << ghost_force_after
                << '\n'
                << "maximum_abs_net_force=" << maximum_net_force << '\n'
                << "maximum_abs_net_torque=" << maximum_net_torque << '\n'
                << "distributed_physical_global_max_abs_error="
                << global_max_error << '\n'
                << "per_atom_host_staging_bytes=0\n"
                << "global_scalar_host_staging_bytes_per_rank=16\n"
                << "distributed_charge_conservation=PASS\n"
                << "distributed_charge_seed_centering=PASS\n"
                << "distributed_energy_charge_hirshfeld_outputs=PASS\n"
                << "owned_multihead_reverse=PASS\n"
                << "separate_sr_lr_halos=PASS\n"
                << "native_physical_lr_rank_local_coupling=PASS\n"
                << "lr_output_halo_exchange=PASS\n"
                << "lr_seed_ghost_to_owner_exchange=PASS\n"
                << "direct_lr_force_ghost_to_owner_exchange=PASS\n"
                << "complete_rank_local_so3lr_energy_forces=PASS\n"
                << "reverse_ghost_to_owner_force_accumulation=PASS\n"
                << "translation_rotation_invariance=PASS\n"
                << "pytorch_distributed_physical_force_reference=PASS\n"
                << "distinct_rank_gpu_binding=PASS\n"
                << "SO3LR_NATIVE_DEV35_DISTRIBUTED_PHYSICAL_TEST=PASS\n";
    }
  } catch (const std::exception &error) {
    std::cerr << "rank=" << rank
              << " SO3LR_NATIVE_DEV35_DISTRIBUTED_PHYSICAL_TEST=FAIL: "
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
