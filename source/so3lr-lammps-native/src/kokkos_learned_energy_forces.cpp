#include "so3lr/kokkos_learned_energy_forces.hpp"

#include <Kokkos_Timer.hpp>

#include <cmath>
#include <stdexcept>

namespace so3lr {
namespace {

constexpr std::size_t ev_width = 24;
constexpr double pi = 3.141592653589793238462643383279502884;

KOKKOS_INLINE_FUNCTION
void evaluate_real_spherical_harmonics(double x, double y, double z,
                                       double *value) {
  const double c1 = Kokkos::sqrt(3.0 / (4.0 * pi));
  value[0] = c1 * y;
  value[1] = c1 * z;
  value[2] = c1 * x;

  const double a2 = 0.5 * Kokkos::sqrt(15.0 / pi);
  const double b2 = 0.25 * Kokkos::sqrt(5.0 / pi);
  const double c2 = 0.25 * Kokkos::sqrt(15.0 / pi);
  value[3] = a2 * x * y;
  value[4] = a2 * y * z;
  value[5] = b2 * (3.0 * z * z - 1.0);
  value[6] = a2 * x * z;
  value[7] = c2 * (x * x - y * y);

  const double a3 = 0.25 * Kokkos::sqrt(35.0 / (2.0 * pi));
  const double b3 = 0.5 * Kokkos::sqrt(105.0 / pi);
  const double c3 = 0.25 * Kokkos::sqrt(21.0 / (2.0 * pi));
  const double d3 = 0.25 * Kokkos::sqrt(7.0 / pi);
  const double e3 = 0.25 * Kokkos::sqrt(105.0 / pi);
  value[8] = a3 * y * (3.0 * x * x - y * y);
  value[9] = b3 * x * y * z;
  value[10] = c3 * y * (5.0 * z * z - 1.0);
  value[11] = d3 * (5.0 * z * z * z - 3.0 * z);
  value[12] = c3 * x * (5.0 * z * z - 1.0);
  value[13] = e3 * (x * x - y * y) * z;
  value[14] = a3 * x * (x * x - 3.0 * y * y);

  const double a4 = 0.75 * Kokkos::sqrt(35.0 / pi);
  const double b4 = 0.75 * Kokkos::sqrt(35.0 / (2.0 * pi));
  const double c4 = 0.75 * Kokkos::sqrt(5.0 / pi);
  const double d4 = 0.75 * Kokkos::sqrt(5.0 / (2.0 * pi));
  const double e4 = 0.1875 * Kokkos::sqrt(1.0 / pi);
  const double f4 = 0.375 * Kokkos::sqrt(5.0 / pi);
  const double g4 = 0.1875 * Kokkos::sqrt(35.0 / pi);
  value[15] = a4 * x * y * (x * x - y * y);
  value[16] = b4 * y * (3.0 * x * x - y * y) * z;
  value[17] = c4 * x * y * (7.0 * z * z - 1.0);
  value[18] = d4 * y * (7.0 * z * z * z - 3.0 * z);
  value[19] = e4 *
              (35.0 * z * z * z * z - 30.0 * z * z + 3.0);
  value[20] = d4 * x * (7.0 * z * z * z - 3.0 * z);
  value[21] = f4 * (x * x - y * y) * (7.0 * z * z - 1.0);
  value[22] = b4 * x * (x * x - 3.0 * y * y) * z;
  value[23] = g4 *
              (x * x * (x * x - 3.0 * y * y) -
               y * y * (3.0 * x * x - y * y));
}

KOKKOS_INLINE_FUNCTION
void spherical_harmonic_input_vjp(double x, double y, double z,
                                  const double *gradient,
                                  double &gx, double &gy, double &gz) {
  gx = 0.0;
  gy = 0.0;
  gz = 0.0;
  const double c1 = Kokkos::sqrt(3.0 / (4.0 * pi));
  gy += gradient[0] * c1;
  gz += gradient[1] * c1;
  gx += gradient[2] * c1;

  const double a2 = 0.5 * Kokkos::sqrt(15.0 / pi);
  const double b2 = 0.25 * Kokkos::sqrt(5.0 / pi);
  const double c2 = 0.25 * Kokkos::sqrt(15.0 / pi);
  gx += gradient[3] * a2 * y;
  gy += gradient[3] * a2 * x;
  gy += gradient[4] * a2 * z;
  gz += gradient[4] * a2 * y;
  gz += gradient[5] * 6.0 * b2 * z;
  gx += gradient[6] * a2 * z;
  gz += gradient[6] * a2 * x;
  gx += gradient[7] * 2.0 * c2 * x;
  gy -= gradient[7] * 2.0 * c2 * y;

  const double a3 = 0.25 * Kokkos::sqrt(35.0 / (2.0 * pi));
  const double b3 = 0.5 * Kokkos::sqrt(105.0 / pi);
  const double c3 = 0.25 * Kokkos::sqrt(21.0 / (2.0 * pi));
  const double d3 = 0.25 * Kokkos::sqrt(7.0 / pi);
  const double e3 = 0.25 * Kokkos::sqrt(105.0 / pi);
  gx += gradient[8] * 6.0 * a3 * x * y;
  gy += gradient[8] * 3.0 * a3 * (x * x - y * y);
  gx += gradient[9] * b3 * y * z;
  gy += gradient[9] * b3 * x * z;
  gz += gradient[9] * b3 * x * y;
  gy += gradient[10] * c3 * (5.0 * z * z - 1.0);
  gz += gradient[10] * 10.0 * c3 * y * z;
  gz += gradient[11] * d3 * (15.0 * z * z - 3.0);
  gx += gradient[12] * c3 * (5.0 * z * z - 1.0);
  gz += gradient[12] * 10.0 * c3 * x * z;
  gx += gradient[13] * 2.0 * e3 * x * z;
  gy -= gradient[13] * 2.0 * e3 * y * z;
  gz += gradient[13] * e3 * (x * x - y * y);
  gx += gradient[14] * 3.0 * a3 * (x * x - y * y);
  gy -= gradient[14] * 6.0 * a3 * x * y;

  const double a4 = 0.75 * Kokkos::sqrt(35.0 / pi);
  const double b4 = 0.75 * Kokkos::sqrt(35.0 / (2.0 * pi));
  const double c4 = 0.75 * Kokkos::sqrt(5.0 / pi);
  const double d4 = 0.75 * Kokkos::sqrt(5.0 / (2.0 * pi));
  const double e4 = 0.1875 * Kokkos::sqrt(1.0 / pi);
  const double f4 = 0.375 * Kokkos::sqrt(5.0 / pi);
  const double g4 = 0.1875 * Kokkos::sqrt(35.0 / pi);
  gx += gradient[15] * a4 * y * (3.0 * x * x - y * y);
  gy += gradient[15] * a4 * x * (x * x - 3.0 * y * y);
  gx += gradient[16] * 6.0 * b4 * x * y * z;
  gy += gradient[16] * 3.0 * b4 * (x * x - y * y) * z;
  gz += gradient[16] * b4 * y * (3.0 * x * x - y * y);
  gx += gradient[17] * c4 * y * (7.0 * z * z - 1.0);
  gy += gradient[17] * c4 * x * (7.0 * z * z - 1.0);
  gz += gradient[17] * 14.0 * c4 * x * y * z;
  gy += gradient[18] * d4 * (7.0 * z * z * z - 3.0 * z);
  gz += gradient[18] * d4 * y * (21.0 * z * z - 3.0);
  gz += gradient[19] * e4 * (140.0 * z * z * z - 60.0 * z);
  gx += gradient[20] * d4 * (7.0 * z * z * z - 3.0 * z);
  gz += gradient[20] * d4 * x * (21.0 * z * z - 3.0);
  gx += gradient[21] * 2.0 * f4 * x * (7.0 * z * z - 1.0);
  gy -= gradient[21] * 2.0 * f4 * y * (7.0 * z * z - 1.0);
  gz += gradient[21] * 14.0 * f4 * z * (x * x - y * y);
  gx += gradient[22] * 3.0 * b4 * (x * x - y * y) * z;
  gy -= gradient[22] * 6.0 * b4 * x * y * z;
  gz += gradient[22] * b4 * x * (x * x - 3.0 * y * y);
  gx += gradient[23] * 4.0 * g4 * x * (x * x - 3.0 * y * y);
  gy += gradient[23] * 4.0 * g4 * y * (y * y - 3.0 * x * x);
}

std::size_t force_workspace_bytes(std::size_t nodes, std::size_t edges,
                                  std::size_t persistent_bytes) {
  constexpr std::size_t block_node_doubles = 4516;
  constexpr std::size_t block_edge_doubles = 1027;
  constexpr std::size_t shared_reverse_node_doubles = 2256;
  constexpr std::size_t shared_reverse_edge_doubles = 549;
  constexpr std::size_t chain_node_doubles =
      3 * block_node_doubles - 2 * shared_reverse_node_doubles +
      128 + 24 + 24 + 1220;
  constexpr std::size_t chain_edge_doubles =
      3 * block_edge_doubles - 2 * shared_reverse_edge_doubles +
      1 + 24 + 1 + 24;
  // Add Cartesian edge vectors, edge-energy gradients, and atom forces. The
  // chain count already includes z, indices, distances and SH vectors.
  return persistent_bytes +
         nodes * ((chain_node_doubles + 3) * sizeof(double) +
                  sizeof(std::int64_t)) +
         edges * ((chain_edge_doubles + 6) * sizeof(double) +
                  2 * sizeof(std::size_t));
}

}  // namespace

GeometryVJPWorkspace::GeometryVJPWorkspace(
    std::size_t nodes, std::size_t edges,
    const std::vector<std::size_t> &feature_to_sh_map)
    : distances("so3lr_geometry_distances", edges),
      sh_vectors("so3lr_geometry_sh",
                 edges * (feature_to_sh_map.empty() ? ev_width
                                                    : feature_to_sh_map.size())),
      edge_energy_gradients("so3lr_geometry_edge_gradients", edges * 3),
      atomic_forces("so3lr_geometry_atomic_forces", nodes * 3) {
  if (nodes == 0 || edges == 0)
    throw std::runtime_error("SO3LR geometry VJP workspace is empty");
  if (!feature_to_sh_map.empty()) {
    for (const std::size_t channel : feature_to_sh_map)
      if (channel >= ev_width)
        throw std::runtime_error(
            "SO3LR geometry: feature channel maps past the degree-4 harmonics");
    feature_to_sh = IndexView("so3lr_geometry_feature_to_sh",
                              feature_to_sh_map.size());
    auto host = Kokkos::create_mirror_view(feature_to_sh);
    for (std::size_t i = 0; i < feature_to_sh_map.size(); ++i)
      host(i) = feature_to_sh_map[i];
    Kokkos::deep_copy(feature_to_sh, host);
  }
}

void launch_so3lr_geometry_forward_device(
    const Kokkos::View<double *> &edge_vectors,
    const GeometryVJPWorkspace &workspace) {
  const std::size_t edges = workspace.distances.extent(0);
  const std::size_t feature_width = workspace.feature_to_sh.extent(0);
  if (edges == 0 || edge_vectors.extent(0) != edges * 3 ||
      workspace.sh_vectors.extent(0) !=
          edges * (feature_width == 0 ? ev_width : feature_width))
    throw std::runtime_error("SO3LR geometry-forward view mismatch");
  const auto distance = workspace.distances;
  const auto sh = workspace.sh_vectors;
  if (feature_width != 0) {
    // Duplicated degrees: evaluate the 24 distinct harmonics once per edge and
    // replicate them into the feature layout the attention kernels read.
    const auto map = workspace.feature_to_sh;
    Kokkos::parallel_for(
        "so3lr_geometry_forward_expanded", Kokkos::RangePolicy<>(0, edges),
        KOKKOS_LAMBDA(const std::size_t edge) {
          const double vx = edge_vectors(edge * 3);
          const double vy = edge_vectors(edge * 3 + 1);
          const double vz = edge_vectors(edge * 3 + 2);
          const double r = Kokkos::sqrt(vx * vx + vy * vy + vz * vz);
          distance(edge) = r;
          double values[ev_width];
          const double inverse = r > 1.0e-14 ? 1.0 / r : 0.0;
          evaluate_real_spherical_harmonics(vx * inverse, vy * inverse,
                                            vz * inverse, values);
          for (std::size_t channel = 0; channel < feature_width; ++channel)
            sh(edge * feature_width + channel) = values[map(channel)];
        });
    return;
  }
  Kokkos::parallel_for(
      "so3lr_geometry_forward", Kokkos::RangePolicy<>(0, edges),
      KOKKOS_LAMBDA(const std::size_t edge) {
        const double vx = edge_vectors(edge * 3);
        const double vy = edge_vectors(edge * 3 + 1);
        const double vz = edge_vectors(edge * 3 + 2);
        const double r = Kokkos::sqrt(vx * vx + vy * vy + vz * vz);
        distance(edge) = r;
        double values[ev_width];
        // The native adapter stores v = x_receiver - x_sender.
        // LAMMPS MLIAP supplies rij = x_sender - x_receiver and the
        // PyTorch SO3LR model evaluates Y(-rij/r), hence Y(v/r).
        const double inverse = r > 1.0e-14 ? 1.0 / r : 0.0;
        evaluate_real_spherical_harmonics(vx * inverse, vy * inverse,
                                          vz * inverse, values);
        for (std::size_t channel = 0; channel < ev_width; ++channel)
          sh(edge * ev_width + channel) = values[channel];
      });
}

void launch_so3lr_geometry_reverse_device(
    const Kokkos::View<double *> &edge_vectors,
    const Kokkos::View<std::size_t *> &senders,
    const Kokkos::View<std::size_t *> &receivers,
    const Kokkos::View<double *> &grad_distances,
    const Kokkos::View<double *> &grad_sh,
    const GeometryVJPWorkspace &workspace) {
  const std::size_t edges = workspace.distances.extent(0);
  const std::size_t nodes = workspace.atomic_forces.extent(0) / 3;
  if (edges == 0 || nodes == 0 || edge_vectors.extent(0) != edges * 3 ||
      senders.extent(0) != edges || receivers.extent(0) != edges ||
      grad_distances.extent(0) != edges ||
      grad_sh.extent(0) != edges * ev_width ||
      workspace.edge_energy_gradients.extent(0) != edges * 3)
    throw std::runtime_error("SO3LR geometry-reverse view mismatch");
  Kokkos::deep_copy(workspace.atomic_forces, 0.0);
  const auto edge_gradient = workspace.edge_energy_gradients;
  const auto atom_force = workspace.atomic_forces;
  Kokkos::parallel_for(
      "so3lr_geometry_reverse_scatter", Kokkos::RangePolicy<>(0, edges),
      KOKKOS_LAMBDA(const std::size_t edge) {
        const double vx = edge_vectors(edge * 3);
        const double vy = edge_vectors(edge * 3 + 1);
        const double vz = edge_vectors(edge * 3 + 2);
        const double r = Kokkos::sqrt(vx * vx + vy * vy + vz * vz);
        if (r <= 1.0e-14) {
          edge_gradient(edge * 3) = 0.0;
          edge_gradient(edge * 3 + 1) = 0.0;
          edge_gradient(edge * 3 + 2) = 0.0;
          return;
        }
        const double nx = vx / r;
        const double ny = vy / r;
        const double nz = vz / r;
        const double ux = nx;
        const double uy = ny;
        const double uz = nz;
        double sh_gradient[ev_width];
        for (std::size_t channel = 0; channel < ev_width; ++channel)
          sh_gradient[channel] = grad_sh(edge * ev_width + channel);
        double gux, guy, guz;
        spherical_harmonic_input_vjp(ux, uy, uz, sh_gradient,
                                     gux, guy, guz);
        const double radial_component = ux * gux + uy * guy + uz * guz;
        const double inverse_r = 1.0 / r;
        const double gx = grad_distances(edge) * nx +
                          (gux - ux * radial_component) * inverse_r;
        const double gy = grad_distances(edge) * ny +
                          (guy - uy * radial_component) * inverse_r;
        const double gz = grad_distances(edge) * nz +
                          (guz - uz * radial_component) * inverse_r;
        edge_gradient(edge * 3) = gx;
        edge_gradient(edge * 3 + 1) = gy;
        edge_gradient(edge * 3 + 2) = gz;
        const std::size_t sender = senders(edge);
        const std::size_t receiver = receivers(edge);
        if (sender < nodes && receiver < nodes) {
          Kokkos::atomic_add(&atom_force(sender * 3), gx);
          Kokkos::atomic_add(&atom_force(sender * 3 + 1), gy);
          Kokkos::atomic_add(&atom_force(sender * 3 + 2), gz);
          Kokkos::atomic_add(&atom_force(receiver * 3), -gx);
          Kokkos::atomic_add(&atom_force(receiver * 3 + 1), -gy);
          Kokkos::atomic_add(&atom_force(receiver * 3 + 2), -gz);
        }
      });
}

LearnedEnergyForcesWorkspace::LearnedEnergyForcesWorkspace(
    std::size_t nodes, std::size_t edges)
    : geometry(nodes, edges), chain(nodes, edges) {}

KokkosLearnedEnergyForces::KokkosLearnedEnergyForces(
    const NativeModel &model)
    : chain_(model) {
  force_contract_verified_ = chain_.chain_contract_verified() &&
                             chain_.shared_kokkos_stream();
  if (!force_contract_verified_)
    throw std::runtime_error("SO3LR learned-energy force contract failed");
}

void KokkosLearnedEnergyForces::launch_device(
    const Int64View &atomic_numbers, const DoubleView &edge_vectors,
    const IndexView &senders, const IndexView &receivers,
    const LearnedEnergyForcesWorkspace &workspace) const {
  launch_so3lr_geometry_forward_device(edge_vectors, workspace.geometry);
  chain_.launch_forward_device(
      atomic_numbers, workspace.geometry.distances,
      workspace.geometry.sh_vectors, senders, receivers, workspace.chain);
  chain_.launch_reverse_device(
      atomic_numbers, workspace.geometry.distances,
      workspace.geometry.sh_vectors, senders, receivers, workspace.chain);
  launch_so3lr_geometry_reverse_device(
      edge_vectors, senders, receivers, chain_.grad_distances(workspace.chain),
      chain_.grad_sh(workspace.chain), workspace.geometry);
}

LearnedEnergyForcesBenchmark KokkosLearnedEnergyForces::benchmark(
    std::size_t nodes, std::size_t edges,
    std::size_t repetitions) const {
  if (nodes == 0 || edges == 0 || repetitions == 0)
    throw std::runtime_error("SO3LR learned-force benchmark invalid");
  Int64View z("so3lr_force_bench_z", nodes);
  DoubleView vectors("so3lr_force_bench_vectors", edges * 3);
  IndexView senders("so3lr_force_bench_senders", edges);
  IndexView receivers("so3lr_force_bench_receivers", edges);
  Kokkos::parallel_for(
      "so3lr_force_bench_nodes", Kokkos::RangePolicy<>(0, nodes),
      KOKKOS_LAMBDA(const std::size_t i) { z(i) = i % 3 == 0 ? 8 : 1; });
  Kokkos::parallel_for(
      "so3lr_force_bench_edges", Kokkos::RangePolicy<>(0, edges),
      KOKKOS_LAMBDA(const std::size_t edge) {
        std::size_t sender = (edge * 17 + 11) % nodes;
        std::size_t receiver = edge % nodes;
        if (receiver == sender) receiver = (receiver + 1) % nodes;
        senders(edge) = sender;
        receivers(edge) = receiver;
        const double ax = Kokkos::sin(0.017 * static_cast<double>(edge + 1));
        const double ay = Kokkos::cos(0.013 * static_cast<double>(edge + 3));
        const double az = 1.25 +
            0.35 * Kokkos::sin(0.011 * static_cast<double>(edge + 7));
        const double norm = Kokkos::sqrt(ax * ax + ay * ay + az * az);
        const double radius =
            0.15 + 4.29 * static_cast<double>(edge % 8192) / 8191.0;
        vectors(edge * 3) = radius * ax / norm;
        vectors(edge * 3 + 1) = radius * ay / norm;
        vectors(edge * 3 + 2) = radius * az / norm;
      });
  LearnedEnergyForcesWorkspace workspace(nodes, edges);
  launch_device(z, vectors, senders, receivers, workspace);
  Kokkos::fence();

  Kokkos::Timer timer;
  for (std::size_t i = 0; i < repetitions; ++i)
    launch_so3lr_geometry_forward_device(vectors, workspace.geometry);
  Kokkos::fence();
  const double geometry_forward_ms =
      timer.seconds() * 1000.0 / static_cast<double>(repetitions);

  timer.reset();
  for (std::size_t i = 0; i < repetitions; ++i)
    launch_so3lr_geometry_reverse_device(
        vectors, senders, receivers, chain_.grad_distances(workspace.chain),
        chain_.grad_sh(workspace.chain), workspace.geometry);
  Kokkos::fence();
  const double geometry_reverse_ms =
      timer.seconds() * 1000.0 / static_cast<double>(repetitions);

  timer.reset();
  for (std::size_t i = 0; i < repetitions; ++i)
    launch_device(z, vectors, senders, receivers, workspace);
  Kokkos::fence();
  const double full_ms =
      timer.seconds() * 1000.0 / static_cast<double>(repetitions);
  double checksum = 0.0;
  const auto atom_force = workspace.geometry.atomic_forces;
  const auto edge_gradient = workspace.geometry.edge_energy_gradients;
  const auto energy = workspace.chain.heads.atomic_energies;
  Kokkos::parallel_reduce(
      "so3lr_force_bench_checksum", Kokkos::RangePolicy<>(0, 3),
      KOKKOS_LAMBDA(const int which, double &update) {
        if (which == 0) update += energy(nodes / 3);
        if (which == 1) update += atom_force((nodes / 2) * 3 + 1);
        if (which == 2) update += edge_gradient((edges / 4) * 3 + 2);
      }, checksum);
  Kokkos::fence();

  LearnedEnergyForcesBenchmark result;
  result.nodes = nodes;
  result.edges = edges;
  result.repetitions = repetitions;
  result.persistent_device_bytes = persistent_device_bytes();
  result.workspace_bytes =
      force_workspace_bytes(nodes, edges, persistent_device_bytes());
  result.host_boundary_bytes_per_iteration = 0;
  result.geometry_forward_milliseconds = geometry_forward_ms;
  result.geometry_reverse_milliseconds = geometry_reverse_ms;
  result.full_energy_force_milliseconds = full_ms;
  result.edges_per_second = static_cast<double>(edges) / (full_ms / 1000.0);
  result.checksum = checksum;
  return result;
}

}  // namespace so3lr
