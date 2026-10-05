#include "so3lr/kokkos_local_cartesian_forces.hpp"

#include <stdexcept>
#include <vector>

namespace so3lr {
namespace {

constexpr std::size_t sh_width = 24;
// Upper bound on interaction depth: per-block views are captured by value in
// a Kokkos::Array so a single fused kernel can sum them.
constexpr std::size_t kMaxInteractionBlocks = 8;

// The expansion map to hand to the geometry workspace: empty when the feature
// layout is exactly the 24 distinct harmonics (every model without repeated
// degrees), so those models keep the original geometry kernel.
std::vector<std::size_t> expansion_map(const So3lrArchitecture &arch) {
  bool identity = arch.feature_to_sh_channel.size() == sh_width;
  for (std::size_t i = 0; identity && i < sh_width; ++i)
    identity = arch.feature_to_sh_channel[i] == i;
  return identity ? std::vector<std::size_t>{} : arch.feature_to_sh_channel;
}

}  // namespace

LocalCartesianForcesWorkspace::LocalCartesianForcesWorkspace(
    std::size_t nodes, std::size_t local_edges, std::size_t layers)
    : reverse(nodes, local_edges, layers),
      geometry(nodes, local_edges),
      accumulated_grad_distances(
          "so3lr_local_cartesian_accumulated_grad_distances", local_edges),
      accumulated_grad_sh("so3lr_local_cartesian_accumulated_grad_sh",
                          local_edges * sh_width) {
  if (nodes == 0 || local_edges == 0)
    throw std::runtime_error("SO3LR local Cartesian workspace is empty");
}

LocalCartesianForcesWorkspace::LocalCartesianForcesWorkspace(
    std::size_t nodes, std::size_t local_edges, const So3lrArchitecture &arch)
    : reverse(nodes, local_edges, arch.layers, arch.dims()),
      geometry(nodes, local_edges, expansion_map(arch)),
      accumulated_grad_distances(
          "so3lr_local_cartesian_accumulated_grad_distances", local_edges),
      accumulated_grad_sh("so3lr_local_cartesian_accumulated_grad_sh",
                          local_edges * sh_width) {
  if (nodes == 0 || local_edges == 0)
    throw std::runtime_error("SO3LR local Cartesian workspace is empty");
  const std::vector<std::size_t> map = expansion_map(arch);
  if (map.empty()) return;
  std::vector<std::size_t> offsets(sh_width + 1, 0);
  for (const std::size_t channel : map) ++offsets[channel + 1];
  for (std::size_t c = 0; c < sh_width; ++c) offsets[c + 1] += offsets[c];
  std::vector<std::size_t> features(map.size());
  std::vector<std::size_t> cursor(offsets.begin(), offsets.end() - 1);
  for (std::size_t f = 0; f < map.size(); ++f) features[cursor[map[f]]++] = f;
  sh_to_feature_offsets =
      IndexView("so3lr_local_cartesian_sh_to_feature_offsets", sh_width + 1);
  sh_to_feature = IndexView("so3lr_local_cartesian_sh_to_feature", map.size());
  auto host_offsets = Kokkos::create_mirror_view(sh_to_feature_offsets);
  auto host_features = Kokkos::create_mirror_view(sh_to_feature);
  for (std::size_t c = 0; c <= sh_width; ++c) host_offsets(c) = offsets[c];
  for (std::size_t f = 0; f < map.size(); ++f) host_features(f) = features[f];
  Kokkos::deep_copy(sh_to_feature_offsets, host_offsets);
  Kokkos::deep_copy(sh_to_feature, host_features);
}

void KokkosLocalCartesianForces::launch_geometry_forward_device(
    const DoubleView &local_edge_vectors,
    const LocalCartesianForcesWorkspace &workspace) const {
  launch_so3lr_geometry_forward_device(local_edge_vectors, workspace.geometry);
}

void KokkosLocalCartesianForces::launch_geometry_reverse_device(
    const DoubleView &local_edge_vectors, const IndexView &local_senders,
    const IndexView &local_receivers,
    const LocalCartesianForcesWorkspace &workspace) const {
  launch_geometry_reverse_with_extra_radial_device(
      local_edge_vectors, local_senders, local_receivers, DoubleView(),
      workspace);
}

void KokkosLocalCartesianForces::launch_geometry_reverse_with_extra_radial_device(
    const DoubleView &local_edge_vectors, const IndexView &local_senders,
    const IndexView &local_receivers, const DoubleView &extra_radial,
    const LocalCartesianForcesWorkspace &workspace) const {
  const std::size_t edges = local_senders.extent(0);
  const bool has_extra = extra_radial.extent(0) != 0;
  if (edges == 0 || local_receivers.extent(0) != edges ||
      local_edge_vectors.extent(0) != edges * 3 ||
      workspace.accumulated_grad_distances.extent(0) != edges ||
      workspace.accumulated_grad_sh.extent(0) != edges * sh_width ||
      (has_extra && extra_radial.extent(0) != edges))
    throw std::runtime_error("SO3LR local Cartesian reverse shape mismatch");

  // Sum every block's geometry adjoint. Accumulating left to right from zero
  // reproduces d0 + d1 + d2 exactly for a three-block model.
  const std::size_t layers = reverse_.layers();
  if (layers == 0 || layers > kMaxInteractionBlocks)
    throw std::runtime_error("SO3LR interaction depth exceeds the compiled maximum");
  Kokkos::Array<DoubleView, kMaxInteractionBlocks> distance_terms;
  Kokkos::Array<DoubleView, kMaxInteractionBlocks> sh_terms;
  for (std::size_t b = 0; b < layers; ++b) {
    distance_terms[b] = reverse_.block_grad_distances(workspace.reverse, b);
    sh_terms[b] = reverse_.block_grad_sh(workspace.reverse, b);
  }
  const auto accumulated_distance = workspace.accumulated_grad_distances;
  Kokkos::parallel_for(
      "so3lr_local_cartesian_accumulate_distance_vjp",
      Kokkos::RangePolicy<>(0, edges),
      KOKKOS_LAMBDA(const std::size_t edge) {
        double sum = 0.0;
        for (std::size_t b = 0; b < layers; ++b) sum += distance_terms[b](edge);
        accumulated_distance(edge) = sum + (has_extra ? extra_radial(edge) : 0.0);
      });

  const auto accumulated_sh = workspace.accumulated_grad_sh;
  const std::size_t feature_width = workspace.sh_to_feature.extent(0);
  if (feature_width == 0) {
    Kokkos::parallel_for(
        "so3lr_local_cartesian_accumulate_sh_vjp",
        Kokkos::RangePolicy<>(0, edges * sh_width),
        KOKKOS_LAMBDA(const std::size_t flat) {
          double sum = 0.0;
          for (std::size_t b = 0; b < layers; ++b) sum += sh_terms[b](flat);
          accumulated_sh(flat) = sum;
        });
  } else {
    // Duplicated degrees: every feature channel that replicates harmonic c
    // contributes to dE/dY_c (adjoint of the forward expansion).
    const auto offsets = workspace.sh_to_feature_offsets;
    const auto features = workspace.sh_to_feature;
    Kokkos::parallel_for(
        "so3lr_local_cartesian_contract_sh_vjp",
        Kokkos::RangePolicy<>(0, edges * sh_width),
        KOKKOS_LAMBDA(const std::size_t flat) {
          const std::size_t edge = flat / sh_width;
          const std::size_t harmonic = flat % sh_width;
          double sum = 0.0;
          for (std::size_t b = 0; b < layers; ++b)
            for (std::size_t k = offsets(harmonic); k < offsets(harmonic + 1);
                 ++k)
              sum += sh_terms[b](edge * feature_width + features(k));
          accumulated_sh(flat) = sum;
        });
  }

  launch_so3lr_geometry_reverse_device(
      local_edge_vectors, local_senders, local_receivers,
      workspace.accumulated_grad_distances, workspace.accumulated_grad_sh,
      workspace.geometry);
}

void KokkosLocalCartesianForces::pack_selected_forces_device(
    const IndexView &nodes, const DoubleView &packed,
    const LocalCartesianForcesWorkspace &workspace) const {
  if (packed.extent(0) != nodes.extent(0) * 3)
    throw std::runtime_error("SO3LR local force pack shape mismatch");
  const auto forces = workspace.geometry.atomic_forces;
  const std::size_t total_nodes = forces.extent(0) / 3;
  Kokkos::parallel_for(
      "so3lr_local_cartesian_pack_forces",
      Kokkos::RangePolicy<>(0, packed.extent(0)),
      KOKKOS_LAMBDA(const std::size_t flat) {
        const std::size_t item = flat / 3;
        const std::size_t component = flat % 3;
        const std::size_t node = nodes(item);
        packed(flat) = node < total_nodes
                           ? forces(node * 3 + component)
                           : 0.0;
      });
}

void KokkosLocalCartesianForces::zero_selected_forces_device(
    const IndexView &nodes,
    const LocalCartesianForcesWorkspace &workspace) const {
  const auto forces = workspace.geometry.atomic_forces;
  const std::size_t total_nodes = forces.extent(0) / 3;
  Kokkos::parallel_for(
      "so3lr_local_cartesian_zero_ghost_forces",
      Kokkos::RangePolicy<>(0, nodes.extent(0) * 3),
      KOKKOS_LAMBDA(const std::size_t flat) {
        const std::size_t item = flat / 3;
        const std::size_t component = flat % 3;
        const std::size_t node = nodes(item);
        if (node < total_nodes) forces(node * 3 + component) = 0.0;
      });
}

void KokkosLocalCartesianForces::accumulate_packed_forces_device(
    const DoubleView &packed, const IndexView &owners,
    const LocalCartesianForcesWorkspace &workspace) const {
  if (packed.extent(0) != owners.extent(0) * 3)
    throw std::runtime_error("SO3LR local force unpack shape mismatch");
  const auto forces = workspace.geometry.atomic_forces;
  const std::size_t total_nodes = forces.extent(0) / 3;
  Kokkos::parallel_for(
      "so3lr_local_cartesian_accumulate_owner_forces",
      Kokkos::RangePolicy<>(0, packed.extent(0)),
      KOKKOS_LAMBDA(const std::size_t flat) {
        const std::size_t item = flat / 3;
        const std::size_t component = flat % 3;
        const std::size_t node = owners(item);
        if (node < total_nodes)
          Kokkos::atomic_add(&forces(node * 3 + component), packed(flat));
      });
}

}  // namespace so3lr
