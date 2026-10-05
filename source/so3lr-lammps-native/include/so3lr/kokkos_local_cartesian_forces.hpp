#pragma once

#include "so3lr/kokkos_learned_energy_forces.hpp"
#include "so3lr/kokkos_local_energy_reverse.hpp"

#include <Kokkos_Core.hpp>

#include <cstddef>

namespace so3lr {

struct LocalCartesianForcesWorkspace {
  using DoubleView = Kokkos::View<double *>;
  using IndexView = Kokkos::View<std::size_t *>;

  // SO3LR v1 widths; `layers` is the interaction depth.
  LocalCartesianForcesWorkspace(std::size_t nodes, std::size_t local_edges,
                                std::size_t layers = 3);
  // Any supported architecture: depth, widths and the spherical-harmonic
  // expansion all come from the descriptor.
  LocalCartesianForcesWorkspace(std::size_t nodes, std::size_t local_edges,
                                const So3lrArchitecture &arch);
  LocalEnergyReverseWorkspace reverse;
  GeometryVJPWorkspace geometry;
  DoubleView accumulated_grad_distances;
  // Always the 24 distinct harmonics: with duplicated degrees the
  // feature-layout SH gradients of every block are summed into their
  // harmonic here, before the Cartesian VJP.
  DoubleView accumulated_grad_sh;
  // Inverse of geometry.feature_to_sh as CSR (harmonic -> feature channels).
  // Empty when the expansion is the identity.
  IndexView sh_to_feature_offsets, sh_to_feature;
};

class KokkosLocalCartesianForces {
 public:
  using DoubleView = Kokkos::View<double *>;
  using IndexView = Kokkos::View<std::size_t *>;

  explicit KokkosLocalCartesianForces(const NativeModel &model)
      : reverse_(model) {}

  KokkosLocalEnergyReverse const &reverse() const { return reverse_; }

  void launch_geometry_forward_device(
      const DoubleView &local_edge_vectors,
      const LocalCartesianForcesWorkspace &workspace) const;

  void launch_geometry_reverse_device(
      const DoubleView &local_edge_vectors, const IndexView &local_senders,
      const IndexView &local_receivers,
      const LocalCartesianForcesWorkspace &workspace) const;

  // Add a native radial energy derivative (currently ZBL) to the three
  // learned block VJPs before converting the result to Cartesian forces.
  void launch_geometry_reverse_with_extra_radial_device(
      const DoubleView &local_edge_vectors, const IndexView &local_senders,
      const IndexView &local_receivers, const DoubleView &extra_radial,
      const LocalCartesianForcesWorkspace &workspace) const;

  void zero_selected_forces_device(
      const IndexView &nodes,
      const LocalCartesianForcesWorkspace &workspace) const;
  void accumulate_packed_forces_device(
      const DoubleView &packed, const IndexView &owners,
      const LocalCartesianForcesWorkspace &workspace) const;
  void pack_selected_forces_device(
      const IndexView &nodes, const DoubleView &packed,
      const LocalCartesianForcesWorkspace &workspace) const;

  const DoubleView &distances(
      const LocalCartesianForcesWorkspace &workspace) const {
    return workspace.geometry.distances;
  }
  const DoubleView &sh_vectors(
      const LocalCartesianForcesWorkspace &workspace) const {
    return workspace.geometry.sh_vectors;
  }
  const DoubleView &edge_energy_gradients(
      const LocalCartesianForcesWorkspace &workspace) const {
    return workspace.geometry.edge_energy_gradients;
  }
  const DoubleView &atomic_forces(
      const LocalCartesianForcesWorkspace &workspace) const {
    return workspace.geometry.atomic_forces;
  }

  bool distributed_cartesian_force_contract() const {
    return reverse_.staged_reverse_contract_verified();
  }
  bool receiver_owned_edge_contract() const {
    return reverse_.receiver_owned_edge_contract();
  }

 private:
  KokkosLocalEnergyReverse reverse_;
};

}  // namespace so3lr
