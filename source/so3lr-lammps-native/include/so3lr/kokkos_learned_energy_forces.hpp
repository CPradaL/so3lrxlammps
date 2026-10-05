#pragma once

#include "so3lr/kokkos_learned_energy_reverse.hpp"

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace so3lr {

struct GeometryVJPWorkspace {
  using DoubleView = Kokkos::View<double *>;
  using IndexView = Kokkos::View<std::size_t *>;

  // `feature_to_sh` maps each equivariant feature channel to the distinct
  // spherical harmonic it carries. Empty (the default) means the identity on
  // the 24 harmonics of degrees 1..4, which is every model whose degree list
  // has no repeats; {1,1,2,2,3,3,4,4} needs the 48-entry map.
  GeometryVJPWorkspace(std::size_t nodes, std::size_t edges,
                       const std::vector<std::size_t> &feature_to_sh = {});
  DoubleView distances, sh_vectors;
  DoubleView edge_energy_gradients, atomic_forces;
  IndexView feature_to_sh;
};

void launch_so3lr_geometry_forward_device(
    const Kokkos::View<double *> &edge_vectors,
    const GeometryVJPWorkspace &workspace);

void launch_so3lr_geometry_reverse_device(
    const Kokkos::View<double *> &edge_vectors,
    const Kokkos::View<std::size_t *> &senders,
    const Kokkos::View<std::size_t *> &receivers,
    const Kokkos::View<double *> &grad_distances,
    const Kokkos::View<double *> &grad_sh,
    const GeometryVJPWorkspace &workspace);

struct LearnedEnergyForcesWorkspace {
  LearnedEnergyForcesWorkspace(std::size_t nodes, std::size_t edges);
  GeometryVJPWorkspace geometry;
  LearnedEnergyReverseWorkspace chain;
};

struct LearnedEnergyForcesBenchmark {
  std::size_t nodes = 0;
  std::size_t edges = 0;
  std::size_t repetitions = 0;
  std::size_t persistent_device_bytes = 0;
  std::size_t workspace_bytes = 0;
  std::size_t host_boundary_bytes_per_iteration = 0;
  double geometry_forward_milliseconds = 0.0;
  double geometry_reverse_milliseconds = 0.0;
  double full_energy_force_milliseconds = 0.0;
  double edges_per_second = 0.0;
  double checksum = 0.0;
};

class KokkosLearnedEnergyForces {
 public:
  using DoubleView = Kokkos::View<double *>;
  using Int64View = Kokkos::View<std::int64_t *>;
  using IndexView = Kokkos::View<std::size_t *>;

  explicit KokkosLearnedEnergyForces(const NativeModel &model);

  void launch_device(
      const Int64View &atomic_numbers, const DoubleView &edge_vectors,
      const IndexView &senders, const IndexView &receivers,
      const LearnedEnergyForcesWorkspace &workspace) const;
  LearnedEnergyForcesBenchmark benchmark(
      std::size_t nodes, std::size_t edges,
      std::size_t repetitions) const;

  const DoubleView &distances(
      const LearnedEnergyForcesWorkspace &workspace) const {
    return workspace.geometry.distances;
  }
  const DoubleView &sh_vectors(
      const LearnedEnergyForcesWorkspace &workspace) const {
    return workspace.geometry.sh_vectors;
  }
  const DoubleView &edge_energy_gradients(
      const LearnedEnergyForcesWorkspace &workspace) const {
    return workspace.geometry.edge_energy_gradients;
  }
  const DoubleView &atomic_forces(
      const LearnedEnergyForcesWorkspace &workspace) const {
    return workspace.geometry.atomic_forces;
  }
  const DoubleView &atomic_energies(
      const LearnedEnergyForcesWorkspace &workspace) const {
    return chain_.atomic_energies(workspace.chain);
  }
  const DoubleView &reductions(
      const LearnedEnergyForcesWorkspace &workspace) const {
    return chain_.reductions(workspace.chain);
  }

  bool force_contract_verified() const { return force_contract_verified_; }
  bool shared_kokkos_stream() const { return chain_.shared_kokkos_stream(); }
  std::size_t persistent_device_bytes() const {
    return chain_.persistent_device_bytes();
  }

 private:
  KokkosLearnedEnergyReverseChain chain_;
  bool force_contract_verified_ = false;
};

}  // namespace so3lr
