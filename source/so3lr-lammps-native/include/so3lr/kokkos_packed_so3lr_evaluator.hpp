#pragma once

#include "so3lr/kokkos_image_segments.hpp"
#include "so3lr/kokkos_local_cartesian_forces.hpp"
#include "so3lr/kokkos_physical_long_range.hpp"
#include "so3lr/kokkos_physical_long_range_model.hpp"
#include "so3lr/kokkos_physical_zbl.hpp"
#include "so3lr/so3lr_repulsion_setup.hpp"

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdint>

namespace so3lr {

struct PackedSo3lrWorkspace {
  using DoubleView = Kokkos::View<double *>;
  using IndexView = Kokkos::View<std::size_t *>;

  // Sized for the model's architecture (widths, layers, degree layout).
  PackedSo3lrWorkspace(std::size_t nodes, std::size_t sr_edges,
                       std::size_t lr_pairs, std::size_t images,
                       const So3lrArchitecture &arch);

  LocalCartesianForcesWorkspace forces;
  PhysicalZblWorkspace zbl;
  PhysicalLongRangeWorkspace long_range;
  ImageSegmentWorkspace segments;
  IndexView owned_mask;
  DoubleView energy_seeds;
  DoubleView atomic_energies, atomic_forces, image_energies;
};

// Complete physical SO3LR evaluation for a packed collection of mutually
// disconnected systems (turbo images), for every model the generalized
// so3lr/native/mpi path supports. It is that path on one rank with every node
// owned, so no feature exchange is needed; the global charge constraint and
// its adjoint are applied per image instead of over the whole pack.
class KokkosPackedSo3lrEvaluator {
 public:
  using DoubleView = Kokkos::View<double *>;
  using Int64View = Kokkos::View<std::int64_t *>;
  using IndexView = Kokkos::View<std::size_t *>;

  explicit KokkosPackedSo3lrEvaluator(const NativeModel &model);

  void launch_device(
      const Int64View &atomic_numbers,
      const DoubleView &sr_edge_vectors,
      const IndexView &sr_senders, const IndexView &sr_receivers,
      const DoubleView &lr_pair_vectors,
      const IndexView &lr_senders, const IndexView &lr_receivers,
      const Int64View &image_ids, const DoubleView &total_charges,
      const PackedSo3lrWorkspace &workspace) const;

  const So3lrArchitecture &arch() const { return arch_; }
  const DoubleView &atomic_energies(const PackedSo3lrWorkspace &w) const {
    return w.atomic_energies;
  }
  const DoubleView &atomic_forces(const PackedSo3lrWorkspace &w) const {
    return w.atomic_forces;
  }
  const DoubleView &partial_charges(const PackedSo3lrWorkspace &w) const {
    return w.forces.reverse.heads.partial_charges;
  }
  // SO3LR v1: Hirshfeld ratios; a0 ratios for models whose head predicts those.
  const DoubleView &hirshfeld_ratios(const PackedSo3lrWorkspace &w) const {
    return w.forces.reverse.heads.hirshfeld_ratios;
  }
  bool has_c6_head() const { return forces_.reverse().has_c6_head(); }
  const DoubleView &c6_ratios(const PackedSo3lrWorkspace &w) const {
    return w.forces.reverse.heads.c6_ratios;
  }
  const DoubleView &image_energies(const PackedSo3lrWorkspace &w) const {
    return w.image_energies;
  }
  const PhysicalLongRangeWorkspace &long_range(
      const PackedSo3lrWorkspace &w) const { return w.long_range; }
  bool contract_verified() const { return contract_verified_; }

 private:
  So3lrArchitecture arch_;
  KokkosLocalCartesianForces forces_;
  KokkosPhysicalLongRangeModel long_range_;
  So3lrRepulsion repulsion_;
  bool contract_verified_ = false;
};

}  // namespace so3lr
