#pragma once

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdint>

namespace so3lr {

struct PhysicalLongRangeParameters {
  double ke = 14.399645351950548;
  double electrostatic_sigma = 4.0;
  double cutoff = 12.0;
  double electrostatic_cuton = 5.4;
  double fine_structure = 0.0072973525693;
  double bohr = 0.529177210903;
  double hartree = 27.211386245988;
  double dispersion_scale = 1.2;
  double dispersion_cuton = 10.0;
  double pair_scale = 0.5;
};

struct PhysicalLongRangeWorkspace {
  using DoubleView = Kokkos::View<double *>;

  PhysicalLongRangeWorkspace(std::size_t nodes, std::size_t pairs);
  DoubleView atomic_energy;
  DoubleView charge_gradient;
  DoubleView hirshfeld_gradient;
  DoubleView electrostatic_pair_energy;
  DoubleView dispersion_pair_energy;
  DoubleView pair_radial_gradient;
  DoubleView pair_force_vectors;
  DoubleView atomic_forces;
  // dE/d(C6 ratio), for models that predict it with a separate head.
  // Allocated lazily; unused for v1, whose C6 is derived from the Hirshfeld
  // ratio and whose whole dispersion gradient lands in hirshfeld_gradient.
  mutable DoubleView c6_gradient;
};

void launch_so3lr_physical_long_range_device(
    const Kokkos::View<std::int64_t *> &atomic_numbers,
    const Kokkos::View<double *> &partial_charges,
    const Kokkos::View<double *> &hirshfeld_ratios,
    const Kokkos::View<double *> &reference_alphas,
    const Kokkos::View<double *> &reference_c6,
    const Kokkos::View<double *> &pair_vectors,
    const Kokkos::View<std::size_t *> &senders,
    const Kokkos::View<std::size_t *> &receivers,
    const PhysicalLongRangeParameters &parameters,
    const PhysicalLongRangeWorkspace &workspace,
    const Kokkos::View<double *> &c6_ratios = Kokkos::View<double *>());

std::size_t physical_long_range_workspace_bytes(std::size_t nodes,
                                                std::size_t pairs);

}  // namespace so3lr
