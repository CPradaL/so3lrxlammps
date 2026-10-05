#pragma once

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdint>

namespace so3lr {

struct PhysicalZblParameters {
  double ke = 0.0;
  double cutoff = 0.0;
  double switch_off = 0.0;
  double a[4]{};
  double c[4]{};
  double p = 0.0;
  double d = 0.0;
  // 0: learnable ZBL (SO3LR v1). 1: parameter-free NLH, whose
  // screening sum is sum_k A_k(zi,zj) exp(-B_k(zi,zj) r) from pair tables.
  // Everything else -- cutoff, 1.5 A switch, Zi Zj ke / r prefactor, the 1/2
  // for directed edges -- is shared.
  int kind = 0;
  std::size_t nlh_z_capacity = 0;
};

struct PhysicalZblWorkspace {
  using DoubleView = Kokkos::View<double *>;

  PhysicalZblWorkspace(std::size_t owned_nodes, std::size_t edges);
  DoubleView atomic_energy;
  DoubleView edge_energy;
  DoubleView edge_radial_gradient;
};

void launch_so3lr_zbl_device(
    const Kokkos::View<std::int64_t *> &atomic_numbers,
    std::size_t owned_nodes,
    const Kokkos::View<double *> &distances,
    const Kokkos::View<std::size_t *> &senders,
    const Kokkos::View<std::size_t *> &receivers,
    const PhysicalZblParameters &parameters,
    const PhysicalZblWorkspace &workspace,
    const Kokkos::View<double *> &nlh_a = Kokkos::View<double *>(),
    const Kokkos::View<double *> &nlh_b = Kokkos::View<double *>());

std::size_t physical_zbl_workspace_bytes(std::size_t owned_nodes,
                                         std::size_t edges);

}  // namespace so3lr
