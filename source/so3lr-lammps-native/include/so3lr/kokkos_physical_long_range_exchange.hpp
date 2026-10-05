#pragma once

#include "so3lr/kokkos_physical_long_range.hpp"

#include <Kokkos_Core.hpp>

namespace so3lr {

// GPU-resident packing operations used around CUDA-aware MPI.  MPI itself is
// intentionally left to the caller/LAMMPS communication layer.
class KokkosPhysicalLongRangeExchange {
 public:
  using DoubleView = Kokkos::View<double *>;
  using IndexView = Kokkos::View<std::size_t *>;

  void pack_charge_hirshfeld_device(const DoubleView &charges,
                                    const DoubleView &hirshfeld,
                                    const IndexView &selected,
                                    const DoubleView &packed) const;
  void unpack_charge_hirshfeld_overwrite_device(
      const DoubleView &packed, const IndexView &selected,
      const DoubleView &charges, const DoubleView &hirshfeld) const;

  void pack_reverse_fields_device(const PhysicalLongRangeWorkspace &workspace,
                                  const IndexView &selected,
                                  const DoubleView &packed) const;
  void zero_reverse_fields_device(const PhysicalLongRangeWorkspace &workspace,
                                  const IndexView &selected) const;
  void unpack_reverse_fields_accumulate_device(
      const DoubleView &packed, const IndexView &selected,
      const DoubleView &atomic_energy, const DoubleView &charge_gradient,
      const DoubleView &hirshfeld_gradient) const;
};

}  // namespace so3lr
