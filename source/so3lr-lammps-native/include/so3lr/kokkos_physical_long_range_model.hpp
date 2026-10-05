#pragma once

#include "so3lr/kokkos_physical_long_range.hpp"
#include "so3lr/native_model.hpp"

#include <Kokkos_Core.hpp>

namespace so3lr {

// The self-contained physical LR part of a native model.  Unlike the complete
// evaluator, this object does not instantiate the GNN or its workspaces.
class KokkosPhysicalLongRangeModel {
 public:
  using DoubleView = Kokkos::View<double *>;
  using Int64View = Kokkos::View<std::int64_t *>;
  using IndexView = Kokkos::View<std::size_t *>;

  explicit KokkosPhysicalLongRangeModel(const NativeModel &model);

  void launch_device(const Int64View &atomic_numbers,
                     const DoubleView &partial_charges,
                     const DoubleView &hirshfeld_ratios,
                     const DoubleView &pair_vectors,
                     const IndexView &senders, const IndexView &receivers,
                     const PhysicalLongRangeWorkspace &workspace,
                     const DoubleView &c6_ratios = DoubleView()) const;

  const PhysicalLongRangeParameters &parameters() const { return parameters_; }
  std::size_t persistent_device_bytes() const {
    return 2 * reference_alphas_.extent(0) * sizeof(double);
  }

 private:
  DoubleView reference_alphas_;
  DoubleView reference_c6_;
  PhysicalLongRangeParameters parameters_;
};

}  // namespace so3lr
