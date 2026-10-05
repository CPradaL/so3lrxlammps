#pragma once

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdint>

namespace so3lr {

// Stage-4 primitive: reductions that are independent for every packed MD image.
// image_ids may be interleaved; no contiguous-image ordering is assumed.
struct ImageSegmentWorkspace {
  using DoubleView = Kokkos::View<double *>;
  using Int64View = Kokkos::View<std::int64_t *>;

  ImageSegmentWorkspace(std::size_t nodes, std::size_t images);

  std::size_t nodes = 0;
  std::size_t images = 0;
  Int64View counts;
  Int64View invalid_labels;
  DoubleView raw_charge_sums;
  DoubleView energy_sums;
  DoubleView charge_corrections;
  DoubleView corrected_charges;
  DoubleView charge_seed_sums;
  DoubleView charge_seed_means;
  DoubleView raw_charge_seeds;
};

// Computes per-image energy sums and enforces each image's requested total
// charge independently. Invalid image labels are counted and ignored.
void launch_image_segment_forward(
    const ImageSegmentWorkspace::DoubleView &raw_charges,
    const ImageSegmentWorkspace::DoubleView &atomic_energies,
    const ImageSegmentWorkspace::Int64View &image_ids,
    const ImageSegmentWorkspace::DoubleView &total_charges,
    const ImageSegmentWorkspace &workspace);

// Applies the VJP of the per-image charge correction:
// dL/dq_raw(i) = dL/dq(i) - mean_image(dL/dq).
void launch_image_segment_reverse(
    const ImageSegmentWorkspace::DoubleView &partial_charge_seeds,
    const ImageSegmentWorkspace::Int64View &image_ids,
    const ImageSegmentWorkspace &workspace);

}  // namespace so3lr
