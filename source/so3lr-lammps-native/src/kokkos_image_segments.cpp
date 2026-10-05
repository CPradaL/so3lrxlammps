#include "so3lr/kokkos_image_segments.hpp"

#include <stdexcept>

namespace so3lr {

ImageSegmentWorkspace::ImageSegmentWorkspace(std::size_t nodes_in,
                                             std::size_t images_in)
    : nodes(nodes_in),
      images(images_in),
      counts("image_counts", images_in),
      invalid_labels("invalid_image_labels", 1),
      raw_charge_sums("image_raw_charge_sums", images_in),
      energy_sums("image_energy_sums", images_in),
      charge_corrections("image_charge_corrections", images_in),
      corrected_charges("image_corrected_charges", nodes_in),
      charge_seed_sums("image_charge_seed_sums", images_in),
      charge_seed_means("image_charge_seed_means", images_in),
      raw_charge_seeds("image_raw_charge_seeds", nodes_in) {
  if (nodes == 0 || images == 0)
    throw std::invalid_argument("image segmentation requires nodes and images");
}

namespace {

void validate_forward(const ImageSegmentWorkspace::DoubleView &raw_charges,
                      const ImageSegmentWorkspace::DoubleView &atomic_energies,
                      const ImageSegmentWorkspace::Int64View &image_ids,
                      const ImageSegmentWorkspace::DoubleView &total_charges,
                      const ImageSegmentWorkspace &workspace) {
  if (raw_charges.extent(0) != workspace.nodes ||
      atomic_energies.extent(0) != workspace.nodes ||
      image_ids.extent(0) != workspace.nodes ||
      total_charges.extent(0) != workspace.images)
    throw std::invalid_argument("image forward view extent mismatch");
}

void validate_reverse(
    const ImageSegmentWorkspace::DoubleView &partial_charge_seeds,
    const ImageSegmentWorkspace::Int64View &image_ids,
    const ImageSegmentWorkspace &workspace) {
  if (partial_charge_seeds.extent(0) != workspace.nodes ||
      image_ids.extent(0) != workspace.nodes)
    throw std::invalid_argument("image reverse view extent mismatch");
}

}  // namespace

void launch_image_segment_forward(
    const ImageSegmentWorkspace::DoubleView &raw_charges,
    const ImageSegmentWorkspace::DoubleView &atomic_energies,
    const ImageSegmentWorkspace::Int64View &image_ids,
    const ImageSegmentWorkspace::DoubleView &total_charges,
    const ImageSegmentWorkspace &workspace) {
  validate_forward(raw_charges, atomic_energies, image_ids, total_charges,
                   workspace);
  Kokkos::deep_copy(workspace.counts, std::int64_t{0});
  Kokkos::deep_copy(workspace.invalid_labels, std::int64_t{0});
  Kokkos::deep_copy(workspace.raw_charge_sums, 0.0);
  Kokkos::deep_copy(workspace.energy_sums, 0.0);

  const auto counts = workspace.counts;
  const auto invalid = workspace.invalid_labels;
  const auto charge_sums = workspace.raw_charge_sums;
  const auto energy_sums = workspace.energy_sums;
  const std::int64_t images = static_cast<std::int64_t>(workspace.images);
  Kokkos::parallel_for(
      "so3lr_image_segment_accumulate",
      Kokkos::RangePolicy<>(0, workspace.nodes), KOKKOS_LAMBDA(const int i) {
        const std::int64_t image = image_ids(i);
        if (image < 0 || image >= images) {
          Kokkos::atomic_add(&invalid(0), std::int64_t{1});
          return;
        }
        Kokkos::atomic_add(&counts(image), std::int64_t{1});
        Kokkos::atomic_add(&charge_sums(image), raw_charges(i));
        Kokkos::atomic_add(&energy_sums(image), atomic_energies(i));
      });

  const auto corrections = workspace.charge_corrections;
  Kokkos::parallel_for(
      "so3lr_image_segment_corrections",
      Kokkos::RangePolicy<>(0, workspace.images),
      KOKKOS_LAMBDA(const int image) {
        corrections(image) = counts(image) > 0
                                 ? (total_charges(image) - charge_sums(image)) /
                                       static_cast<double>(counts(image))
                                 : 0.0;
      });

  const auto corrected = workspace.corrected_charges;
  Kokkos::parallel_for(
      "so3lr_image_segment_apply_charge",
      Kokkos::RangePolicy<>(0, workspace.nodes), KOKKOS_LAMBDA(const int i) {
        const std::int64_t image = image_ids(i);
        corrected(i) = (image >= 0 && image < images)
                           ? raw_charges(i) + corrections(image)
                           : raw_charges(i);
      });
}

void launch_image_segment_reverse(
    const ImageSegmentWorkspace::DoubleView &partial_charge_seeds,
    const ImageSegmentWorkspace::Int64View &image_ids,
    const ImageSegmentWorkspace &workspace) {
  validate_reverse(partial_charge_seeds, image_ids, workspace);
  Kokkos::deep_copy(workspace.charge_seed_sums, 0.0);

  const auto seed_sums = workspace.charge_seed_sums;
  const std::int64_t images = static_cast<std::int64_t>(workspace.images);
  Kokkos::parallel_for(
      "so3lr_image_segment_seed_accumulate",
      Kokkos::RangePolicy<>(0, workspace.nodes), KOKKOS_LAMBDA(const int i) {
        const std::int64_t image = image_ids(i);
        if (image >= 0 && image < images)
          Kokkos::atomic_add(&seed_sums(image), partial_charge_seeds(i));
      });

  const auto counts = workspace.counts;
  const auto means = workspace.charge_seed_means;
  Kokkos::parallel_for(
      "so3lr_image_segment_seed_means",
      Kokkos::RangePolicy<>(0, workspace.images),
      KOKKOS_LAMBDA(const int image) {
        means(image) = counts(image) > 0
                           ? seed_sums(image) /
                                 static_cast<double>(counts(image))
                           : 0.0;
      });

  const auto raw_seeds = workspace.raw_charge_seeds;
  Kokkos::parallel_for(
      "so3lr_image_segment_apply_seed",
      Kokkos::RangePolicy<>(0, workspace.nodes), KOKKOS_LAMBDA(const int i) {
        const std::int64_t image = image_ids(i);
        raw_seeds(i) = (image >= 0 && image < images)
                           ? partial_charge_seeds(i) - means(image)
                           : partial_charge_seeds(i);
      });
}

}  // namespace so3lr
