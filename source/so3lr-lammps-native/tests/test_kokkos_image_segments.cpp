#include "so3lr/kokkos_image_segments.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

template <class View, class Vector>
void copy_to_device(const View &view, const Vector &values) {
  if (view.extent(0) != values.size())
    throw std::runtime_error("copy_to_device extent mismatch");
  auto host = Kokkos::create_mirror_view(view);
  for (std::size_t i = 0; i < values.size(); ++i) host(i) = values[i];
  Kokkos::deep_copy(view, host);
}

template <class View>
auto copy_to_host_vector(const View &view) {
  using Value = typename View::non_const_value_type;
  std::vector<Value> values(view.extent(0));
  auto host = Kokkos::create_mirror_view(view);
  Kokkos::deep_copy(host, view);
  for (std::size_t i = 0; i < values.size(); ++i) values[i] = host(i);
  return values;
}

double max_error(const std::vector<double> &a, const std::vector<double> &b) {
  if (a.size() != b.size()) throw std::runtime_error("comparison size mismatch");
  double result = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i)
    result = std::max(result, std::abs(a[i] - b[i]));
  return result;
}

void require(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error(message);
}

}  // namespace

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  int result_code = 0;
  try {
#ifndef KOKKOS_ENABLE_CUDA
    throw std::runtime_error("dev_1 must be compiled with Kokkos CUDA");
#endif
    const std::vector<std::size_t> sizes{3, 5, 2, 7};
    const std::size_t images = sizes.size();
    const std::size_t nodes = 17;
    std::vector<std::int64_t> image_ids;
    std::vector<std::size_t> used(images, 0);
    while (image_ids.size() < nodes) {
      for (std::size_t image = 0; image < images; ++image) {
        if (used[image] < sizes[image]) {
          image_ids.push_back(static_cast<std::int64_t>(image));
          ++used[image];
        }
      }
    }

    std::vector<double> raw_charges(nodes), atomic_energies(nodes), seeds(nodes);
    for (std::size_t i = 0; i < nodes; ++i) {
      raw_charges[i] = 0.17 * std::sin(0.31 * static_cast<double>(i + 1)) - 0.04;
      atomic_energies[i] = -0.23 + 0.019 * static_cast<double>(i) +
                           0.007 * static_cast<double>(image_ids[i]);
      seeds[i] = std::cos(0.27 * static_cast<double>(i + 2)) +
                 0.13 * static_cast<double>(image_ids[i]);
    }
    const std::vector<double> target_charges{0.0, 1.0, -1.0, 2.0};

    std::vector<double> ref_charge_sums(images, 0.0), ref_energies(images, 0.0);
    std::vector<double> ref_seed_sums(images, 0.0);
    for (std::size_t i = 0; i < nodes; ++i) {
      const std::size_t image = static_cast<std::size_t>(image_ids[i]);
      ref_charge_sums[image] += raw_charges[i];
      ref_energies[image] += atomic_energies[i];
      ref_seed_sums[image] += seeds[i];
    }
    std::vector<double> ref_corrected(nodes), ref_raw_seeds(nodes);
    for (std::size_t i = 0; i < nodes; ++i) {
      const std::size_t image = static_cast<std::size_t>(image_ids[i]);
      ref_corrected[i] = raw_charges[i] +
                         (target_charges[image] - ref_charge_sums[image]) /
                             static_cast<double>(sizes[image]);
      ref_raw_seeds[i] = seeds[i] -
                         ref_seed_sums[image] / static_cast<double>(sizes[image]);
    }

    using DoubleView = so3lr::ImageSegmentWorkspace::DoubleView;
    using Int64View = so3lr::ImageSegmentWorkspace::Int64View;
    DoubleView d_raw("test_raw_charges", nodes);
    DoubleView d_energy("test_atomic_energies", nodes);
    DoubleView d_targets("test_total_charges", images);
    DoubleView d_seeds("test_partial_charge_seeds", nodes);
    Int64View d_ids("test_image_ids", nodes);
    copy_to_device(d_raw, raw_charges);
    copy_to_device(d_energy, atomic_energies);
    copy_to_device(d_targets, target_charges);
    copy_to_device(d_seeds, seeds);
    copy_to_device(d_ids, image_ids);

    so3lr::ImageSegmentWorkspace workspace(nodes, images);
    so3lr::launch_image_segment_forward(d_raw, d_energy, d_ids, d_targets,
                                        workspace);
    so3lr::launch_image_segment_reverse(d_seeds, d_ids, workspace);
    Kokkos::fence();

    const auto actual_counts = copy_to_host_vector(workspace.counts);
    const auto actual_energies = copy_to_host_vector(workspace.energy_sums);
    const auto actual_corrected = copy_to_host_vector(workspace.corrected_charges);
    const auto actual_raw_seeds = copy_to_host_vector(workspace.raw_charge_seeds);
    const auto invalid = copy_to_host_vector(workspace.invalid_labels);
    for (std::size_t image = 0; image < images; ++image)
      require(actual_counts[image] == static_cast<std::int64_t>(sizes[image]),
              "per-image count mismatch");
    require(invalid[0] == 0, "valid fixture produced invalid labels");
    const double energy_error = max_error(actual_energies, ref_energies);
    const double charge_error = max_error(actual_corrected, ref_corrected);
    const double reverse_error = max_error(actual_raw_seeds, ref_raw_seeds);
    require(energy_error < 2.0e-13, "per-image energy mismatch");
    require(charge_error < 2.0e-13, "per-image charge correction mismatch");
    require(reverse_error < 2.0e-13, "per-image charge VJP mismatch");

    std::vector<double> corrected_sums(images, 0.0), reverse_sums(images, 0.0);
    for (std::size_t i = 0; i < nodes; ++i) {
      const std::size_t image = static_cast<std::size_t>(image_ids[i]);
      corrected_sums[image] += actual_corrected[i];
      reverse_sums[image] += actual_raw_seeds[i];
    }
    require(max_error(corrected_sums, target_charges) < 5.0e-13,
            "per-image charge conservation failed");
    require(max_error(reverse_sums, std::vector<double>(images, 0.0)) < 5.0e-13,
            "per-image reverse projection failed");

    constexpr std::size_t bench_images = 64;
    constexpr std::size_t nodes_per_image = 162;
    constexpr std::size_t bench_nodes = bench_images * nodes_per_image;
    constexpr int repetitions = 500;
    DoubleView b_raw("benchmark_raw", bench_nodes);
    DoubleView b_energy("benchmark_energy", bench_nodes);
    DoubleView b_targets("benchmark_targets", bench_images);
    DoubleView b_seeds("benchmark_seeds", bench_nodes);
    Int64View b_ids("benchmark_ids", bench_nodes);
    Kokkos::parallel_for(
        "initialize_segment_benchmark", Kokkos::RangePolicy<>(0, bench_nodes),
        KOKKOS_LAMBDA(const int i) {
          b_ids(i) = static_cast<std::int64_t>(i % bench_images);
          b_raw(i) = 1.0e-3 * static_cast<double>((i % 31) - 15);
          b_energy(i) = -0.1 + 1.0e-5 * static_cast<double>(i % 101);
          b_seeds(i) = 0.02 * static_cast<double>((i % 19) - 9);
        });
    Kokkos::deep_copy(b_targets, 0.0);
    so3lr::ImageSegmentWorkspace benchmark_workspace(bench_nodes, bench_images);
    Kokkos::fence();
    Kokkos::Timer timer;
    for (int repetition = 0; repetition < repetitions; ++repetition) {
      so3lr::launch_image_segment_forward(b_raw, b_energy, b_ids, b_targets,
                                          benchmark_workspace);
      so3lr::launch_image_segment_reverse(b_seeds, b_ids, benchmark_workspace);
    }
    Kokkos::fence();
    const double milliseconds = 1000.0 * timer.seconds() /
                                static_cast<double>(repetitions);
    require(std::isfinite(milliseconds) && milliseconds > 0.0,
            "invalid segmented benchmark time");

    std::cout << std::setprecision(12)
              << "execution_space=" << Kokkos::DefaultExecutionSpace::name() << '\n'
              << "correctness_images=" << images << '\n'
              << "correctness_nodes=" << nodes << '\n'
              << "interleaved_image_ids=1\n"
              << "per_image_energy_max_abs_error=" << energy_error << '\n'
              << "per_image_charge_max_abs_error=" << charge_error << '\n'
              << "per_image_reverse_max_abs_error=" << reverse_error << '\n'
              << "benchmark_images=" << bench_images << '\n'
              << "benchmark_nodes=" << bench_nodes << '\n'
              << "benchmark_repetitions=" << repetitions << '\n'
              << "segmented_forward_reverse_ms=" << milliseconds << '\n'
              << "per_image_charge_conservation=PASS\n"
              << "per_image_energy_reduction=PASS\n"
              << "per_image_charge_vjp_projection=PASS\n"
              << "SO3LR_STAGE4_DEV1_SEGMENTED_CORRECTNESS=PASS\n"
              << "SO3LR_STAGE4_DEV1=PASS\n";
  } catch (const std::exception &error) {
    std::cerr << "SO3LR_STAGE4_DEV1=FAIL reason=" << error.what() << '\n';
    result_code = 1;
  }
  Kokkos::finalize();
  return result_code;
}
