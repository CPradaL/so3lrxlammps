#include "so3lr/kokkos_input_ops.hpp"

#include <Kokkos_Timer.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace so3lr {
namespace {

template <class View, class Reader>
void fill_device_view(const View &device, std::size_t count, Reader reader) {
  auto host = Kokkos::create_mirror_view(device);
  for (std::size_t index = 0; index < count; ++index) host(index) = reader(index);
  Kokkos::deep_copy(device, host);
}

std::size_t product(const std::vector<std::size_t> &shape) {
  std::size_t result = 1;
  for (const auto extent : shape) result *= extent;
  return result;
}

// NVCC extended device lambdas cannot be lexically enclosed by a private
// member function.  Keep the public class interface small and place the
// actual launches in this translation-unit-local free function instead.
void launch_input_kernels(
    const Kokkos::View<std::int64_t *> &atomic_numbers,
    const Kokkos::View<double *> &distances,
    const Kokkos::View<double *> &embedding,
    const Kokkos::View<double *> &cutoff,
    const Kokkos::View<double *> &radial_basis,
    const Kokkos::View<double *> &weight,
    const Kokkos::View<double *> &b,
    const Kokkos::View<std::int64_t *> &k,
    const Kokkos::View<std::int64_t *> &k_reverse,
    double inverse_embedding_scale, double gamma, double rmax) {
  const std::size_t atoms = atomic_numbers.extent(0);
  const std::size_t edges = distances.extent(0);
  Kokkos::parallel_for(
      "so3lr_invariant_embedding", Kokkos::RangePolicy<>(0, atoms * 128),
      KOKKOS_LAMBDA(const std::size_t flat) {
        const std::size_t atom = flat / 128;
        const std::size_t feature = flat % 128;
        const std::int64_t z = atomic_numbers(atom);
        embedding(flat) = (z > 0 && z <= 118)
                              ? weight(feature * 118 + static_cast<std::size_t>(z - 1)) *
                                    inverse_embedding_scale
                              : 0.0;
      });
  Kokkos::parallel_for(
      "so3lr_physnet_cutoff", Kokkos::RangePolicy<>(0, edges),
      KOKKOS_LAMBDA(const std::size_t edge) {
        const double distance = distances(edge);
        if (distance < rmax) {
          const double x = distance / rmax;
          const double x2 = x * x;
          const double x3 = x2 * x;
          cutoff(edge) = 1.0 - 10.0 * x3 + 15.0 * x3 * x -
                         6.0 * x3 * x2;
        } else {
          cutoff(edge) = 0.0;
        }
      });
  Kokkos::parallel_for(
      "so3lr_bernstein_rbf", Kokkos::RangePolicy<>(0, edges * 32),
      KOKKOS_LAMBDA(const std::size_t flat) {
        const std::size_t edge = flat / 32;
        const std::size_t channel = flat % 32;
        double x = Kokkos::exp(-gamma * distances(edge));
        if (x < 1.0e-6) x = 1.0e-6;
        if (x > 1.0 - 1.0e-6) x = 1.0 - 1.0e-6;
        const double log_poly =
            b(channel) + static_cast<double>(k(channel)) * Kokkos::log(x) +
            static_cast<double>(k_reverse(channel)) * Kokkos::log(1.0 - x);
        radial_basis(flat) = Kokkos::exp(log_poly);
      });
}

}  // namespace

KokkosInputOps::KokkosInputOps(const NativeModel &model)
    : embedding_weight_("so3lr_embedding_weight", 128 * 118),
      bernstein_b_("so3lr_bernstein_b", 32),
      bernstein_k_("so3lr_bernstein_k", 32),
      bernstein_k_reverse_("so3lr_bernstein_k_reverse", 32),
      inverse_embedding_scale_(
          1.0 / model.model_number("embedding_scale")),
      gamma_(model.float64(
          model.tensor("model.radial_embedding.radial_basis_fn.gamma"), 0)),
      cutoff_radius_(model.architecture_number("short_range_cutoff_angstrom")) {
  const auto &weight = model.tensor("model.inv_feature_embedding.embedding.weight");
  const auto &b = model.tensor("model.radial_embedding.radial_basis_fn.b");
  const auto &k = model.tensor("model.radial_embedding.radial_basis_fn.k");
  const auto &k_reverse = model.tensor("model.radial_embedding.radial_basis_fn.k_rev");
  if (!std::isfinite(inverse_embedding_scale_) ||
      inverse_embedding_scale_ <= 0.0)
    throw std::runtime_error("SO3LR embedding scale must be positive and finite");
  if (weight.shape != std::vector<std::size_t>({128, 118}) ||
      b.shape != std::vector<std::size_t>({32}) ||
      k.shape != std::vector<std::size_t>({32}) ||
      k_reverse.shape != std::vector<std::size_t>({32}) ||
      product(weight.shape) != 128 * 118) {
    throw std::runtime_error("SO3LR Kokkos input tensor contract mismatch");
  }
  fill_device_view(embedding_weight_, 128 * 118,
                   [&](std::size_t i) { return model.float64(weight, i); });
  fill_device_view(bernstein_b_, 32,
                   [&](std::size_t i) { return model.float64(b, i); });
  fill_device_view(bernstein_k_, 32,
                   [&](std::size_t i) { return model.int64(k, i); });
  fill_device_view(bernstein_k_reverse_, 32,
                   [&](std::size_t i) { return model.int64(k_reverse, i); });
  Kokkos::fence();
  persistent_device_bytes_ = 128 * 118 * sizeof(double) + 32 * sizeof(double) +
                             64 * sizeof(std::int64_t);
}

void KokkosInputOps::launch(const IntView &atomic_numbers,
                            const DoubleView &distances,
                            const DoubleView &embedding,
                            const DoubleView &cutoff,
                            const DoubleView &radial_basis) const {
  launch_input_kernels(atomic_numbers, distances, embedding, cutoff,
                       radial_basis, embedding_weight_, bernstein_b_,
                       bernstein_k_, bernstein_k_reverse_,
                       inverse_embedding_scale_, gamma_, cutoff_radius_);
}

InputResults KokkosInputOps::evaluate(
    const std::vector<std::int64_t> &atomic_numbers,
    const std::vector<double> &distances) const {
  IntView device_z("so3lr_atomic_numbers", atomic_numbers.size());
  DoubleView device_distances("so3lr_distances", distances.size());
  DoubleView embedding("so3lr_embedding", atomic_numbers.size() * 128);
  DoubleView cutoff("so3lr_cutoff", distances.size());
  DoubleView radial_basis("so3lr_radial_basis", distances.size() * 32);
  fill_device_view(device_z, atomic_numbers.size(),
                   [&](std::size_t i) { return atomic_numbers[i]; });
  fill_device_view(device_distances, distances.size(),
                   [&](std::size_t i) { return distances[i]; });
  launch(device_z, device_distances, embedding, cutoff, radial_basis);
  Kokkos::fence();

  const auto embedding_host = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), embedding);
  const auto cutoff_host = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), cutoff);
  const auto rbf_host = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), radial_basis);
  InputResults result;
  result.embedding.resize(embedding.extent(0));
  result.cutoff.resize(cutoff.extent(0));
  result.radial_basis.resize(radial_basis.extent(0));
  for (std::size_t i = 0; i < result.embedding.size(); ++i)
    result.embedding[i] = embedding_host(i);
  for (std::size_t i = 0; i < result.cutoff.size(); ++i)
    result.cutoff[i] = cutoff_host(i);
  for (std::size_t i = 0; i < result.radial_basis.size(); ++i)
    result.radial_basis[i] = rbf_host(i);
  return result;
}

InputBenchmark KokkosInputOps::benchmark(std::size_t atoms, std::size_t edges,
                                         std::size_t repetitions) const {
  if (atoms == 0 || edges == 0 || repetitions == 0)
    throw std::runtime_error("SO3LR benchmark sizes must be positive");
  IntView device_z("so3lr_benchmark_z", atoms);
  DoubleView distances("so3lr_benchmark_distances", edges);
  DoubleView embedding("so3lr_benchmark_embedding", atoms * 128);
  DoubleView cutoff("so3lr_benchmark_cutoff", edges);
  DoubleView radial_basis("so3lr_benchmark_rbf", edges * 32);
  fill_device_view(device_z, atoms,
                   [](std::size_t i) { return static_cast<std::int64_t>(i % 3 == 0 ? 8 : 1); });
  fill_device_view(distances, edges, [](std::size_t i) {
    return 0.05 + 4.449 * static_cast<double>(i % 8192) / 8191.0;
  });
  for (int warmup = 0; warmup < 3; ++warmup)
    launch(device_z, distances, embedding, cutoff, radial_basis);
  Kokkos::fence();
  Kokkos::Timer timer;
  for (std::size_t iteration = 0; iteration < repetitions; ++iteration)
    launch(device_z, distances, embedding, cutoff, radial_basis);
  Kokkos::fence();
  const double elapsed_ms = timer.seconds() * 1000.0;

  const auto embedding_host = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), embedding);
  const auto cutoff_host = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), cutoff);
  const auto rbf_host = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), radial_basis);
  InputBenchmark result;
  result.atoms = atoms;
  result.edges = edges;
  result.repetitions = repetitions;
  result.workspace_bytes = persistent_device_bytes_ +
      atoms * (sizeof(std::int64_t) + 128 * sizeof(double)) +
      edges * (sizeof(double) + sizeof(double) + 32 * sizeof(double));
  result.total_milliseconds = elapsed_ms;
  result.milliseconds_per_iteration = elapsed_ms / static_cast<double>(repetitions);
  result.checksum = embedding_host(0) + cutoff_host(edges / 2) +
                    rbf_host((edges / 2) * 32 + 7);
  return result;
}

}  // namespace so3lr
