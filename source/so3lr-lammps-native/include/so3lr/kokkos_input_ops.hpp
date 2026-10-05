#pragma once

#include "so3lr/native_model.hpp"

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace so3lr {

struct InputResults {
  std::vector<double> embedding;
  std::vector<double> cutoff;
  std::vector<double> radial_basis;
};

struct InputBenchmark {
  std::size_t atoms = 0;
  std::size_t edges = 0;
  std::size_t repetitions = 0;
  std::size_t workspace_bytes = 0;
  double total_milliseconds = 0.0;
  double milliseconds_per_iteration = 0.0;
  double checksum = 0.0;
};

class KokkosInputOps {
 public:
  explicit KokkosInputOps(const NativeModel &model);

  InputResults evaluate(const std::vector<std::int64_t> &atomic_numbers,
                        const std::vector<double> &distances) const;
  InputBenchmark benchmark(std::size_t atoms, std::size_t edges,
                           std::size_t repetitions) const;
  std::size_t persistent_device_bytes() const { return persistent_device_bytes_; }

 private:
  using DoubleView = Kokkos::View<double *>;
  using IntView = Kokkos::View<std::int64_t *>;

  void launch(const IntView &atomic_numbers, const DoubleView &distances,
              const DoubleView &embedding, const DoubleView &cutoff,
              const DoubleView &radial_basis) const;

  DoubleView embedding_weight_;
  DoubleView bernstein_b_;
  IntView bernstein_k_;
  IntView bernstein_k_reverse_;
  double inverse_embedding_scale_ = 1.0;
  double gamma_ = 0.0;
  double cutoff_radius_ = 0.0;
  std::size_t persistent_device_bytes_ = 0;
};

}  // namespace so3lr

