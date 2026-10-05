#include "so3lr/kokkos_filter_block0.hpp"

#include <Kokkos_Timer.hpp>

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

namespace so3lr {
namespace {

using DoubleView = Kokkos::View<double *>;

template <class Reader>
void fill_view(const DoubleView &device, std::size_t count, Reader reader) {
  auto host = Kokkos::create_mirror_view(device);
  for (std::size_t i = 0; i < count; ++i) host(i) = reader(i);
  Kokkos::deep_copy(device, host);
}

void require_shape(const TensorRecord &tensor,
                   const std::vector<std::size_t> &shape) {
  if (tensor.dtype != "float64" || tensor.shape != shape)
    throw std::runtime_error("SO3LR filter tensor contract mismatch: " +
                             tensor.state_key);
}

DoubleView load_tensor(const NativeModel &model, const std::string &key,
                       const std::vector<std::size_t> &shape) {
  const auto &tensor = model.tensor(key);
  require_shape(tensor, shape);
  std::size_t count = 1;
  for (const auto extent : shape) count *= extent;
  DoubleView result(key, count);
  fill_view(result, count,
            [&](std::size_t i) { return model.float64(tensor, i); });
  return result;
}

FilterDeviceParameters load_filter(const NativeModel &model,
                                   const std::string &prefix) {
  FilterDeviceParameters result;
  result.rbf_weight_0 = load_tensor(
      model, prefix + ".mlp_rbf.0.weight", {128, 32});
  result.rbf_bias_0 = load_tensor(
      model, prefix + ".mlp_rbf.0.bias", {128});
  result.rbf_weight_1 = load_tensor(
      model, prefix + ".mlp_rbf.mlp_rbf_layer_1.0.weight", {128, 128});
  result.rbf_bias_1 = load_tensor(
      model, prefix + ".mlp_rbf.mlp_rbf_layer_1.0.bias", {128});
  result.ev_weight_0 = load_tensor(
      model, prefix + ".mlp_ev.0.weight", {32, 4});
  result.ev_bias_0 = load_tensor(
      model, prefix + ".mlp_ev.0.bias", {32});
  result.ev_weight_1 = load_tensor(
      model, prefix + ".mlp_ev.mlp_ev_layer_1.0.weight", {128, 32});
  result.ev_bias_1 = load_tensor(
      model, prefix + ".mlp_ev.mlp_ev_layer_1.0.bias", {128});
  return result;
}

bool verify_aliases(const NativeModel &model, const std::string &direct,
                    const std::string &attention) {
  constexpr std::array<const char *, 8> suffixes = {
      ".mlp_rbf.0.weight", ".mlp_rbf.0.bias",
      ".mlp_rbf.mlp_rbf_layer_1.0.weight",
      ".mlp_rbf.mlp_rbf_layer_1.0.bias", ".mlp_ev.0.weight",
      ".mlp_ev.0.bias", ".mlp_ev.mlp_ev_layer_1.0.weight",
      ".mlp_ev.mlp_ev_layer_1.0.bias"};
  for (const auto suffix : suffixes) {
    const auto &left = model.tensor(direct + suffix);
    const auto &right = model.tensor(attention + suffix);
    if (left.sha256 != right.sha256 || left.shape != right.shape ||
        left.dtype != right.dtype)
      return false;
  }
  return true;
}

void dense_silu(const DoubleView &input, std::size_t rows,
                std::size_t input_width, std::size_t output_width,
                const DoubleView &weight, const DoubleView &bias,
                const DoubleView &output, const std::string &label) {
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<>(0, rows * output_width),
      KOKKOS_LAMBDA(const std::size_t flat) {
        const std::size_t row = flat / output_width;
        const std::size_t out = flat % output_width;
        double value = bias(out);
        for (std::size_t in = 0; in < input_width; ++in)
          value += input(row * input_width + in) *
                   weight(out * input_width + in);
        output(flat) = value / (1.0 + Kokkos::exp(-value));
      });
}

void dense_assign(const DoubleView &input, std::size_t rows,
                  std::size_t input_width, std::size_t output_width,
                  const DoubleView &weight, const DoubleView &bias,
                  const DoubleView &output, const std::string &label) {
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<>(0, rows * output_width),
      KOKKOS_LAMBDA(const std::size_t flat) {
        const std::size_t row = flat / output_width;
        const std::size_t out = flat % output_width;
        double value = bias(out);
        for (std::size_t in = 0; in < input_width; ++in)
          value += input(row * input_width + in) *
                   weight(out * input_width + in);
        output(flat) = value;
      });
}

void dense_add(const DoubleView &input, std::size_t rows,
               std::size_t input_width, std::size_t output_width,
               const DoubleView &weight, const DoubleView &bias,
               const DoubleView &output, const std::string &label) {
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<>(0, rows * output_width),
      KOKKOS_LAMBDA(const std::size_t flat) {
        const std::size_t row = flat / output_width;
        const std::size_t out = flat % output_width;
        double value = bias(out);
        for (std::size_t in = 0; in < input_width; ++in)
          value += input(row * input_width + in) *
                   weight(out * input_width + in);
        output(flat) += value;
      });
}

void launch_filter(const FilterDeviceParameters &parameters,
                   const DoubleView &rbf, const DoubleView &ev,
                   std::size_t edges, const DoubleView &rbf_hidden,
                   const DoubleView &ev_hidden, const DoubleView &output,
                   const std::string &prefix) {
  dense_silu(rbf, edges, 32, 128, parameters.rbf_weight_0,
             parameters.rbf_bias_0, rbf_hidden, prefix + "_rbf_dense_silu");
  dense_assign(rbf_hidden, edges, 128, 128, parameters.rbf_weight_1,
               parameters.rbf_bias_1, output, prefix + "_rbf_dense");
  dense_silu(ev, edges, 4, 32, parameters.ev_weight_0,
             parameters.ev_bias_0, ev_hidden, prefix + "_ev_dense_silu");
  dense_add(ev_hidden, edges, 32, 128, parameters.ev_weight_1,
            parameters.ev_bias_1, output, prefix + "_ev_dense_add");
}

}  // namespace

KokkosFilterBlock0::KokkosFilterBlock0(const NativeModel &model) {
  const std::string block = "model.euclidean_transformers.0";
  const std::string inv = block + ".filter_net_inv";
  const std::string ev = block + ".filter_net_ev";
  aliases_verified_ =
      verify_aliases(model, inv,
                     block + ".euclidean_attention_block.filter_net_inv") &&
      verify_aliases(model, ev,
                     block + ".euclidean_attention_block.filter_net_ev");
  if (!aliases_verified_)
    throw std::runtime_error("SO3LR block-0 filter tensor aliases differ");
  invariant_ = load_filter(model, inv);
  equivariant_ = load_filter(model, ev);
  persistent_device_bytes_ = 2 * 25120 * sizeof(double);
  Kokkos::fence();
}

FilterBlock0Results KokkosFilterBlock0::evaluate(
    const std::vector<double> &radial_basis,
    const std::vector<double> &ev_invariants, std::size_t edges) const {
  if (radial_basis.size() != edges * 32 ||
      ev_invariants.size() != edges * 4)
    throw std::runtime_error("SO3LR block-0 filter input shape mismatch");
  DoubleView rbf("so3lr_filter_rbf", radial_basis.size());
  DoubleView ev("so3lr_filter_ev_invariants", ev_invariants.size());
  DoubleView rbf_hidden("so3lr_filter_rbf_hidden", edges * 128);
  DoubleView ev_hidden("so3lr_filter_ev_hidden", edges * 32);
  DoubleView inv_output("so3lr_filter_inv_output", edges * 128);
  DoubleView ev_output("so3lr_filter_ev_output", edges * 128);
  fill_view(rbf, radial_basis.size(),
            [&](std::size_t i) { return radial_basis[i]; });
  fill_view(ev, ev_invariants.size(),
            [&](std::size_t i) { return ev_invariants[i]; });
  launch_filter(invariant_, rbf, ev, edges, rbf_hidden, ev_hidden,
                inv_output, "so3lr_filter0_inv");
  launch_filter(equivariant_, rbf, ev, edges, rbf_hidden, ev_hidden,
                ev_output, "so3lr_filter0_ev");
  Kokkos::fence();
  const auto inv_host = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), inv_output);
  const auto ev_host = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), ev_output);
  FilterBlock0Results result;
  result.invariant_filter.resize(edges * 128);
  result.equivariant_filter.resize(edges * 128);
  for (std::size_t i = 0; i < edges * 128; ++i) {
    result.invariant_filter[i] = inv_host(i);
    result.equivariant_filter[i] = ev_host(i);
  }
  return result;
}

FilterBlock0Benchmark KokkosFilterBlock0::benchmark(
    std::size_t edges, std::size_t repetitions) const {
  if (edges == 0 || repetitions == 0)
    throw std::runtime_error("SO3LR filter benchmark sizes must be positive");
  DoubleView rbf("so3lr_filter_benchmark_rbf", edges * 32);
  DoubleView ev("so3lr_filter_benchmark_ev", edges * 4);
  DoubleView rbf_hidden("so3lr_filter_benchmark_rbf_hidden", edges * 128);
  DoubleView ev_hidden("so3lr_filter_benchmark_ev_hidden", edges * 32);
  DoubleView inv_output("so3lr_filter_benchmark_inv_output", edges * 128);
  DoubleView ev_output("so3lr_filter_benchmark_ev_output", edges * 128);
  fill_view(rbf, edges * 32, [](std::size_t i) {
    return 0.01 + static_cast<double>(i % 97) / 101.0;
  });
  fill_view(ev, edges * 4, [](std::size_t i) {
    return (static_cast<double>(i % 31) - 15.0) / 17.0;
  });
  for (int warmup = 0; warmup < 2; ++warmup) {
    launch_filter(invariant_, rbf, ev, edges, rbf_hidden, ev_hidden,
                  inv_output, "so3lr_filter0_inv_warmup");
    launch_filter(equivariant_, rbf, ev, edges, rbf_hidden, ev_hidden,
                  ev_output, "so3lr_filter0_ev_warmup");
  }
  Kokkos::fence();
  Kokkos::Timer timer;
  for (std::size_t iteration = 0; iteration < repetitions; ++iteration) {
    launch_filter(invariant_, rbf, ev, edges, rbf_hidden, ev_hidden,
                  inv_output, "so3lr_filter0_inv_benchmark");
    launch_filter(equivariant_, rbf, ev, edges, rbf_hidden, ev_hidden,
                  ev_output, "so3lr_filter0_ev_benchmark");
  }
  Kokkos::fence();
  const double elapsed_ms = timer.seconds() * 1000.0;
  double checksum = 0.0;
  Kokkos::parallel_reduce(
      "so3lr_filter0_checksum", Kokkos::RangePolicy<>(0, 2),
      KOKKOS_LAMBDA(const int which, double &update) {
        update += which == 0 ? inv_output(0) : ev_output((edges / 2) * 128 + 7);
      }, checksum);
  Kokkos::fence();
  FilterBlock0Benchmark result;
  result.edges = edges;
  result.repetitions = repetitions;
  result.workspace_bytes = persistent_device_bytes_ +
      edges * (32 + 4 + 128 + 32 + 128 + 128) * sizeof(double);
  result.total_milliseconds = elapsed_ms;
  result.milliseconds_per_iteration =
      elapsed_ms / static_cast<double>(repetitions);
  result.checksum = checksum;
  return result;
}

}  // namespace so3lr

