#include "so3lr/kokkos_post_attention_reverse.hpp"

#include <Kokkos_Timer.hpp>

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace so3lr {
namespace {

using DoubleView = Kokkos::View<double *>;
using IndexView = Kokkos::View<std::size_t *>;

// Upper bound on degree slots (eight for {1,1,2,2,3,3,4,4}); sizes
// the per-node gate-gradient register array.
constexpr std::size_t kMaxDegreeSlots = 8;

// Same width binding as kokkos_post_attention_block0.cpp.
#define SO3LR_POST_ATTENTION_WIDTHS(dims)                                   \
  const std::size_t inv_width = (dims).invariant_width;                    \
  const std::size_t ev_width = (dims).equivariant_width;                   \
  const std::size_t degree_count = (dims).ev_heads;                        \
  const std::size_t interaction_width = (dims).interaction_width;          \
  static_cast<void>(inv_width);                                            \
  static_cast<void>(ev_width);                                             \
  static_cast<void>(degree_count);                                         \
  static_cast<void>(interaction_width)

void check_cublas(cublasStatus_t status, const char *operation) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string("cuBLAS failure in ") + operation +
                             ": status=" +
                             std::to_string(static_cast<int>(status)));
}

int checked_int(std::size_t value, const char *name) {
  if (value > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::runtime_error(std::string("cuBLAS dimension exceeds int: ") +
                             name);
  return static_cast<int>(value);
}

template <class View, class Reader>
void fill_view(const View &device, std::size_t count, Reader reader) {
  auto host = Kokkos::create_mirror_view(device);
  for (std::size_t i = 0; i < count; ++i) host(i) = reader(i);
  Kokkos::deep_copy(device, host);
}

void require_tensor(const TensorRecord &tensor,
                    const std::vector<std::size_t> &shape,
                    const std::string &dtype = "float64") {
  if (tensor.dtype != dtype || tensor.shape != shape)
    throw std::runtime_error("SO3LR post-attention reverse tensor mismatch: " +
                             tensor.state_key);
}

DoubleView load_double(const NativeModel &model, const std::string &key,
                       const std::vector<std::size_t> &shape) {
  const auto &tensor = model.tensor(key);
  require_tensor(tensor, shape);
  std::size_t count = 1;
  for (const auto extent : shape) count *= extent;
  DoubleView result(key + "_post_reverse", count);
  fill_view(result, count,
            [&](std::size_t i) { return model.float64(tensor, i); });
  return result;
}

void linear(cublasHandle_t handle, const DoubleView &input,
            std::size_t rows, std::size_t input_width,
            std::size_t output_width, const DoubleView &weight,
            const DoubleView &output) {
  const int m = checked_int(output_width, "output_width");
  const int n = checked_int(rows, "rows");
  const int k = checked_int(input_width, "input_width");
  constexpr double alpha = 1.0;
  constexpr double beta = 0.0;
  check_cublas(
      cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha,
                  weight.data(), k, input.data(), k, &beta, output.data(), m),
      "cublasDgemm(post-attention-recompute)");
}

void linear_input_gradient(cublasHandle_t handle,
                           const DoubleView &output_gradient,
                           std::size_t rows, std::size_t output_width,
                           std::size_t input_width,
                           const DoubleView &weight,
                           const DoubleView &input_gradient) {
  const int m = checked_int(input_width, "reverse_input_width");
  const int n = checked_int(rows, "reverse_rows");
  const int k = checked_int(output_width, "reverse_output_width");
  constexpr double alpha = 1.0;
  constexpr double beta = 0.0;
  check_cublas(
      cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
                  weight.data(), m, output_gradient.data(), k, &beta,
                  input_gradient.data(), m),
      "cublasDgemm(post-attention-reverse)");
}

void layer_norm_backward(const DoubleView &input,
                         const DoubleView &weight,
                         const DoubleView &output_gradient,
                         std::size_t nodes, double epsilon,
                         const DoubleView &input_gradient,
                         const char *label, const ArchDims &dims) {
  SO3LR_POST_ATTENTION_WIDTHS(dims);
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<>(0, nodes),
      KOKKOS_LAMBDA(const std::size_t node) {
        const std::size_t offset = node * inv_width;
        double mean = 0.0;
        for (std::size_t channel = 0; channel < inv_width; ++channel)
          mean += input(offset + channel);
        mean /= static_cast<double>(inv_width);
        double variance = 0.0;
        for (std::size_t channel = 0; channel < inv_width; ++channel) {
          const double difference = input(offset + channel) - mean;
          variance += difference * difference;
        }
        variance /= static_cast<double>(inv_width);
        const double inverse_sigma = 1.0 / Kokkos::sqrt(variance + epsilon);
        double sum_scaled_gradient = 0.0;
        double sum_scaled_gradient_xhat = 0.0;
        for (std::size_t channel = 0; channel < inv_width; ++channel) {
          const double xhat =
              (input(offset + channel) - mean) * inverse_sigma;
          const double scaled =
              output_gradient(offset + channel) * weight(channel);
          sum_scaled_gradient += scaled;
          sum_scaled_gradient_xhat += scaled * xhat;
        }
        for (std::size_t channel = 0; channel < inv_width; ++channel) {
          const double xhat =
              (input(offset + channel) - mean) * inverse_sigma;
          const double scaled =
              output_gradient(offset + channel) * weight(channel);
          input_gradient(offset + channel) =
              inverse_sigma / static_cast<double>(inv_width) *
              (static_cast<double>(inv_width) * scaled -
               sum_scaled_gradient - xhat * sum_scaled_gradient_xhat);
        }
      });
}

// Adjoint of y = x r, r = (mean(x^2) + eps)^(-1/2), over the feature axis:
//   dx_c = r dy_c - x_c r^3 mean(x dy)
void rms_norm_backward(const DoubleView &input,
                       const DoubleView &output_gradient, std::size_t nodes,
                       double epsilon, const DoubleView &input_gradient,
                       const char *label, const ArchDims &dims) {
  SO3LR_POST_ATTENTION_WIDTHS(dims);
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<>(0, nodes),
      KOKKOS_LAMBDA(const std::size_t node) {
        const std::size_t offset = node * inv_width;
        double mean_square = 0.0;
        double mean_x_dy = 0.0;
        for (std::size_t channel = 0; channel < inv_width; ++channel) {
          const double value = input(offset + channel);
          mean_square += value * value;
          mean_x_dy += value * output_gradient(offset + channel);
        }
        mean_square /= static_cast<double>(inv_width);
        mean_x_dy /= static_cast<double>(inv_width);
        const double r = 1.0 / Kokkos::sqrt(mean_square + epsilon);
        const double r3_mean = r * r * r * mean_x_dy;
        for (std::size_t channel = 0; channel < inv_width; ++channel)
          input_gradient(offset + channel) =
              r * output_gradient(offset + channel) -
              input(offset + channel) * r3_mean;
      });
}

// Dispatch mirroring the forward's apply_norm. The LayerAffine branch is the
// unchanged v1 adjoint.
void norm_backward(NormMode mode, const DoubleView &input,
                   const DoubleView &weight, const DoubleView &output_gradient,
                   std::size_t nodes, double epsilon,
                   const DoubleView &input_gradient, const char *label,
                   const ArchDims &dims) {
  switch (mode) {
    case NormMode::LayerAffine:
      layer_norm_backward(input, weight, output_gradient, nodes, epsilon,
                          input_gradient, label, dims);
      return;
    case NormMode::RmsNoScale:
      rms_norm_backward(input, output_gradient, nodes, epsilon,
                        input_gradient, label, dims);
      return;
    case NormMode::Identity:
      Kokkos::deep_copy(input_gradient, output_gradient);
      return;
  }
}

std::size_t complete_benchmark_bytes(std::size_t nodes, const ArchDims &dims) {
  SO3LR_POST_ATTENTION_WIDTHS(dims);
  // Four inputs, the post-attention forward workspace, two output gradients
  // and the reverse workspace (304 + 1340 + 152 + 1312 doubles/node in v1).
  const std::size_t doubles_per_node =
      (2 * inv_width + 2 * ev_width) +
      (8 * inv_width + 2 * ev_width + degree_count + 2 * interaction_width) +
      (inv_width + ev_width) +
      (8 * inv_width + 2 * interaction_width + ev_width);
  return nodes * doubles_per_node * sizeof(double);
}

}  // namespace

PostAttentionReverseDeviceWorkspace::PostAttentionReverseDeviceWorkspace(
    std::size_t nodes, const ArchDims &dims)
    : PostAttentionReverseDeviceWorkspace(nodes, dims.invariant_width,
                                          dims.equivariant_width,
                                          dims.interaction_width) {}

PostAttentionReverseDeviceWorkspace::PostAttentionReverseDeviceWorkspace(
    std::size_t nodes, std::size_t inv_width, std::size_t ev_width,
    std::size_t interaction_width)
    : hidden_preact("so3lr_post_reverse_hidden_preact", nodes * inv_width),
      grad_pre_norm2("so3lr_post_reverse_grad_pre_norm2", nodes * inv_width),
      grad_post_mlp("so3lr_post_reverse_grad_post_mlp", nodes * inv_width),
      grad_transformed("so3lr_post_reverse_grad_transformed",
                       nodes * interaction_width),
      grad_concatenated("so3lr_post_reverse_grad_concatenated",
                        nodes * interaction_width),
      grad_attention_ev("so3lr_post_reverse_grad_attention_ev",
                        nodes * ev_width),
      grad_hidden("so3lr_post_reverse_grad_hidden", nodes * inv_width),
      grad_hidden_preact("so3lr_post_reverse_grad_hidden_preact",
                         nodes * inv_width),
      grad_activation("so3lr_post_reverse_grad_activation", nodes * inv_width),
      grad_norm1("so3lr_post_reverse_grad_norm1", nodes * inv_width),
      grad_attention_inv("so3lr_post_reverse_grad_attention_inv",
                         nodes * inv_width) {
  if (nodes == 0)
    throw std::runtime_error("SO3LR post-attention reverse workspace is empty");
}

KokkosPostAttentionReverseBlock::KokkosPostAttentionReverseBlock(
    const NativeModel &model, std::size_t block_index)
    : dims_(model.arch().dims()), forward_(model, block_index),
      block_index_(block_index) {
  SO3LR_POST_ATTENTION_WIDTHS(dims_);
  if (degree_count > kMaxDegreeSlots)
    throw std::runtime_error("SO3LR post-attention reverse: too many degree slots");
  const auto blocks = static_cast<std::size_t>(
      model.architecture_number("interaction_blocks"));
  if (block_index_ >= blocks)
    throw std::runtime_error("SO3LR post-attention reverse index out of range");
  const std::string block =
      "model.euclidean_transformers." + std::to_string(block_index_);
  // The forward block has already validated the norm contract; only an affine
  // LayerNorm has a weight for the adjoint to use.
  if (forward_.norm_1_mode() == NormMode::LayerAffine)
    layer_norm_1_weight_ =
        load_double(model, block + ".layer_norm_inv_1.weight", {inv_width});
  if (forward_.norm_2_mode() == NormMode::LayerAffine)
    layer_norm_2_weight_ =
        load_double(model, block + ".layer_norm_inv_2.weight", {inv_width});
  mlp_1_weight_1_ =
      load_double(model, block + ".mlp_1.1.weight", {inv_width, inv_width});
  mlp_1_bias_1_ = load_double(model, block + ".mlp_1.1.bias", {inv_width});
  mlp_1_weight_2_ =
      load_double(model, block + ".mlp_1.3.weight", {inv_width, inv_width});
  const std::string interaction = block + ".interaction_block";
  interaction_weight_ =
      load_double(model, interaction + ".linear_layer.weight",
                  {interaction_width, interaction_width});
  interaction_cg_rep_ = load_double(
      model, interaction + ".so3_conv_invariants.cg_rep", {ev_width});

  const auto &arch = model.arch();
  const auto &repeats = model.tensor(interaction + ".degree_repeats");
  require_tensor(repeats, {degree_count}, "int64");
  for (std::size_t degree = 0; degree < degree_count; ++degree)
    if (model.int64(repeats, degree) !=
        static_cast<std::int64_t>(arch.degree_repeats[degree]))
      throw std::runtime_error("SO3LR reverse degree repeats changed");
  degree_offsets_ = IndexView("so3lr_post_reverse_degree_offsets",
                              degree_count);
  fill_view(degree_offsets_, degree_count,
            [&](std::size_t i) { return arch.degree_offsets[i]; });

  check_cublas(cublasCreate(&handle_), "cublasCreate");
  const auto stream = Kokkos::Cuda().cuda_stream();
  check_cublas(cublasSetStream(handle_, stream), "cublasSetStream");
  cudaStream_t configured = nullptr;
  check_cublas(cublasGetStream(handle_, &configured), "cublasGetStream");
  shared_kokkos_stream_ = configured == stream;
  const std::size_t reverse_persistent_doubles =
      2 * inv_width + 2 * (inv_width * inv_width) + inv_width +
      interaction_width * interaction_width + ev_width;
  persistent_device_bytes_ =
      forward_.persistent_device_bytes() +
      reverse_persistent_doubles * sizeof(double) +
      degree_count * sizeof(std::size_t);
  checkpoint_contract_verified_ =
      forward_.checkpoint_contract_verified() &&
      forward_.shared_kokkos_stream() && shared_kokkos_stream_ &&
      forward_.block_index() == block_index_;
  if (!checkpoint_contract_verified_)
    throw std::runtime_error("SO3LR post-attention reverse contract failed");
  Kokkos::fence();
}

KokkosPostAttentionReverseBlock::~KokkosPostAttentionReverseBlock() {
  if (handle_ != nullptr) {
    Kokkos::fence();
    static_cast<void>(cublasDestroy(handle_));
  }
}

void KokkosPostAttentionReverseBlock::launch_forward_device(
    const DoubleView &inv_features, const DoubleView &ev_features,
    const DoubleView &d_attention_inv, const DoubleView &d_attention_ev,
    const PostAttentionBlock0DeviceWorkspace &workspace) const {
  forward_.launch_device(inv_features, ev_features, d_attention_inv,
                         d_attention_ev, workspace);
}

void KokkosPostAttentionReverseBlock::launch_reverse_device(
    const DoubleView &grad_final_inv, const DoubleView &grad_final_ev,
    const PostAttentionBlock0DeviceWorkspace &forward,
    const PostAttentionReverseDeviceWorkspace &reverse) const {
  SO3LR_POST_ATTENTION_WIDTHS(dims_);
  const std::size_t nodes = grad_final_inv.extent(0) / inv_width;
  if (nodes == 0 || grad_final_inv.extent(0) != nodes * inv_width ||
      grad_final_ev.extent(0) != nodes * ev_width ||
      forward.final_inv.extent(0) != nodes * inv_width ||
      forward.final_ev.extent(0) != nodes * ev_width ||
      reverse.grad_attention_inv.extent(0) != nodes * inv_width ||
      reverse.grad_attention_ev.extent(0) != nodes * ev_width)
    throw std::runtime_error("SO3LR post-attention reverse view mismatch");

  norm_backward(forward_.norm_2_mode(), forward.pre_layer_norm_2_inv,
                layer_norm_2_weight_, grad_final_inv, nodes,
                layer_norm_epsilon_, reverse.grad_pre_norm2,
                "so3lr_post_reverse_layer_norm_2", dims_);

  const auto grad_pre = reverse.grad_pre_norm2;
  const auto grad_post = reverse.grad_post_mlp;
  const auto grad_transformed = reverse.grad_transformed;
  Kokkos::parallel_for(
      "so3lr_post_reverse_final_inv_residual",
      Kokkos::RangePolicy<>(0, nodes * inv_width),
      KOKKOS_LAMBDA(const std::size_t flat) {
        const std::size_t node = flat / inv_width;
        const std::size_t channel = flat % inv_width;
        const double value = grad_pre(flat);
        grad_post(flat) = value;
        grad_transformed(node * interaction_width + channel) = value;
      });

  const auto offsets = degree_offsets_;
  const auto att_ev = forward.attention_residual_ev;
  const auto transformed = forward.interaction_transformed;
  const auto grad_att_ev = reverse.grad_attention_ev;
  Kokkos::parallel_for(
      "so3lr_post_reverse_final_ev_gate",
      Kokkos::RangePolicy<>(0, nodes),
      KOKKOS_LAMBDA(const std::size_t node) {
        double gate_gradient[kMaxDegreeSlots] = {};
        for (std::size_t channel = 0; channel < ev_width; ++channel) {
          std::size_t degree = 0;
          for (std::size_t slot = 1; slot < degree_count; ++slot)
            if (channel >= offsets(slot)) degree = slot;
          const std::size_t index = node * ev_width + channel;
          const double upstream = grad_final_ev(index);
          const double gate =
              transformed(node * interaction_width + inv_width + degree);
          grad_att_ev(index) = upstream * (1.0 + gate);
          gate_gradient[degree] += upstream * att_ev(index);
        }
        for (std::size_t degree = 0; degree < degree_count; ++degree)
          grad_transformed(node * interaction_width + inv_width + degree) =
              gate_gradient[degree];
      });

  linear_input_gradient(handle_, grad_transformed, nodes, interaction_width,
                        interaction_width, interaction_weight_,
                        reverse.grad_concatenated);

  const auto grad_concat = reverse.grad_concatenated;
  const auto cg = interaction_cg_rep_;
  Kokkos::parallel_for(
      "so3lr_post_reverse_interaction_inputs",
      Kokkos::RangePolicy<>(0, nodes),
      KOKKOS_LAMBDA(const std::size_t node) {
        for (std::size_t channel = 0; channel < inv_width; ++channel)
          grad_post(node * inv_width + channel) +=
              grad_concat(node * interaction_width + channel);
        for (std::size_t channel = 0; channel < ev_width; ++channel) {
          std::size_t degree = 0;
          for (std::size_t slot = 1; slot < degree_count; ++slot)
            if (channel >= offsets(slot)) degree = slot;
          const std::size_t index = node * ev_width + channel;
          grad_att_ev(index) +=
              2.0 * att_ev(index) * cg(channel) *
              grad_concat(node * interaction_width + inv_width + degree);
        }
      });

  linear_input_gradient(handle_, grad_post, nodes, inv_width, inv_width,
                        mlp_1_weight_2_, reverse.grad_hidden);
  linear(handle_, forward.mlp_activation, nodes, inv_width, inv_width,
         mlp_1_weight_1_, reverse.hidden_preact);
  const auto hidden_preact = reverse.hidden_preact;
  const auto grad_hidden = reverse.grad_hidden;
  const auto grad_hidden_preact = reverse.grad_hidden_preact;
  const auto bias1 = mlp_1_bias_1_;
  Kokkos::parallel_for(
      "so3lr_post_reverse_hidden_silu",
      Kokkos::RangePolicy<>(0, nodes * inv_width),
      KOKKOS_LAMBDA(const std::size_t i) {
        const double value = hidden_preact(i) + bias1(i % inv_width);
        hidden_preact(i) = value;
        const double sigmoid = 1.0 / (1.0 + Kokkos::exp(-value));
        const double derivative =
            sigmoid * (1.0 + value * (1.0 - sigmoid));
        grad_hidden_preact(i) = grad_hidden(i) * derivative;
      });
  linear_input_gradient(handle_, grad_hidden_preact, nodes, inv_width,
                        inv_width, mlp_1_weight_1_, reverse.grad_activation);

  const auto norm1 = forward.layer_norm_1;
  const auto grad_activation = reverse.grad_activation;
  const auto grad_norm1 = reverse.grad_norm1;
  Kokkos::parallel_for(
      "so3lr_post_reverse_input_silu_residual",
      Kokkos::RangePolicy<>(0, nodes * inv_width),
      KOKKOS_LAMBDA(const std::size_t i) {
        const double value = norm1(i);
        const double sigmoid = 1.0 / (1.0 + Kokkos::exp(-value));
        const double derivative =
            sigmoid * (1.0 + value * (1.0 - sigmoid));
        grad_norm1(i) = grad_post(i) + grad_activation(i) * derivative;
      });
  norm_backward(forward_.norm_1_mode(), forward.attention_residual_inv,
                layer_norm_1_weight_, grad_norm1, nodes, layer_norm_epsilon_,
                reverse.grad_attention_inv,
                "so3lr_post_reverse_layer_norm_1", dims_);
}

PostAttentionReverseBenchmark KokkosPostAttentionReverseBlock::benchmark(
    std::size_t nodes, std::size_t repetitions) const {
  SO3LR_POST_ATTENTION_WIDTHS(dims_);
  if (nodes == 0 || repetitions == 0)
    throw std::runtime_error("SO3LR post-attention reverse benchmark invalid");
  DoubleView inv("so3lr_post_reverse_bench_inv", nodes * inv_width);
  DoubleView ev("so3lr_post_reverse_bench_ev", nodes * ev_width);
  DoubleView d_inv("so3lr_post_reverse_bench_d_inv", nodes * inv_width);
  DoubleView d_ev("so3lr_post_reverse_bench_d_ev", nodes * ev_width);
  DoubleView grad_final_inv("so3lr_post_reverse_bench_final_inv_grad",
                            nodes * inv_width);
  DoubleView grad_final_ev("so3lr_post_reverse_bench_final_ev_grad",
                           nodes * ev_width);
  Kokkos::parallel_for(
      "so3lr_post_reverse_bench_inv",
      Kokkos::RangePolicy<>(0, nodes * inv_width),
      KOKKOS_LAMBDA(const std::size_t i) {
        const double value =
            (static_cast<double>(i % 257) - 128.0) / 181.0;
        inv(i) = value;
        d_inv(i) = 0.15 * value + 0.02;
        grad_final_inv(i) =
            (static_cast<double>(i % 71) - 35.0) / 97.0;
      });
  Kokkos::parallel_for(
      "so3lr_post_reverse_bench_ev",
      Kokkos::RangePolicy<>(0, nodes * ev_width),
      KOKKOS_LAMBDA(const std::size_t i) {
        const double value =
            (static_cast<double>(i % 61) - 30.0) / 43.0;
        ev(i) = value;
        d_ev(i) = -0.12 * value + 0.01;
        grad_final_ev(i) =
            (static_cast<double>(i % 37) - 18.0) / 83.0;
      });
  PostAttentionBlock0DeviceWorkspace forward(nodes, dims_);
  PostAttentionReverseDeviceWorkspace reverse(nodes, dims_);
  launch_forward_device(inv, ev, d_inv, d_ev, forward);
  for (int warmup = 0; warmup < 2; ++warmup)
    launch_reverse_device(grad_final_inv, grad_final_ev, forward, reverse);
  Kokkos::fence();
  Kokkos::Timer timer;
  for (std::size_t i = 0; i < repetitions; ++i)
    launch_reverse_device(grad_final_inv, grad_final_ev, forward, reverse);
  Kokkos::fence();
  const double reverse_ms =
      timer.seconds() * 1000.0 / static_cast<double>(repetitions);
  for (int warmup = 0; warmup < 2; ++warmup) {
    launch_forward_device(inv, ev, d_inv, d_ev, forward);
    launch_reverse_device(grad_final_inv, grad_final_ev, forward, reverse);
  }
  Kokkos::fence();
  timer.reset();
  for (std::size_t i = 0; i < repetitions; ++i) {
    launch_forward_device(inv, ev, d_inv, d_ev, forward);
    launch_reverse_device(grad_final_inv, grad_final_ev, forward, reverse);
  }
  Kokkos::fence();
  const double forward_reverse_ms =
      timer.seconds() * 1000.0 / static_cast<double>(repetitions);
  double checksum = 0.0;
  const auto grad_inv = reverse.grad_attention_inv;
  const auto grad_ev = reverse.grad_attention_ev;
  Kokkos::parallel_reduce(
      "so3lr_post_reverse_bench_checksum", Kokkos::RangePolicy<>(0, 2),
      KOKKOS_LAMBDA(const int which, double &update) {
        if (which == 0) update += grad_inv((nodes / 3) * inv_width + 17);
        if (which == 1) update += grad_ev((nodes / 2) * ev_width + 9);
      },
      checksum);
  Kokkos::fence();
  PostAttentionReverseBenchmark result;
  result.block_index = block_index_;
  result.nodes = nodes;
  result.repetitions = repetitions;
  result.persistent_device_bytes = persistent_device_bytes_;
  result.workspace_bytes =
      persistent_device_bytes_ + complete_benchmark_bytes(nodes, dims_);
  result.host_boundary_bytes_per_iteration = 0;
  result.reverse_milliseconds = reverse_ms;
  result.forward_reverse_milliseconds = forward_reverse_ms;
  result.nodes_per_second =
      static_cast<double>(nodes) / (reverse_ms / 1000.0);
  result.checksum = checksum;
  return result;
}

}  // namespace so3lr
