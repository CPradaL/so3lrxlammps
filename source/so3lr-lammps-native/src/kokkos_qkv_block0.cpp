#include "so3lr/kokkos_qkv_block0.hpp"

#include <Kokkos_Timer.hpp>

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <string>

namespace so3lr {
namespace {

using DoubleView = Kokkos::View<double *>;
using IndexView = Kokkos::View<std::size_t *>;

constexpr std::size_t heads = 4;
constexpr std::size_t head_width = 32;
constexpr std::size_t feature_width = heads * head_width;
constexpr std::size_t weight_elements = heads * head_width * head_width;

void check_cublas(cublasStatus_t status, const char *operation) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string("cuBLAS failure in ") + operation +
                             ": status=" +
                             std::to_string(static_cast<int>(status)));
}

int checked_int(std::size_t value, const char *name) {
  if (value > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::runtime_error(std::string("cuBLAS dimension exceeds int: ") + name);
  return static_cast<int>(value);
}

template <class Reader>
void fill_view(const DoubleView &device, std::size_t count, Reader reader) {
  auto host = Kokkos::create_mirror_view(device);
  for (std::size_t i = 0; i < count; ++i) host(i) = reader(i);
  Kokkos::deep_copy(device, host);
}

DoubleView load_weight(const NativeModel &model, const std::string &key) {
  const auto &tensor = model.tensor(key);
  const std::vector<std::size_t> expected = {heads, head_width, head_width};
  if (tensor.dtype != "float64" || tensor.shape != expected)
    throw std::runtime_error("SO3LR Q/K/V tensor contract mismatch: " + key);
  DoubleView result(key + "_native", weight_elements);
  fill_view(result, weight_elements,
            [&](std::size_t i) { return model.float64(tensor, i); });
  return result;
}

bool has_identity_qk(const NativeModel &model) {
  // Was: match the module type against "torch.nn.modules.linear.Identity".
  // Deciding a kernel path from a Python class name is the wrong contract, and
  // a flax-exported model has no module tree at all.
  return model.arch().qk_nonlinearity_identity;
}

// PyTorch uses einsum("nhd,hde->nhe").  A head's row-major W[d,e]
// is the same memory as the column-major W^T[e,d] consumed below.  The
// node-major feature and output buffers use a padded leading dimension of 128,
// so all four heads are projected without transposing the feature matrix.
void project(cublasHandle_t handle, const DoubleView &input,
             std::size_t nodes, const DoubleView &weight,
             const DoubleView &output) {
  const int m = checked_int(head_width, "head_width");
  const int n = checked_int(nodes, "nodes");
  const int k = checked_int(head_width, "head_width");
  const int node_stride = checked_int(feature_width, "feature_width");
  constexpr double alpha = 1.0;
  constexpr double beta = 0.0;
  for (std::size_t head = 0; head < heads; ++head) {
    check_cublas(
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
                    weight.data() + head * head_width * head_width, m,
                    input.data() + head * head_width, node_stride, &beta,
                    output.data() + head * head_width, node_stride),
        "cublasDgemm(Q/K/V head)");
  }
}

void launch_cublas(cublasHandle_t handle, const DoubleView &features,
                   std::size_t nodes, const std::array<DoubleView, 5> &weights,
                   const std::array<DoubleView, 5> &outputs) {
  for (std::size_t projection = 0; projection < weights.size(); ++projection)
    project(handle, features, nodes, weights[projection], outputs[projection]);
}

void baseline_projection(const DoubleView &features, std::size_t nodes,
                         const DoubleView &weight, const DoubleView &output,
                         const std::string &label) {
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<>(0, nodes * feature_width),
      KOKKOS_LAMBDA(const std::size_t flat) {
        const std::size_t node = flat / feature_width;
        const std::size_t channel = flat % feature_width;
        const std::size_t head = channel / head_width;
        const std::size_t out = channel % head_width;
        double value = 0.0;
        for (std::size_t in = 0; in < head_width; ++in)
          value += features(node * feature_width + head * head_width + in) *
                   weight(head * head_width * head_width +
                          in * head_width + out);
        output(flat) = value;
      });
}

void launch_baseline(const DoubleView &features, std::size_t nodes,
                     const std::array<DoubleView, 5> &weights,
                     const std::array<DoubleView, 5> &outputs,
                     const std::string &suffix) {
  constexpr std::array<const char *, 5> names = {
      "q_inv", "k_inv", "v_inv", "q_ev", "k_ev"};
  for (std::size_t projection = 0; projection < weights.size(); ++projection)
    baseline_projection(features, nodes, weights[projection], outputs[projection],
                        "so3lr_qkv0_baseline_" +
                            std::string(names[projection]) + suffix);
}

std::vector<double> copy_host(const DoubleView &device) {
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), device);
  std::vector<double> result(device.extent(0));
  for (std::size_t i = 0; i < result.size(); ++i) result[i] = host(i);
  return result;
}

QkvBlock0Benchmark finish_benchmark(
    const std::array<DoubleView, 5> &outputs, std::size_t nodes,
    std::size_t repetitions, std::size_t persistent_bytes,
    double elapsed_ms) {
  const auto q_inv = outputs[0];
  const auto k_inv = outputs[1];
  const auto v_inv = outputs[2];
  const auto q_ev = outputs[3];
  const auto k_ev = outputs[4];
  double checksum = 0.0;
  Kokkos::parallel_reduce(
      "so3lr_qkv0_checksum", Kokkos::RangePolicy<>(0, 5),
      KOKKOS_LAMBDA(const int projection, double &update) {
        const std::size_t channel = static_cast<std::size_t>(projection) * 19 %
                                    feature_width;
        const std::size_t node =
            (static_cast<std::size_t>(projection) * 7919) % nodes;
        const std::size_t offset = node * feature_width + channel;
        if (projection == 0) update += q_inv(offset);
        if (projection == 1) update += k_inv(offset);
        if (projection == 2) update += v_inv(offset);
        if (projection == 3) update += q_ev(offset);
        if (projection == 4) update += k_ev(offset);
      },
      checksum);
  Kokkos::fence();
  QkvBlock0Benchmark result;
  result.nodes = nodes;
  result.repetitions = repetitions;
  result.workspace_bytes =
      persistent_bytes + nodes * feature_width * 6 * sizeof(double);
  result.total_milliseconds = elapsed_ms;
  result.milliseconds_per_iteration =
      elapsed_ms / static_cast<double>(repetitions);
  result.checksum = checksum;
  return result;
}

}  // namespace

KokkosQkvBlock0::KokkosQkvBlock0(const NativeModel &model) {
  if (model.architecture_number("attention_heads") != heads ||
      model.architecture_number("attention_head_width") != head_width)
    throw std::runtime_error("SO3LR block-0 Q/K/V architecture mismatch");
  identity_qk_activation_ = has_identity_qk(model);
  if (!identity_qk_activation_)
    throw std::runtime_error("SO3LR block-0 Q/K activation is not identity");
  const std::string prefix =
      "model.euclidean_transformers.0.euclidean_attention_block.";
  w_q_inv_ = load_weight(model, prefix + "W_q_inv");
  w_k_inv_ = load_weight(model, prefix + "W_k_inv");
  w_v_inv_ = load_weight(model, prefix + "W_v_inv");
  w_q_ev_ = load_weight(model, prefix + "W_q_ev");
  w_k_ev_ = load_weight(model, prefix + "W_k_ev");
  persistent_device_bytes_ = 5 * weight_elements * sizeof(double);
  check_cublas(cublasCreate(&handle_), "cublasCreate");
  const auto stream = Kokkos::Cuda().cuda_stream();
  check_cublas(cublasSetStream(handle_, stream), "cublasSetStream");
  cudaStream_t configured = nullptr;
  check_cublas(cublasGetStream(handle_, &configured), "cublasGetStream");
  shared_kokkos_stream_ = configured == stream;
  Kokkos::fence();
}

KokkosQkvBlock0::~KokkosQkvBlock0() {
  if (handle_ != nullptr) {
    Kokkos::fence();
    static_cast<void>(cublasDestroy(handle_));
  }
}

QkvBlock0Results KokkosQkvBlock0::evaluate(
    const std::vector<double> &features,
    const std::vector<std::size_t> &senders,
    const std::vector<std::size_t> &receivers, std::size_t nodes) const {
  if (nodes == 0 || features.size() != nodes * feature_width ||
      senders.size() != receivers.size())
    throw std::runtime_error("SO3LR block-0 Q/K/V input contract mismatch");
  if (std::any_of(senders.begin(), senders.end(),
                  [&](std::size_t i) { return i >= nodes; }) ||
      std::any_of(receivers.begin(), receivers.end(),
                  [&](std::size_t i) { return i >= nodes; }))
    throw std::runtime_error("SO3LR block-0 Q/K/V edge index out of range");

  DoubleView input("so3lr_qkv0_features", features.size());
  fill_view(input, features.size(), [&](std::size_t i) { return features[i]; });
  std::array<DoubleView, 5> weights = {
      w_q_inv_, w_k_inv_, w_v_inv_, w_q_ev_, w_k_ev_};
  std::array<DoubleView, 5> node_outputs = {
      DoubleView("so3lr_qkv0_q_inv_nodes", nodes * feature_width),
      DoubleView("so3lr_qkv0_k_inv_nodes", nodes * feature_width),
      DoubleView("so3lr_qkv0_v_inv_nodes", nodes * feature_width),
      DoubleView("so3lr_qkv0_q_ev_nodes", nodes * feature_width),
      DoubleView("so3lr_qkv0_k_ev_nodes", nodes * feature_width)};
  launch_cublas(handle_, input, nodes, weights, node_outputs);

  const std::size_t edges = senders.size();
  IndexView sender_view("so3lr_qkv0_senders", edges);
  IndexView receiver_view("so3lr_qkv0_receivers", edges);
  auto sender_host = Kokkos::create_mirror_view(sender_view);
  auto receiver_host = Kokkos::create_mirror_view(receiver_view);
  for (std::size_t edge = 0; edge < edges; ++edge) {
    sender_host(edge) = senders[edge];
    receiver_host(edge) = receivers[edge];
  }
  Kokkos::deep_copy(sender_view, sender_host);
  Kokkos::deep_copy(receiver_view, receiver_host);
  std::array<DoubleView, 5> edge_outputs = {
      DoubleView("so3lr_qkv0_q_inv_edges", edges * feature_width),
      DoubleView("so3lr_qkv0_k_inv_edges", edges * feature_width),
      DoubleView("so3lr_qkv0_v_inv_edges", edges * feature_width),
      DoubleView("so3lr_qkv0_q_ev_edges", edges * feature_width),
      DoubleView("so3lr_qkv0_k_ev_edges", edges * feature_width)};
  const auto q_inv_nodes = node_outputs[0];
  const auto k_inv_nodes = node_outputs[1];
  const auto v_inv_nodes = node_outputs[2];
  const auto q_ev_nodes = node_outputs[3];
  const auto k_ev_nodes = node_outputs[4];
  const auto q_inv_edges = edge_outputs[0];
  const auto k_inv_edges = edge_outputs[1];
  const auto v_inv_edges = edge_outputs[2];
  const auto q_ev_edges = edge_outputs[3];
  const auto k_ev_edges = edge_outputs[4];
  Kokkos::parallel_for(
      "so3lr_qkv0_reference_gather",
      Kokkos::RangePolicy<>(0, edges * feature_width),
      KOKKOS_LAMBDA(const std::size_t flat) {
        const std::size_t edge = flat / feature_width;
        const std::size_t channel = flat % feature_width;
        q_inv_edges(flat) =
            q_inv_nodes(receiver_view(edge) * feature_width + channel);
        k_inv_edges(flat) =
            k_inv_nodes(sender_view(edge) * feature_width + channel);
        v_inv_edges(flat) =
            v_inv_nodes(sender_view(edge) * feature_width + channel);
        q_ev_edges(flat) =
            q_ev_nodes(receiver_view(edge) * feature_width + channel);
        k_ev_edges(flat) =
            k_ev_nodes(sender_view(edge) * feature_width + channel);
      });
  Kokkos::fence();

  QkvBlock0Results result;
  result.q_inv_nodes = copy_host(node_outputs[0]);
  result.k_inv_nodes = copy_host(node_outputs[1]);
  result.v_inv_nodes = copy_host(node_outputs[2]);
  result.q_ev_nodes = copy_host(node_outputs[3]);
  result.k_ev_nodes = copy_host(node_outputs[4]);
  result.q_inv_edges = copy_host(edge_outputs[0]);
  result.k_inv_edges = copy_host(edge_outputs[1]);
  result.v_inv_edges = copy_host(edge_outputs[2]);
  result.q_ev_edges = copy_host(edge_outputs[3]);
  result.k_ev_edges = copy_host(edge_outputs[4]);
  return result;
}

QkvBlock0Benchmark KokkosQkvBlock0::benchmark(
    std::size_t nodes, std::size_t repetitions) const {
  if (nodes == 0 || repetitions == 0)
    throw std::runtime_error("SO3LR Q/K/V benchmark sizes must be positive");
  DoubleView features("so3lr_qkv0_benchmark_features", nodes * feature_width);
  fill_view(features, nodes * feature_width, [](std::size_t i) {
    return (static_cast<double>(i % 251) - 125.0) / 127.0;
  });
  const std::array<DoubleView, 5> weights = {
      w_q_inv_, w_k_inv_, w_v_inv_, w_q_ev_, w_k_ev_};
  const std::array<DoubleView, 5> outputs = {
      DoubleView("so3lr_qkv0_benchmark_q_inv", nodes * feature_width),
      DoubleView("so3lr_qkv0_benchmark_k_inv", nodes * feature_width),
      DoubleView("so3lr_qkv0_benchmark_v_inv", nodes * feature_width),
      DoubleView("so3lr_qkv0_benchmark_q_ev", nodes * feature_width),
      DoubleView("so3lr_qkv0_benchmark_k_ev", nodes * feature_width)};
  for (int warmup = 0; warmup < 3; ++warmup)
    launch_cublas(handle_, features, nodes, weights, outputs);
  Kokkos::fence();
  Kokkos::Timer timer;
  for (std::size_t iteration = 0; iteration < repetitions; ++iteration)
    launch_cublas(handle_, features, nodes, weights, outputs);
  Kokkos::fence();
  return finish_benchmark(outputs, nodes, repetitions,
                          persistent_device_bytes_, timer.seconds() * 1000.0);
}

QkvBlock0Benchmark KokkosQkvBlock0::benchmark_baseline(
    std::size_t nodes, std::size_t repetitions) const {
  if (nodes == 0 || repetitions == 0)
    throw std::runtime_error("SO3LR Q/K/V benchmark sizes must be positive");
  DoubleView features("so3lr_qkv0_baseline_features", nodes * feature_width);
  fill_view(features, nodes * feature_width, [](std::size_t i) {
    return (static_cast<double>(i % 251) - 125.0) / 127.0;
  });
  const std::array<DoubleView, 5> weights = {
      w_q_inv_, w_k_inv_, w_v_inv_, w_q_ev_, w_k_ev_};
  const std::array<DoubleView, 5> outputs = {
      DoubleView("so3lr_qkv0_baseline_q_inv", nodes * feature_width),
      DoubleView("so3lr_qkv0_baseline_k_inv", nodes * feature_width),
      DoubleView("so3lr_qkv0_baseline_v_inv", nodes * feature_width),
      DoubleView("so3lr_qkv0_baseline_q_ev", nodes * feature_width),
      DoubleView("so3lr_qkv0_baseline_k_ev", nodes * feature_width)};
  for (int warmup = 0; warmup < 2; ++warmup)
    launch_baseline(features, nodes, weights, outputs, "_warmup");
  Kokkos::fence();
  Kokkos::Timer timer;
  for (std::size_t iteration = 0; iteration < repetitions; ++iteration)
    launch_baseline(features, nodes, weights, outputs, "_benchmark");
  Kokkos::fence();
  return finish_benchmark(outputs, nodes, repetitions,
                          persistent_device_bytes_, timer.seconds() * 1000.0);
}

}  // namespace so3lr
