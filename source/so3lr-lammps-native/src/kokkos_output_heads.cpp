#include "so3lr/kokkos_output_heads.hpp"

#include <Kokkos_Timer.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace so3lr {
namespace {

using DoubleView = Kokkos::View<double *>;

// The input feature width F and the energy/charge hidden width come from the
// architecture descriptor (F = 128 for SO3LR v1, up to 256, where
// energy and charge heads map 256 -> 128). The ratio heads are 64 wide in
// every released model.
constexpr std::size_t hirshfeld_width = 64;
constexpr std::size_t energy_elements = 118;
constexpr std::size_t lr_elements = 100;

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

std::vector<double> copy_host(const DoubleView &device) {
  const auto host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), device);
  std::vector<double> result(device.extent(0));
  for (std::size_t i = 0; i < result.size(); ++i) result[i] = host(i);
  return result;
}

void require_tensor(const TensorRecord &tensor,
                    const std::vector<std::size_t> &shape) {
  if (tensor.dtype != "float64" || tensor.shape != shape)
    throw std::runtime_error("SO3LR output-head tensor mismatch: " +
                             tensor.state_key);
}

DoubleView load_double(const NativeModel &model, const std::string &key,
                       const std::vector<std::size_t> &shape) {
  const auto &tensor = model.tensor(key);
  require_tensor(tensor, shape);
  std::size_t count = 1;
  for (const auto extent : shape) count *= extent;
  DoubleView result(key + "_native_head", count);
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
      "cublasDgemm(output-head)");
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
      "cublasDgemm(energy-head-reverse)");
}

// Shapes come from the tensors, not from constants: H = rows of W1 and
// K = rows of W2 (64 and 64 for the v1 Hirshfeld head, 64 and 1 for the
// a0 and C6 heads). The workspace buffers are sized for H = 64.
RatioHeadParameters load_ratio_head(const NativeModel &model,
                                    const std::string &prefix,
                                    std::size_t inv_width) {
  const auto &w1 = model.tensor(prefix + ".transform_features.0.weight");
  const auto &w2 = model.tensor(prefix + ".transform_features.2.weight");
  if (w1.shape.size() != 2 || w1.shape[1] != inv_width || w2.shape.size() != 2)
    throw std::runtime_error("SO3LR ratio head has unexpected weight ranks: " + prefix);
  RatioHeadParameters head;
  head.hidden_width = w1.shape[0];
  head.key_width = w2.shape[0];
  if (w2.shape[1] != head.hidden_width)
    throw std::runtime_error("SO3LR ratio head layer widths disagree: " + prefix);
  if (head.hidden_width != hirshfeld_width || head.key_width == 0 ||
      head.key_width > hirshfeld_width)
    throw std::runtime_error("SO3LR ratio head width is not supported: " + prefix);
  const std::size_t H = head.hidden_width, K = head.key_width;
  head.v_shift = load_double(model, prefix + ".v_shift_embedding.weight", {lr_elements, 1});
  head.q_embedding = load_double(model, prefix + ".q_embedding.weight", {lr_elements, K});
  head.weight_1 = load_double(model, prefix + ".transform_features.0.weight", {H, inv_width});
  head.bias_1 = load_double(model, prefix + ".transform_features.0.bias", {H});
  head.weight_2 = load_double(model, prefix + ".transform_features.2.weight", {K, H});
  head.bias_2 = load_double(model, prefix + ".transform_features.2.bias", {K});
  head.present = true;
  return head;
}

// Forward of one ratio head. For the v1 Hirshfeld head (K = 64) this performs
// the same operations, in the same order, as the fused kernel it replaces, so
// the result is bit-identical: sqrt(64) is exactly 8.
void ratio_head_forward(cublasHandle_t handle, const RatioHeadParameters &head,
                        const DoubleView &inv_features,
                        const Kokkos::View<std::int64_t *> &atomic_numbers,
                        std::size_t nodes, const DoubleView &preact_1,
                        const DoubleView &hidden_1, const DoubleView &hidden_2,
                        const DoubleView &ratios, std::size_t inv_width) {
  const std::size_t H = head.hidden_width;
  const std::size_t K = head.key_width;
  const double sqrt_k = std::sqrt(static_cast<double>(K));
  linear(handle, inv_features, nodes, inv_width, H, head.weight_1, preact_1);
  const auto bias_1 = head.bias_1;
  Kokkos::parallel_for(
      "so3lr_head_hirshfeld_silu", Kokkos::RangePolicy<>(0, nodes * H),
      KOKKOS_LAMBDA(const std::size_t i) {
        const double value = preact_1(i) + bias_1(i % H);
        preact_1(i) = value;
        hidden_1(i) = value / (1.0 + Kokkos::exp(-value));
      });
  linear(handle, hidden_1, nodes, H, K, head.weight_2, hidden_2);
  const auto bias_2 = head.bias_2;
  const auto q_embedding = head.q_embedding;
  const auto v_shift = head.v_shift;
  Kokkos::parallel_for(
      "so3lr_head_ratio_finalize", Kokkos::RangePolicy<>(0, nodes),
      KOKKOS_LAMBDA(const std::size_t node) {
        const std::int64_t z_value = atomic_numbers(node);
        if (z_value <= 0 ||
            z_value >= static_cast<std::int64_t>(lr_elements) ||
            z_value > static_cast<std::int64_t>(energy_elements))
          return;
        const std::size_t z = static_cast<std::size_t>(z_value);
        double qk = 0.0;
        for (std::size_t channel = 0; channel < K; ++channel) {
          const double k = hidden_2(node * K + channel) + bias_2(channel);
          hidden_2(node * K + channel) = k;
          qk += q_embedding(z * K + channel) * k;
        }
        ratios(node) = Kokkos::abs(v_shift(z) + qk / sqrt_k);
      });
}

// Adjoint of ratio_head_forward, accumulated into `combined`.
void ratio_head_reverse(cublasHandle_t handle, const RatioHeadParameters &head,
                        const Kokkos::View<std::int64_t *> &atomic_numbers,
                        std::size_t nodes, const DoubleView &seeds,
                        const DoubleView &preact_1, const DoubleView &hidden_1,
                        const DoubleView &hidden_2, const DoubleView &scratch,
                        const DoubleView &combined, std::size_t inv_width) {
  const std::size_t H = head.hidden_width;
  const std::size_t K = head.key_width;
  const double sqrt_k = std::sqrt(static_cast<double>(K));
  const auto q_embedding = head.q_embedding;
  const auto v_shift = head.v_shift;
  Kokkos::parallel_for(
      "so3lr_multihead_hirshfeld_output_reverse",
      Kokkos::RangePolicy<>(0, nodes),
      KOKKOS_LAMBDA(const std::size_t node) {
        const std::size_t z = static_cast<std::size_t>(atomic_numbers(node));
        double effective = v_shift(z);
        for (std::size_t channel = 0; channel < K; ++channel)
          effective += q_embedding(z * K + channel) *
                       hidden_2(node * K + channel) / sqrt_k;
        const double sign = effective > 0.0 ? 1.0 :
                            (effective < 0.0 ? -1.0 : 0.0);
        const double seed = seeds(node) * sign / sqrt_k;
        for (std::size_t channel = 0; channel < K; ++channel)
          hidden_2(node * K + channel) = seed * q_embedding(z * K + channel);
      });
  linear_input_gradient(handle, hidden_2, nodes, K, H, head.weight_2, hidden_1);
  Kokkos::parallel_for(
      "so3lr_multihead_hirshfeld_reverse_silu",
      Kokkos::RangePolicy<>(0, nodes * H),
      KOKKOS_LAMBDA(const std::size_t i) {
        const double value = preact_1(i);
        const double sigmoid = 1.0 / (1.0 + Kokkos::exp(-value));
        const double derivative = sigmoid * (1.0 + value * (1.0 - sigmoid));
        hidden_1(i) *= derivative;
      });
  linear_input_gradient(handle, hidden_1, nodes, H, inv_width, head.weight_1,
                        scratch);
  Kokkos::parallel_for(
      "so3lr_multihead_accumulate_hirshfeld",
      Kokkos::RangePolicy<>(0, nodes * inv_width),
      KOKKOS_LAMBDA(const std::size_t i) { combined(i) += scratch(i); });
}

std::size_t dynamic_bytes(std::size_t nodes, std::size_t inv_width,
                          std::size_t hidden_width) {
  const std::size_t doubles_per_node =
      3 * hidden_width + inv_width + 1 +  // energy preact/activation/grads/output
      2 * hidden_width + 1 + 1 +   // charge preact/activation/raw/corrected
      3 * hirshfeld_width + 1 +    // Hirshfeld preact/two hidden/output
      2 * inv_width;               // combined input gradient and scratch
  return (nodes * doubles_per_node + 3) * sizeof(double);
}

}  // namespace

OutputHeadsDeviceWorkspace::OutputHeadsDeviceWorkspace(std::size_t nodes,
                                                       const ArchDims &dims)
    : OutputHeadsDeviceWorkspace(nodes, dims.invariant_width,
                                 dims.head_hidden_width) {}

OutputHeadsDeviceWorkspace::OutputHeadsDeviceWorkspace(
    std::size_t nodes, std::size_t inv_width, std::size_t hidden_width)
    : energy_preact("so3lr_head_energy_preact", nodes * hidden_width),
      energy_hidden("so3lr_head_energy_hidden", nodes * hidden_width),
      atomic_energies("so3lr_head_atomic_energies", nodes),
      // Shared by the energy and charge head adjoints (same hidden width).
      energy_hidden_grad("so3lr_head_energy_hidden_grad",
                         nodes * hidden_width),
      energy_input_grad("so3lr_head_energy_input_grad", nodes * inv_width),
      charge_preact("so3lr_head_charge_preact", nodes * hidden_width),
      charge_hidden("so3lr_head_charge_hidden", nodes * hidden_width),
      raw_charges("so3lr_head_raw_charges", nodes),
      partial_charges("so3lr_head_partial_charges", nodes),
      hirshfeld_preact_1("so3lr_head_hirshfeld_preact1",
                         nodes * hirshfeld_width),
      hirshfeld_hidden_1("so3lr_head_hirshfeld_hidden1",
                         nodes * hirshfeld_width),
      hirshfeld_hidden_2("so3lr_head_hirshfeld_hidden2",
                         nodes * hirshfeld_width),
      hirshfeld_ratios("so3lr_head_hirshfeld_ratios", nodes),
      combined_input_grad("so3lr_head_combined_input_grad", nodes * inv_width),
      reverse_input_scratch("so3lr_head_reverse_input_scratch", nodes * inv_width),
      reductions("so3lr_head_reductions", 2),
      reverse_reductions("so3lr_head_reverse_reductions", 1) {
  if (nodes == 0)
    throw std::runtime_error("SO3LR output-head workspace is empty");
}

KokkosOutputHeads::KokkosOutputHeads(const NativeModel &model)
    : input_width_(model.arch().invariant_width),
      hidden_width_(model.arch().head_hidden_width) {
  const std::size_t inv_width = input_width_;
  const std::size_t hidden_width = hidden_width_;
  const auto &heads = model.manifest().at("architecture").at("output_heads").array();
  // The energy, charge and Hirshfeld/a0 heads are always present, in this
  // order; a model may add a fourth, the C6 head.
  const bool three = heads.size() == 3;
  const bool four = heads.size() == 4 &&
                    heads[3].string() == "c6_ratios_output_block";
  if (!(three || four) ||
      heads[0].string() != "atomic_energy_output_block" ||
      heads[1].string() != "partial_charges_output_block" ||
      heads[2].string() != "hirshfeld_output_block")
    throw std::runtime_error("SO3LR output-head architecture changed");

  const std::string energy = "model.atomic_energy_output_block";
  energy_weight_1_ = load_double(model, energy + ".layers.0.weight",
                                 {hidden_width, inv_width});
  energy_bias_1_ =
      load_double(model, energy + ".layers.0.bias", {hidden_width});
  energy_weight_2_ =
      load_double(model, energy + ".final_layer.weight", {1, hidden_width});
  energy_bias_2_ = load_double(model, energy + ".final_layer.bias", {1});
  energy_scales_ =
      load_double(model, energy + ".energy_scales.weight", {1, 118});
  energy_shifts_ = load_double(model, energy + ".energy_shifts", {118});

  const std::string charge = "model.partial_charges_output_block";
  charge_embedding_ =
      load_double(model, charge + ".atomic_embedding.weight", {100, 1});
  charge_weight_1_ = load_double(
      model, charge + ".transform_inv_features.0.weight",
      {hidden_width, inv_width});
  charge_bias_1_ = load_double(
      model, charge + ".transform_inv_features.0.bias", {hidden_width});
  charge_weight_2_ = load_double(
      model, charge + ".transform_inv_features.2.weight", {1, hidden_width});
  charge_bias_2_ = load_double(
      model, charge + ".transform_inv_features.2.bias", {1});

  hirshfeld_head_ =
      load_ratio_head(model, "model.hirshfeld_output_block", inv_width);
  if (four)
    c6_head_ =
        load_ratio_head(model, "model.c6_ratios_output_block", inv_width);

  check_cublas(cublasCreate(&handle_), "cublasCreate");
  const auto stream = Kokkos::Cuda().cuda_stream();
  check_cublas(cublasSetStream(handle_, stream), "cublasSetStream");
  cudaStream_t configured = nullptr;
  check_cublas(cublasGetStream(handle_, &configured), "cublasGetStream");
  shared_kokkos_stream_ = configured == stream;
  // Energy, charge and Hirshfeld parameters (16877 + 16741 + 18916 in v1).
  const std::size_t persistent_doubles =
      (hidden_width * inv_width + 2 * hidden_width + 1 + 2 * 118) +
      (100 + hidden_width * inv_width + 2 * hidden_width + 1) +
      (2 * 100 * 1 + 100 * (hirshfeld_head_.key_width - 1) +
       hirshfeld_head_.hidden_width * (inv_width + 1) +
       hirshfeld_head_.key_width * (hirshfeld_head_.hidden_width + 1));
  persistent_device_bytes_ = persistent_doubles * sizeof(double);
  checkpoint_contract_verified_ = shared_kokkos_stream_;
  Kokkos::fence();
}

KokkosOutputHeads::~KokkosOutputHeads() {
  if (handle_ != nullptr) {
    Kokkos::fence();
    static_cast<void>(cublasDestroy(handle_));
  }
}

void KokkosOutputHeads::launch_device(
    const DoubleView &inv_features, const Int64View &atomic_numbers,
    double total_charge, const OutputHeadsDeviceWorkspace &workspace) const {
  const std::size_t inv_width = input_width_;
  const std::size_t hidden_width = hidden_width_;
  const std::size_t nodes = atomic_numbers.extent(0);
  if (nodes == 0 || inv_features.extent(0) != nodes * inv_width ||
      workspace.energy_hidden.extent(0) != nodes * hidden_width ||
      workspace.charge_hidden.extent(0) != nodes * hidden_width ||
      workspace.hirshfeld_hidden_1.extent(0) !=
          nodes * hirshfeld_width ||
      workspace.reductions.extent(0) != 2)
    throw std::runtime_error("SO3LR output-head device-view contract mismatch");

  // Copy all views used by CUDA lambdas into the closure. Capturing this would
  // expose the host-resident C++ object to device code.
  const auto energy_preact = workspace.energy_preact;
  const auto energy_hidden = workspace.energy_hidden;
  const auto atomic_energies = workspace.atomic_energies;
  const auto charge_preact = workspace.charge_preact;
  const auto charge_hidden = workspace.charge_hidden;
  const auto raw_charges = workspace.raw_charges;
  const auto partial_charges = workspace.partial_charges;
  const auto reductions = workspace.reductions;
  const auto energy_bias_1 = energy_bias_1_;
  const auto energy_bias_2 = energy_bias_2_;
  const auto energy_scales = energy_scales_;
  const auto energy_shifts = energy_shifts_;
  const auto charge_bias_1 = charge_bias_1_;
  const auto charge_bias_2 = charge_bias_2_;
  const auto charge_embedding = charge_embedding_;

  linear(handle_, inv_features, nodes, inv_width, hidden_width,
         energy_weight_1_, energy_preact);
  Kokkos::parallel_for(
      "so3lr_head_energy_silu",
      Kokkos::RangePolicy<>(0, nodes * hidden_width),
      KOKKOS_LAMBDA(const std::size_t i) {
        const double value =
            energy_preact(i) + energy_bias_1(i % hidden_width);
        energy_preact(i) = value;
        energy_hidden(i) = value / (1.0 + Kokkos::exp(-value));
      });
  linear(handle_, energy_hidden, nodes, hidden_width, 1, energy_weight_2_,
         atomic_energies);

  linear(handle_, inv_features, nodes, inv_width, hidden_width,
         charge_weight_1_, charge_preact);
  Kokkos::parallel_for(
      "so3lr_head_charge_silu",
      Kokkos::RangePolicy<>(0, nodes * hidden_width),
      KOKKOS_LAMBDA(const std::size_t i) {
        const double value =
            charge_preact(i) + charge_bias_1(i % hidden_width);
        charge_preact(i) = value;
        charge_hidden(i) = value / (1.0 + Kokkos::exp(-value));
      });
  linear(handle_, charge_hidden, nodes, hidden_width, 1, charge_weight_2_,
         raw_charges);

  ratio_head_forward(handle_, hirshfeld_head_, inv_features, atomic_numbers,
                     nodes, workspace.hirshfeld_preact_1,
                     workspace.hirshfeld_hidden_1, workspace.hirshfeld_hidden_2,
                     workspace.hirshfeld_ratios, inv_width);
  if (c6_head_.present) {
    const std::size_t capacity = workspace.hirshfeld_ratios.extent(0);
    if (workspace.c6_ratios.extent(0) != capacity) {
      workspace.c6_preact_1 = DoubleView("so3lr_head_c6_preact1", capacity * hirshfeld_width);
      workspace.c6_hidden_1 = DoubleView("so3lr_head_c6_hidden1", capacity * hirshfeld_width);
      workspace.c6_hidden_2 = DoubleView("so3lr_head_c6_hidden2", capacity * hirshfeld_width);
      workspace.c6_ratios = DoubleView("so3lr_head_c6_ratios", capacity);
    }
    ratio_head_forward(handle_, c6_head_, inv_features, atomic_numbers, nodes,
                       workspace.c6_preact_1, workspace.c6_hidden_1,
                       workspace.c6_hidden_2, workspace.c6_ratios, inv_width);
  }

  Kokkos::parallel_for(
      "so3lr_head_zero_reductions", Kokkos::RangePolicy<>(0, 2),
      KOKKOS_LAMBDA(const std::size_t i) { reductions(i) = 0.0; });
  Kokkos::parallel_for(
      "so3lr_head_finalize_outputs", Kokkos::RangePolicy<>(0, nodes),
      KOKKOS_LAMBDA(const std::size_t node) {
        const std::int64_t z_value = atomic_numbers(node);
        if (z_value <= 0 ||
            z_value >= static_cast<std::int64_t>(lr_elements) ||
            z_value > static_cast<std::int64_t>(energy_elements))
          return;
        // Learned one-hot tables use class Z-1. Physical charge,
        // Hirshfeld and reference-element tables use atomic number Z.
        const std::size_t z = static_cast<std::size_t>(z_value);
        const std::size_t element = static_cast<std::size_t>(z_value - 1);
        const double energy =
            (atomic_energies(node) + energy_bias_2(0)) *
                energy_scales(element) +
            energy_shifts(element);
        atomic_energies(node) = energy;
        const double raw_charge =
            raw_charges(node) + charge_bias_2(0) + charge_embedding(z);
        raw_charges(node) = raw_charge;
        Kokkos::atomic_add(&reductions(0), raw_charge);
        Kokkos::atomic_add(&reductions(1), energy);
      });
  Kokkos::parallel_for(
      "so3lr_head_charge_conservation", Kokkos::RangePolicy<>(0, nodes),
      KOKKOS_LAMBDA(const std::size_t node) {
        const double correction =
            (total_charge - reductions(0)) / static_cast<double>(nodes);
        partial_charges(node) = raw_charges(node) + correction;
      });
}

void KokkosOutputHeads::launch_energy_reverse_device(
    const Int64View &atomic_numbers,
    const OutputHeadsDeviceWorkspace &workspace) const {
  const std::size_t inv_width = input_width_;
  const std::size_t hidden_width = hidden_width_;
  const std::size_t nodes = atomic_numbers.extent(0);
  if (nodes == 0 ||
      workspace.energy_preact.extent(0) != nodes * hidden_width ||
      workspace.energy_hidden_grad.extent(0) != nodes * hidden_width ||
      workspace.energy_input_grad.extent(0) != nodes * inv_width)
    throw std::runtime_error("SO3LR energy-head reverse contract mismatch");
  const auto preact = workspace.energy_preact;
  const auto hidden_grad = workspace.energy_hidden_grad;
  const auto input_grad = workspace.energy_input_grad;
  const auto scales = energy_scales_;
  const auto final_weight = energy_weight_2_;
  Kokkos::parallel_for(
      "so3lr_energy_head_reverse_silu",
      Kokkos::RangePolicy<>(0, nodes * hidden_width),
      KOKKOS_LAMBDA(const std::size_t i) {
        const std::size_t node = i / hidden_width;
        const std::size_t channel = i % hidden_width;
        const std::int64_t z_value = atomic_numbers(node);
        if (z_value <= 0 || z_value > 118) {
          hidden_grad(i) = 0.0;
          return;
        }
        const std::size_t element =
            static_cast<std::size_t>(z_value - 1);
        const double value = preact(i);
        const double sigmoid = 1.0 / (1.0 + Kokkos::exp(-value));
        const double silu_prime =
            sigmoid * (1.0 + value * (1.0 - sigmoid));
        hidden_grad(i) =
            scales(element) * final_weight(channel) * silu_prime;
      });
  linear_input_gradient(handle_, hidden_grad, nodes, hidden_width, inv_width,
                        energy_weight_1_, input_grad);
}

void KokkosOutputHeads::launch_multihead_reverse_device(
    const Int64View &atomic_numbers, const DoubleView &energy_seeds,
    const DoubleView &partial_charge_seeds,
    const DoubleView &hirshfeld_seeds,
    const OutputHeadsDeviceWorkspace &workspace,
    double global_charge_seed_mean,
    bool use_global_charge_seed_mean, const DoubleView &c6_seeds) const {
  const std::size_t inv_width = input_width_;
  const std::size_t hidden_width = hidden_width_;
  const std::size_t nodes = atomic_numbers.extent(0);
  if (nodes == 0 || energy_seeds.extent(0) != nodes ||
      partial_charge_seeds.extent(0) != nodes ||
      hirshfeld_seeds.extent(0) != nodes ||
      workspace.combined_input_grad.extent(0) != nodes * inv_width ||
      workspace.reverse_input_scratch.extent(0) != nodes * inv_width)
    throw std::runtime_error("SO3LR multihead reverse contract mismatch");

  const auto hidden_grad = workspace.energy_hidden_grad;
  const auto combined = workspace.combined_input_grad;
  const auto scratch = workspace.reverse_input_scratch;
  const auto energy_preact = workspace.energy_preact;
  const auto energy_scales = energy_scales_;
  const auto energy_final_weight = energy_weight_2_;
  Kokkos::parallel_for(
      "so3lr_multihead_energy_reverse_silu",
      Kokkos::RangePolicy<>(0, nodes * hidden_width),
      KOKKOS_LAMBDA(const std::size_t i) {
        const std::size_t node = i / hidden_width;
        const std::size_t channel = i % hidden_width;
        const std::int64_t z_value = atomic_numbers(node);
        if (z_value <= 0 || z_value > 118) {
          hidden_grad(i) = 0.0;
          return;
        }
        const std::size_t element =
            static_cast<std::size_t>(z_value - 1);
        const double value = energy_preact(i);
        const double sigmoid = 1.0 / (1.0 + Kokkos::exp(-value));
        const double derivative = sigmoid * (1.0 + value * (1.0 - sigmoid));
        hidden_grad(i) = energy_seeds(node) * energy_scales(element) *
                         energy_final_weight(channel) * derivative;
      });
  linear_input_gradient(handle_, hidden_grad, nodes, hidden_width, inv_width,
                        energy_weight_1_, combined);

  Kokkos::deep_copy(workspace.reverse_reductions, 0.0);
  const auto reverse_reductions = workspace.reverse_reductions;
  Kokkos::parallel_for(
      "so3lr_multihead_charge_seed_sum", Kokkos::RangePolicy<>(0, nodes),
      KOKKOS_LAMBDA(const std::size_t node) {
        Kokkos::atomic_add(&reverse_reductions(0),
                           partial_charge_seeds(node));
      });
  const auto charge_preact = workspace.charge_preact;
  const auto charge_final_weight = charge_weight_2_;
  Kokkos::parallel_for(
      "so3lr_multihead_charge_reverse_silu",
      Kokkos::RangePolicy<>(0, nodes * hidden_width),
      KOKKOS_LAMBDA(const std::size_t i) {
        const std::size_t node = i / hidden_width;
        const std::size_t channel = i % hidden_width;
        const double raw_seed =
            partial_charge_seeds(node) -
            (use_global_charge_seed_mean
                 ? global_charge_seed_mean
                 : reverse_reductions(0) / static_cast<double>(nodes));
        const double value = charge_preact(i);
        const double sigmoid = 1.0 / (1.0 + Kokkos::exp(-value));
        const double derivative = sigmoid * (1.0 + value * (1.0 - sigmoid));
        hidden_grad(i) = raw_seed * charge_final_weight(channel) * derivative;
      });
  linear_input_gradient(handle_, hidden_grad, nodes, hidden_width, inv_width,
                        charge_weight_1_, scratch);
  Kokkos::parallel_for(
      "so3lr_multihead_accumulate_charge", Kokkos::RangePolicy<>(0, nodes * inv_width),
      KOKKOS_LAMBDA(const std::size_t i) { combined(i) += scratch(i); });

  ratio_head_reverse(handle_, hirshfeld_head_, atomic_numbers, nodes,
                     hirshfeld_seeds, workspace.hirshfeld_preact_1,
                     workspace.hirshfeld_hidden_1, workspace.hirshfeld_hidden_2,
                     scratch, combined, inv_width);
  if (c6_head_.present) {
    if (c6_seeds.extent(0) != nodes)
      throw std::runtime_error("SO3LR model has a C6 head but no C6 seeds were given");
    ratio_head_reverse(handle_, c6_head_, atomic_numbers, nodes, c6_seeds,
                       workspace.c6_preact_1, workspace.c6_hidden_1,
                       workspace.c6_hidden_2, scratch, combined, inv_width);
  }
}

OutputHeadsResults KokkosOutputHeads::evaluate(
    const std::vector<double> &inv_features,
    const std::vector<std::int64_t> &atomic_numbers,
    double total_charge) const {
  const std::size_t inv_width = input_width_;
  const std::size_t hidden_width = hidden_width_;
  const std::size_t nodes = atomic_numbers.size();
  if (nodes == 0 || inv_features.size() != nodes * inv_width)
    throw std::runtime_error("SO3LR output-head host contract mismatch");
  if (std::any_of(atomic_numbers.begin(), atomic_numbers.end(),
                  [](std::int64_t z) { return z < 0 || z >= 100; }))
    throw std::runtime_error("SO3LR output-head atomic number out of range");
  DoubleView inv("so3lr_head_eval_inv", inv_features.size());
  Int64View z("so3lr_head_eval_z", nodes);
  fill_view(inv, inv_features.size(),
            [&](std::size_t i) { return inv_features[i]; });
  fill_view(z, nodes, [&](std::size_t i) { return atomic_numbers[i]; });
  OutputHeadsDeviceWorkspace workspace(nodes, inv_width, hidden_width);
  launch_device(inv, z, total_charge, workspace);
  Kokkos::fence();
  OutputHeadsResults result;
  result.atomic_energies = copy_host(workspace.atomic_energies);
  result.raw_charges = copy_host(workspace.raw_charges);
  result.partial_charges = copy_host(workspace.partial_charges);
  result.hirshfeld_ratios = copy_host(workspace.hirshfeld_ratios);
  const auto reductions = copy_host(workspace.reductions);
  result.charge_sum = 0.0;
  for (const double charge : result.partial_charges)
    result.charge_sum += charge;
  result.learned_sr_energy = reductions[1];
  return result;
}

std::vector<double> KokkosOutputHeads::evaluate_energy_input_gradient(
    const std::vector<double> &inv_features,
    const std::vector<std::int64_t> &atomic_numbers) const {
  const std::size_t inv_width = input_width_;
  const std::size_t hidden_width = hidden_width_;
  const std::size_t nodes = atomic_numbers.size();
  if (nodes == 0 || inv_features.size() != nodes * inv_width ||
      std::any_of(atomic_numbers.begin(), atomic_numbers.end(),
                  [](std::int64_t z) { return z < 0 || z >= 100; }))
    throw std::runtime_error("SO3LR energy-head reverse host contract mismatch");
  DoubleView inv("so3lr_energy_reverse_eval_inv", inv_features.size());
  Int64View z("so3lr_energy_reverse_eval_z", nodes);
  fill_view(inv, inv_features.size(),
            [&](std::size_t i) { return inv_features[i]; });
  fill_view(z, nodes, [&](std::size_t i) { return atomic_numbers[i]; });
  OutputHeadsDeviceWorkspace workspace(nodes, inv_width, hidden_width);
  launch_device(inv, z, 0.0, workspace);
  launch_energy_reverse_device(z, workspace);
  Kokkos::fence();
  return copy_host(workspace.energy_input_grad);
}

OutputHeadsBenchmark KokkosOutputHeads::benchmark(
    std::size_t nodes, std::size_t repetitions) const {
  const std::size_t inv_width = input_width_;
  const std::size_t hidden_width = hidden_width_;
  if (nodes == 0 || repetitions == 0)
    throw std::runtime_error("SO3LR output-head benchmark invalid");
  DoubleView inv("so3lr_head_benchmark_inv", nodes * inv_width);
  Int64View z("so3lr_head_benchmark_z", nodes);
  Kokkos::parallel_for(
      "so3lr_head_benchmark_features",
      Kokkos::RangePolicy<>(0, nodes * inv_width),
      KOKKOS_LAMBDA(const std::size_t i) {
        inv(i) = (static_cast<double>(i % 67) - 33.0) / 41.0;
      });
  Kokkos::parallel_for(
      "so3lr_head_benchmark_z", Kokkos::RangePolicy<>(0, nodes),
      KOKKOS_LAMBDA(const std::size_t i) { z(i) = i % 3 == 0 ? 8 : 1; });
  OutputHeadsDeviceWorkspace workspace(nodes, inv_width, hidden_width);
  for (int warmup = 0; warmup < 2; ++warmup)
    launch_device(inv, z, 0.0, workspace);
  Kokkos::fence();
  Kokkos::Timer timer;
  for (std::size_t i = 0; i < repetitions; ++i)
    launch_device(inv, z, 0.0, workspace);
  Kokkos::fence();
  const double milliseconds =
      timer.seconds() * 1000.0 / static_cast<double>(repetitions);
  double checksum = 0.0;
  const auto energy = workspace.atomic_energies;
  const auto charge = workspace.partial_charges;
  const auto hirshfeld = workspace.hirshfeld_ratios;
  Kokkos::parallel_reduce(
      "so3lr_head_benchmark_checksum", Kokkos::RangePolicy<>(0, 3),
      KOKKOS_LAMBDA(const int which, double &update) {
        if (which == 0) update += energy(nodes / 3);
        if (which == 1) update += charge(nodes / 2);
        if (which == 2) update += hirshfeld(nodes / 4);
      },
      checksum);
  Kokkos::fence();
  OutputHeadsBenchmark result;
  result.nodes = nodes;
  result.repetitions = repetitions;
  result.persistent_device_bytes = persistent_device_bytes_;
  result.workspace_bytes =
      persistent_device_bytes_ + dynamic_bytes(nodes, inv_width, hidden_width);
  result.host_boundary_bytes_per_iteration = 0;
  result.milliseconds = milliseconds;
  result.nodes_per_second =
      static_cast<double>(nodes) / (milliseconds / 1000.0);
  result.checksum = checksum;
  return result;
}

EnergyHeadReverseBenchmark KokkosOutputHeads::benchmark_energy_reverse(
    std::size_t nodes, std::size_t repetitions) const {
  const std::size_t inv_width = input_width_;
  const std::size_t hidden_width = hidden_width_;
  if (nodes == 0 || repetitions == 0)
    throw std::runtime_error("SO3LR energy-head reverse benchmark invalid");
  DoubleView inv("so3lr_energy_reverse_bench_inv", nodes * inv_width);
  Int64View z("so3lr_energy_reverse_bench_z", nodes);
  Kokkos::parallel_for(
      "so3lr_energy_reverse_bench_features",
      Kokkos::RangePolicy<>(0, nodes * inv_width),
      KOKKOS_LAMBDA(const std::size_t i) {
        inv(i) = (static_cast<double>(i % 67) - 33.0) / 41.0;
      });
  Kokkos::parallel_for(
      "so3lr_energy_reverse_bench_z", Kokkos::RangePolicy<>(0, nodes),
      KOKKOS_LAMBDA(const std::size_t i) { z(i) = i % 3 == 0 ? 8 : 1; });
  OutputHeadsDeviceWorkspace workspace(nodes, inv_width, hidden_width);
  launch_device(inv, z, 0.0, workspace);
  for (int warmup = 0; warmup < 2; ++warmup)
    launch_energy_reverse_device(z, workspace);
  Kokkos::fence();
  Kokkos::Timer timer;
  for (std::size_t i = 0; i < repetitions; ++i)
    launch_energy_reverse_device(z, workspace);
  Kokkos::fence();
  const double milliseconds =
      timer.seconds() * 1000.0 / static_cast<double>(repetitions);
  double checksum = 0.0;
  const auto gradient = workspace.energy_input_grad;
  Kokkos::parallel_reduce(
      "so3lr_energy_reverse_bench_checksum", Kokkos::RangePolicy<>(0, 3),
      KOKKOS_LAMBDA(const int which, double &update) {
        update += gradient((nodes / static_cast<std::size_t>(which + 2)) *
                               inv_width +
                           static_cast<std::size_t>(which * 17));
      },
      checksum);
  Kokkos::fence();
  EnergyHeadReverseBenchmark result;
  result.nodes = nodes;
  result.repetitions = repetitions;
  result.persistent_device_bytes = persistent_device_bytes_;
  result.workspace_bytes = persistent_device_bytes_ + dynamic_bytes(nodes, inv_width, hidden_width);
  result.host_boundary_bytes_per_iteration = 0;
  result.milliseconds = milliseconds;
  result.nodes_per_second =
      static_cast<double>(nodes) / (milliseconds / 1000.0);
  result.checksum = checksum;
  return result;
}

}  // namespace so3lr
