#include "so3lr/so3lr_charge_spin.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace so3lr {
namespace {

const char *const kPrefix = "model.";  // state-key prefix

bool has_tensor(const NativeModel &model, const std::string &key) {
  try {
    model.tensor(key);
    return true;
  } catch (const std::exception &) {
    return false;
  }
}

std::vector<double> read(const NativeModel &model, const std::string &key,
                         const std::vector<std::size_t> &shape) {
  const auto &t = model.tensor(key);
  if (t.dtype != "float64" || t.shape != shape)
    throw std::runtime_error("SO3LR charge/spin embedding tensor has an "
                             "unexpected shape: " + key);
  std::size_t n = 1;
  for (const auto s : shape) n *= s;
  std::vector<double> out(n);
  for (std::size_t i = 0; i < n; ++i) out[i] = model.float64(t, i);
  return out;
}

bool load_module(const NativeModel &model, const std::string &name,
                 std::size_t width, ChargeSpinEmbedding::Module &m) {
  const std::string base = std::string(kPrefix) + name + ".";
  if (!has_tensor(model, base + "Wq.weight")) return false;
  m.wq = read(model, base + "Wq.weight", {width, 118});
  m.wk = read(model, base + "Wk", {2, width});
  m.wv = read(model, base + "Wv", {2, width});
  m.w1 = read(model, base + "mlp.1.weight", {width, width});
  m.w3 = read(model, base + "mlp.3.weight", {width, width});
  return true;
}

double softplus(double x) {
  return std::max(x, 0.0) + std::log1p(std::exp(-std::abs(x)));
}

double silu(double x) { return x / (1.0 + std::exp(-x)); }

void add_module(const ChargeSpinEmbedding::Module &m, std::size_t width,
                const ElementCounts &counts, double psi,
                std::vector<double> &table) {
  if (psi == 0.0) return;  // the embedding is exactly zero
  // psi // inf in the reference: 0 for psi >= 0, -1 (the last row) below.
  const std::size_t row = psi < 0.0 ? 1 : 0;
  const double *k = m.wk.data() + row * width;
  const double *v = m.wv.data() + row * width;
  const double inv_sqrt = 1.0 / std::sqrt(static_cast<double>(width));
  std::array<double, 119> y{};
  double denominator = 0.0;
  for (std::size_t z = 1; z <= 118; ++z) {
    if (counts[z] <= 0) continue;
    double qk = 0.0;
    for (std::size_t c = 0; c < width; ++c) qk += m.wq[c * 118 + (z - 1)] * k[c];
    y[z] = softplus(qk * inv_sqrt);
    denominator += static_cast<double>(counts[z]) * y[z];
  }
  std::vector<double> x(width), h(width), g(width);
  for (std::size_t z = 1; z <= 118; ++z) {
    if (counts[z] <= 0) continue;
    const double a = psi * y[z] / denominator;
    for (std::size_t c = 0; c < width; ++c) x[c] = a * v[c];
    for (std::size_t o = 0; o < width; ++o) {
      double s = 0.0;
      for (std::size_t i = 0; i < width; ++i) s += m.w1[o * width + i] * silu(x[i]);
      h[o] = s;
    }
    for (std::size_t o = 0; o < width; ++o) {
      double s = 0.0;
      for (std::size_t i = 0; i < width; ++i) s += m.w3[o * width + i] * silu(h[i]);
      g[o] = s;
    }
    double *out = table.data() + (z - 1) * width;
    for (std::size_t c = 0; c < width; ++c) out[c] += x[c] + g[c];
  }
}

}  // namespace

ChargeSpinEmbedding load_charge_spin_embedding(const NativeModel &model) {
  ChargeSpinEmbedding e;
  e.width = model.arch().invariant_width;
  e.has_charge = load_module(model, "charge_embedding", e.width, e.charge);
  e.has_spin = load_module(model, "spin_embedding", e.width, e.spin);
  return e;
}

std::vector<double> charge_spin_offset_table(const ChargeSpinEmbedding &e,
                                             const ElementCounts &counts,
                                             double total_charge,
                                             double unpaired_electrons) {
  if (total_charge != 0.0 && !e.has_charge)
    throw std::runtime_error("SO3LR: this model has no total-charge embedding; "
                             "it can only evaluate neutral systems");
  if (unpaired_electrons != 0.0 && !e.has_spin)
    throw std::runtime_error("SO3LR: this model has no spin embedding; it can "
                             "only evaluate singlets (multiplicity 1)");
  std::vector<double> table(118 * e.width, 0.0);
  if (e.has_charge) add_module(e.charge, e.width, counts, total_charge, table);
  if (e.has_spin) add_module(e.spin, e.width, counts, unpaired_electrons, table);
  return table;
}

std::string charge_multiplicity_problem(const ElementCounts &counts,
                                        double total_charge,
                                        double multiplicity) {
  if (!std::isfinite(total_charge) || std::round(total_charge) != total_charge)
    return "the total charge must be an integer";
  if (!std::isfinite(multiplicity) || std::round(multiplicity) != multiplicity ||
      multiplicity < 1.0)
    return "the spin multiplicity must be an integer >= 1";
  std::int64_t protons = 0;
  for (std::size_t z = 1; z <= 118; ++z)
    protons += counts[z] * static_cast<std::int64_t>(z);
  const std::int64_t electrons = protons - static_cast<std::int64_t>(total_charge);
  const std::int64_t unpaired = static_cast<std::int64_t>(multiplicity) - 1;
  if (electrons < 0)
    return "the total charge leaves a negative number of electrons";
  if (unpaired > electrons)
    return "multiplicity " + std::to_string(static_cast<long long>(multiplicity)) +
           " needs more unpaired electrons than the system's " +
           std::to_string(static_cast<long long>(electrons)) + " electrons";
  if ((electrons - unpaired) % 2 != 0)
    return std::to_string(static_cast<long long>(electrons)) +
           " electrons cannot have multiplicity " +
           std::to_string(static_cast<long long>(multiplicity)) +
           " (an even electron count needs an odd multiplicity and vice versa)";
  return std::string();
}

}  // namespace so3lr
