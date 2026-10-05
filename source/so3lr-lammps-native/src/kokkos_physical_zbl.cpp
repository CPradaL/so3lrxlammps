#include "so3lr/kokkos_physical_zbl.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace so3lr {
namespace {

void require_positive(double value, const char *name) {
  if (!std::isfinite(value) || !(value > 0.0))
    throw std::runtime_error(std::string("SO3LR invalid ZBL parameter ") + name);
}

}  // namespace

PhysicalZblWorkspace::PhysicalZblWorkspace(std::size_t owned_nodes,
                                           std::size_t edges)
    : atomic_energy("so3lr_zbl_atomic_energy", owned_nodes),
      edge_energy("so3lr_zbl_edge_energy", edges),
      edge_radial_gradient("so3lr_zbl_edge_radial_gradient", edges) {
  if (owned_nodes == 0 || edges == 0)
    throw std::runtime_error("SO3LR ZBL workspace is empty");
}

std::size_t physical_zbl_workspace_bytes(std::size_t owned_nodes,
                                         std::size_t edges) {
  return (owned_nodes + 2 * edges) * sizeof(double);
}

void launch_so3lr_zbl_device(
    const Kokkos::View<std::int64_t *> &atomic_numbers,
    std::size_t owned_nodes,
    const Kokkos::View<double *> &distances,
    const Kokkos::View<std::size_t *> &senders,
    const Kokkos::View<std::size_t *> &receivers,
    const PhysicalZblParameters &parameters,
    const PhysicalZblWorkspace &workspace,
    const Kokkos::View<double *> &nlh_a,
    const Kokkos::View<double *> &nlh_b) {
  const std::size_t nodes = atomic_numbers.extent(0);
  const std::size_t edges = distances.extent(0);
  if (nodes == 0 || owned_nodes == 0 || owned_nodes > nodes || edges == 0 ||
      senders.extent(0) != edges || receivers.extent(0) != edges ||
      workspace.atomic_energy.extent(0) != owned_nodes ||
      workspace.edge_energy.extent(0) != edges ||
      workspace.edge_radial_gradient.extent(0) != edges)
    throw std::runtime_error("SO3LR ZBL graph/workspace mismatch");
  require_positive(parameters.ke, "ke");
  require_positive(parameters.cutoff, "cutoff");
  require_positive(parameters.switch_off, "switch_off");
  if (parameters.kind == 1) {
    const std::size_t cap = parameters.nlh_z_capacity;
    if (cap == 0 || nlh_a.extent(0) != cap * cap * 3 ||
        nlh_b.extent(0) != cap * cap * 3)
      throw std::runtime_error("SO3LR NLH repulsion tables are missing or mis-sized");
  } else {
    require_positive(parameters.p, "p");
    require_positive(parameters.d, "d");
    double c_sum = 0.0;
    for (int k = 0; k < 4; ++k) {
      require_positive(parameters.a[k], "a");
      require_positive(parameters.c[k], "c");
      c_sum += parameters.c[k];
    }
    if (std::abs(c_sum - 1.0) > 2.0e-12)
      throw std::runtime_error("SO3LR normalized ZBL coefficients changed");
  }
  const auto table_a = nlh_a;
  const auto table_b = nlh_b;

  Kokkos::deep_copy(workspace.atomic_energy, 0.0);
  const auto z = atomic_numbers;
  const auto distance = distances;
  const auto sender = senders;
  const auto receiver = receivers;
  const auto atomic_energy = workspace.atomic_energy;
  const auto edge_energy = workspace.edge_energy;
  const auto radial_gradient = workspace.edge_radial_gradient;
  const PhysicalZblParameters params = parameters;
  Kokkos::parallel_for(
      "so3lr_native_zbl", Kokkos::RangePolicy<>(0, edges),
      KOKKOS_LAMBDA(const std::size_t edge) {
        const std::size_t i = receiver(edge);
        const std::size_t j = sender(edge);
        const double r = distance(edge);
        if (i >= owned_nodes || j >= nodes || !(r > 0.0) ||
            r > params.cutoff * (1.0 + 1.0e-12)) {
          edge_energy(edge) = 0.0;
          radial_gradient(edge) = 0.0;
          return;
        }
        const double zi = static_cast<double>(z(i));
        const double zj = static_cast<double>(z(j));
        const double zz = zi * zj;

        // Exact PhysNetCutoff used by this model and by the native GNN path:
        // 1 - 10*x^3 + 15*x^4 - 6*x^5 for x=r/r_cut < 1.
        const double normalized = r / params.cutoff;
        const double x2 = normalized * normalized;
        const double x3 = x2 * normalized;
        const double cutoff = r < params.cutoff
            ? 1.0 - 10.0 * x3 + 15.0 * x3 * normalized -
                  6.0 * x3 * x2
            : 0.0;
        const double dcutoff = r < params.cutoff
            ? (-30.0 * x2 + 60.0 * x3 - 30.0 * x3 * normalized) /
                  params.cutoff
            : 0.0;

        // Exact smooth switching_fn(r, x_on=0, x_off=1.5), including
        // the 1e-12 denominator stabilizer used by PyTorch.
        const double c = r / params.switch_off;
        const double left = 1.0 - c;
        const double right = c;
        const double sigma_left =
            left > 0.0 ? Kokkos::exp(-1.0 / (left > 1.0e-12 ? left : 1.0e-12)) : 0.0;
        const double sigma_right =
            right > 0.0 ? Kokkos::exp(-1.0 / (right > 1.0e-12 ? right : 1.0e-12)) : 0.0;
        const double denominator = sigma_left + sigma_right + 1.0e-12;
        const double dleft_dc = left > 0.0
            ? -sigma_left / ((left > 1.0e-12 ? left : 1.0e-12) *
                             (left > 1.0e-12 ? left : 1.0e-12))
            : 0.0;
        const double dright_dc = right > 0.0
            ? sigma_right / ((right > 1.0e-12 ? right : 1.0e-12) *
                             (right > 1.0e-12 ? right : 1.0e-12))
            : 0.0;
        const double switching = sigma_left / denominator;
        const double dswitch_dc =
            (dleft_dc * denominator -
             sigma_left * (dleft_dc + dright_dc)) /
            (denominator * denominator);
        const double dswitch = dswitch_dc / params.switch_off;

        const double safe_r = r > 1.0e-6 ? r : 1.0e-6;
        const double inverse_r = 1.0 / safe_r;
        const double dinverse_r = r > 1.0e-6 ? -1.0 / (r * r) : 0.0;
        const double x = params.ke * cutoff * zz * inverse_r;
        const double dx = params.ke * zz *
            (dcutoff * inverse_r + cutoff * dinverse_r);
        double y = 0.0;
        double dy = 0.0;
        if (params.kind == 1) {
          // NLH: tables are indexed (min(Z), max(Z)), both from 1.
          const std::int64_t za = z(i) < z(j) ? z(i) : z(j);
          const std::int64_t zb = z(i) < z(j) ? z(j) : z(i);
          const std::int64_t cap = static_cast<std::int64_t>(params.nlh_z_capacity);
          if (za >= 1 && zb <= cap) {
            const std::size_t base =
                (static_cast<std::size_t>(za - 1) * params.nlh_z_capacity +
                 static_cast<std::size_t>(zb - 1)) * 3;
            for (std::size_t k = 0; k < 3; ++k) {
              const double amplitude = table_a(base + k);
              const double rate = table_b(base + k);
              const double exponential = Kokkos::exp(-rate * r);
              y += amplitude * exponential;
              dy -= rate * amplitude * exponential;
            }
          }
        } else {
          const double screening_rate =
              (Kokkos::pow(zi, params.p) + Kokkos::pow(zj, params.p)) *
              params.d;
          const double rzd = r * screening_rate;
          for (int k = 0; k < 4; ++k) {
            const double exponential = Kokkos::exp(-params.a[k] * rzd);
            y += params.c[k] * exponential;
            dy -= params.c[k] * params.a[k] * screening_rate * exponential;
          }
        }
        const double directed_energy = 0.5 * switching * x * y;
        const double directed_gradient = 0.5 *
            (dswitch * x * y + switching * dx * y + switching * x * dy);
        edge_energy(edge) = directed_energy;
        radial_gradient(edge) = directed_gradient;
        Kokkos::atomic_add(&atomic_energy(i), directed_energy);
      });
}

}  // namespace so3lr
