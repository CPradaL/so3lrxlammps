#include "so3lr/kokkos_physical_long_range.hpp"

#include <cmath>
#include <stdexcept>

namespace so3lr {
namespace {

constexpr double inv_sqrt_pi = 0.56418958354775628695;

KOKKOS_INLINE_FUNCTION
void smooth_switch(double r, double cuton, double cutoff,
                   double &value, double &derivative) {
  if (r >= cutoff) {
    value = 0.0;
    derivative = 0.0;
    return;
  }
  const double width = cutoff - cuton;
  const double c = (r - cuton) / width;
  const double left = 1.0 - c;
  const double right = c;
  const double safe_left = left > 1.0e-12 ? left : 1.0e-12;
  const double safe_right = right > 1.0e-12 ? right : 1.0e-12;
  const double sigma_left = left > 0.0 ? Kokkos::exp(-1.0 / safe_left) : 0.0;
  const double sigma_right = right > 0.0 ? Kokkos::exp(-1.0 / safe_right) : 0.0;
  const double denominator = sigma_left + sigma_right + 1.0e-12;
  value = sigma_left / denominator;
  const double derivative_left =
      left > 0.0 ? sigma_left / (safe_left * safe_left) : 0.0;
  const double derivative_right =
      right > 0.0 ? sigma_right / (safe_right * safe_right) : 0.0;
  const double dleft_dc = -derivative_left;
  const double dright_dc = derivative_right;
  const double derivative_c =
      (dleft_dc * denominator - sigma_left * (dleft_dc + dright_dc)) /
      (denominator * denominator);
  derivative = derivative_c / width;
}

}  // namespace

PhysicalLongRangeWorkspace::PhysicalLongRangeWorkspace(
    std::size_t nodes, std::size_t pairs)
    : atomic_energy("so3lr_lr_atomic_energy", nodes),
      charge_gradient("so3lr_lr_charge_gradient", nodes),
      hirshfeld_gradient("so3lr_lr_hirshfeld_gradient", nodes),
      electrostatic_pair_energy("so3lr_lr_electrostatic_pair_energy", pairs),
      dispersion_pair_energy("so3lr_lr_dispersion_pair_energy", pairs),
      pair_radial_gradient("so3lr_lr_pair_radial_gradient", pairs),
      pair_force_vectors("so3lr_lr_pair_force_vectors", pairs * 3),
      atomic_forces("so3lr_lr_atomic_forces", nodes * 3) {
  if (nodes == 0 || pairs == 0)
    throw std::runtime_error("SO3LR physical LR workspace is empty");
}

std::size_t physical_long_range_workspace_bytes(std::size_t nodes,
                                                std::size_t pairs) {
  return (6 * nodes + 6 * pairs) * sizeof(double);
}

void launch_so3lr_physical_long_range_device(
    const Kokkos::View<std::int64_t *> &atomic_numbers,
    const Kokkos::View<double *> &partial_charges,
    const Kokkos::View<double *> &hirshfeld_ratios,
    const Kokkos::View<double *> &reference_alphas,
    const Kokkos::View<double *> &reference_c6,
    const Kokkos::View<double *> &pair_vectors,
    const Kokkos::View<std::size_t *> &senders,
    const Kokkos::View<std::size_t *> &receivers,
    const PhysicalLongRangeParameters &p,
    const PhysicalLongRangeWorkspace &workspace,
    const Kokkos::View<double *> &c6_ratios) {
  const std::size_t nodes = atomic_numbers.extent(0);
  const std::size_t pairs = senders.extent(0);
  if (nodes == 0 || pairs == 0 || partial_charges.extent(0) != nodes ||
      hirshfeld_ratios.extent(0) != nodes || receivers.extent(0) != pairs ||
      pair_vectors.extent(0) != pairs * 3 ||
      workspace.atomic_energy.extent(0) != nodes ||
      workspace.charge_gradient.extent(0) != nodes ||
      workspace.hirshfeld_gradient.extent(0) != nodes ||
      workspace.electrostatic_pair_energy.extent(0) != pairs ||
      workspace.dispersion_pair_energy.extent(0) != pairs ||
      workspace.pair_radial_gradient.extent(0) != pairs ||
      workspace.pair_force_vectors.extent(0) != pairs * 3 ||
      workspace.atomic_forces.extent(0) != nodes * 3 ||
      reference_alphas.extent(0) != reference_c6.extent(0) ||
      reference_alphas.extent(0) == 0 || !(p.cutoff > p.electrostatic_cuton) ||
      !(p.cutoff > p.dispersion_cuton) || !(p.electrostatic_sigma > 0.0) ||
      !(p.bohr > 0.0) || !(p.pair_scale > 0.0))
    throw std::runtime_error("SO3LR physical LR view/parameter mismatch");

  Kokkos::deep_copy(workspace.atomic_energy, 0.0);
  Kokkos::deep_copy(workspace.charge_gradient, 0.0);
  Kokkos::deep_copy(workspace.hirshfeld_gradient, 0.0);
  Kokkos::deep_copy(workspace.electrostatic_pair_energy, 0.0);
  Kokkos::deep_copy(workspace.dispersion_pair_energy, 0.0);
  Kokkos::deep_copy(workspace.pair_radial_gradient, 0.0);
  Kokkos::deep_copy(workspace.pair_force_vectors, 0.0);
  Kokkos::deep_copy(workspace.atomic_forces, 0.0);

  const auto z = atomic_numbers;
  const auto q = partial_charges;
  const auto h = hirshfeld_ratios;
  const auto alpha_ref = reference_alphas;
  const auto c6_ref = reference_c6;
  const auto vectors = pair_vectors;
  const auto sender = senders;
  const auto receiver = receivers;
  const auto atomic_energy = workspace.atomic_energy;
  const auto q_gradient = workspace.charge_gradient;
  const auto h_gradient = workspace.hirshfeld_gradient;
  const auto electrostatic_energy = workspace.electrostatic_pair_energy;
  const auto dispersion_energy = workspace.dispersion_pair_energy;
  const auto radial_gradient = workspace.pair_radial_gradient;
  const auto pair_force = workspace.pair_force_vectors;
  const auto atom_force = workspace.atomic_forces;
  // With a C6 head: alpha comes from the a0 ratio (in the Hirshfeld slot) and C6 from
  // its own ratio. v1 passes no C6 ratios and derives C6 = C6_ref h^2.
  const bool separate_c6 = c6_ratios.extent(0) != 0;
  if (separate_c6) {
    if (c6_ratios.extent(0) != nodes)
      throw std::runtime_error("SO3LR physical LR C6-ratio shape mismatch");
    if (workspace.c6_gradient.extent(0) != nodes)
      workspace.c6_gradient = Kokkos::View<double *>("so3lr_lr_c6_gradient", nodes);
    Kokkos::deep_copy(workspace.c6_gradient, 0.0);
  }
  const auto c6r = c6_ratios;
  const auto c6_gradient = workspace.c6_gradient;

  const double cutoff_ratio = p.cutoff / p.electrostatic_sigma;
  const double cutoff_erf = erf(cutoff_ratio);
  const double cutoff_exp = Kokkos::exp(-(cutoff_ratio * cutoff_ratio));
  const double electrostatic_shift = cutoff_erf / p.cutoff;
  const double electrostatic_force_shift =
      (2.0 * p.cutoff * cutoff_exp * inv_sqrt_pi /
           p.electrostatic_sigma -
       cutoff_erf) /
      (p.cutoff * p.cutoff);
  const double fine_structure_factor =
      Kokkos::pow(p.fine_structure, -4.0 / 21.0);
  constexpr double b0 = -0.00433008;
  constexpr double b1 = 0.24428889;
  constexpr double b2 = 0.04125273;
  constexpr double b3 = -0.00078893;

  Kokkos::parallel_for(
      "so3lr_native_physical_long_range", Kokkos::RangePolicy<>(0, pairs),
      KOKKOS_LAMBDA(const std::size_t pair) {
        const std::size_t i = sender(pair);
        const std::size_t j = receiver(pair);
        if (i >= nodes || j >= nodes || i == j) return;
        const double dx = vectors(pair * 3);
        const double dy = vectors(pair * 3 + 1);
        const double dz = vectors(pair * 3 + 2);
        const double raw_r = Kokkos::sqrt(dx * dx + dy * dy + dz * dz);
        if (!(raw_r > 1.0e-12) || raw_r >= p.cutoff) return;
        const double r = raw_r;

        double electrostatic_switch = 0.0;
        double electrostatic_switch_derivative = 0.0;
        smooth_switch(r, p.electrostatic_cuton, p.cutoff,
                      electrostatic_switch,
                      electrostatic_switch_derivative);
        const double ratio = r / p.electrostatic_sigma;
        const double erf_term = erf(ratio);
        const double exp_term = Kokkos::exp(-(ratio * ratio));
        const double potential = erf_term / r;
        const double potential_derivative =
            (2.0 * r * exp_term * inv_sqrt_pi /
                 p.electrostatic_sigma -
             erf_term) /
            (r * r);
        const double displacement = r - p.cutoff;
        const double shifted_potential =
            potential - electrostatic_shift -
            electrostatic_force_shift * displacement;
        const double electrostatic_kernel =
            electrostatic_switch * (potential - electrostatic_shift) +
            (1.0 - electrostatic_switch) * shifted_potential;
        const double electrostatic_radial_kernel =
            potential_derivative -
            (1.0 - electrostatic_switch) * electrostatic_force_shift +
            electrostatic_switch_derivative * electrostatic_force_shift *
                displacement;
        const double full_scale = 2.0 * p.pair_scale;
        const double electrostatic_full_energy =
            full_scale * p.ke * q(i) * q(j) * electrostatic_kernel;
        const double charge_prefactor =
            full_scale * p.ke * electrostatic_kernel;
        const double electrostatic_radial =
            full_scale * p.ke * q(i) * q(j) *
            electrostatic_radial_kernel;

        double dispersion_full_energy = 0.0;
        double dispersion_radial = 0.0;
        double dhi = 0.0;
        double dhj = 0.0;
        double dci = 0.0;
        double dcj = 0.0;
        const std::int64_t zi64 = z(i) - 1;
        const std::int64_t zj64 = z(j) - 1;
        if (zi64 >= 0 && zj64 >= 0 &&
            static_cast<std::size_t>(zi64) < alpha_ref.extent(0) &&
            static_cast<std::size_t>(zj64) < alpha_ref.extent(0)) {
          const std::size_t zi = static_cast<std::size_t>(zi64);
          const std::size_t zj = static_cast<std::size_t>(zj64);
          const double hi = h(i);
          const double hj = h(j);
          const double ai0 = alpha_ref(zi);
          const double aj0 = alpha_ref(zj);
          const double ci0 = c6_ref(zi);
          const double cj0 = c6_ref(zj);
          const double ai = ai0 * hi;
          const double aj = aj0 * hj;
          // v1 expression order is kept exactly, so v1 stays bit-identical.
          const double c6i = separate_c6 ? ci0 * c6r(i) : ci0 * hi * hi;
          const double c6j = separate_c6 ? cj0 * c6r(j) : cj0 * hj * hj;
          const double alpha = 0.5 * (ai + aj);
          const double mix_numerator = 2.0 * c6i * c6j * aj * ai;
          const double mix_denominator = ai * ai * c6j + aj * aj * c6i;
          if (alpha > 0.0 && mix_denominator > 1.0e-30) {
            const double c6 = mix_numerator / mix_denominator;
            const double inverse_mix_squared =
                1.0 / (mix_denominator * mix_denominator);
            // dc6_dh*: derivative of the mixed C6 w.r.t. the ratio in the
            // Hirshfeld slot. For v1 that ratio drives both alpha and C6; for
            // a C6-head model it drives alpha only, and dc6_dc* carries the C6 ratio.
            double dc6_dhi = 0.0, dc6_dhj = 0.0, dc6_dci = 0.0, dc6_dcj = 0.0;
            if (!separate_c6) {
              const double dc6i_dhi = 2.0 * ci0 * hi;
              const double dc6j_dhj = 2.0 * cj0 * hj;
              const double dn_dhi = 2.0 *
                  (dc6i_dhi * c6j * aj * ai + c6i * c6j * aj * ai0);
              const double dn_dhj = 2.0 *
                  (c6i * dc6j_dhj * aj * ai + c6i * c6j * aj0 * ai);
              const double dd_dhi =
                  2.0 * ai * ai0 * c6j + aj * aj * dc6i_dhi;
              const double dd_dhj =
                  ai * ai * dc6j_dhj + 2.0 * aj * aj0 * c6i;
              dc6_dhi = (dn_dhi * mix_denominator - mix_numerator * dd_dhi) *
                        inverse_mix_squared;
              dc6_dhj = (dn_dhj * mix_denominator - mix_numerator * dd_dhj) *
                        inverse_mix_squared;
            } else {
              // through alpha_i = ai0 a_i only
              const double dn_dai = 2.0 * c6i * c6j * aj * ai0;
              const double dd_dai = 2.0 * ai * ai0 * c6j;
              const double dn_daj = 2.0 * c6i * c6j * aj0 * ai;
              const double dd_daj = 2.0 * aj * aj0 * c6i;
              dc6_dhi = (dn_dai * mix_denominator - mix_numerator * dd_dai) *
                        inverse_mix_squared;
              dc6_dhj = (dn_daj * mix_denominator - mix_numerator * dd_daj) *
                        inverse_mix_squared;
              // through C6_i = ci0 c_i only
              const double dn_dci = 2.0 * ci0 * c6j * aj * ai;
              const double dd_dci = aj * aj * ci0;
              const double dn_dcj = 2.0 * c6i * cj0 * aj * ai;
              const double dd_dcj = ai * ai * cj0;
              dc6_dci = (dn_dci * mix_denominator - mix_numerator * dd_dci) *
                        inverse_mix_squared;
              dc6_dcj = (dn_dcj * mix_denominator - mix_numerator * dd_dcj) *
                        inverse_mix_squared;
            }
            const double alpha_seventh = Kokkos::pow(alpha, 1.0 / 7.0);
            const double vdw_radius =
                fine_structure_factor * alpha_seventh;
            const double sigma_fit =
                b3 * vdw_radius * vdw_radius * vdw_radius +
                b2 * vdw_radius * vdw_radius + b1 * vdw_radius + b0;
            const double gamma = 0.5 / (sigma_fit * sigma_fit);
            const double d_vdw_radius_d_alpha =
                vdw_radius / (7.0 * alpha);
            const double d_sigma_d_alpha =
                (3.0 * b3 * vdw_radius * vdw_radius +
                 2.0 * b2 * vdw_radius + b1) *
                d_vdw_radius_d_alpha;
            const double d_gamma_d_alpha =
                -2.0 * gamma * d_sigma_d_alpha / sigma_fit;
            const double c8 = 5.0 * c6 / gamma;
            const double c10 = 245.0 * c6 / (8.0 * gamma * gamma);
            const double damping_length =
                p.dispersion_scale * 2.0 * 2.54 * alpha_seventh;
            const double d_damping_d_alpha =
                damping_length / (7.0 * alpha);
            const double distance_au = r / p.bohr;
            const double r2 = distance_au * distance_au;
            const double r4 = r2 * r2;
            const double r6 = r4 * r2;
            const double r8 = r4 * r4;
            const double r10 = r8 * r2;
            const double p2 = damping_length * damping_length;
            const double p4 = p2 * p2;
            const double p6 = p4 * p2;
            const double p8 = p4 * p4;
            const double p10 = p8 * p2;
            const double d6 = r6 + p6;
            const double d8 = r8 + p8;
            const double d10 = r10 + p10;
            const double qdo_potential =
                -c6 / d6 - c8 / d8 - c10 / d10;
            const double radial_au =
                6.0 * c6 * r4 * distance_au / (d6 * d6) +
                8.0 * c8 * r6 * distance_au / (d8 * d8) +
                10.0 * c10 * r8 * distance_au / (d10 * d10);
            const double dc8_dalpha =
                -c8 * d_gamma_d_alpha / gamma;
            const double dc10_dalpha =
                -2.0 * c10 * d_gamma_d_alpha / gamma;
            const double dd6_dalpha =
                6.0 * p4 * damping_length * d_damping_d_alpha;
            const double dd8_dalpha =
                8.0 * p6 * damping_length * d_damping_d_alpha;
            const double dd10_dalpha =
                10.0 * p8 * damping_length * d_damping_d_alpha;
            const double dpotential_dalpha =
                c6 * dd6_dalpha / (d6 * d6) - dc8_dalpha / d8 +
                c8 * dd8_dalpha / (d8 * d8) - dc10_dalpha / d10 +
                c10 * dd10_dalpha / (d10 * d10);
            const double dpotential_dc6 =
                -1.0 / d6 - (5.0 / gamma) / d8 -
                (245.0 / (8.0 * gamma * gamma)) / d10;
            const double dpotential_dhi =
                dpotential_dc6 * dc6_dhi +
                dpotential_dalpha * 0.5 * ai0;
            const double dpotential_dhj =
                dpotential_dc6 * dc6_dhj +
                dpotential_dalpha * 0.5 * aj0;
            double dispersion_switch = 0.0;
            double dispersion_switch_derivative = 0.0;
            smooth_switch(r, p.dispersion_cuton, p.cutoff,
                          dispersion_switch,
                          dispersion_switch_derivative);
            dispersion_full_energy =
                full_scale * p.hartree * qdo_potential * dispersion_switch;
            dispersion_radial = full_scale * p.hartree *
                (radial_au * dispersion_switch / p.bohr +
                 qdo_potential * dispersion_switch_derivative);
            dhi = full_scale * p.hartree * dispersion_switch *
                  dpotential_dhi;
            dhj = full_scale * p.hartree * dispersion_switch *
                  dpotential_dhj;
            dci = full_scale * p.hartree * dispersion_switch *
                  dpotential_dc6 * dc6_dci;
            dcj = full_scale * p.hartree * dispersion_switch *
                  dpotential_dc6 * dc6_dcj;
          }
        }

        electrostatic_energy(pair) = electrostatic_full_energy;
        dispersion_energy(pair) = dispersion_full_energy;
        const double pair_energy =
            electrostatic_full_energy + dispersion_full_energy;
        Kokkos::atomic_add(&atomic_energy(i), 0.5 * pair_energy);
        Kokkos::atomic_add(&atomic_energy(j), 0.5 * pair_energy);
        Kokkos::atomic_add(&q_gradient(i), charge_prefactor * q(j));
        Kokkos::atomic_add(&q_gradient(j), charge_prefactor * q(i));
        Kokkos::atomic_add(&h_gradient(i), dhi);
        Kokkos::atomic_add(&h_gradient(j), dhj);
        if (separate_c6) {
          Kokkos::atomic_add(&c6_gradient(i), dci);
          Kokkos::atomic_add(&c6_gradient(j), dcj);
        }

        const double radial = electrostatic_radial + dispersion_radial;
        radial_gradient(pair) = radial;
        const double inverse_r = 1.0 / r;
        const double fx = radial * dx * inverse_r;
        const double fy = radial * dy * inverse_r;
        const double fz = radial * dz * inverse_r;
        pair_force(pair * 3) = fx;
        pair_force(pair * 3 + 1) = fy;
        pair_force(pair * 3 + 2) = fz;
        Kokkos::atomic_add(&atom_force(i * 3), fx);
        Kokkos::atomic_add(&atom_force(i * 3 + 1), fy);
        Kokkos::atomic_add(&atom_force(i * 3 + 2), fz);
        Kokkos::atomic_add(&atom_force(j * 3), -fx);
        Kokkos::atomic_add(&atom_force(j * 3 + 1), -fy);
        Kokkos::atomic_add(&atom_force(j * 3 + 2), -fz);
      });
}

}  // namespace so3lr
