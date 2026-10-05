#include "so3lr/kokkos_physical_long_range_model.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace so3lr {
namespace {
template <class Reader>
void fill_device(const Kokkos::View<double *> &device, std::size_t count,
                 Reader reader) {
  auto host = Kokkos::create_mirror_view(device);
  for (std::size_t i = 0; i < count; ++i) host(i) = reader(i);
  Kokkos::deep_copy(device, host);
}

void positive(double value, const char *name) {
  if (!std::isfinite(value) || value <= 0.0)
    throw std::runtime_error(std::string("SO3LR invalid LR parameter ") + name);
}
}  // namespace

KokkosPhysicalLongRangeModel::KokkosPhysicalLongRangeModel(
    const NativeModel &model) {
  if (model.manifest().at("schema").string() != "so3lr-native-model-v2")
    throw std::runtime_error("SO3LR physical model requires model v2");
  const auto &alpha = model.tensor("physical.reference_alphas");
  const auto &c6 = model.tensor("physical.reference_c6");
  if (alpha.dtype != "float64" || c6.dtype != "float64" ||
      alpha.shape.size() != 1 || c6.shape != alpha.shape || alpha.shape[0] < 8)
    throw std::runtime_error("SO3LR physical reference-table contract changed");
  const std::size_t table_size = alpha.shape[0];
  reference_alphas_ = DoubleView("so3lr_lr_model_alphas", table_size);
  reference_c6_ = DoubleView("so3lr_lr_model_c6", table_size);
  fill_device(reference_alphas_, table_size,
              [&](std::size_t i) { return model.float64(alpha, i); });
  fill_device(reference_c6_, table_size,
              [&](std::size_t i) { return model.float64(c6, i); });

  parameters_.ke = model.architecture_number("lr_electrostatic_ke");
  parameters_.electrostatic_sigma =
      model.architecture_number("lr_electrostatic_sigma");
  parameters_.cutoff =
      model.architecture_number("long_range_cutoff_angstrom");
  parameters_.electrostatic_cuton =
      model.architecture_number("lr_electrostatic_cuton_angstrom");
  parameters_.fine_structure =
      model.architecture_number("lr_fine_structure");
  parameters_.bohr = model.architecture_number("lr_bohr_angstrom");
  parameters_.hartree = model.architecture_number("lr_hartree_ev");
  parameters_.dispersion_scale =
      model.architecture_number("lr_dispersion_scale");
  parameters_.dispersion_cuton =
      model.architecture_number("lr_dispersion_cuton_angstrom");
  parameters_.pair_scale = model.architecture_number("lr_pair_scale");
  positive(parameters_.ke, "ke");
  positive(parameters_.electrostatic_sigma, "sigma");
  positive(parameters_.cutoff, "cutoff");
  positive(parameters_.fine_structure, "fine_structure");
  positive(parameters_.bohr, "bohr");
  positive(parameters_.hartree, "hartree");
  positive(parameters_.dispersion_scale, "dispersion_scale");
  positive(parameters_.pair_scale, "pair_scale");
  if (!(parameters_.cutoff > parameters_.electrostatic_cuton) ||
      !(parameters_.electrostatic_cuton > 0.0) ||
      !(parameters_.cutoff > parameters_.dispersion_cuton) ||
      !(parameters_.dispersion_cuton > 0.0) ||
      std::abs(parameters_.pair_scale - 0.5) > 1.0e-12)
    throw std::runtime_error("SO3LR LR cutoff/half-pair contract changed");
  Kokkos::fence();
}

void KokkosPhysicalLongRangeModel::launch_device(
    const Int64View &atomic_numbers, const DoubleView &partial_charges,
    const DoubleView &hirshfeld_ratios, const DoubleView &pair_vectors,
    const IndexView &senders, const IndexView &receivers,
    const PhysicalLongRangeWorkspace &workspace,
    const DoubleView &c6_ratios) const {
  launch_so3lr_physical_long_range_device(
      atomic_numbers, partial_charges, hirshfeld_ratios, reference_alphas_,
      reference_c6_, pair_vectors, senders, receivers, parameters_, workspace,
      c6_ratios);
}

}  // namespace so3lr
