#include "so3lr/so3lr_repulsion_setup.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace so3lr {

PhysicalZblParameters load_zbl_parameters(const NativeModel &model) {
  PhysicalZblParameters result;
  if (model.manifest().at("architecture").at("zbl_cutoff_function").string() !=
      "phys")
    throw std::runtime_error("SO3LR native repulsion cutoff contract changed");
  result.ke = model.architecture_number("zbl_ke");
  result.cutoff = model.architecture_number("short_range_cutoff_angstrom");
  result.switch_off = model.architecture_number("zbl_switch_off_angstrom");
  if (model.arch().repulsion == RepulsionKind::NlhTable) {
    // Parameter-free NLH: the pair tables are loaded separately.
    result.kind = 1;
    result.nlh_z_capacity = model.tensor("physical.nlh_a").shape.at(0);
    return result;
  }
  if (model.arch().repulsion != RepulsionKind::LearnedZbl ||
      model.architecture_number("zbl_enabled") != 1.0)
    throw std::runtime_error("SO3LR native ZBL contract changed");
  result.p = model.architecture_number("zbl_p");
  result.d = model.architecture_number("zbl_d");
  for (int k = 0; k < 4; ++k) {
    result.a[k] = model.architecture_number("zbl_a" + std::to_string(k + 1));
    result.c[k] = model.architecture_number("zbl_c" + std::to_string(k + 1));
  }
  return result;
}

So3lrRepulsion load_so3lr_repulsion(const NativeModel &model) {
  So3lrRepulsion result;
  result.parameters = load_zbl_parameters(model);
  if (result.parameters.kind != 1) return result;
  const std::size_t cap = result.parameters.nlh_z_capacity;
  const auto load_table = [&](const char *key) {
    const auto &tensor = model.tensor(key);
    if (tensor.dtype != "float64" ||
        tensor.shape != std::vector<std::size_t>({cap, cap, 3}))
      throw std::runtime_error(
          std::string("SO3LR NLH table has an unexpected shape: ") + key);
    Kokkos::View<double *> device(std::string(key), cap * cap * 3);
    auto host = Kokkos::create_mirror_view(device);
    for (std::size_t i = 0; i < cap * cap * 3; ++i)
      host(i) = model.float64(tensor, i);
    Kokkos::deep_copy(device, host);
    return device;
  };
  result.nlh_a = load_table("physical.nlh_a");
  result.nlh_b = load_table("physical.nlh_b");
  return result;
}

}  // namespace so3lr
