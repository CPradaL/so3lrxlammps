// Prints the charge/spin embedding offset table of one system, for comparison
// with the reference implementation (CPU only, no GPU needed).
//
//   so3lr_charge_spin_table MODEL CHARGE MULTIPLICITY Z1 [Z2 ...]
//
// One line per element present: "Z v_0 v_1 ... v_{F-1}" (before the
// embedding scale), or the reason a (charge, multiplicity) is invalid.
#include "so3lr/native_model.hpp"
#include "so3lr/so3lr_charge_spin.hpp"

#include <cstdio>
#include <cstdlib>
#include <exception>

int main(int argc, char **argv) {
  if (argc < 5) {
    std::fprintf(stderr, "usage: %s MODEL CHARGE MULTIPLICITY Z1 [Z2 ...]\n", argv[0]);
    return 2;
  }
  try {
    const auto model = so3lr::NativeModel::load(argv[1]);
    const double charge = std::atof(argv[2]);
    const double multiplicity = std::atof(argv[3]);
    so3lr::ElementCounts counts{};
    for (int i = 4; i < argc; ++i) ++counts[static_cast<std::size_t>(std::atoi(argv[i]))];
    const auto problem = so3lr::charge_multiplicity_problem(counts, charge, multiplicity);
    if (!problem.empty()) {
      std::printf("INVALID %s\n", problem.c_str());
      return 3;
    }
    const auto e = so3lr::load_charge_spin_embedding(model);
    const auto table = so3lr::charge_spin_offset_table(e, counts, charge, multiplicity - 1.0);
    for (std::size_t z = 1; z <= 118; ++z) {
      if (counts[z] == 0) continue;
      std::printf("%zu", z);
      for (std::size_t c = 0; c < e.width; ++c) std::printf(" %.17g", table[(z - 1) * e.width + c]);
      std::printf("\n");
    }
  } catch (const std::exception &exc) {
    std::fprintf(stderr, "ERROR %s\n", exc.what());
    return 1;
  }
  return 0;
}
