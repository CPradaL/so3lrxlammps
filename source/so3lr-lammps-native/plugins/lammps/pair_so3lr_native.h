#ifdef PAIR_CLASS
// clang-format off
PairStyle(so3lr/native,PairSO3LRNative);
// clang-format on
#else

#ifndef LMP_PAIR_SO3LR_NATIVE_H
#define LMP_PAIR_SO3LR_NATIVE_H

#include "pair.h"

#include "so3lr/kokkos_so3lr_evaluator.hpp"
#include "so3lr/native_model.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace LAMMPS_NS {

class PairSO3LRNative : public Pair {
 public:
  explicit PairSO3LRNative(class LAMMPS *);
  ~PairSO3LRNative() override;

  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  void init_list(int, class NeighList *) override;
  double init_one(int, int) override;

 private:
  std::string model_path_;
  double short_range_cutoff_ = 0.0;
  double long_range_cutoff_ = 0.0;
  std::vector<std::int64_t> type_to_atomic_number_;
  class NeighList *short_range_list_ = nullptr;
  class NeighList *long_range_list_ = nullptr;
  std::unique_ptr<so3lr::NativeModel> model_;
  std::unique_ptr<so3lr::KokkosSo3lrEvaluator> evaluator_;
  bool initialized_kokkos_here_ = false;
  bool reported_ = false;
  std::size_t evaluation_count_ = 0;

  void allocate();
};

}  // namespace LAMMPS_NS

#endif
#endif
