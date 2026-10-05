#ifdef PAIR_CLASS
// clang-format off
PairStyle(so3lr/turbo,PairSO3LRTurbo);
// clang-format on
#else

#ifndef LMP_PAIR_SO3LR_TURBO_H
#define LMP_PAIR_SO3LR_TURBO_H

#include "pair.h"

#include "so3lr/mpi_turbo_replica_broker.hpp"
#include "so3lr/native_model.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace LAMMPS_NS {

class PairSO3LRTurbo : public Pair {
 public:
  explicit PairSO3LRTurbo(class LAMMPS *);
  ~PairSO3LRTurbo() override;

  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  void init_list(int, class NeighList *) override;
  double init_one(int, int) override;

 private:
  std::string model_path_;
  double short_range_cutoff_ = 0.0;
  // This image's total charge and spin multiplicity (pair_style args 2, 3).
  double total_charge_ = 0.0;
  double multiplicity_ = 1.0;
  double long_range_cutoff_ = 0.0;
  std::vector<std::int64_t> type_to_atomic_number_;
  class NeighList *short_range_list_ = nullptr;
  class NeighList *long_range_list_ = nullptr;
  std::unique_ptr<so3lr::NativeModel> model_;
  std::unique_ptr<so3lr::MpiTurboReplicaBroker> broker_;
  bool initialized_kokkos_here_ = false;
  bool reported_ = false;
  std::size_t evaluation_count_ = 0;
  int universe_rank_ = -1;
  int universe_size_ = 0;

  void allocate();
};

}  // namespace LAMMPS_NS

#endif
#endif
