#pragma once

#include "pair.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace LAMMPS_NS {

class PairSO3LRAdapter : public Pair {
 public:
  explicit PairSO3LRAdapter(class LAMMPS *);
  ~PairSO3LRAdapter() override;

  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  void init_list(int, class NeighList *) override;
  double init_one(int, int) override;

 private:
  double short_range_cutoff_ = 0.0;
  double long_range_cutoff_ = 0.0;
  std::vector<std::int64_t> type_to_atomic_number_;
  class NeighList *short_range_list_ = nullptr;
  class NeighList *long_range_list_ = nullptr;
  std::size_t calls_ = 0;
  std::uint64_t previous_owned_layout_ = 0;
  std::uint64_t previous_sr_list_topology_ = 0;
  std::uint64_t previous_lr_list_topology_ = 0;
  std::uint64_t previous_sr_active_topology_ = 0;
  std::uint64_t previous_lr_active_topology_ = 0;

  void allocate();
};

}  // namespace LAMMPS_NS
