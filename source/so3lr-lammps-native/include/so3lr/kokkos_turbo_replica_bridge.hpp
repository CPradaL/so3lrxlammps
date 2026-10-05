#pragma once

#include "so3lr/kokkos_packed_so3lr_evaluator.hpp"
#include "so3lr/so3lr_charge_spin.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace so3lr {

// Host-side graph contract exposed to independent simulation images.  Indices
// are local to one replica.  The bridge alone applies packed global offsets.
struct TurboReplicaGraph {
  std::vector<std::int64_t> atomic_numbers;
  std::vector<double> sr_edge_vectors;
  std::vector<std::size_t> sr_senders, sr_receivers;
  std::vector<double> lr_pair_vectors;
  std::vector<std::size_t> lr_senders, lr_receivers;
  double total_charge = 0.0;
  double unpaired_electrons = 0.0;  // spin multiplicity - 1
};

struct TurboReplicaResult {
  std::vector<double> atomic_energies;
  std::vector<double> partial_charges;
  std::vector<double> hirshfeld_ratios;
  std::vector<double> atomic_forces;
  double total_energy = 0.0;
};

struct TurboReplicaBatchResult {
  std::vector<TurboReplicaResult> replicas;
  std::size_t packed_nodes = 0;
  std::size_t packed_sr_edges = 0;
  std::size_t packed_lr_pairs = 0;
};

// A correctness-first LAMMPS-facing boundary.  Every caller may build its own
// graph and retain its own integrator/thermostat state.  The bridge performs
// one packed device evaluation and scatters the outputs without mixing images.
class KokkosTurboReplicaBridge {
 public:
  explicit KokkosTurboReplicaBridge(const NativeModel &model);

  TurboReplicaBatchResult evaluate(
      const std::vector<TurboReplicaGraph> &replicas) const;

  bool contract_verified() const { return evaluator_.contract_verified(); }

 private:
  KokkosPackedSo3lrEvaluator evaluator_;
  ChargeSpinEmbedding charge_spin_;
};

}  // namespace so3lr
