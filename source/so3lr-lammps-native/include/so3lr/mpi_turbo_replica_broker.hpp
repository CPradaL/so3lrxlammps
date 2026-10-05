#pragma once

#include "so3lr/kokkos_turbo_replica_bridge.hpp"

#include <mpi.h>

#include <memory>

namespace so3lr {

// Cross-process replica broker. Every MPI rank owns one independent simulation
// image. Only the coordinator owns the CUDA evaluator; workers participate in
// gather/scatter collectives and never instantiate a model or Kokkos object.
class MpiTurboReplicaBroker {
 public:
  MpiTurboReplicaBroker(MPI_Comm communicator, int coordinator,
                        const NativeModel *coordinator_model);

  TurboReplicaResult evaluate(const TurboReplicaGraph &local_graph);

  int rank() const { return rank_; }
  int size() const { return size_; }
  int coordinator() const { return coordinator_; }
  bool is_coordinator() const { return rank_ == coordinator_; }

  // Valid on the coordinator after evaluate(); empty on workers.
  const TurboReplicaBatchResult &coordinator_batch() const {
    return coordinator_batch_;
  }

 private:
  MPI_Comm communicator_ = MPI_COMM_NULL;
  int coordinator_ = 0;
  int rank_ = -1;
  int size_ = 0;
  std::unique_ptr<KokkosTurboReplicaBridge> bridge_;
  TurboReplicaBatchResult coordinator_batch_;
};

}  // namespace so3lr
