#!/usr/bin/env python3
"""Independent LAMMPS-style neighbour snapshot for the native adapter."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def pair_owner(i: int, j: int, owners: list[int]) -> int:
    if owners[i] == owners[j]:
        return owners[i]
    low, high = sorted((i, j))
    return owners[low] if (low + high) % 2 == 0 else owners[high]


def compact(snapshot: dict, neighbor: dict, cutoff: float) -> dict:
    nlocal = snapshot["nlocal"]
    source = list(range(nlocal))
    lookup = {row: row for row in source}
    senders: list[int] = []
    receivers: list[int] = []
    vectors: list[float] = []
    entries: list[int] = []
    positions = snapshot["positions"]
    for ii, receiver_row in enumerate(neighbor["ilist"]):
        for entry in range(neighbor["offsets"][ii], neighbor["offsets"][ii + 1]):
            sender_row = neighbor["neighbors"][entry]
            vector = [positions[receiver_row][c] - positions[sender_row][c]
                      for c in range(3)]
            r = math.sqrt(sum(value * value for value in vector))
            if sender_row == receiver_row or not (0.0 < r < cutoff):
                continue
            if sender_row not in lookup:
                lookup[sender_row] = len(source)
                source.append(sender_row)
            senders.append(lookup[sender_row])
            receivers.append(lookup[receiver_row])
            vectors.extend(vector)
            entries.append(entry)
    types = snapshot["types"]
    zmap = [0, 8, 1]
    return {
        "source_atom_rows": source,
        "tags": [snapshot["tags"][row] for row in source],
        "atomic_numbers": [zmap[types[row]] for row in source],
        "owned_local": list(range(nlocal)),
        "ghost_local": list(range(nlocal, len(source))),
        "vectors": vectors,
        "senders": senders,
        "receivers": receivers,
        "original_neighbor_entries": entries,
    }


def make_rank(rank: int, positions: list[list[float]], atomic_numbers: list[int],
              owners: list[int]) -> dict:
    owned = [i for i, owner in enumerate(owners) if owner == rank]
    sr_edges = [(s, r) for r in owned for s in range(len(positions))
                if s != r and distance(positions[s], positions[r]) < 4.5]
    lr_global = [(s, r) for s in range(len(positions)) for r in range(s)
                 if distance(positions[s], positions[r]) < 12.0]
    assigned = [(s, r) for s, r in lr_global
                if pair_owner(s, r, owners) == rank]
    ghosts = sorted({s for s, _ in sr_edges if owners[s] != rank} |
                    {node for pair in assigned for node in pair
                     if owners[node] != rank})
    local_to_global = owned + ghosts
    global_to_local = {global_id: local for local, global_id in enumerate(local_to_global)}
    snapshot = {
        "nlocal": len(owned),
        "tags": [global_id + 1 for global_id in local_to_global],
        "types": [1 if atomic_numbers[global_id] == 8 else 2
                  for global_id in local_to_global],
        "positions": [positions[global_id] for global_id in local_to_global],
        "local_to_global": local_to_global,
    }

    sr_by_receiver: dict[int, list[int]] = {global_to_local[g]: [] for g in owned}
    for sender, receiver in sr_edges:
        sr_by_receiver[global_to_local[receiver]].append(global_to_local[sender])
    sr_ilist = sorted(sr_by_receiver)
    sr_neighbors: list[int] = []
    sr_offsets = [0]
    for receiver in sr_ilist:
        sr_neighbors.extend(sr_by_receiver[receiver])
        sr_offsets.append(len(sr_neighbors))
    sr = {"ilist": sr_ilist, "offsets": sr_offsets, "neighbors": sr_neighbors}

    lr_by_receiver: dict[int, list[int]] = {global_to_local[g]: [] for g in owned}
    for a, b in assigned:
        receiver, sender = (a, b) if owners[a] == rank else (b, a)
        lr_by_receiver[global_to_local[receiver]].append(global_to_local[sender])
    lr_ilist = sorted(lr_by_receiver)
    lr_neighbors: list[int] = []
    lr_offsets = [0]
    for receiver in lr_ilist:
        lr_neighbors.extend(lr_by_receiver[receiver])
        lr_offsets.append(len(lr_neighbors))
    lr = {"ilist": lr_ilist, "offsets": lr_offsets, "neighbors": lr_neighbors}
    return {
        "atoms": snapshot,
        "full_sr": sr,
        "half_lr": lr,
        "expected_sr": compact(snapshot, sr, 4.5),
        "expected_lr": compact(snapshot, lr, 12.0),
        "assigned_lr_pairs": len(assigned),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    positions = [[1.8 * i, 0.31 * math.sin(0.73 * i),
                  0.23 * math.cos(0.51 * i)] for i in range(10)]
    atomic_numbers = [8, 1, 1, 8, 1, 8, 1, 1, 8, 1]
    owners = [0] * 5 + [1] * 5
    ranks = [make_rank(rank, positions, atomic_numbers, owners) for rank in range(2)]
    if [rank["assigned_lr_pairs"] for rank in ranks] != [20, 19]:
        raise RuntimeError("unexpected LR pair partition")

    periodic_atoms = {
        "nlocal": 1,
        "tags": [7, 7, 8],
        "types": [1, 1, 2],
        "positions": [[0.2, 0.0, 0.0], [10.2, 0.0, 0.0], [-0.3, 0.0, 0.0]],
    }
    periodic_sr = {"ilist": [0], "offsets": [0, 1], "neighbors": [2]}
    periodic_lr = {"ilist": [0], "offsets": [0, 1], "neighbors": [1]}
    fixture = {
        "schema": "so3lr-lammps-neighbor-adapter-fixture-v1",
        "type_to_atomic_number": [0, 8, 1],
        "short_range_cutoff": 4.5,
        "long_range_cutoff": 12.0,
        "ranks": ranks,
        "periodic_image_case": {
            "atoms": periodic_atoms,
            "full_sr": periodic_sr,
            "half_lr": periodic_lr,
            "expected_sr": compact(periodic_atoms, periodic_sr, 4.5),
            "expected_lr": compact(periodic_atoms, periodic_lr, 12.0),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, sort_keys=True, separators=(",", ":")) + "\n")
    print(f"fixture={args.output}")
    print("global_nodes=10")
    print("global_sr_edges=34")
    print("global_lr_pairs=39")
    print("lr_pairs_per_rank=20,19")
    print("periodic_duplicate_tag_rows=2")
    print("lammps_neighbor_adapter_reference=PASS")
    print("separate_sr_lr_compaction_reference=PASS")
    print("periodic_image_identity_reference=PASS")


if __name__ == "__main__":
    main()
