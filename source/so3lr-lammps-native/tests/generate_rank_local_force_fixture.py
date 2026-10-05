#!/usr/bin/env python3
"""Generate a sparse PyTorch force oracle for true owned-plus-ghost ranks."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


def discover_root(wrapper: object) -> torch.nn.Module:
    roots: list[torch.nn.Module] = []
    seen = {id(wrapper)}

    def visit(value: object) -> None:
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, torch.nn.Module):
            roots.append(value)
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    for value in vars(wrapper).values():
        visit(value)
    if len(roots) != 1:
        raise RuntimeError(f"expected one root module, found {len(roots)}")
    return roots[0]


def find_so3lr(root: torch.nn.Module) -> torch.nn.Module:
    for module in root.modules():
        if type(module).__name__ == "SO3LR":
            return module
    raise RuntimeError("checkpoint does not contain SO3LR")


def flat(value: torch.Tensor) -> list[float]:
    return value.detach().cpu().contiguous().reshape(-1).tolist()


def scatter_forces(
    edge_gradients: torch.Tensor,
    senders: torch.Tensor,
    receivers: torch.Tensor,
    nodes: int,
) -> torch.Tensor:
    forces = torch.zeros((nodes, 3), dtype=edge_gradients.dtype)
    forces.index_add_(0, senders, edge_gradients)
    forces.index_add_(0, receivers, -edge_gradients)
    return forces


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    wrapper = torch.load(args.model, map_location="cpu", weights_only=False)
    model = find_so3lr(discover_root(wrapper)).eval()
    if list(model.spherical_harmonics.degrees) != [1, 2, 3, 4]:
        raise RuntimeError("unexpected SO3LR spherical-harmonic basis")
    if len(model.euclidean_transformers) != 3:
        raise RuntimeError("expected exactly three transformer blocks")

    # A slightly curved ten-atom chain. The 1.8 A spacing makes only first and
    # second neighbours fall inside 4.5 A. Splitting after atom 4 therefore
    # gives each rank five owned atoms and only two ghosts, rather than all ten
    # global rows. The non-collinear perturbation exercises all force axes.
    positions0 = torch.tensor(
        [
            [1.8 * i, 0.31 * math.sin(0.73 * i),
             0.23 * math.cos(0.51 * i)]
            for i in range(10)
        ],
        dtype=torch.float64,
    )
    atomic_numbers = torch.tensor(
        [8, 1, 1, 8, 1, 8, 1, 1, 8, 1], dtype=torch.long
    )
    node_owner = torch.tensor([0] * 5 + [1] * 5, dtype=torch.long)
    senders_list: list[int] = []
    receivers_list: list[int] = []
    for sender in range(positions0.shape[0]):
        for receiver in range(positions0.shape[0]):
            if sender == receiver:
                continue
            distance = torch.linalg.vector_norm(
                positions0[receiver] - positions0[sender]
            )
            if float(distance) < 4.5:
                senders_list.append(sender)
                receivers_list.append(receiver)
    senders = torch.tensor(senders_list, dtype=torch.long)
    receivers = torch.tensor(receivers_list, dtype=torch.long)
    nodes = int(atomic_numbers.numel())
    edges = int(senders.numel())
    if edges != 34:
        raise RuntimeError(f"unexpected sparse fixture edge count: {edges}")

    one_hot = torch.nn.functional.one_hot(
        atomic_numbers - 1, num_classes=118
    ).to(torch.float64)
    with torch.no_grad():
        embedded = model.inv_feature_embedding(one_hot)
    data = {"node_attrs": one_hot}

    def learned_energy(
        edge_vectors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        distances = torch.linalg.vector_norm(edge_vectors, dim=1, keepdim=True)
        sh_vectors = model.spherical_harmonics(+edge_vectors / distances)
        inv = embedded
        ev = torch.zeros((nodes, 24), dtype=torch.float64)
        radial = model.radial_embedding(distances)
        cutoffs = model.cutoff_fn(distances)
        for block in model.euclidean_transformers:
            inv, ev = block(
                inv, ev, radial, senders, receivers, sh_vectors, cutoffs,
                return_att=False,
            )
        atomic_energies = model.atomic_energy_output_block(
            inv, data, atomic_numbers=atomic_numbers
        ).squeeze(-1)
        return atomic_energies.sum(), atomic_energies

    base_vectors = positions0[receivers] - positions0[senders]
    edge_vectors = base_vectors.detach().clone().requires_grad_(True)
    total_energy, atomic_energies = learned_energy(edge_vectors)
    edge_gradients = torch.autograd.grad(total_energy, edge_vectors)[0]
    scattered_forces = scatter_forces(
        edge_gradients, senders, receivers, nodes
    )

    positions = positions0.detach().clone().requires_grad_(True)
    position_energy, _ = learned_energy(
        positions[receivers] - positions[senders]
    )
    position_forces = -torch.autograd.grad(position_energy, positions)[0]
    route_error = float((position_forces - scattered_forces).abs().max())
    if route_error > 2.0e-12:
        raise RuntimeError(f"edge-to-position force route failed: {route_error}")

    epsilon = 1.0e-5
    selected = [0, 2, 7, 11, 14, 16, 21, 25, 29]
    finite_difference: list[dict[str, float | int]] = []
    maximum_scaled_error = 0.0
    with torch.no_grad():
        for index in selected:
            plus = positions0.clone().reshape(-1)
            minus = positions0.clone().reshape(-1)
            plus[index] += epsilon
            minus[index] -= epsilon
            plus = plus.reshape_as(positions0)
            minus = minus.reshape_as(positions0)
            plus_energy = learned_energy(
                plus[receivers] - plus[senders]
            )[0]
            minus_energy = learned_energy(
                minus[receivers] - minus[senders]
            )[0]
            numerical = float((plus_energy - minus_energy) / (2 * epsilon))
            analytic = float(-position_forces.reshape(-1)[index])
            scaled = abs(numerical - analytic) / (1.0 + abs(analytic))
            maximum_scaled_error = max(maximum_scaled_error, scaled)
            finite_difference.append(
                {
                    "position_flat_index": index,
                    "numerical_energy_gradient": numerical,
                    "negative_force": analytic,
                    "scaled_error": scaled,
                }
            )
    if maximum_scaled_error > 2.0e-5:
        raise RuntimeError(
            f"Cartesian finite difference failed: {maximum_scaled_error}"
        )

    net_force = position_forces.sum(dim=0)
    torque = torch.linalg.cross(positions0, position_forces).sum(dim=0)
    maximum_net_force = float(net_force.abs().max())
    maximum_net_torque = float(torque.abs().max())
    if maximum_net_force > 2.0e-10 or maximum_net_torque > 2.0e-10:
        raise RuntimeError("sparse fixture invariance failed")

    rank_local_nodes: list[list[int]] = []
    rank_local_edges: list[int] = []
    for rank in [0, 1]:
        owned = {i for i, owner in enumerate(node_owner.tolist()) if owner == rank}
        selected_edges = [
            edge for edge, receiver in enumerate(receivers.tolist())
            if receiver in owned
        ]
        ghosts = sorted(
            {
                int(senders[edge]) for edge in selected_edges
                if int(senders[edge]) not in owned
            }
        )
        rank_local_nodes.append(sorted(owned) + ghosts)
        rank_local_edges.append(len(selected_edges))
    if rank_local_nodes != [[0, 1, 2, 3, 4, 5, 6],
                            [5, 6, 7, 8, 9, 3, 4]]:
        raise RuntimeError(f"unexpected owned-plus-ghost tables: {rank_local_nodes}")

    fixture: dict[str, object] = {
        "schema": "so3lr-native-rank-local-cartesian-force-fixture-v1",
        "geometry": "sparse_nonperiodic_curved_chain",
        "vector_convention": "receiver_minus_sender",
        "nodes": nodes,
        "edges": edges,
        "atomic_numbers": atomic_numbers.tolist(),
        "node_owner": node_owner.tolist(),
        "senders": senders.tolist(),
        "receivers": receivers.tolist(),
        "positions": flat(positions0),
        "edge_vectors": flat(base_vectors),
        "atomic_energies": flat(atomic_energies),
        "edge_energy_gradients": flat(edge_gradients),
        "atomic_forces": flat(position_forces),
        "total_energy": float(total_energy),
        "rank_local_nodes": rank_local_nodes,
        "rank_local_edges": rank_local_edges,
        "edge_to_position_force_max_abs_error": route_error,
        "maximum_abs_net_force": maximum_net_force,
        "maximum_abs_net_torque": maximum_net_torque,
        "finite_difference_epsilon": epsilon,
        "finite_difference_checks": finite_difference,
        "finite_difference_max_scaled_error": maximum_scaled_error,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(fixture, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(f"fixture={args.output}")
    print("geometry=sparse_nonperiodic_curved_chain")
    print(f"nodes={nodes}")
    print(f"edges={edges}")
    print("owned_nodes_per_rank=5")
    print("ghost_nodes_per_rank=2")
    print("local_nodes_per_rank=7")
    print("global_node_row_replication_factor=1.4")
    print(f"rank_local_edges={rank_local_edges[0]},{rank_local_edges[1]}")
    print(f"total_energy={float(total_energy):.17g}")
    print(f"edge_to_position_force_max_abs_error={route_error:.17g}")
    print(f"maximum_abs_net_force={maximum_net_force:.17g}")
    print(f"maximum_abs_net_torque={maximum_net_torque:.17g}")
    print(f"finite_difference_max_scaled_error={maximum_scaled_error:.17g}")
    print("pytorch_sparse_rank_local_cartesian_force_reference=PASS")
    print("translation_rotation_invariance=PASS")


if __name__ == "__main__":
    main()
