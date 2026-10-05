#!/usr/bin/env python3
"""Generate a sparse distributed energy/charge/Hirshfeld force oracle."""

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
    if len(model.euclidean_transformers) != 3:
        raise RuntimeError("expected three transformer blocks")

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
    sender_values: list[int] = []
    receiver_values: list[int] = []
    for sender in range(10):
        for receiver in range(10):
            if sender == receiver:
                continue
            if float(torch.linalg.vector_norm(
                positions0[receiver] - positions0[sender]
            )) < 4.5:
                sender_values.append(sender)
                receiver_values.append(receiver)
    senders = torch.tensor(sender_values, dtype=torch.long)
    receivers = torch.tensor(receiver_values, dtype=torch.long)
    nodes = int(atomic_numbers.numel())
    edges = int(senders.numel())
    if edges != 34:
        raise RuntimeError(f"unexpected sparse edge count: {edges}")

    one_hot = torch.nn.functional.one_hot(
        atomic_numbers - 1, num_classes=118
    ).to(torch.float64)
    with torch.no_grad():
        embedding = model.inv_feature_embedding(one_hot)
    initial_ev = torch.zeros((nodes, 24), dtype=torch.float64)
    batch = torch.zeros(nodes, dtype=torch.long)
    total_charge = torch.tensor([0.0], dtype=torch.float64)
    data = {"node_attrs": one_hot}

    energy_seeds = torch.tensor(
        [0.31, -0.17, 0.43, -0.29, 0.23,
         -0.11, 0.37, -0.21, 0.19, -0.07], dtype=torch.float64
    )
    charge_seeds = torch.tensor(
        [-0.37, 0.19, 0.41, -0.13, 0.07,
         -0.29, 0.33, -0.23, 0.17, 0.09], dtype=torch.float64
    )
    hirshfeld_seeds = torch.tensor(
        [0.27, -0.33, 0.16, 0.45, -0.21,
         0.12, -0.18, 0.29, -0.25, 0.38], dtype=torch.float64
    )
    charge_seed_mean = float(charge_seeds.mean())
    if abs(charge_seed_mean) < 1.0e-4:
        raise RuntimeError("charge seed accidentally bypasses conservation")

    def heads(inv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        energies = model.atomic_energy_output_block(
            inv, data, atomic_numbers=atomic_numbers
        ).squeeze(-1)
        charges = model.partial_charges_output_block(
            inv_features=inv,
            atomic_numbers=atomic_numbers,
            total_charge=total_charge,
            batch_segments=batch,
            num_graphs=1,
        ).reshape(-1)
        hirshfeld = model.hirshfeld_output_block(
            inv, atomic_numbers
        ).reshape(-1)
        return energies, charges, hirshfeld

    def objective_and_outputs(
        edge_vectors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        distances = torch.linalg.vector_norm(edge_vectors, dim=1, keepdim=True)
        sh = model.spherical_harmonics(+edge_vectors / distances)
        inv = embedding
        ev = initial_ev
        radial = model.radial_embedding(distances)
        cutoff = model.cutoff_fn(distances)
        for block in model.euclidean_transformers:
            inv, ev = block(
                inv, ev, radial, senders, receivers, sh, cutoff,
                return_att=False,
            )
        energies, charges, hirshfeld = heads(inv)
        objective = (
            torch.dot(energy_seeds, energies)
            + torch.dot(charge_seeds, charges)
            + torch.dot(hirshfeld_seeds, hirshfeld)
        )
        return objective, energies, charges, hirshfeld

    base_vectors = positions0[receivers] - positions0[senders]
    edge_vectors = base_vectors.detach().clone().requires_grad_(True)
    objective, energies, charges, hirshfeld = objective_and_outputs(
        edge_vectors
    )
    edge_gradients = torch.autograd.grad(objective, edge_vectors)[0]
    scattered_forces = scatter_forces(
        edge_gradients, senders, receivers, nodes
    )
    if abs(float(charges.sum())) > 2.0e-12:
        raise RuntimeError("global partial-charge conservation failed")

    positions = positions0.detach().clone().requires_grad_(True)
    position_objective = objective_and_outputs(
        positions[receivers] - positions[senders]
    )[0]
    position_forces = -torch.autograd.grad(position_objective, positions)[0]
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
            plus_value = objective_and_outputs(
                plus[receivers] - plus[senders]
            )[0]
            minus_value = objective_and_outputs(
                minus[receivers] - minus[senders]
            )[0]
            numerical = float((plus_value - minus_value) / (2 * epsilon))
            analytic = float(-position_forces.reshape(-1)[index])
            scaled = abs(numerical - analytic) / (1.0 + abs(analytic))
            maximum_scaled_error = max(maximum_scaled_error, scaled)
            finite_difference.append(
                {
                    "position_flat_index": index,
                    "numerical_objective_gradient": numerical,
                    "negative_force": analytic,
                    "scaled_error": scaled,
                }
            )
    if maximum_scaled_error > 2.0e-5:
        raise RuntimeError(
            f"multihead finite difference failed: {maximum_scaled_error}"
        )

    net_force = position_forces.sum(dim=0)
    torque = torch.linalg.cross(positions0, position_forces).sum(dim=0)
    maximum_net_force = float(net_force.abs().max())
    maximum_net_torque = float(torque.abs().max())
    if maximum_net_force > 2.0e-10 or maximum_net_torque > 2.0e-10:
        raise RuntimeError("multihead translation/rotation invariance failed")

    fixture: dict[str, object] = {
        "schema": "so3lr-native-rank-local-multihead-force-fixture-v1",
        "geometry": "sparse_nonperiodic_curved_chain",
        "objective": "seeded_energy_plus_partial_charge_plus_hirshfeld",
        "vector_convention": "receiver_minus_sender",
        "nodes": nodes,
        "edges": edges,
        "atomic_numbers": atomic_numbers.tolist(),
        "node_owner": node_owner.tolist(),
        "senders": senders.tolist(),
        "receivers": receivers.tolist(),
        "positions": flat(positions0),
        "edge_vectors": flat(base_vectors),
        "atomic_energies": flat(energies),
        "partial_charges": flat(charges),
        "hirshfeld_ratios": flat(hirshfeld),
        "energy_seeds": flat(energy_seeds),
        "partial_charge_seeds": flat(charge_seeds),
        "hirshfeld_seeds": flat(hirshfeld_seeds),
        "charge_seed_mean": charge_seed_mean,
        "combined_objective": float(objective),
        "combined_edge_gradients": flat(edge_gradients),
        "combined_atomic_forces": flat(position_forces),
        "edge_to_position_force_max_abs_error": route_error,
        "maximum_abs_net_force": maximum_net_force,
        "maximum_abs_net_torque": maximum_net_torque,
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
    print(f"combined_objective={float(objective):.17g}")
    print(f"charge_sum={float(charges.sum()):.17g}")
    print(f"charge_seed_mean={charge_seed_mean:.17g}")
    print(f"edge_to_position_force_max_abs_error={route_error:.17g}")
    print(f"maximum_abs_net_force={maximum_net_force:.17g}")
    print(f"maximum_abs_net_torque={maximum_net_torque:.17g}")
    print(f"finite_difference_max_scaled_error={maximum_scaled_error:.17g}")
    print("pytorch_distributed_multihead_cartesian_force_reference=PASS")
    print("global_charge_conservation_reference=PASS")
    print("translation_rotation_invariance=PASS")


if __name__ == "__main__":
    main()
