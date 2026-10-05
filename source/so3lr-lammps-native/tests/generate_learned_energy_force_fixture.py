#!/usr/bin/env python3
"""Generate full learned-energy Cartesian-force and geometry-VJP oracles."""

from __future__ import annotations

import argparse
import json
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


def scatter_atomic_forces(
    edge_energy_gradients: torch.Tensor,
    senders: torch.Tensor,
    receivers: torch.Tensor,
    nodes: int,
) -> torch.Tensor:
    forces = torch.zeros((nodes, 3), dtype=edge_energy_gradients.dtype)
    forces.index_add_(0, senders, edge_energy_gradients)
    forces.index_add_(0, receivers, -edge_energy_gradients)
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
        raise RuntimeError("expected exactly three EuclideanTransformer blocks")

    reference_positions = torch.tensor(
        [
            [0.000000, 0.000000, 0.000000],
            [0.957200, 0.000000, 0.000000],
            [-0.239987, 0.927297, 0.000000],
            [2.800000, 0.200000, 0.100000],
            [3.650000, 0.550000, 0.050000],
            [2.180000, 0.920000, 0.180000],
        ],
        dtype=torch.float64,
    )
    atomic_numbers = torch.tensor([8, 1, 1, 8, 1, 1], dtype=torch.long)
    senders_list: list[int] = []
    receivers_list: list[int] = []
    with torch.no_grad():
        for sender in range(reference_positions.shape[0]):
            for receiver in range(reference_positions.shape[0]):
                if sender == receiver:
                    continue
                vector = reference_positions[receiver] - reference_positions[sender]
                if float(torch.linalg.vector_norm(vector)) < 4.5:
                    senders_list.append(sender)
                    receivers_list.append(receiver)
    senders = torch.tensor(senders_list, dtype=torch.long)
    receivers = torch.tensor(receivers_list, dtype=torch.long)
    nodes = int(atomic_numbers.numel())
    edges = int(senders.numel())
    if edges != nodes * (nodes - 1):
        raise RuntimeError("water-dimer fixture is not a complete directed graph")

    one_hot = torch.nn.functional.one_hot(
        atomic_numbers - 1, num_classes=118
    ).to(torch.float64)
    with torch.no_grad():
        embedded = model.inv_feature_embedding(one_hot)
    data = {"node_attrs": one_hot}

    def learned_energy(
        edge_vectors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        distances = torch.linalg.vector_norm(edge_vectors, dim=1, keepdim=True)
        sh_vectors = model.spherical_harmonics(+edge_vectors / distances)
        inv = embedded
        ev = torch.zeros((nodes, 24), dtype=torch.float64)
        radial = model.radial_embedding(distances)
        cutoffs = model.cutoff_fn(distances)
        for block in model.euclidean_transformers:
            inv, ev = block(
                inv,
                ev,
                radial,
                senders,
                receivers,
                sh_vectors,
                cutoffs,
                return_att=False,
            )
        atomic_energies = model.atomic_energy_output_block(
            inv, data, atomic_numbers=atomic_numbers
        ).squeeze(-1)
        return atomic_energies.sum(), atomic_energies, distances, sh_vectors

    base_vectors = (
        reference_positions[receivers] - reference_positions[senders]
    )
    edge_vectors = base_vectors.detach().clone().requires_grad_(True)
    total_energy, atomic_energies, distances, sh_vectors = learned_energy(
        edge_vectors
    )
    edge_gradients = torch.autograd.grad(total_energy, edge_vectors)[0]
    scattered_forces = scatter_atomic_forces(
        edge_gradients, senders, receivers, nodes
    )

    positions = reference_positions.detach().clone().requires_grad_(True)
    position_vectors = positions[receivers] - positions[senders]
    position_energy, _, _, _ = learned_energy(position_vectors)
    position_forces = -torch.autograd.grad(position_energy, positions)[0]
    force_route_error = float((position_forces - scattered_forces).abs().max())
    if force_route_error > 2.0e-12:
        raise RuntimeError(f"edge-to-atom force route failed: {force_route_error}")

    probe_vectors = base_vectors.detach().clone().requires_grad_(True)
    probe_distances = torch.linalg.vector_norm(probe_vectors, dim=1, keepdim=True)
    probe_sh_values = model.spherical_harmonics(+probe_vectors / probe_distances)
    edge_index = torch.arange(edges, dtype=torch.float64)[:, None]
    channel = torch.arange(24, dtype=torch.float64)[None, :]
    probe_grad_distances = (
        0.17 * torch.sin((edge_index + 0.4) / 3.1)
        - 0.09 * torch.cos((edge_index + 1.3) / 4.7)
    )
    probe_grad_sh = (
        0.13 * torch.sin((edge_index + 0.8) * (channel + 0.6) / 17.0)
        + 0.07 * torch.cos((edge_index + 1.1) * (channel + 0.3) / 11.0)
    )
    probe_objective = (
        probe_distances * probe_grad_distances
    ).sum() + (probe_sh_values * probe_grad_sh).sum()
    probe_edge_gradients = torch.autograd.grad(probe_objective, probe_vectors)[0]
    probe_atomic_forces = scatter_atomic_forces(
        probe_edge_gradients, senders, receivers, nodes
    )

    selected_coordinates = [0, 1, 5, 7, 9, 11, 13, 16, 17]
    epsilon = 1.0e-5
    finite_difference: list[dict[str, float | int]] = []
    maximum_scaled_error = 0.0
    with torch.no_grad():
        for index in selected_coordinates:
            plus = reference_positions.clone().reshape(-1)
            minus = reference_positions.clone().reshape(-1)
            plus[index] += epsilon
            minus[index] -= epsilon
            plus_vectors = plus.reshape_as(reference_positions)[receivers] - \
                plus.reshape_as(reference_positions)[senders]
            minus_vectors = minus.reshape_as(reference_positions)[receivers] - \
                minus.reshape_as(reference_positions)[senders]
            plus_energy = learned_energy(plus_vectors)[0]
            minus_energy = learned_energy(minus_vectors)[0]
            numerical_gradient = float(
                (plus_energy - minus_energy) / (2.0 * epsilon)
            )
            analytic_gradient = float(-position_forces.reshape(-1)[index])
            scaled_error = abs(numerical_gradient - analytic_gradient) / (
                1.0 + abs(analytic_gradient)
            )
            maximum_scaled_error = max(maximum_scaled_error, scaled_error)
            finite_difference.append(
                {
                    "position_flat_index": index,
                    "numerical_energy_gradient": numerical_gradient,
                    "negative_force": analytic_gradient,
                    "scaled_error": scaled_error,
                }
            )
    if maximum_scaled_error > 2.0e-5:
        raise RuntimeError(
            f"Cartesian-force finite difference failed: {maximum_scaled_error}"
        )

    net_force = position_forces.sum(dim=0)
    torque = torch.linalg.cross(reference_positions, position_forces).sum(dim=0)
    maximum_net_force = float(net_force.abs().max())
    maximum_net_torque = float(torque.abs().max())
    if maximum_net_force > 2.0e-10:
        raise RuntimeError(f"translation invariance failed: {maximum_net_force}")
    if maximum_net_torque > 2.0e-10:
        raise RuntimeError(f"rotation invariance failed: {maximum_net_torque}")

    tensors = {
        "positions": reference_positions,
        "edge_vectors": base_vectors,
        "distances": distances,
        "sh_vectors": sh_vectors,
        "atomic_energies": atomic_energies,
        "edge_energy_gradients": edge_gradients,
        "atomic_forces": position_forces,
        "probe_grad_distances": probe_grad_distances,
        "probe_grad_sh": probe_grad_sh,
        "probe_edge_energy_gradients": probe_edge_gradients,
        "probe_atomic_forces": probe_atomic_forces,
    }
    for label, value in tensors.items():
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"nonfinite fixture tensor: {label}")

    fixture: dict[str, object] = {
        "schema": "so3lr-native-learned-energy-cartesian-force-fixture-v1",
        "geometry": "physical_water_dimer_nonperiodic",
        "vector_convention": "receiver_minus_sender",
        "spherical_harmonic_argument": "negative_normalized_edge_vector",
        "spherical_harmonic_degrees": [1, 2, 3, 4],
        "nodes": nodes,
        "edges": edges,
        "atomic_numbers": atomic_numbers.tolist(),
        "senders": senders.tolist(),
        "receivers": receivers.tolist(),
        "total_energy": float(total_energy),
        "edge_to_position_force_max_abs_error": force_route_error,
        "maximum_abs_net_force": maximum_net_force,
        "maximum_abs_net_torque": maximum_net_torque,
        "finite_difference_epsilon": epsilon,
        "finite_difference_checks": finite_difference,
        "finite_difference_max_scaled_error": maximum_scaled_error,
    }
    fixture.update({name: flat(value) for name, value in tensors.items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(fixture, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(f"fixture={args.output}")
    print("geometry=physical_water_dimer_nonperiodic")
    print("vector_convention=receiver_minus_sender")
    print("spherical_harmonic_degrees=1,2,3,4")
    print(f"nodes={nodes}")
    print(f"edges={edges}")
    print(f"total_energy={float(total_energy):.17g}")
    print(f"edge_to_position_force_max_abs_error={force_route_error:.17g}")
    print(f"maximum_abs_net_force={maximum_net_force:.17g}")
    print(f"maximum_abs_net_torque={maximum_net_torque:.17g}")
    print(f"finite_difference_max_scaled_error={maximum_scaled_error:.17g}")
    print("pytorch_geometry_vjp_probe=PASS")
    print("pytorch_learned_energy_cartesian_force_reference=PASS")
    print("translation_rotation_invariance=PASS")


if __name__ == "__main__":
    main()
