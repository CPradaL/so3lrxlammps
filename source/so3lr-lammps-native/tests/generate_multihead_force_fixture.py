#!/usr/bin/env python3
"""Generate independent PyTorch multi-head reverse and Cartesian-force oracles."""

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
        raise RuntimeError("expected three EuclideanTransformer blocks")
    if list(model.spherical_harmonics.degrees) != [1, 2, 3, 4]:
        raise RuntimeError("unexpected spherical-harmonic contract")

    positions0 = torch.tensor(
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
    sender_values: list[int] = []
    receiver_values: list[int] = []
    for sender in range(positions0.shape[0]):
        for receiver in range(positions0.shape[0]):
            if sender == receiver:
                continue
            vector = positions0[receiver] - positions0[sender]
            if float(torch.linalg.vector_norm(vector)) < 4.5:
                sender_values.append(sender)
                receiver_values.append(receiver)
    senders = torch.tensor(sender_values, dtype=torch.long)
    receivers = torch.tensor(receiver_values, dtype=torch.long)
    nodes = int(atomic_numbers.numel())
    edges = int(senders.numel())
    if edges != nodes * (nodes - 1):
        raise RuntimeError("water-dimer graph is not complete and directed")

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
        [0.31, -0.17, 0.43, -0.29, 0.23, -0.11], dtype=torch.float64
    )
    charge_seeds = torch.tensor(
        [-0.37, 0.19, 0.41, -0.13, 0.07, -0.29], dtype=torch.float64
    )
    hirshfeld_seeds = torch.tensor(
        [0.27, -0.33, 0.16, 0.45, -0.21, 0.12], dtype=torch.float64
    )
    centered_charge_seeds = charge_seeds - charge_seeds.mean()
    if abs(float(charge_seeds.mean())) < 1.0e-4:
        raise RuntimeError("charge seed accidentally bypasses conservation VJP")

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
        hirshfeld = model.hirshfeld_output_block(inv, atomic_numbers).reshape(-1)
        return energies, charges, hirshfeld

    def objectives(inv: torch.Tensor) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor,
    ]:
        energies, charges, hirshfeld = heads(inv)
        e_obj = torch.dot(energy_seeds, energies)
        q_obj = torch.dot(charge_seeds, charges)
        h_obj = torch.dot(hirshfeld_seeds, hirshfeld)
        return e_obj, q_obj, h_obj, e_obj + q_obj + h_obj, energies, charges, hirshfeld

    final_inv = embedding.detach().clone()
    base_vectors = positions0[receivers] - positions0[senders]
    base_distances = torch.linalg.vector_norm(base_vectors, dim=1, keepdim=True)
    base_sh = model.spherical_harmonics(+base_vectors / base_distances)
    radial = model.radial_embedding(base_distances)
    cutoff = model.cutoff_fn(base_distances)
    with torch.no_grad():
        final_ev = initial_ev
        for block in model.euclidean_transformers:
            final_inv, final_ev = block(
                final_inv, final_ev, radial, senders, receivers, base_sh,
                cutoff, return_att=False,
            )

    final_leaf = final_inv.detach().clone().requires_grad_(True)
    e_obj, q_obj, h_obj, combined_obj, energies, charges, hirshfeld = objectives(
        final_leaf
    )
    energy_input_grad = torch.autograd.grad(
        e_obj, final_leaf, retain_graph=True
    )[0]
    charge_input_grad = torch.autograd.grad(
        q_obj, final_leaf, retain_graph=True
    )[0]
    hirshfeld_input_grad = torch.autograd.grad(
        h_obj, final_leaf, retain_graph=True
    )[0]
    combined_input_grad = torch.autograd.grad(combined_obj, final_leaf)[0]
    component_sum_error = float(
        (
            combined_input_grad
            - energy_input_grad
            - charge_input_grad
            - hirshfeld_input_grad
        ).abs().max()
    )
    if component_sum_error > 2.0e-12:
        raise RuntimeError(f"head-gradient sum failed: {component_sum_error}")
    if abs(float(charges.sum())) > 2.0e-12:
        raise RuntimeError("partial-charge conservation failed")

    def complete_objective(
        edge_vectors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        distances = torch.linalg.vector_norm(edge_vectors, dim=1, keepdim=True)
        sh = model.spherical_harmonics(+edge_vectors / distances)
        inv = embedding
        ev = initial_ev
        radial_values = model.radial_embedding(distances)
        cutoff_values = model.cutoff_fn(distances)
        for block in model.euclidean_transformers:
            inv, ev = block(
                inv, ev, radial_values, senders, receivers, sh, cutoff_values,
                return_att=False,
            )
        result = objectives(inv)[3]
        return result, distances, sh, inv

    edge_vectors = base_vectors.detach().clone().requires_grad_(True)
    objective, distances, sh_vectors, _ = complete_objective(edge_vectors)
    edge_gradients = torch.autograd.grad(objective, edge_vectors)[0]
    atomic_forces = scatter_forces(edge_gradients, senders, receivers, nodes)

    positions = positions0.detach().clone().requires_grad_(True)
    position_vectors = positions[receivers] - positions[senders]
    position_objective = complete_objective(position_vectors)[0]
    position_forces = -torch.autograd.grad(position_objective, positions)[0]
    force_route_error = float((atomic_forces - position_forces).abs().max())
    if force_route_error > 2.0e-12:
        raise RuntimeError(f"edge-to-position force route failed: {force_route_error}")

    selected_coordinates = [0, 1, 5, 7, 9, 11, 13, 16, 17]
    epsilon = 1.0e-5
    maximum_scaled_error = 0.0
    finite_difference: list[dict[str, float | int]] = []
    with torch.no_grad():
        for index in selected_coordinates:
            plus = positions0.clone().reshape(-1)
            minus = positions0.clone().reshape(-1)
            plus[index] += epsilon
            minus[index] -= epsilon
            plus_positions = plus.reshape_as(positions0)
            minus_positions = minus.reshape_as(positions0)
            plus_value = complete_objective(
                plus_positions[receivers] - plus_positions[senders]
            )[0]
            minus_value = complete_objective(
                minus_positions[receivers] - minus_positions[senders]
            )[0]
            numerical = float((plus_value - minus_value) / (2.0 * epsilon))
            analytic = float(-position_forces.reshape(-1)[index])
            scaled_error = abs(numerical - analytic) / (1.0 + abs(analytic))
            maximum_scaled_error = max(maximum_scaled_error, scaled_error)
            finite_difference.append(
                {
                    "position_flat_index": index,
                    "numerical_objective_gradient": numerical,
                    "negative_force": analytic,
                    "scaled_error": scaled_error,
                }
            )
    if maximum_scaled_error > 2.0e-5:
        raise RuntimeError(
            f"multihead Cartesian-force finite difference failed: {maximum_scaled_error}"
        )

    net_force = position_forces.sum(dim=0)
    torque = torch.linalg.cross(positions0, position_forces).sum(dim=0)
    maximum_net_force = float(net_force.abs().max())
    maximum_net_torque = float(torque.abs().max())
    if maximum_net_force > 2.0e-10 or maximum_net_torque > 2.0e-10:
        raise RuntimeError("multihead translation/rotation invariance failed")

    tensors = {
        "positions": positions0,
        "edge_vectors": base_vectors,
        "distances": distances,
        "sh_vectors": sh_vectors,
        "atomic_energies": energies,
        "partial_charges": charges,
        "hirshfeld_ratios": hirshfeld,
        "energy_seeds": energy_seeds,
        "partial_charge_seeds": charge_seeds,
        "centered_partial_charge_seeds": centered_charge_seeds,
        "hirshfeld_seeds": hirshfeld_seeds,
        "energy_input_grad": energy_input_grad,
        "charge_input_grad": charge_input_grad,
        "hirshfeld_input_grad": hirshfeld_input_grad,
        "combined_input_grad": combined_input_grad,
        "combined_edge_gradients": edge_gradients,
        "combined_atomic_forces": position_forces,
    }
    for label, tensor in tensors.items():
        if not bool(torch.isfinite(tensor).all()):
            raise RuntimeError(f"nonfinite fixture tensor: {label}")

    fixture: dict[str, object] = {
        "schema": "so3lr-native-multihead-cartesian-force-fixture-v1",
        "geometry": "physical_water_dimer_nonperiodic",
        "objective": "seeded_energy_plus_partial_charge_plus_hirshfeld",
        "vector_convention": "receiver_minus_sender",
        "spherical_harmonic_argument": "negative_normalized_edge_vector",
        "nodes": nodes,
        "edges": edges,
        "atomic_numbers": atomic_numbers.tolist(),
        "senders": senders.tolist(),
        "receivers": receivers.tolist(),
        "energy_objective": float(e_obj),
        "charge_objective": float(q_obj),
        "hirshfeld_objective": float(h_obj),
        "combined_objective": float(combined_obj),
        "charge_seed_mean": float(charge_seeds.mean()),
        "component_gradient_sum_max_abs_error": component_sum_error,
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
    print("objective=seeded_energy_plus_partial_charge_plus_hirshfeld")
    print(f"nodes={nodes}")
    print(f"edges={edges}")
    print(f"charge_seed_mean={float(charge_seeds.mean()):.17g}")
    print(f"component_gradient_sum_max_abs_error={component_sum_error:.17g}")
    print(f"edge_to_position_force_max_abs_error={force_route_error:.17g}")
    print(f"maximum_abs_net_force={maximum_net_force:.17g}")
    print(f"maximum_abs_net_torque={maximum_net_torque:.17g}")
    print(f"finite_difference_max_scaled_error={maximum_scaled_error:.17g}")
    print("pytorch_charge_conservation_vjp=PASS")
    print("pytorch_multihead_cartesian_force_reference=PASS")
    print("translation_rotation_invariance=PASS")


if __name__ == "__main__":
    main()
