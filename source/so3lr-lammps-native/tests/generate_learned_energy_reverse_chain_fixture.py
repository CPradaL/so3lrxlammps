#!/usr/bin/env python3
"""Generate the full learned-energy reverse-chain oracle for dev_22."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    wrapper = torch.load(args.model, map_location="cpu", weights_only=False)
    model = find_so3lr(discover_root(wrapper)).eval()
    if len(model.euclidean_transformers) != 3:
        raise RuntimeError("expected exactly three EuclideanTransformer blocks")

    positions = torch.tensor(
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
    vectors_list: list[torch.Tensor] = []
    for sender in range(positions.shape[0]):
        for receiver in range(positions.shape[0]):
            if sender == receiver:
                continue
            vector = positions[receiver] - positions[sender]
            if float(torch.linalg.vector_norm(vector)) < 4.5:
                senders_list.append(sender)
                receivers_list.append(receiver)
                vectors_list.append(vector)
    senders = torch.tensor(senders_list, dtype=torch.long)
    receivers = torch.tensor(receivers_list, dtype=torch.long)
    vectors = torch.stack(vectors_list)
    base_distances = torch.linalg.vector_norm(vectors, dim=1, keepdim=True)
    base_sh = model.spherical_harmonics(+vectors / base_distances)
    nodes = int(atomic_numbers.numel())
    edges = int(senders.numel())
    if edges != nodes * (nodes - 1):
        raise RuntimeError("water-dimer fixture is not a complete directed graph")

    one_hot = torch.nn.functional.one_hot(
        atomic_numbers - 1, num_classes=118
    ).to(torch.float64)
    with torch.no_grad():
        embedded = model.inv_feature_embedding(one_hot)
    initial_inv = embedded.detach().clone().requires_grad_(True)
    initial_ev = torch.zeros((nodes, 24), dtype=torch.float64).requires_grad_(True)
    # Separate equal-valued leaves expose each block's direct geometric VJP.
    # Their sum is the gradient for the single shared production geometry.
    distances = [
        base_distances.detach().clone().requires_grad_(True) for _ in range(3)
    ]
    sh_vectors = [base_sh.detach().clone().requires_grad_(True) for _ in range(3)]

    inv, ev = initial_inv, initial_ev
    for index, block in enumerate(model.euclidean_transformers):
        rbf = model.radial_embedding(distances[index])
        cutoffs = model.cutoff_fn(distances[index])
        inv, ev = block(
            inv,
            ev,
            rbf,
            senders,
            receivers,
            sh_vectors[index],
            cutoffs,
            return_att=False,
        )
    data = {"node_attrs": one_hot}
    atomic_energies = model.atomic_energy_output_block(
        inv, data, atomic_numbers=atomic_numbers
    ).squeeze(-1)
    total_energy = atomic_energies.sum()
    inputs = [initial_inv, initial_ev, *distances, *sh_vectors]
    gradients = torch.autograd.grad(total_energy, inputs)
    grad_inv, grad_ev = gradients[0], gradients[1]
    grad_distance_blocks = list(gradients[2:5])
    grad_sh_blocks = list(gradients[5:8])
    grad_distances = torch.stack(grad_distance_blocks).sum(dim=0)
    grad_sh = torch.stack(grad_sh_blocks).sum(dim=0)

    values = {
        "atomic_energies": atomic_energies,
        "grad_embedding": grad_inv,
        "grad_initial_ev": grad_ev,
        "grad_distances": grad_distances,
        "grad_sh_vectors": grad_sh,
    }
    for block in range(3):
        values[f"grad_distances_block{block}"] = grad_distance_blocks[block]
        values[f"grad_sh_vectors_block{block}"] = grad_sh_blocks[block]
    for label, value in values.items():
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"nonfinite reference tensor: {label}")
    for block in range(3):
        if float(grad_distance_blocks[block].abs().max()) <= 1.0e-12:
            raise RuntimeError(f"block {block} has no direct distance sensitivity")
        if float(grad_sh_blocks[block].abs().max()) <= 1.0e-12:
            raise RuntimeError(f"block {block} has no direct SH sensitivity")

    def shared_objective(
        inv_input: torch.Tensor,
        ev_input: torch.Tensor,
        distance_input: torch.Tensor,
        sh_input: torch.Tensor,
    ) -> torch.Tensor:
        inv_value, ev_value = inv_input, ev_input
        radial = model.radial_embedding(distance_input)
        cutoff = model.cutoff_fn(distance_input)
        for block in model.euclidean_transformers:
            inv_value, ev_value = block(
                inv_value,
                ev_value,
                radial,
                senders,
                receivers,
                sh_input,
                cutoff,
                return_att=False,
            )
        return model.atomic_energy_output_block(
            inv_value, data, atomic_numbers=atomic_numbers
        ).sum()

    shared_inputs = [
        initial_inv.detach(),
        initial_ev.detach(),
        base_distances.detach(),
        base_sh.detach(),
    ]
    shared_gradients = [grad_inv, grad_ev, grad_distances, grad_sh]
    names = ["embedding", "initial_ev", "distances", "sh_vectors"]
    selected = [
        (0, 17),
        (0, 4 * 128 + 93),
        (1, 2 * 24 + 9),
        (1, 5 * 24 + 21),
        (2, 0),
        (2, 14),
        (2, 29),
        (3, 3 * 24 + 7),
        (3, 24 * 24 + 19),
    ]
    epsilon = 1.0e-5
    checks: list[dict[str, object]] = []
    maximum_scaled_error = 0.0
    with torch.no_grad():
        for which, index in selected:
            plus = [value.clone() for value in shared_inputs]
            minus = [value.clone() for value in shared_inputs]
            plus[which].reshape(-1)[index] += epsilon
            minus[which].reshape(-1)[index] -= epsilon
            numerical = float(
                (shared_objective(*plus) - shared_objective(*minus))
                / (2.0 * epsilon)
            )
            analytic = float(shared_gradients[which].reshape(-1)[index])
            scaled_error = abs(numerical - analytic) / (1.0 + abs(analytic))
            maximum_scaled_error = max(maximum_scaled_error, scaled_error)
            checks.append(
                {
                    "input": names[which],
                    "flat_index": index,
                    "numerical": numerical,
                    "autograd": analytic,
                    "scaled_error": scaled_error,
                }
            )
    if maximum_scaled_error > 2.0e-5:
        raise RuntimeError(
            f"learned-energy reverse finite difference failed: {maximum_scaled_error}"
        )

    fixture: dict[str, object] = {
        "schema": "so3lr-native-learned-energy-reverse-chain-fixture-v1",
        "geometry": "physical_water_dimer_nonperiodic",
        "reverse_scope": "energy_head_through_transformer_blocks_2_1_0",
        "nodes": nodes,
        "edges": edges,
        "atomic_numbers": atomic_numbers.tolist(),
        "senders": senders.tolist(),
        "receivers": receivers.tolist(),
        "distances": flat(base_distances),
        "sh_vectors": flat(base_sh),
        "total_energy": float(total_energy),
        "finite_difference_epsilon": epsilon,
        "finite_difference_checks": checks,
        "finite_difference_max_scaled_error": maximum_scaled_error,
    }
    fixture.update({name: flat(value) for name, value in values.items()})
    for block in range(3):
        fixture[f"maximum_abs_distance_gradient_block{block}"] = float(
            grad_distance_blocks[block].abs().max()
        )
        fixture[f"maximum_abs_sh_gradient_block{block}"] = float(
            grad_sh_blocks[block].abs().max()
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(fixture, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(f"fixture={args.output}")
    print("geometry=physical_water_dimer_nonperiodic")
    print("reverse_scope=energy_head_through_transformer_blocks_2_1_0")
    print(f"nodes={nodes}")
    print(f"edges={edges}")
    print(f"total_energy={float(total_energy):.17g}")
    for block in range(3):
        print(
            f"maximum_abs_distance_gradient_block{block}="
            f"{float(grad_distance_blocks[block].abs().max()):.17g}"
        )
        print(
            f"maximum_abs_sh_gradient_block{block}="
            f"{float(grad_sh_blocks[block].abs().max()):.17g}"
        )
    print(f"finite_difference_max_scaled_error={maximum_scaled_error:.17g}")
    print("pytorch_learned_energy_reverse_chain_reference=PASS")
    print("per_block_geometry_contributions=PASS")


if __name__ == "__main__":
    main()
