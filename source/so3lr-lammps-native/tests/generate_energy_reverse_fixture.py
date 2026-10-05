#!/usr/bin/env python3
"""Generate PyTorch-autograd and finite-difference references for dE/dh."""

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
    distances = torch.linalg.vector_norm(vectors, dim=1, keepdim=True)
    sh_vectors = model.spherical_harmonics(+vectors / distances)

    with torch.no_grad():
        one_hot = torch.nn.functional.one_hot(
            atomic_numbers - 1, num_classes=118
        ).to(torch.float64)
        inv = model.inv_feature_embedding(one_hot)
        ev = torch.zeros((atomic_numbers.numel(), 24), dtype=torch.float64)
        rbf = model.radial_embedding(distances)
        cutoffs = model.cutoff_fn(distances)
        for block in model.euclidean_transformers:
            inv, ev = block(
                inv, ev, rbf, senders, receivers, sh_vectors, cutoffs
            )

    final_inv = inv.detach().requires_grad_(True)
    data = {"node_attrs": one_hot}
    atomic_energies = model.atomic_energy_output_block(
        final_inv, data, atomic_numbers=atomic_numbers
    ).squeeze(-1)
    total_energy = atomic_energies.sum()
    gradient = torch.autograd.grad(total_energy, final_inv)[0]

    flat_indices = [0, 37, 127, 128 + 11, 3 * 128 + 64, 5 * 128 + 127]
    epsilon = 1.0e-5
    finite_difference: list[float] = []
    with torch.no_grad():
        base = final_inv.detach()
        for index in flat_indices:
            plus = base.clone().reshape(-1)
            minus = base.clone().reshape(-1)
            plus[index] += epsilon
            minus[index] -= epsilon
            plus_energy = model.atomic_energy_output_block(
                plus.reshape_as(base), data, atomic_numbers=atomic_numbers
            ).sum()
            minus_energy = model.atomic_energy_output_block(
                minus.reshape_as(base), data, atomic_numbers=atomic_numbers
            ).sum()
            finite_difference.append(
                float((plus_energy - minus_energy) / (2.0 * epsilon))
            )
    analytic_selected = [float(gradient.reshape(-1)[i]) for i in flat_indices]
    maximum_fd_error = max(
        abs(left - right)
        for left, right in zip(finite_difference, analytic_selected)
    )
    if maximum_fd_error > 2.0e-7:
        raise RuntimeError(
            f"energy-head finite difference failed: {maximum_fd_error}"
        )

    fixture = {
        "schema": "so3lr-native-energy-head-reverse-fixture-v1",
        "geometry": "physical_water_dimer_nonperiodic",
        "nodes": atomic_numbers.numel(),
        "atomic_numbers": atomic_numbers.tolist(),
        "final_inv": flat(final_inv),
        "atomic_energies": flat(atomic_energies),
        "total_energy": float(total_energy),
        "energy_input_gradient": flat(gradient),
        "finite_difference_epsilon": epsilon,
        "finite_difference_indices": flat_indices,
        "finite_difference_values": finite_difference,
        "autograd_values_at_finite_difference_indices": analytic_selected,
        "finite_difference_max_abs_error": maximum_fd_error,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(fixture, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(f"fixture={args.output}")
    print("geometry=physical_water_dimer_nonperiodic")
    print("reverse_target=learned_energy_to_final_invariant_features")
    print(f"nodes={atomic_numbers.numel()}")
    print(f"total_energy={float(total_energy):.17g}")
    print(f"finite_difference_max_abs_error={maximum_fd_error:.17g}")
    print("pytorch_energy_head_reverse_fixture=PASS")


if __name__ == "__main__":
    main()
