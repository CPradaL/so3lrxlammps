#!/usr/bin/env python3
"""Generate block-2 post-attention VJP references with PyTorch autograd."""

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
    block = model.euclidean_transformers[2]
    interaction = block.interaction_block
    if interaction.degree_repeats.detach().cpu().tolist() != [3, 5, 7, 9]:
        raise RuntimeError("unexpected block-2 interaction degree contract")

    nodes = 7
    node = torch.arange(nodes, dtype=torch.float64)[:, None]
    inv_channel = torch.arange(128, dtype=torch.float64)[None, :]
    ev_channel = torch.arange(24, dtype=torch.float64)[None, :]
    initial = [
        0.31 * torch.sin((node + 0.7) * (inv_channel + 0.5) / 29.0)
        + 0.12 * torch.cos((node + 0.2) * (inv_channel + 1.0) / 17.0),
        0.16 * torch.sin((node + 0.4) * (ev_channel + 0.75) / 8.0)
        - 0.09 * torch.cos((node + 1.1) * (ev_channel + 1.0) / 6.0),
        0.055 * torch.cos((node + 0.8) * (inv_channel + 1.5) / 21.0)
        - 0.025 * torch.sin((node + 1.3) * (inv_channel + 0.25) / 15.0),
        0.043 * torch.cos((node + 0.3) * (ev_channel + 1.25) / 7.0)
        + 0.019 * torch.sin((node + 1.2) * (ev_channel + 0.5) / 5.0),
    ]
    inputs = [value.detach().requires_grad_(True) for value in initial]
    atomic_numbers = torch.tensor([8, 1, 1, 8, 1, 1, 8], dtype=torch.long)
    one_hot = torch.nn.functional.one_hot(
        atomic_numbers - 1, num_classes=118
    ).to(torch.float64)

    def post(values: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        inv_features, ev_features, d_att_inv, d_att_ev = values
        att_inv = inv_features + d_att_inv
        att_ev = ev_features + d_att_ev
        norm1 = block.layer_norm_inv_1(att_inv)
        post_mlp = norm1 + block.mlp_1(norm1)
        ev_invariants = interaction.so3_conv_invariants(att_ev)
        transformed = interaction.linear_layer(
            torch.concatenate([post_mlp, ev_invariants], dim=-1)
        )
        d_inv = transformed[:, :128]
        gates = torch.repeat_interleave(
            transformed[:, 128:],
            interaction.degree_repeats,
            dim=-1,
            output_size=24,
        )
        final_inv = block.layer_norm_inv_2(post_mlp + d_inv)
        final_ev = att_ev * (1.0 + gates)
        return final_inv, final_ev

    final_inv, final_ev = post(inputs)
    data = {"node_attrs": one_hot}
    atomic_energies = model.atomic_energy_output_block(
        final_inv, data, atomic_numbers=atomic_numbers
    ).squeeze(-1)
    total_energy = atomic_energies.sum()
    energy_final_inv_gradient = torch.autograd.grad(
        total_energy, final_inv, retain_graph=True
    )[0]
    energy_input_gradients = torch.autograd.grad(
        total_energy, inputs, retain_graph=True
    )

    generic_inv_seed = (
        0.013 * torch.sin((node + 0.6) * (inv_channel + 0.4) / 23.0)
        - 0.007 * torch.cos((node + 1.4) * (inv_channel + 0.9) / 13.0)
    )
    generic_ev_seed = (
        0.017 * torch.cos((node + 0.5) * (ev_channel + 0.8) / 7.0)
        + 0.009 * torch.sin((node + 1.1) * (ev_channel + 0.3) / 4.0)
    )
    generic_scalar = (
        (final_inv * generic_inv_seed).sum()
        + (final_ev * generic_ev_seed).sum()
    )
    generic_input_gradients = torch.autograd.grad(generic_scalar, inputs)

    tensors_to_check = {
        "final_inv": final_inv,
        "final_ev": final_ev,
        "energy_final_inv_gradient": energy_final_inv_gradient,
    }
    for index, gradient in enumerate(energy_input_gradients):
        tensors_to_check[f"energy_input_gradient_{index}"] = gradient
    for index, gradient in enumerate(generic_input_gradients):
        tensors_to_check[f"generic_input_gradient_{index}"] = gradient
    for label, value in tensors_to_check.items():
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"nonfinite PyTorch reference: {label}")

    input_names = ["inv_features", "ev_features", "d_att_inv", "d_att_ev"]
    selected = [
        (0, 0),
        (0, 5 * 128 + 127),
        (1, 3 * 24 + 11),
        (2, 2 * 128 + 64),
        (3, 6 * 24 + 23),
    ]
    epsilon = 1.0e-5
    finite_difference: list[dict[str, object]] = []
    maximum_scaled_error = 0.0
    with torch.no_grad():
        for which, index in selected:
            plus = [value.detach().clone() for value in inputs]
            minus = [value.detach().clone() for value in inputs]
            plus[which].reshape(-1)[index] += epsilon
            minus[which].reshape(-1)[index] -= epsilon
            plus_inv, _ = post(plus)
            minus_inv, _ = post(minus)
            plus_energy = model.atomic_energy_output_block(
                plus_inv, data, atomic_numbers=atomic_numbers
            ).sum()
            minus_energy = model.atomic_energy_output_block(
                minus_inv, data, atomic_numbers=atomic_numbers
            ).sum()
            numerical = float((plus_energy - minus_energy) / (2.0 * epsilon))
            analytic = float(energy_input_gradients[which].reshape(-1)[index])
            scaled_error = abs(numerical - analytic) / (1.0 + abs(analytic))
            maximum_scaled_error = max(maximum_scaled_error, scaled_error)
            finite_difference.append(
                {
                    "input": input_names[which],
                    "flat_index": index,
                    "numerical": numerical,
                    "autograd": analytic,
                    "scaled_error": scaled_error,
                }
            )
    if maximum_scaled_error > 2.0e-5:
        raise RuntimeError(
            f"block-2 post-attention finite difference failed: "
            f"{maximum_scaled_error}"
        )

    fixture: dict[str, object] = {
        "schema": "so3lr-native-block2-post-attention-reverse-fixture-v1",
        "block_index": 2,
        "nodes": nodes,
        "atomic_numbers": atomic_numbers.tolist(),
        "inv_features": flat(inputs[0]),
        "ev_features": flat(inputs[1]),
        "d_att_inv": flat(inputs[2]),
        "d_att_ev": flat(inputs[3]),
        "final_inv": flat(final_inv),
        "final_ev": flat(final_ev),
        "total_energy": float(total_energy),
        "energy_final_inv_gradient": flat(energy_final_inv_gradient),
        "energy_final_ev_gradient": [0.0] * (nodes * 24),
        "energy_grad_inv_features": flat(energy_input_gradients[0]),
        "energy_grad_ev_features": flat(energy_input_gradients[1]),
        "energy_grad_d_att_inv": flat(energy_input_gradients[2]),
        "energy_grad_d_att_ev": flat(energy_input_gradients[3]),
        "generic_final_inv_seed": flat(generic_inv_seed),
        "generic_final_ev_seed": flat(generic_ev_seed),
        "generic_grad_inv_features": flat(generic_input_gradients[0]),
        "generic_grad_ev_features": flat(generic_input_gradients[1]),
        "generic_grad_d_att_inv": flat(generic_input_gradients[2]),
        "generic_grad_d_att_ev": flat(generic_input_gradients[3]),
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
    print("block_index=2")
    print("reverse_scope=complete_post_attention_boundary")
    print(f"nodes={nodes}")
    print(f"total_energy={float(total_energy):.17g}")
    print(f"finite_difference_max_scaled_error={maximum_scaled_error:.17g}")
    print("pytorch_block2_post_attention_reverse_fixture=PASS")


if __name__ == "__main__":
    main()
