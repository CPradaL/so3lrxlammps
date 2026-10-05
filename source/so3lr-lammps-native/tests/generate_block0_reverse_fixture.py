#!/usr/bin/env python3
"""Generate the complete block-0 VJP oracle at its production input boundary."""

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
    block = model.euclidean_transformers[0]
    if not (
        block.layer_normalization_1
        and block.layer_normalization_2
        and block.residual_mlp_1
        and not block.residual_mlp_2
    ):
        raise RuntimeError("unexpected block-0 residual contract")

    atomic_numbers = torch.tensor([8, 1, 1, 8, 1, 1, 8], dtype=torch.long)
    nodes = int(atomic_numbers.numel())
    senders = torch.tensor([0, 1, 2, 4, 5, 6, 3, 0, 5, 2, 6], dtype=torch.long)
    receivers = torch.tensor([1, 2, 0, 3, 3, 4, 5, 6, 1, 4, 2], dtype=torch.long)
    edges = int(senders.numel())
    node = torch.arange(nodes, dtype=torch.float64)[:, None]
    inv_channel = torch.arange(128, dtype=torch.float64)[None, :]
    ev_channel = torch.arange(24, dtype=torch.float64)[None, :]
    edge = torch.arange(edges, dtype=torch.float64)[:, None]
    one_hot = torch.nn.functional.one_hot(
        atomic_numbers - 1, num_classes=118
    ).to(torch.float64)
    with torch.no_grad():
        embedded = model.inv_feature_embedding(one_hot)
    initial = [
        embedded,
        torch.zeros((nodes, 24), dtype=torch.float64),
        torch.tensor(
            [0.22, 0.48, 0.79, 1.16, 1.61, 2.08, 2.57, 3.02, 3.46, 3.91, 4.34],
            dtype=torch.float64,
        )[:, None],
        0.19 * torch.sin((edge + 0.7) * (ev_channel + 0.9) / 12.0)
        + 0.07 * torch.cos((edge + 1.2) * (ev_channel + 0.4) / 7.0),
    ]
    inputs = [value.detach().requires_grad_(True) for value in initial]

    def complete(values: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        inv, ev, distances, sh = values
        radial_basis = model.radial_embedding(distances)
        cutoffs = model.cutoff_fn(distances)
        return block(
            inv,
            ev,
            radial_basis,
            senders,
            receivers,
            sh,
            cutoffs,
            return_att=False,
        )

    final_inv, final_ev = complete(inputs)
    inv_seed = (
        0.029 * torch.sin((node + 0.3) * (inv_channel + 0.7) / 22.0)
        - 0.012 * torch.cos((node + 1.4) * (inv_channel + 0.2) / 14.0)
    )
    ev_seed = (
        0.025 * torch.cos((node + 0.8) * (ev_channel + 0.6) / 9.0)
        + 0.013 * torch.sin((node + 1.0) * (ev_channel + 0.4) / 6.0)
    )
    objective = (final_inv * inv_seed).sum() + (final_ev * ev_seed).sum()
    gradients = torch.autograd.grad(objective, inputs)
    for label, value in {
        "final_inv": final_inv,
        "final_ev": final_ev,
        **{f"gradient_{i}": value for i, value in enumerate(gradients)},
    }.items():
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"nonfinite reference tensor: {label}")
    if float(inputs[1].detach().abs().max()) != 0.0:
        raise RuntimeError("block-0 production equivariant input is not zero")
    if float(gradients[2].abs().max()) <= 1.0e-12:
        raise RuntimeError("block-0 fixture has no distance sensitivity")
    if float(gradients[3].abs().max()) <= 1.0e-12:
        raise RuntimeError("block-0 fixture has no SH sensitivity")

    names = ["inv_features", "ev_features", "distances", "sh_vectors"]
    selected = [
        (0, 17), (0, 6 * 128 + 113),
        (1, 2 * 24 + 9), (1, 5 * 24 + 21),
        (2, 0), (2, 5), (2, 10),
        (3, 3 * 24 + 7), (3, 9 * 24 + 19),
    ]
    epsilon = 1.0e-5
    checks: list[dict[str, object]] = []
    maximum_scaled_error = 0.0
    with torch.no_grad():
        for which, index in selected:
            plus = [value.detach().clone() for value in inputs]
            minus = [value.detach().clone() for value in inputs]
            plus[which].reshape(-1)[index] += epsilon
            minus[which].reshape(-1)[index] -= epsilon
            plus_inv, plus_ev = complete(plus)
            minus_inv, minus_ev = complete(minus)
            plus_objective = (plus_inv * inv_seed).sum() + (plus_ev * ev_seed).sum()
            minus_objective = (minus_inv * inv_seed).sum() + (minus_ev * ev_seed).sum()
            numerical = float((plus_objective - minus_objective) / (2.0 * epsilon))
            analytic = float(gradients[which].reshape(-1)[index])
            scaled_error = abs(numerical - analytic) / (1.0 + abs(analytic))
            maximum_scaled_error = max(maximum_scaled_error, scaled_error)
            checks.append({
                "input": names[which],
                "flat_index": index,
                "numerical": numerical,
                "autograd": analytic,
                "scaled_error": scaled_error,
            })
    if maximum_scaled_error > 2.0e-5:
        raise RuntimeError(f"complete block-0 finite difference failed: {maximum_scaled_error}")

    fixture: dict[str, object] = {
        "schema": "so3lr-native-complete-block0-reverse-fixture-v1",
        "block_index": 0,
        "input_contract": "atomic_embedding_plus_zero_equivariants",
        "nodes": nodes,
        "edges": edges,
        "atomic_numbers": atomic_numbers.tolist(),
        "senders": senders.tolist(),
        "receivers": receivers.tolist(),
        "inv_features": flat(inputs[0]),
        "ev_features": flat(inputs[1]),
        "distances": flat(inputs[2]),
        "sh_vectors": flat(inputs[3]),
        "final_inv": flat(final_inv),
        "final_ev": flat(final_ev),
        "grad_final_inv": flat(inv_seed),
        "grad_final_ev": flat(ev_seed),
        "grad_inv_features": flat(gradients[0]),
        "grad_ev_features": flat(gradients[1]),
        "grad_distances": flat(gradients[2]),
        "grad_sh_vectors": flat(gradients[3]),
        "maximum_abs_initial_ev": float(inputs[1].detach().abs().max()),
        "maximum_abs_distance_gradient": float(gradients[2].abs().max()),
        "maximum_abs_sh_gradient": float(gradients[3].abs().max()),
        "finite_difference_epsilon": epsilon,
        "finite_difference_checks": checks,
        "finite_difference_max_scaled_error": maximum_scaled_error,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(fixture, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(f"fixture={args.output}")
    print("block_index=0")
    print("input_contract=atomic_embedding_plus_zero_equivariants")
    print("reverse_scope=complete_transformer_to_features_distance_sh")
    print(f"nodes={nodes}")
    print(f"edges={edges}")
    print(f"maximum_abs_distance_gradient={float(gradients[2].abs().max()):.17g}")
    print(f"maximum_abs_sh_gradient={float(gradients[3].abs().max()):.17g}")
    print(f"finite_difference_max_scaled_error={maximum_scaled_error:.17g}")
    print("pytorch_complete_block0_vjp_reference=PASS")


if __name__ == "__main__":
    main()
