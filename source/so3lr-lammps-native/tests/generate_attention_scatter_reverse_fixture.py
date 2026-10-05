#!/usr/bin/env python3
"""Generate the block-2 fused attention/scatter VJP oracle."""

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
    attention = model.euclidean_transformers[2].euclidean_attention_block
    degree_repeats = attention.degree_repeats.detach().cpu().tolist()
    if degree_repeats != [3, 5, 7, 9]:
        raise RuntimeError(f"unexpected block-2 degree repeats: {degree_repeats}")

    nodes = 7
    senders = torch.tensor([0, 1, 2, 4, 5, 6, 3, 0, 5, 2, 6], dtype=torch.long)
    receivers = torch.tensor([1, 2, 0, 3, 3, 4, 5, 6, 1, 4, 2], dtype=torch.long)
    edges = int(senders.numel())
    node = torch.arange(nodes, dtype=torch.float64)[:, None, None]
    head = torch.arange(4, dtype=torch.float64)[None, :, None]
    channel = torch.arange(32, dtype=torch.float64)[None, None, :]
    edge = torch.arange(edges, dtype=torch.float64)[:, None, None]
    ev_channel = torch.arange(24, dtype=torch.float64)[None, :]

    initial = [
        0.21 * torch.sin((node + 0.3) * (head + 0.8) * (channel + 1.0) / 31.0)
        + 0.07 * torch.cos((node + 1.1) * (channel + 0.4) / 19.0),
        0.18 * torch.cos((node + 0.5) * (head + 1.2) * (channel + 0.7) / 29.0)
        - 0.06 * torch.sin((node + 0.8) * (channel + 1.1) / 17.0),
        0.24 * torch.sin((node + 1.2) * (channel + 0.2) / 23.0)
        + 0.04 * torch.cos((head + 0.6) * (channel + 1.5) / 11.0),
        0.15 * torch.cos((node + 0.7) * (head + 0.4) * (channel + 1.3) / 27.0)
        + 0.05 * torch.sin((node + 0.2) * (channel + 0.9) / 13.0),
        0.17 * torch.sin((node + 0.9) * (head + 1.0) * (channel + 0.6) / 25.0)
        - 0.03 * torch.cos((node + 1.4) * (channel + 0.3) / 15.0),
        0.14 * torch.sin((edge + 0.4) * (head + 0.9) * (channel + 0.5) / 37.0)
        + 0.08 * torch.cos((edge + 1.3) * (channel + 1.0) / 21.0),
        0.16 * torch.cos((edge + 0.6) * (head + 0.5) * (channel + 1.2) / 35.0)
        - 0.05 * torch.sin((edge + 0.2) * (channel + 0.7) / 18.0),
        0.19 * torch.sin((edge[:, :, 0] + 0.8) * (ev_channel + 0.6) / 14.0)
        + 0.09 * torch.cos((edge[:, :, 0] + 0.3) * (ev_channel + 1.1) / 9.0),
        0.35
        + 0.55 * (torch.arange(edges, dtype=torch.float64) + 1.0) / (edges + 2.0),
    ]
    inputs = [value.detach().requires_grad_(True) for value in initial]

    def fused(values: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        q_inv, k_inv, v_inv, q_ev, k_ev, filter_inv, filter_ev, sh, cutoff = values
        alpha_inv = (
            q_inv[receivers] * k_inv[senders] * filter_inv
        ).sum(dim=-1) / float(attention.att_norm_inv)
        messages_inv = cutoff[:, None, None] * alpha_inv[:, :, None] * v_inv[senders]
        d_inv = torch.zeros((nodes, 4, 32), dtype=torch.float64)
        d_inv = d_inv.index_add(0, receivers, messages_inv)
        alpha_ev = (
            q_ev[receivers] * k_ev[senders] * filter_ev
        ).sum(dim=-1) / float(attention.att_norm_ev)
        expanded = torch.repeat_interleave(
            alpha_ev, attention.degree_repeats, dim=-1, output_size=24
        )
        messages_ev = cutoff[:, None] * expanded * sh
        d_ev = torch.zeros((nodes, 24), dtype=torch.float64)
        d_ev = d_ev.index_add(0, receivers, messages_ev)
        return d_inv, d_ev

    d_inv, d_ev = fused(inputs)
    inv_seed = (
        0.031 * torch.sin((node + 0.4) * (head + 0.7) * (channel + 0.6) / 28.0)
        - 0.014 * torch.cos((node + 1.0) * (channel + 0.3) / 16.0)
    )
    ev_node = torch.arange(nodes, dtype=torch.float64)[:, None]
    ev_seed = (
        0.027 * torch.cos((ev_node + 0.5) * (ev_channel + 0.8) / 10.0)
        + 0.011 * torch.sin((ev_node + 1.2) * (ev_channel + 0.4) / 6.0)
    )
    objective = (d_inv * inv_seed).sum() + (d_ev * ev_seed).sum()
    gradients = torch.autograd.grad(objective, inputs)
    for label, value in {
        "d_inv": d_inv,
        "d_ev": d_ev,
        **{f"gradient_{i}": value for i, value in enumerate(gradients)},
    }.items():
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"nonfinite reference tensor: {label}")

    input_names = [
        "q_inv", "k_inv", "v_inv", "q_ev", "k_ev",
        "filter_inv", "filter_ev", "sh", "cutoff",
    ]
    selected = [
        (0, 17), (1, 4 * 128 + 71), (2, 6 * 128 + 127),
        (3, 2 * 128 + 9), (4, 5 * 128 + 82), (5, 3 * 128 + 41),
        (6, 8 * 128 + 103), (7, 7 * 24 + 19), (8, 9),
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
            plus_inv, plus_ev = fused(plus)
            minus_inv, minus_ev = fused(minus)
            plus_objective = (plus_inv * inv_seed).sum() + (plus_ev * ev_seed).sum()
            minus_objective = (minus_inv * inv_seed).sum() + (minus_ev * ev_seed).sum()
            numerical = float((plus_objective - minus_objective) / (2.0 * epsilon))
            analytic = float(gradients[which].reshape(-1)[index])
            scaled_error = abs(numerical - analytic) / (1.0 + abs(analytic))
            maximum_scaled_error = max(maximum_scaled_error, scaled_error)
            checks.append({
                "input": input_names[which],
                "flat_index": index,
                "numerical": numerical,
                "autograd": analytic,
                "scaled_error": scaled_error,
            })
    if maximum_scaled_error > 2.0e-6:
        raise RuntimeError(
            f"attention/scatter finite difference failed: {maximum_scaled_error}"
        )

    fixture: dict[str, object] = {
        "schema": "so3lr-native-block2-attention-scatter-reverse-fixture-v1",
        "block_index": 2,
        "nodes": nodes,
        "edges": edges,
        "senders": senders.tolist(),
        "receivers": receivers.tolist(),
        "degree_repeats": degree_repeats,
        "attention_norm_inv": float(attention.att_norm_inv),
        "attention_norm_ev": float(attention.att_norm_ev),
        "q_inv": flat(inputs[0]),
        "k_inv": flat(inputs[1]),
        "v_inv": flat(inputs[2]),
        "q_ev": flat(inputs[3]),
        "k_ev": flat(inputs[4]),
        "filter_inv": flat(inputs[5]),
        "filter_ev": flat(inputs[6]),
        "sh": flat(inputs[7]),
        "cutoff": flat(inputs[8]),
        "d_inv": flat(d_inv),
        "d_ev": flat(d_ev),
        "grad_d_inv": flat(inv_seed),
        "grad_d_ev": flat(ev_seed),
        **{f"grad_{name}": flat(value) for name, value in zip(input_names, gradients)},
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
    print("block_index=2")
    print("reverse_scope=fused_attention_scatter")
    print(f"nodes={nodes}")
    print(f"edges={edges}")
    print(f"finite_difference_max_scaled_error={maximum_scaled_error:.17g}")
    print("pytorch_attention_scatter_vjp_reference=PASS")


if __name__ == "__main__":
    main()
