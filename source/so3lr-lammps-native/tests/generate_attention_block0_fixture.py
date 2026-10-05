#!/usr/bin/env python3
"""Generate an independent PyTorch reference for fused block-0 attention."""

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
    attention = model.euclidean_transformers[0].euclidean_attention_block
    if not isinstance(attention.qk_non_linearity, torch.nn.Identity):
        raise RuntimeError("dev_7 requires identity Q/K activation")
    degree_repeats = attention.degree_repeats.detach().cpu().tolist()
    if degree_repeats != [3, 5, 7, 9]:
        raise RuntimeError(f"unexpected degree repeats: {degree_repeats}")

    nodes = 7
    senders = torch.tensor([0, 1, 2, 3, 4, 5, 6, 0, 3, 6, 2], dtype=torch.long)
    receivers = torch.tensor([1, 2, 3, 4, 5, 6, 0, 4, 1, 5, 6], dtype=torch.long)
    edges = senders.numel()
    node = torch.arange(nodes, dtype=torch.float64)[:, None]
    inv_channel = torch.arange(128, dtype=torch.float64)[None, :]
    ev_channel = torch.arange(24, dtype=torch.float64)[None, :]
    edge = torch.arange(edges, dtype=torch.float64)[:, None]
    inv_features = (
        torch.sin((node + 1.0) * (inv_channel + 0.5) / 29.0)
        + 0.25 * torch.cos((node + 0.75) * (inv_channel + 1.0) / 17.0)
    )
    ev_features = (
        0.35 * torch.sin((node + 0.5) * (ev_channel + 1.0) / 11.0)
        - 0.15 * torch.cos((node + 1.25) * (ev_channel + 0.5) / 7.0)
    )
    sh_vectors = (
        torch.sin((edge + 1.0) * (ev_channel + 0.75) / 13.0)
        + 0.1 * torch.cos((edge + 0.25) * (ev_channel + 1.0) / 5.0)
    )
    distances = torch.tensor(
        [0.15, 0.45, 0.85, 1.25, 1.8, 2.3, 2.85, 3.35, 3.8, 4.2, 4.45],
        dtype=torch.float64,
    )[:, None]

    with torch.no_grad():
        rbf = model.radial_embedding(distances)
        cutoffs = model.cutoff_fn(distances)
        ev_differences = ev_features[senders] - ev_features[receivers]
        ev_invariants = attention.so3_conv_invariants(ev_differences)
        filter_inv = attention.filter_net_inv(rbf, ev_invariants)
        filter_ev = attention.filter_net_ev(rbf, ev_invariants)
        split = inv_features.view(nodes, 4, 32)
        q_inv_nodes = torch.einsum("nhd,hde->nhe", split, attention.W_q_inv)
        k_inv_nodes = torch.einsum("nhd,hde->nhe", split, attention.W_k_inv)
        v_inv_nodes = torch.einsum("nhd,hde->nhe", split, attention.W_v_inv)
        q_ev_nodes = torch.einsum("nhd,hde->nhe", split, attention.W_q_ev)
        k_ev_nodes = torch.einsum("nhd,hde->nhe", split, attention.W_k_ev)
        d_inv, d_ev, (alpha_inv, alpha_ev) = attention(
            inv_features,
            ev_features,
            rbf,
            senders,
            receivers,
            sh_vectors,
            cutoffs,
            return_att=True,
        )

    fixture = {
        "schema": "so3lr-native-block0-attention-fixture-v1",
        "nodes": nodes,
        "edges": edges,
        "heads": 4,
        "head_width": 32,
        "degree_repeats": degree_repeats,
        "attention_norm_inv": float(attention.att_norm_inv),
        "attention_norm_ev": float(attention.att_norm_ev),
        "senders": senders.tolist(),
        "receivers": receivers.tolist(),
        "cutoffs": flat(cutoffs),
        "sh_vectors": flat(sh_vectors),
        "filter_inv": flat(filter_inv),
        "filter_ev": flat(filter_ev),
        "q_inv_nodes": flat(q_inv_nodes),
        "k_inv_nodes": flat(k_inv_nodes),
        "v_inv_nodes": flat(v_inv_nodes),
        "q_ev_nodes": flat(q_ev_nodes),
        "k_ev_nodes": flat(k_ev_nodes),
        "d_inv": flat(d_inv),
        "d_ev": flat(d_ev),
        "alpha_inv": flat(alpha_inv),
        "alpha_ev": flat(alpha_ev),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(fixture, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(f"fixture={args.output}")
    print(f"nodes={nodes}")
    print(f"edges={edges}")
    print(f"attention_norm_inv={fixture['attention_norm_inv']}")
    print(f"attention_norm_ev={fixture['attention_norm_ev']}")
    print("degree_repeats=3,5,7,9")
    print("pytorch_block0_attention_fixture=PASS")


if __name__ == "__main__":
    main()

