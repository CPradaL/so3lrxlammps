#!/usr/bin/env python3
"""Generate independent PyTorch references for block-0 Q/K/V projections."""

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
        raise RuntimeError("dev_6 requires identity Q/K activation")
    if (attention.inv_heads, attention.inv_head_dim,
            attention.ev_heads, attention.ev_head_dim) != (4, 32, 4, 32):
        raise RuntimeError("unexpected block-0 attention dimensions")

    nodes = 7
    node = torch.arange(nodes, dtype=torch.float64)[:, None]
    channel = torch.arange(128, dtype=torch.float64)[None, :]
    features = (
        torch.sin((node + 1.0) * (channel + 0.5) / 29.0)
        + 0.25 * torch.cos((node + 0.75) * (channel + 1.0) / 17.0)
    )
    senders = torch.tensor([0, 1, 2, 3, 4, 5, 6, 0, 3, 6, 2], dtype=torch.long)
    receivers = torch.tensor([1, 2, 3, 4, 5, 6, 0, 4, 1, 5, 6], dtype=torch.long)
    split = features.view(nodes, 4, 32)
    with torch.no_grad():
        q_inv_nodes = torch.einsum("nhd,hde->nhe", split, attention.W_q_inv)
        k_inv_nodes = torch.einsum("nhd,hde->nhe", split, attention.W_k_inv)
        v_inv_nodes = torch.einsum("nhd,hde->nhe", split, attention.W_v_inv)
        q_ev_nodes = torch.einsum("nhd,hde->nhe", split, attention.W_q_ev)
        k_ev_nodes = torch.einsum("nhd,hde->nhe", split, attention.W_k_ev)
        q_inv, k_inv, v_inv, q_ev, k_ev = attention._get_qkv(
            split, split, receivers, senders
        )

    fixture = {
        "schema": "so3lr-native-block0-qkv-fixture-v1",
        "nodes": nodes,
        "edges": senders.numel(),
        "heads": 4,
        "head_width": 32,
        "qk_activation": "identity",
        "features": flat(features),
        "senders": senders.tolist(),
        "receivers": receivers.tolist(),
        "q_inv_nodes": flat(q_inv_nodes),
        "k_inv_nodes": flat(k_inv_nodes),
        "v_inv_nodes": flat(v_inv_nodes),
        "q_ev_nodes": flat(q_ev_nodes),
        "k_ev_nodes": flat(k_ev_nodes),
        "q_inv_edges": flat(q_inv),
        "k_inv_edges": flat(k_inv),
        "v_inv_edges": flat(v_inv),
        "q_ev_edges": flat(q_ev),
        "k_ev_edges": flat(k_ev),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(fixture, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(f"fixture={args.output}")
    print(f"nodes={nodes}")
    print(f"edges={senders.numel()}")
    print("qk_activation=identity")
    print("pytorch_block0_qkv_fixture=PASS")


if __name__ == "__main__":
    main()

