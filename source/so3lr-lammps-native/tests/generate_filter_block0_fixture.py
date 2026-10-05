#!/usr/bin/env python3
"""Generate independent PyTorch references for transformer block-0 filters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def discover_root(wrapper: object) -> torch.nn.Module:
    roots = []
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    wrapper = torch.load(args.model, map_location="cpu", weights_only=False)
    model = find_so3lr(discover_root(wrapper)).eval()
    block = model.euclidean_transformers[0]
    distances = torch.tensor(
        [0.05, 0.25, 0.75, 1.5, 2.25, 3.5, 4.25, 4.499999],
        dtype=torch.float64,
    )
    edge = torch.arange(distances.numel(), dtype=torch.float64)[:, None]
    channel = torch.arange(4, dtype=torch.float64)[None, :]
    ev_invariants = torch.sin((edge + 1.0) * (channel + 0.5)) / (channel + 1.0)
    zero_ev = torch.zeros_like(ev_invariants)
    with torch.no_grad():
        rbf = model.radial_embedding(distances[:, None])
        inv_filter = block.filter_net_inv(rbf, ev_invariants)
        ev_filter = block.filter_net_ev(rbf, ev_invariants)
        inv_filter_zero = block.filter_net_inv(rbf, zero_ev)
        ev_filter_zero = block.filter_net_ev(rbf, zero_ev)
    fixture = {
        "schema": "so3lr-native-block0-filter-fixture-v1",
        "edges": distances.numel(),
        "distances": distances.tolist(),
        "radial_basis": rbf.tolist(),
        "ev_invariants": ev_invariants.tolist(),
        "zero_ev_invariants": zero_ev.tolist(),
        "invariant_filter": inv_filter.tolist(),
        "equivariant_filter": ev_filter.tolist(),
        "invariant_filter_zero_ev": inv_filter_zero.tolist(),
        "equivariant_filter_zero_ev": ev_filter_zero.tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(fixture, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(f"fixture={args.output}")
    print(f"edges={fixture['edges']}")
    print("physical_zero_ev_case=1")
    print("synthetic_nonzero_ev_case=1")
    print("pytorch_block0_filter_fixture=PASS")


if __name__ == "__main__":
    main()

