#!/usr/bin/env python3
"""Generate an independent PyTorch reference for transformer blocks 0 -> 1."""

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
        raise RuntimeError("expected three EuclideanTransformer blocks")

    atomic_numbers = torch.tensor([8, 1, 1, 8, 1, 1, 8], dtype=torch.long)
    senders = torch.tensor([0, 1, 2, 3, 4, 5, 6, 0, 3, 6, 2], dtype=torch.long)
    receivers = torch.tensor([1, 2, 3, 4, 5, 6, 0, 4, 1, 5, 6], dtype=torch.long)
    distances = torch.tensor(
        [0.15, 0.45, 0.85, 1.25, 1.8, 2.3, 2.85, 3.35, 3.8, 4.2, 4.45],
        dtype=torch.float64,
    )[:, None]
    edge = torch.arange(senders.numel(), dtype=torch.float64)[:, None]
    channel = torch.arange(24, dtype=torch.float64)[None, :]
    sh_vectors = (
        torch.sin((edge + 1.0) * (channel + 0.75) / 13.0)
        + 0.1 * torch.cos((edge + 0.25) * (channel + 1.0) / 5.0)
    )

    with torch.no_grad():
        one_hot = torch.nn.functional.one_hot(
            atomic_numbers - 1, num_classes=118
        ).to(torch.float64)
        inv = model.inv_feature_embedding(one_hot)
        ev = torch.zeros((atomic_numbers.numel(), 24), dtype=torch.float64)
        rbf = model.radial_embedding(distances)
        cutoffs = model.cutoff_fn(distances)
        block0_inv, block0_ev = model.euclidean_transformers[0](
            inv, ev, rbf, senders, receivers, sh_vectors, cutoffs
        )
        block1_inv, block1_ev = model.euclidean_transformers[1](
            block0_inv,
            block0_ev,
            rbf,
            senders,
            receivers,
            sh_vectors,
            cutoffs,
        )

    if float(block0_ev.abs().max()) <= 1.0e-12:
        raise RuntimeError("stack fixture lacks nonzero block-0 equivariance")
    fixture = {
        "schema": "so3lr-native-transformer-stack01-fixture-v1",
        "blocks": [0, 1],
        "nodes": atomic_numbers.numel(),
        "edges": senders.numel(),
        "atomic_numbers": atomic_numbers.tolist(),
        "distances": flat(distances),
        "senders": senders.tolist(),
        "receivers": receivers.tolist(),
        "sh_vectors": flat(sh_vectors),
        "block0_final_inv": flat(block0_inv),
        "block0_final_ev": flat(block0_ev),
        "block1_final_inv": flat(block1_inv),
        "block1_final_ev": flat(block1_ev),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(fixture, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(f"fixture={args.output}")
    print("blocks=0,1")
    print(f"nodes={atomic_numbers.numel()}")
    print(f"edges={senders.numel()}")
    print(f"block0_maximum_ev={float(block0_ev.abs().max()):.17g}")
    print("pytorch_transformer_stack01_fixture=PASS")


if __name__ == "__main__":
    main()
