#!/usr/bin/env python3
"""Generate independent PyTorch references for the first native SR operators."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def find_so3lr(root: torch.nn.Module) -> torch.nn.Module:
    for module in root.modules():
        if type(module).__name__ == "SO3LR":
            return module
    raise RuntimeError("checkpoint does not contain SO3LR")


def discover_root(wrapper: object) -> torch.nn.Module:
    found = []
    seen = {id(wrapper)}

    def visit(value: object) -> None:
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, torch.nn.Module):
            found.append(value)
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    for value in vars(wrapper).values():
        visit(value)
    if len(found) != 1:
        raise RuntimeError(f"expected one root module, found {len(found)}")
    return found[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    wrapper = torch.load(args.model, map_location="cpu", weights_only=False)
    model = find_so3lr(discover_root(wrapper)).eval()
    distances = torch.tensor(
        [0.0, 1.0e-7, 0.25, 1.0, 2.25, 4.0, 4.499999, 4.5, 5.0],
        dtype=torch.float64,
    )
    atomic_numbers = torch.tensor([1, 8, 1, 8], dtype=torch.int64)
    one_hot = torch.nn.functional.one_hot(atomic_numbers - 1, num_classes=118).to(torch.float64)
    with torch.no_grad():
        fixture = {
            "schema": "so3lr-native-input-fixture-v1",
            "distances": distances.tolist(),
            "physnet_cutoff": model.cutoff_fn(distances).tolist(),
            # BernsteinBasis stores 32 channel indices.  Its production input
            # is an [N,1] edge-distance tensor so broadcasting produces
            # [N,32], whereas a bare [N] vector conflicts with the channels.
            "bernstein_rbf": model.radial_embedding(distances[:, None]).tolist(),
            "atomic_numbers": atomic_numbers.tolist(),
            "invariant_embedding": model.inv_feature_embedding(one_hot).tolist(),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, sort_keys=True, separators=(",", ":")) + "\n")
    print(f"fixture={args.output}")
    print(f"distances={len(fixture['distances'])}")
    print(f"embedding_atoms={len(fixture['atomic_numbers'])}")
    print("pytorch_input_fixture=PASS")


if __name__ == "__main__":
    main()
