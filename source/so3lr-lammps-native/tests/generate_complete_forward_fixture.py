#!/usr/bin/env python3
"""Generate a physical PyTorch reference for the complete learned forward core."""

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
    if senders.numel() != atomic_numbers.numel() * (atomic_numbers.numel() - 1):
        raise RuntimeError("water-dimer fixture is not a complete directed graph")
    # MLIAP's sender-minus-receiver rij is negated to this vector.
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

        data = {"node_attrs": one_hot}
        atomic_energies = model.atomic_energy_output_block(
            inv, data, atomic_numbers=atomic_numbers
        ).squeeze(-1)
        charge_head = model.partial_charges_output_block
        raw_charges = (
            charge_head.transform_inv_features(inv).squeeze(-1)
            + charge_head.atomic_embedding(atomic_numbers).squeeze(-1)
        )
        total_charge = torch.tensor([0.0], dtype=torch.float64)
        batch = torch.zeros(atomic_numbers.numel(), dtype=torch.long)
        partial_charges = charge_head(
            inv_features=inv,
            atomic_numbers=atomic_numbers,
            total_charge=total_charge,
            batch_segments=batch,
            num_graphs=1,
        )
        hirshfeld = model.hirshfeld_output_block(inv, atomic_numbers)

    values = {
        "final_inv": inv,
        "final_ev": ev,
        "atomic_energies": atomic_energies,
        "raw_charges": raw_charges,
        "partial_charges": partial_charges,
        "hirshfeld_ratios": hirshfeld,
    }
    for name, value in values.items():
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"nonfinite {name} reference")
    if abs(float(partial_charges.sum())) > 1.0e-12:
        raise RuntimeError("PyTorch charge-neutrality reference failed")

    fixture = {
        "schema": "so3lr-native-complete-forward-fixture-v1",
        "geometry": "physical_water_dimer_nonperiodic",
        "scope": "three_transformers_plus_energy_charge_hirshfeld_heads",
        "nodes": atomic_numbers.numel(),
        "edges": senders.numel(),
        "atomic_numbers": atomic_numbers.tolist(),
        "positions": flat(positions),
        "distances": flat(distances),
        "senders": senders.tolist(),
        "receivers": receivers.tolist(),
        "sh_vectors": flat(sh_vectors),
        "total_charge": 0.0,
        "learned_sr_energy": float(atomic_energies.sum()),
    }
    fixture.update({name: flat(value) for name, value in values.items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(fixture, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(f"fixture={args.output}")
    print("geometry=physical_water_dimer_nonperiodic")
    print("pipeline=three_transformers_plus_energy_charge_hirshfeld_heads")
    print(f"nodes={atomic_numbers.numel()}")
    print(f"edges={senders.numel()}")
    print(f"charge_sum={float(partial_charges.sum()):.17g}")
    print(f"learned_sr_energy={float(atomic_energies.sum()):.17g}")
    print("pytorch_complete_forward_fixture=PASS")


if __name__ == "__main__":
    main()
