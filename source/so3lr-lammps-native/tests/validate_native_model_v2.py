#!/usr/bin/env python3
"""Validate the self-contained native SO3LR model-v2 package."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch import nn

from so3krates_torch.blocks.physical_potentials import (
    ALPHAS,
    BOHR,
    C6_COEF,
    FINE_STRUCTURE,
    HARTREE,
)
from so3lr_native_format import (
    NativeModelFormatError,
    read_native_model,
    tensor_bytes,
)


TORCH_DTYPES = {
    torch.float64: "float64",
    torch.float32: "float32",
    torch.int64: "int64",
    torch.int32: "int32",
    torch.uint8: "uint8",
    torch.bool: "bool",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def raw_tensor(tensor: torch.Tensor) -> bytes:
    return (
        tensor.detach().cpu().contiguous().reshape(-1)
        .view(torch.uint8).numpy().tobytes()
    )


def roots(wrapper: Any) -> list[tuple[str, nn.Module]]:
    return [
        (f"wrapper.{name}", value)
        for name, value in vars(wrapper).items()
        if isinstance(value, nn.Module)
    ]


def find_so3lr(root: nn.Module) -> nn.Module:
    for module in root.modules():
        if type(module).__name__ == "SO3LR":
            return module
    raise RuntimeError("source checkpoint does not contain SO3LR")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--native-model", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--corruption-test", action="store_true")
    args = parser.parse_args()

    manifest, payload = read_native_model(args.native_model, verify=True)
    if manifest.get("schema") != "so3lr-native-model-v2":
        raise SystemExit("native model does not use the v2 self-contained schema")
    if manifest.get("exporter_version") != "stage3-dev26-v2":
        raise SystemExit("unexpected self-contained exporter version")
    wrapper = torch.load(args.source_model, map_location="cpu", weights_only=False)
    found = roots(wrapper)
    if len(found) != 1:
        raise SystemExit(f"expected one root module, found {len(found)}")
    root_path, root = found[0]
    model = find_so3lr(root)

    records_by_state = {record["state_key"]: record for record in manifest["tensors"]}
    source_state = root.state_dict()
    expected_state_keys = set(source_state) | {
        "physical.reference_alphas",
        "physical.reference_c6",
    }
    if set(records_by_state) != expected_state_keys:
        missing = sorted(expected_state_keys - set(records_by_state))
        extra = sorted(set(records_by_state) - expected_state_keys)
        raise SystemExit(f"v2 tensor mismatch missing={missing} extra={extra}")

    checked_source_bytes = 0
    for state_key, tensor in source_state.items():
        record = records_by_state[state_key]
        if record["name"] != f"{root_path}.{state_key}":
            raise SystemExit(f"source tensor name changed for {state_key}")
        if record["dtype"] != TORCH_DTYPES[tensor.dtype] or record["shape"] != list(tensor.shape):
            raise SystemExit(f"source tensor contract changed for {state_key}")
        raw = raw_tensor(tensor)
        if tensor_bytes(manifest, payload, record["name"]) != raw:
            raise SystemExit(f"source tensor differs for {state_key}")
        checked_source_bytes += len(raw)

    physical_tensors = {
        "physical.reference_alphas": ALPHAS.to(dtype=torch.float64),
        "physical.reference_c6": C6_COEF.to(dtype=torch.float64),
    }
    physical_bytes = 0
    for state_key, tensor in physical_tensors.items():
        record = records_by_state[state_key]
        if record["role"] != "physical_constant" or record["dtype"] != "float64":
            raise SystemExit(f"physical tensor metadata changed for {state_key}")
        raw = raw_tensor(tensor)
        if tensor_bytes(manifest, payload, record["name"]) != raw:
            raise SystemExit(f"physical tensor differs for {state_key}")
        physical_bytes += len(raw)

    architecture = manifest["architecture"]
    exact_contract = {
        "short_range_cutoff_angstrom": 4.5,
        "long_range_cutoff_angstrom": 12.0,
        "interaction_blocks": 3,
        "invariant_features": 128,
        "radial_basis_features": 32,
        "physical_reference_table_tensor_count": 2,
        "physical_reference_table_state_keys": [
            "physical.reference_alphas",
            "physical.reference_c6",
        ],
    }
    for key, expected in exact_contract.items():
        if architecture.get(key) != expected:
            raise SystemExit(
                f"architecture {key}={architecture.get(key)!r}, expected={expected!r}"
            )
    float_contract = {
        "lr_electrostatic_ke": float(model.electrostatic_potential.ke),
        "lr_electrostatic_sigma": float(model.electrostatic_energy_scale),
        "lr_electrostatic_cuton_angstrom": 0.45 * float(model.r_max_lr),
        "lr_fine_structure": float(FINE_STRUCTURE),
        "lr_bohr_angstrom": float(BOHR),
        "lr_hartree_ev": float(HARTREE),
        "lr_dispersion_scale": float(model.dispersion_energy_scale),
        "lr_dispersion_cuton_angstrom": float(model.r_max_lr) - float(
            model.dispersion_energy_cutoff_lr_damping
        ),
        "lr_pair_scale": float(model.dispersion_potential.c),
    }
    for key, expected in float_contract.items():
        actual = float(architecture.get(key, float("nan")))
        if abs(actual - expected) > 2.0e-14:
            raise SystemExit(f"physical parameter {key}={actual}, expected={expected}")

    if manifest["source"]["implementation_commit"] != args.expected_source_commit:
        raise SystemExit("source implementation commit mismatch")
    if manifest["source"]["checkpoint_sha256"] != file_sha256(args.source_model):
        raise SystemExit("source checkpoint SHA-256 mismatch")
    if checked_source_bytes + physical_bytes != manifest["tensor_bytes"]:
        raise SystemExit("v2 tensor-byte total mismatch")

    corruption_detected = False
    if args.corruption_test:
        corrupted = bytearray(args.native_model.read_bytes())
        corrupted[-1] ^= 0x01
        with tempfile.NamedTemporaryFile(suffix=".so3lr", delete=False) as handle:
            handle.write(corrupted)
            corrupt_path = Path(handle.name)
        try:
            try:
                read_native_model(corrupt_path, verify=True)
            except NativeModelFormatError:
                corruption_detected = True
        finally:
            corrupt_path.unlink(missing_ok=True)
        if not corruption_detected:
            raise SystemExit("corruption test was not detected")

    summary = {
        "schema": "so3lr-stage3-dev26-native-model-validation-v1",
        "status": "PASS",
        "native_model": str(args.native_model),
        "native_sha256": file_sha256(args.native_model),
        "format_version": manifest["format_version"],
        "model_schema": manifest["schema"],
        "tensor_count": manifest["tensor_count"],
        "learned_tensor_count": len(source_state),
        "physical_tensor_count": 2,
        "learned_tensor_bytes": checked_source_bytes,
        "physical_tensor_bytes": physical_bytes,
        "architecture": architecture,
        "corruption_detected": corruption_detected,
    }
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"native_sha256={summary['native_sha256']}")
    print(f"tensor_count={summary['tensor_count']}")
    print(f"learned_tensor_count={summary['learned_tensor_count']}")
    print(f"physical_tensor_count={summary['physical_tensor_count']}")
    print(f"physical_tensor_bytes={summary['physical_tensor_bytes']}")
    print(f"corruption_detected={int(corruption_detected)}")
    print("native_model_v2_self_contained=PASS")
    print("native_model_v2_roundtrip=PASS")


if __name__ == "__main__":
    main()
