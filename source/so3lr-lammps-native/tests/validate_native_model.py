#!/usr/bin/env python3
"""Independent tensor/architecture round-trip validation for native SO3LR."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch import nn

from so3lr_native_format import NativeModelFormatError, read_native_model, tensor_bytes


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
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def roots(wrapper: Any) -> list[tuple[str, nn.Module]]:
    found = []
    for name, value in vars(wrapper).items():
        if isinstance(value, nn.Module):
            found.append((f"wrapper.{name}", value))
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--native-model", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--corruption-test", action="store_true")
    args = parser.parse_args()

    manifest, payload = read_native_model(args.native_model, verify=True)
    wrapper = torch.load(args.source_model, map_location="cpu", weights_only=False)
    found = roots(wrapper)
    if len(found) != 1:
        raise SystemExit(f"expected one root module, found {len(found)}")
    root_path, root = found[0]
    exported = {record["name"]: record for record in manifest["tensors"]}
    source_state = root.state_dict()
    expected_names = {f"{root_path}.{name}" for name in source_state}
    if set(exported) != expected_names:
        missing = sorted(expected_names - set(exported))
        extra = sorted(set(exported) - expected_names)
        raise SystemExit(f"tensor-name mismatch missing={missing} extra={extra}")

    checked_bytes = 0
    for state_key, tensor in source_state.items():
        name = f"{root_path}.{state_key}"
        record = exported[name]
        expected_dtype = TORCH_DTYPES.get(tensor.dtype)
        if expected_dtype != record["dtype"] or list(tensor.shape) != record["shape"]:
            raise SystemExit(f"dtype/shape mismatch for {name}")
        raw = raw_tensor(tensor)
        if raw != tensor_bytes(manifest, payload, name):
            raise SystemExit(f"bitwise tensor mismatch for {name}")
        checked_bytes += len(raw)

    architecture = manifest["architecture"]
    expected_contract = {
        "short_range_cutoff_angstrom": 4.5,
        "long_range_cutoff_angstrom": 12.0,
        "interaction_blocks": 3,
        "invariant_features": 128,
        "atomic_number_capacity": 118,
        "attention_heads": 4,
        "attention_head_width": 32,
        "euclidean_degree_channels": 4,
        "radial_basis_features": 32,
        "state_tensor_count": 211,
        "parameter_elements": 529852,
    }
    for key, expected in expected_contract.items():
        if architecture.get(key) != expected:
            raise SystemExit(
                f"architecture {key}={architecture.get(key)!r}, expected={expected!r}"
            )
    if architecture.get("output_heads") != [
        "atomic_energy_output_block",
        "partial_charges_output_block",
        "hirshfeld_output_block",
    ]:
        raise SystemExit("required energy/charge/Hirshfeld heads are not frozen")
    required_types = {
        "so3krates_torch.modules.models.SO3LR",
        "so3krates_torch.blocks.euclidean_transformer.EuclideanTransformer",
        "so3krates_torch.blocks.euclidean_transformer.EuclideanAttentionBlock",
        "so3krates_torch.blocks.euclidean_transformer.InteractionBlock",
        "so3krates_torch.blocks.radial_basis.BernsteinBasis",
        "so3krates_torch.modules.spherical_harmonics.RealSphericalHarmonics",
    }
    registry = set(architecture["module_type_registry"])
    if not required_types <= registry:
        raise SystemExit(f"missing required module types: {sorted(required_types-registry)}")
    source = manifest["source"]
    if source["implementation_commit"] != args.expected_source_commit:
        raise SystemExit("source implementation commit mismatch")
    if source["checkpoint_sha256"] != file_sha256(args.source_model):
        raise SystemExit("source checkpoint SHA-256 mismatch")
    if checked_bytes != manifest["tensor_bytes"]:
        raise SystemExit("round-trip byte total mismatch")

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
        "schema": "so3lr-stage3-dev1-validation-v1",
        "status": "PASS",
        "native_model": str(args.native_model),
        "native_sha256": file_sha256(args.native_model),
        "tensor_count": len(exported),
        "tensor_bytes": checked_bytes,
        "module_count": len(manifest["modules"]),
        "architecture": architecture,
        "corruption_detected": corruption_detected,
    }
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"tensor_count={len(exported)}")
    print(f"tensor_bytes={checked_bytes}")
    print(f"module_count={len(manifest['modules'])}")
    print(f"native_sha256={summary['native_sha256']}")
    print(f"corruption_detected={int(corruption_detected)}")
    print("so3lr_native_roundtrip=PASS")


if __name__ == "__main__":
    main()
