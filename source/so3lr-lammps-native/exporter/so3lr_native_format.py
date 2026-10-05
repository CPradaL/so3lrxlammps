#!/usr/bin/env python3
"""Standard-library reader/writer for SO3LR native model files."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any, Iterable


MAGIC = b"SO3LRN1\0"
VERSION = 1
ALIGNMENT = 64
HEADER = struct.Struct("<8sIIQQ32s32s32s")
HEADER_SIZE = 128
DTYPE_BYTES = {
    "float64": 8,
    "float32": 4,
    "int64": 8,
    "int32": 4,
    "uint8": 1,
    "bool": 1,
}


class NativeModelFormatError(RuntimeError):
    """The native model file violates the versioned format contract."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def align_up(value: int, alignment: int = ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def _shape_elements(shape: list[int]) -> int:
    result = 1
    for extent in shape:
        if not isinstance(extent, int) or extent < 0:
            raise NativeModelFormatError(f"invalid tensor shape {shape!r}")
        result *= extent
    return result


def write_native_model(
    path: Path,
    manifest_base: dict[str, Any],
    tensors: Iterable[tuple[dict[str, Any], bytes]],
) -> dict[str, Any]:
    """Write a deterministic native model and return its final manifest."""
    path = Path(path)
    records: list[dict[str, Any]] = []
    payload = bytearray()
    previous_name: str | None = None
    for supplied, raw in sorted(tensors, key=lambda item: item[0]["name"]):
        record = dict(supplied)
        name = record.get("name")
        if not isinstance(name, str) or not name:
            raise NativeModelFormatError("every tensor requires a nonempty name")
        if previous_name == name:
            raise NativeModelFormatError(f"duplicate tensor name {name}")
        previous_name = name
        dtype = record.get("dtype")
        if dtype not in DTYPE_BYTES:
            raise NativeModelFormatError(f"unsupported dtype {dtype!r}")
        shape = record.get("shape")
        if not isinstance(shape, list):
            raise NativeModelFormatError(f"tensor {name} has no shape list")
        expected = _shape_elements(shape) * DTYPE_BYTES[dtype]
        if len(raw) != expected:
            raise NativeModelFormatError(
                f"tensor {name}: bytes={len(raw)}, expected={expected}"
            )
        offset = align_up(len(payload))
        payload.extend(b"\0" * (offset - len(payload)))
        payload.extend(raw)
        record.update(
            offset=offset,
            nbytes=len(raw),
            sha256=sha256_bytes(raw),
        )
        records.append(record)

    manifest = dict(manifest_base)
    manifest.update(
        format="so3lr-native",
        format_version=VERSION,
        endianness="little",
        tensor_alignment=ALIGNMENT,
        tensor_count=len(records),
        tensor_bytes=sum(item["nbytes"] for item in records),
        tensors=records,
    )
    manifest_bytes = canonical_json(manifest)
    payload_bytes = bytes(payload)
    header = HEADER.pack(
        MAGIC,
        VERSION,
        0,
        len(manifest_bytes),
        len(payload_bytes),
        hashlib.sha256(manifest_bytes).digest(),
        hashlib.sha256(payload_bytes).digest(),
        b"\0" * 32,
    )
    if len(header) != HEADER_SIZE:
        raise AssertionError("internal header-size mismatch")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + manifest_bytes + payload_bytes)
    return manifest


def read_native_model(path: Path, *, verify: bool = True) -> tuple[dict[str, Any], bytes]:
    path = Path(path)
    raw = path.read_bytes()
    if len(raw) < HEADER_SIZE:
        raise NativeModelFormatError("file is shorter than the fixed header")
    magic, version, flags, manifest_size, payload_size, manifest_hash, payload_hash, reserved = HEADER.unpack(
        raw[:HEADER_SIZE]
    )
    if magic != MAGIC:
        raise NativeModelFormatError(f"wrong magic {magic!r}")
    if version != VERSION:
        raise NativeModelFormatError(f"unsupported format version {version}")
    if flags != 0 or reserved != b"\0" * 32:
        raise NativeModelFormatError("nonzero flags or reserved header bytes")
    expected_size = HEADER_SIZE + manifest_size + payload_size
    if len(raw) != expected_size:
        raise NativeModelFormatError(
            f"file size={len(raw)}, header declares={expected_size}"
        )
    manifest_bytes = raw[HEADER_SIZE : HEADER_SIZE + manifest_size]
    payload = raw[HEADER_SIZE + manifest_size :]
    if verify and hashlib.sha256(manifest_bytes).digest() != manifest_hash:
        raise NativeModelFormatError("manifest SHA-256 mismatch")
    if verify and hashlib.sha256(payload).digest() != payload_hash:
        raise NativeModelFormatError("payload SHA-256 mismatch")
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NativeModelFormatError(f"invalid manifest JSON: {exc}") from exc
    if canonical_json(manifest) != manifest_bytes:
        raise NativeModelFormatError("manifest is not canonical JSON")
    validate_manifest(manifest, payload, verify=verify)
    return manifest, payload


def validate_manifest(manifest: dict[str, Any], payload: bytes, *, verify: bool = True) -> None:
    if manifest.get("format") != "so3lr-native":
        raise NativeModelFormatError("wrong manifest format")
    if manifest.get("format_version") != VERSION:
        raise NativeModelFormatError("manifest/header version mismatch")
    if manifest.get("endianness") != "little":
        raise NativeModelFormatError("version 1 requires little-endian tensors")
    if manifest.get("tensor_alignment") != ALIGNMENT:
        raise NativeModelFormatError("wrong tensor alignment")
    tensors = manifest.get("tensors")
    if not isinstance(tensors, list):
        raise NativeModelFormatError("manifest tensors must be a list")
    if manifest.get("tensor_count") != len(tensors):
        raise NativeModelFormatError("tensor-count mismatch")

    names: set[str] = set()
    previous_end = 0
    total_bytes = 0
    for record in tensors:
        name = record.get("name")
        dtype = record.get("dtype")
        shape = record.get("shape")
        offset = record.get("offset")
        nbytes = record.get("nbytes")
        if not isinstance(name, str) or not name or name in names:
            raise NativeModelFormatError(f"invalid/duplicate tensor name {name!r}")
        names.add(name)
        if dtype not in DTYPE_BYTES or not isinstance(shape, list):
            raise NativeModelFormatError(f"tensor {name}: invalid dtype/shape")
        expected = _shape_elements(shape) * DTYPE_BYTES[dtype]
        if not isinstance(offset, int) or not isinstance(nbytes, int):
            raise NativeModelFormatError(f"tensor {name}: invalid offset/length")
        if offset % ALIGNMENT or nbytes != expected:
            raise NativeModelFormatError(f"tensor {name}: alignment/size mismatch")
        if offset < previous_end or offset + nbytes > len(payload):
            raise NativeModelFormatError(f"tensor {name}: overlapping/out-of-range data")
        if any(payload[previous_end:offset]):
            raise NativeModelFormatError(f"tensor {name}: nonzero alignment padding")
        tensor_raw = payload[offset : offset + nbytes]
        if verify and sha256_bytes(tensor_raw) != record.get("sha256"):
            raise NativeModelFormatError(f"tensor {name}: SHA-256 mismatch")
        previous_end = offset + nbytes
        total_bytes += nbytes
    if previous_end != len(payload):
        raise NativeModelFormatError("unreferenced trailing payload bytes")
    if manifest.get("tensor_bytes") != total_bytes:
        raise NativeModelFormatError("tensor-byte-total mismatch")


def tensor_bytes(manifest: dict[str, Any], payload: bytes, name: str) -> bytes:
    for record in manifest["tensors"]:
        if record["name"] == name:
            return payload[record["offset"] : record["offset"] + record["nbytes"]]
    raise KeyError(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    manifest, payload = read_native_model(args.model, verify=True)
    if args.manifest:
        args.manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    architecture = manifest.get("architecture", {})
    print(f"format_version={manifest['format_version']}")
    print(f"model_family={manifest.get('model_family')}")
    print(f"tensor_count={manifest['tensor_count']}")
    print(f"tensor_bytes={manifest['tensor_bytes']}")
    print(f"payload_bytes={len(payload)}")
    print(f"short_range_cutoff={architecture.get('short_range_cutoff_angstrom')}")
    print(f"long_range_cutoff={architecture.get('long_range_cutoff_angstrom')}")
    print("native_model_format_validation=PASS")


if __name__ == "__main__":
    main()
