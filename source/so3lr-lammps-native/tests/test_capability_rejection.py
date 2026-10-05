#!/usr/bin/env python3
"""Negative tests for capability negotiation (stage A3).

0.2.0-rc1 rejected every non-foundation checkpoint with one message,
"unsupported architecture contract", which told a user nothing about which
part of their model was the problem. This script builds deliberately
unsupported variants of a valid model and asserts that the runtime now names
the offending feature.

Each variant is a *structurally valid* .so3lr file -- manifest re-serialised
canonically, both SHA-256 digests recomputed -- so a rejection can only come
from negotiation, never from an integrity check.

usage: test_capability_rejection.py <probe-binary> <model.so3lr>
"""

import json
import hashlib
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HEADER = struct.Struct("<8sIIQQ32s32s32s")
MAGIC = b"SO3LRN1\0"


def canonical_json(obj):
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def read_model(path):
    raw = Path(path).read_bytes()
    magic, version, flags, mlen, plen, mhash, phash, reserved = HEADER.unpack_from(raw, 0)
    if magic != MAGIC:
        raise SystemExit(f"{path} is not a .so3lr file")
    manifest = json.loads(raw[128:128 + mlen].decode("ascii"))
    payload = raw[128 + mlen:128 + mlen + plen]
    return version, flags, manifest, payload, reserved


def write_model(path, version, flags, manifest, payload, reserved):
    blob = canonical_json(manifest)
    header = HEADER.pack(
        MAGIC, version, flags, len(blob), len(payload),
        hashlib.sha256(blob).digest(), hashlib.sha256(payload).digest(), reserved,
    )
    Path(path).write_bytes(header + blob + payload)


def variants(manifest):
    """(name, mutation, expected substring in the rejection message)."""

    def with_arch(**kw):
        def mutate(m):
            m["architecture"].update(kw)
        return mutate

    yield (
        "too_many_interaction_blocks",
        with_arch(interaction_blocks=9),
        "architecture.interaction_blocks = 9",
    )
    # Widen consistently (192 = 6 heads x 32, divisible by the 4 degree
    # slots), so the rejection comes from negotiation rather than from the
    # descriptor's own consistency check. 128 and 256 are accepted.
    yield (
        "wide_features",
        with_arch(invariant_features=192, attention_heads=6),
        "architecture.invariant_features = 192",
    )
    # 32 and 64 are accepted.
    yield (
        "large_radial_basis",
        with_arch(radial_basis_features=48),
        "architecture.radial_basis_features = 48",
    )
    yield (
        "unknown_architecture_key",
        with_arch(kspace_electrostatics="pme"),
        "architecture.kspace_electrostatics",
    )
    yield (
        "unsupported_module_type",
        lambda m: m["architecture"]["module_type_registry"].update(
            {"so3lr.mlff.nn.layer.RMSNorm": 3}
        ),
        "so3lr.mlff.nn.layer.RMSNorm",
    )
    yield (
        "unknown_output_head",
        lambda m: m["architecture"]["output_heads"].append("quadrupole_output_block"),
        "quadrupole_output_block",
    )
    yield (
        "future_schema",
        lambda m: m.update({"schema": "so3lr-native-model-v3"}),
        "unsupported manifest schema",
    )


def main():
    if len(sys.argv) < 3:
        raise SystemExit("usage: test_capability_rejection.py <probe> <model.so3lr>")
    probe, model_path = sys.argv[1], sys.argv[2]

    version, flags, manifest, payload, reserved = read_model(model_path)

    # The unmodified model must still be accepted, otherwise nothing below
    # distinguishes a working negotiation from a broken loader.
    result = subprocess.run([probe, model_path], capture_output=True, text=True)
    if result.returncode != 0:
        print("baseline model was rejected:\n" + result.stdout)
        raise SystemExit("capability_rejection=FAIL")

    failures = []
    with tempfile.TemporaryDirectory() as work:
        for name, mutate, expected in variants(manifest):
            patched = json.loads(json.dumps(manifest))
            mutate(patched)
            out = Path(work) / f"{name}.so3lr"
            write_model(out, version, flags, patched, payload, reserved)

            result = subprocess.run([probe, str(out)], capture_output=True, text=True)
            if result.returncode == 0:
                failures.append(f"{name}: accepted but should have been refused")
            elif expected not in result.stdout:
                failures.append(
                    f"{name}: message does not mention {expected!r}\n"
                    + "    " + result.stdout.replace("\n", "\n    ")
                )
            else:
                print(f"  ok  {name}")

    if failures:
        for failure in failures:
            print("  FAIL " + failure)
        raise SystemExit("capability_rejection=FAIL")
    print("capability_rejection=PASS")


if __name__ == "__main__":
    main()
