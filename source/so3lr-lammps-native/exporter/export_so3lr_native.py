#!/usr/bin/env python3
"""Export a PyTorch SO3LR/ML-IAP checkpoint into native format v1."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from collections import Counter
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
from so3lr_native_format import write_native_model


EXPORTER_VERSION = "stage3-dev56-native-zbl"
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


def qualified(value: Any) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


def simple_value(value: Any, depth: int = 0) -> Any:
    if depth > 3:
        raise TypeError
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if torch.is_tensor(value) and value.numel() <= 64:
        cpu = value.detach().cpu()
        return {"dtype": str(cpu.dtype).removeprefix("torch."), "shape": list(cpu.shape), "value": cpu.tolist()}
    if isinstance(value, (list, tuple)) and len(value) <= 64:
        return [simple_value(item, depth + 1) for item in value]
    if isinstance(value, dict) and len(value) <= 64:
        return {str(key): simple_value(item, depth + 1) for key, item in value.items()}
    raise TypeError


def discover_roots(wrapper: Any) -> list[tuple[str, nn.Module]]:
    roots: list[tuple[str, nn.Module]] = []
    visited = {id(wrapper)}

    def visit(value: Any, path: str) -> None:
        if id(value) in visited:
            return
        visited.add(id(value))
        if isinstance(value, nn.Module):
            roots.append((path, value))
        elif isinstance(value, dict):
            for key, child in value.items():
                visit(child, f"{path}[{key!r}]")
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    for name, value in vars(wrapper).items():
        visit(value, f"wrapper.{name}")
    return roots


def public_attributes(value: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, item in sorted(vars(value).items()):
        if name.startswith("_") or isinstance(item, nn.Module) or callable(item):
            continue
        try:
            result[name] = simple_value(item)
        except TypeError:
            continue
    for name in ("r_max", "long_range", "num_interactions", "degrees", "max_degree"):
        if name not in result and hasattr(value, name):
            try:
                result[name] = simple_value(getattr(value, name))
            except TypeError:
                pass
    return result


def tensor_raw(tensor: torch.Tensor) -> bytes:
    if sys.byteorder != "little":
        raise RuntimeError("version-1 exporter currently requires a little-endian host")
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def require_scalar(value: Any, name: str) -> float:
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise RuntimeError(f"{name} is not scalar")
        return float(value.detach().cpu())
    return float(value)


def architecture_contract(root: nn.Module, modules: list[dict[str, Any]]) -> dict[str, Any]:
    model = None
    wrapper = root
    for _, module in root.named_modules():
        if qualified(module).endswith(".SO3LR"):
            model = module
            break
    if model is None:
        raise RuntimeError("serialized wrapper does not contain an SO3LR module")
    transformers = getattr(model, "euclidean_transformers", None)
    if transformers is None:
        raise RuntimeError("SO3LR model has no euclidean_transformers")
    attention = transformers[0].euclidean_attention_block
    inv_weight = model.inv_feature_embedding.embedding.weight
    degree_repeats = attention.degree_repeats
    wq = attention.W_q_inv
    state = root.state_dict()
    output_heads = []
    for name in ("atomic_energy_output_block", "partial_charges_output_block", "hirshfeld_output_block"):
        if hasattr(model, name) and getattr(model, name) is not None:
            output_heads.append(name)
    module_types = Counter(item["type"] for item in modules)
    electrostatic = model.electrostatic_potential
    dispersion = model.dispersion_potential
    if not bool(getattr(model, "zbl_repulsion_bool", False)):
        raise RuntimeError("native dev_56 requires an enabled ZBL term")
    if str(model.cutoff_fn_name) != "phys":
        raise RuntimeError(
            "native dev_56 currently supports the PhysNet SR cutoff only"
        )
    zbl = model.zbl_repulsion
    positive = lambda name: require_scalar(
        torch.nn.functional.softplus(getattr(zbl, name + "_raw")),
        "zbl." + name,
    )
    zbl_a = [positive(f"a{index}") for index in range(1, 5)]
    zbl_c = [positive(f"c{index}") for index in range(1, 5)]
    zbl_c_sum = sum(zbl_c)
    zbl_c = [value / zbl_c_sum for value in zbl_c]
    cutoff_lr = require_scalar(model.r_max_lr, "r_max_lr")
    dispersion_damping = require_scalar(
        model.dispersion_energy_cutoff_lr_damping,
        "dispersion_energy_cutoff_lr_damping",
    )
    return {
        "short_range_cutoff_angstrom": require_scalar(model.r_max, "r_max"),
        "long_range_cutoff_angstrom": require_scalar(wrapper.long_range, "long_range"),
        "interaction_blocks": len(transformers),
        "embedding_scale": require_scalar(
            model.embedding_scale, "embedding_scale"
        ),
        "num_embeddings": int(model.num_embeddings),
        "invariant_features": int(inv_weight.shape[0]),
        "atomic_number_capacity": int(inv_weight.shape[1]),
        "attention_heads": int(wq.shape[0]),
        "attention_head_width": int(wq.shape[1]),
        "euclidean_degree_channels": int(degree_repeats.numel()),
        "radial_basis_features": int(transformers[0].filter_net_inv.mlp_rbf[0].weight.shape[1]),
        "output_heads": output_heads,
        "native_long_range_observables": ["partial_charges", "hirshfeld_ratios"],
        "native_long_range_terms": ["electrostatics", "dispersion"],
        "native_short_range_physical_terms": ["zbl_repulsion"],
        "zbl_enabled": 1,
        "zbl_cutoff_function": str(model.cutoff_fn_name),
        "zbl_ke": require_scalar(zbl.ke, "zbl.ke"),
        "zbl_switch_off_angstrom": 1.5,
        "zbl_a1": zbl_a[0],
        "zbl_a2": zbl_a[1],
        "zbl_a3": zbl_a[2],
        "zbl_a4": zbl_a[3],
        "zbl_c1": zbl_c[0],
        "zbl_c2": zbl_c[1],
        "zbl_c3": zbl_c[2],
        "zbl_c4": zbl_c[3],
        "zbl_p": positive("p"),
        "zbl_d": positive("d"),
        "physical_reference_table_state_keys": [
            "physical.reference_alphas",
            "physical.reference_c6",
        ],
        "physical_reference_table_tensor_count": 2,
        "lr_electrostatic_ke": require_scalar(electrostatic.ke, "electrostatic.ke"),
        "lr_electrostatic_sigma": require_scalar(
            model.electrostatic_energy_scale, "electrostatic_energy_scale"
        ),
        "lr_electrostatic_cuton_angstrom": 0.45 * cutoff_lr,
        "lr_fine_structure": float(FINE_STRUCTURE),
        "lr_bohr_angstrom": float(BOHR),
        "lr_hartree_ev": float(HARTREE),
        "lr_dispersion_scale": require_scalar(
            model.dispersion_energy_scale, "dispersion_energy_scale"
        ),
        "lr_dispersion_cuton_angstrom": cutoff_lr - dispersion_damping,
        "lr_pair_scale": require_scalar(dispersion.c, "dispersion.c"),
        "runtime_element_mapping": True,
        "parameter_elements": sum(parameter.numel() for parameter in root.parameters()),
        "state_tensor_count": len(state),
        "module_type_registry": dict(sorted(module_types.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()

    wrapper = torch.load(args.model, map_location="cpu", weights_only=False)
    roots = discover_roots(wrapper)
    if len(roots) != 1:
        raise SystemExit(f"expected one root nn.Module, found {len(roots)}")
    root_path, root = roots[0]

    modules: list[dict[str, Any]] = []
    for name, module in root.named_modules():
        path = root_path if not name else f"{root_path}.{name}"
        try:
            signature = str(inspect.signature(module.forward))
        except (TypeError, ValueError):
            signature = "unavailable"
        modules.append({
            "path": path,
            "type": qualified(module),
            "children": [child_name for child_name, _ in module.named_children()],
            "attributes": public_attributes(module),
            "parameters_direct": [name for name, _ in module.named_parameters(recurse=False)],
            "buffers_direct": [name for name, _ in module.named_buffers(recurse=False)],
            "forward_signature": signature,
        })

    tensor_items: list[tuple[dict[str, Any], bytes]] = []
    for name, tensor in sorted(root.state_dict().items()):
        dtype = TORCH_DTYPES.get(tensor.dtype)
        if dtype is None:
            raise SystemExit(f"unsupported tensor dtype {tensor.dtype} for {name}")
        full_name = f"{root_path}.{name}"
        role = "buffer"
        parameter_names = {f"{root_path}.{key}" for key, _ in root.named_parameters()}
        if full_name in parameter_names:
            role = "parameter"
        tensor_items.append(({
            "name": full_name,
            "state_key": name,
            "role": role,
            "dtype": dtype,
            "shape": list(tensor.shape),
        }, tensor_raw(tensor)))

    for state_key, tensor in (
        ("physical.reference_alphas", ALPHAS.to(dtype=torch.float64)),
        ("physical.reference_c6", C6_COEF.to(dtype=torch.float64)),
    ):
        tensor_items.append(({
            "name": f"wrapper.physical.{state_key.removeprefix('physical.')}",
            "state_key": state_key,
            "role": "physical_constant",
            "dtype": "float64",
            "shape": list(tensor.shape),
        }, tensor_raw(tensor)))

    manifest_base = {
        "schema": "so3lr-native-model-v2",
        "exporter_version": EXPORTER_VERSION,
        "model_family": "SO3LR",
        "source": {
            "checkpoint_name": args.model.name,
            "checkpoint_sha256": file_sha256(args.model),
            "implementation_commit": args.source_commit,
            "wrapper_type": qualified(wrapper),
            "root_path": root_path,
            "root_type": qualified(root),
            "torch_version": torch.__version__,
        },
        "architecture": architecture_contract(root, modules),
        "wrapper_attributes": public_attributes(wrapper),
        "modules": modules,
    }
    manifest = write_native_model(args.output, manifest_base, tensor_items)
    print(f"output={args.output}")
    print(f"output_sha256={file_sha256(args.output)}")
    print(f"modules={len(modules)}")
    print(f"tensors={manifest['tensor_count']}")
    print(f"tensor_bytes={manifest['tensor_bytes']}")
    print("physical_reference_tables=2")
    print("self_contained_physical_model=1")
    print("so3lr_native_export=PASS")


if __name__ == "__main__":
    main()
