#!/usr/bin/env python3
"""Export a JAX/flax SO3LR checkpoint directly to `.so3lr`.

Reads the flax parameters and the training configuration and writes a
self-describing `.so3lr` file: every tensor, the physical reference tables and
an architecture record. Whether this build has kernels for that architecture
is answered separately, by capability negotiation when the model is loaded
(`so3lr_capability_probe` answers it without LAMMPS).

`--verify-against` compares every tensor with an existing `.so3lr` file.

usage:
    export_flax_so3lr.py --model so3lr --output so3lr.so3lr
    export_flax_so3lr.py --model /path/to/workdir --output my-model.so3lr
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import pickle
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from so3lr_native_format import read_native_model, write_native_model  # noqa: E402

EXPORTER_VERSION = "so3lr-lammps-0.3.0-flax-exporter-v1"
SCHEMA = "so3lr-native-model-v2"

# Torch Linear stores (out, in) where flax stores (in, out); these entries are
# element embeddings rather than linear kernels, so they must not be transposed.
NO_TRANSPOSE = {
    "charge_embedding.Wk", "charge_embedding.Wv",
    "spin_embedding.Wk", "spin_embedding.Wv",
    "partial_charges_output_block.atomic_embedding.weight",
    "hirshfeld_output_block.v_shift_embedding.weight",
    "hirshfeld_output_block.q_embedding.weight",
    "c6_ratios_output_block.v_shift_embedding.weight",
}
# flax indexes elements from Z=0; the native runtime indexes from Z=1.
DROP_LEADING_ELEMENT = {
    "inv_feature_embedding.embedding.weight",
    "charge_embedding.Wq.weight",
    "spin_embedding.Wq.weight",
}


def fail(message: str) -> None:
    raise SystemExit(f"export_flax_so3lr: {message}")


def flatten(tree: dict, prefix: str = "") -> dict[str, np.ndarray]:
    flat: dict[str, np.ndarray] = {}
    for key, value in tree.items():
        path = f"{prefix}/{key}" if prefix else key
        if hasattr(value, "items"):
            flat.update(flatten(value, path))
        else:
            flat[path] = np.asarray(value, dtype=np.float64)
    return flat


def load_checkpoint(model: str) -> tuple[dict, dict[str, np.ndarray]]:
    """Resolve a bundled model name or a directory and load config + params."""
    try:
        from so3lr.model_registry import resolve_model
        workdir = resolve_model(model)[0]
    except Exception:
        workdir = pathlib.Path(model).expanduser().resolve()
    workdir = pathlib.Path(workdir)
    if not workdir.is_dir():
        fail(f"{workdir} is not a model directory")

    hyper = workdir / "hyperparameters.json"
    if not hyper.is_file():
        fail(f"{hyper} is missing")
    config = json.loads(hyper.read_text())

    params_file = workdir / "params.pkl"
    if not params_file.is_file():
        fail(f"{params_file} is missing (orbax-only checkpoints are not supported)")
    with params_file.open("rb") as handle:
        raw = pickle.load(handle)

    # s/m/l pickles carry a top-level `adaptive_robust_loss` collection. It is a
    # training artefact, not part of the model, and must not be exported.
    if "params" not in raw:
        fail("checkpoint has no 'params' collection")
    return config, flatten({"params": raw["params"]})


def name_mapping(config: dict) -> dict[str, str]:
    """flax parameter path -> native state key (without the `model.` prefix).

    Reuses the mapping table maintained alongside the PyTorch port, which is
    already generic over depth and over the legacy feature flags. Removing this
    dependency is the remaining half of the direct-exporter work; it is a pure
    name table, and nothing it returns is used at MD time.
    """
    try:
        import ml_collections
        from so3krates_torch.tools.jax_torch_conversion import get_flax_to_torch_mapping
    except ImportError as error:
        fail(
            "the flax->native name table currently comes from so3krates_torch "
            f"({error}). Install it, or point PYTHONPATH at its src/ directory."
        )
    model = dict(config["model"])
    model.setdefault("energy_learn_atomic_type_shifts", False)
    model.setdefault("energy_learn_atomic_type_scales", False)
    model.setdefault("electrostatic_energy_bool", True)
    model.setdefault("dispersion_energy_bool", True)
    model.setdefault("zbl_repulsion_bool", True)
    cfg = ml_collections.ConfigDict({"model": model})
    mapping = get_flax_to_torch_mapping(
        cfg, trainable_rbf=bool(model.get("trainable_rbf", False))
    )
    # Models with use_rms_norm use nn.RMSNorm(use_scale=False), which has no
    # parameters at all, so the scale/bias entries the shared table still emits
    # have nothing behind them in the checkpoint.
    if bool(model.get("use_rms_norm", False)):
        mapping = {k: v for k, v in mapping.items() if "layer_normalization" not in k}
    # The non-legacy Hirshfeld head lost its Embed_1 attention branch and its
    # final layer became scalar, so it is a different module and is emitted
    # explicitly rather than through the legacy table.
    # Non-legacy models replace the learnable ZBL term with the parameter-free NLH
    # table, so the ten softplus parameters are absent too.
    if not bool(model.get("legacy_so3lr_bool", False)):
        mapping = {k: v for k, v in mapping.items()
                   if not k.startswith("params/observables_2/")
                   and "/zbl_repulsion/" not in k}
    return mapping


def reshape(flax_key: str, state_key: str, array: np.ndarray,
            theory_level: int | None) -> np.ndarray:
    """Apply the flax -> native layout rules."""
    # The per-element energy tables are indexed (Z, theory_level) rather than
    # (in, out), so they are neither transposed nor sliced like a kernel.
    if flax_key.endswith("/energy_offset") or flax_key.endswith("/atomic_scales"):
        if array.ndim == 2:
            # Multi-theory-level head -> scalar head. The one-hot theory mask
            # commutes with the per-channel scale and offset, so selecting the
            # production channel here is exact in float64 and leaves a head
            # structurally identical to the single-channel one.
            if theory_level is None:
                fail(f"{flax_key} has {array.shape[1]} channels but no theory level")
            if not 0 <= theory_level < array.shape[1]:
                fail(f"theory level {theory_level} is outside 0..{array.shape[1] - 1}")
            array = array[:, theory_level]
        array = array[1:]  # flax indexes from Z=0, the runtime from Z=1
        if flax_key.endswith("/atomic_scales"):
            array = array.reshape(1, -1)
        return np.ascontiguousarray(array, dtype=np.float64)

    if array.ndim == 2 and state_key not in NO_TRANSPOSE:
        array = array.T
    elif array.ndim == 3:
        array = array.transpose(0, 2, 1)

    if state_key in DROP_LEADING_ELEMENT:
        array = array[:, 1:]

    if theory_level is not None and flax_key.endswith("/energy_dense_final/kernel"):
        if array.shape[0] > 1:
            array = array[theory_level:theory_level + 1, :]
    return np.ascontiguousarray(array, dtype=np.float64)


def derived_constants(config: dict) -> list[tuple[str, np.ndarray]]:
    """Regenerate the buffers the torch exporter copied out of the port.

    These are pure functions of `degrees` and `num_radial_basis_fn`, and the
    reference package can produce them, so the port is not their source of
    truth.
    """
    import jax
    jax.config.update("jax_enable_x64", True)
    import itertools as it
    from so3lr.mlff.sph_ops.contract import init_clebsch_gordan_matrix, indx_fn
    from so3lr.mlff.utils.radial_basis_fn import log_binomial_coefficient

    model = config["model"]
    degrees = list(model["degrees"])
    n_rbf = int(model["num_radial_basis_fn"])

    cg = np.diagonal(
        np.asarray(init_clebsch_gordan_matrix(degrees=list({0, *degrees}), l_out_max=0)),
        axis1=1, axis2=2,
    )[0]
    pieces = []
    for degree, repeat in zip(*np.unique(np.array(degrees), return_counts=True)):
        pieces.append(np.tile(cg[indx_fn(degree - 1):indx_fn(degree)], repeat))
    cg_rep = np.concatenate(pieces).astype(np.float64)

    segment_ids = np.array(
        list(it.chain(*[[n] * int(2 * degrees[n] + 1) for n in range(len(degrees))])),
        dtype=np.int64,
    )
    segment_sum = np.zeros((cg_rep.size, len(degrees)), dtype=np.float64)
    segment_sum[np.arange(cg_rep.size), segment_ids] = 1.0
    degree_repeats = np.array([2 * d + 1 for d in degrees], dtype=np.int64)

    bernstein_b = np.array(
        [log_binomial_coefficient(n_rbf - 1, x) for x in range(n_rbf)], dtype=np.float64
    )
    return [
        ("cg_rep", cg_rep),
        ("segment_ids", segment_ids),
        ("segment_sum_matrix", segment_sum),
        ("degree_repeats", degree_repeats),
        ("bernstein_b", bernstein_b),
        ("bernstein_k", np.arange(n_rbf, dtype=np.int64)),
        ("bernstein_k_rev", np.arange(n_rbf, dtype=np.int64)[::-1].copy()),
    ]


def softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def physical_tables(config: dict) -> list[tuple[str, np.ndarray]]:
    """Free-atom dispersion references, plus NLH tables when that term is used.

    Exported so the `.so3lr` file is self-contained: no reference package is
    needed at MD time.
    """
    import so3lr.mlff.nn.observable.dispersion_ref_data as ref
    legacy = bool(config["model"].get("legacy_so3lr_bool", False))
    tables = [
        ("physical.reference_alphas",
         np.asarray(ref.alphas_legacy if legacy else ref.alphas, dtype=np.float64)),
        ("physical.reference_c6",
         np.asarray(ref.C6_coef_legacy if legacy else ref.C6_coef, dtype=np.float64)),
    ]
    if not legacy:
        tables += [
            ("physical.nlh_a", np.asarray(ref.NLH_AA, dtype=np.float64)),
            ("physical.nlh_b", np.asarray(ref.NLH_BB, dtype=np.float64)),
        ]
    return tables


def architecture(config: dict, flat: dict[str, np.ndarray], theory_level: int | None,
                 lr_cutoff: float, lr_damping: float) -> dict[str, Any]:
    model = config["model"]
    data = config.get("data", {})
    # `avg_num_neighbors` lives in the DATA section, not the model section --
    # like `energy_shifts`. It is the divisor the attention kernels apply, so
    # silently defaulting it to 1 would scale every message by ~13x and be
    # visible only as a few-percent energy error. Fail instead.
    normalization = str(model.get("message_normalization", "avg_num_neighbors"))
    avg_num_neighbors = data.get("avg_num_neighbors")
    if normalization == "avg_num_neighbors" and avg_num_neighbors is None:
        fail("message_normalization is 'avg_num_neighbors' but the checkpoint's "
             "hyperparameters carry no data.avg_num_neighbors")
    degrees = list(model["degrees"])
    legacy = bool(model.get("legacy_so3lr_bool", False))
    inv_weight = flat["params/feature_embeddings_0/Embed_0/embedding"]

    heads = ["atomic_energy_output_block", "partial_charges_output_block",
             "hirshfeld_output_block"]
    if "params/observables_3/Embed_0/embedding" in flat:
        heads.append("c6_ratios_output_block")

    arch: dict[str, Any] = {
        "interaction_blocks": int(model["num_layers"]),
        "invariant_features": int(model["num_features"]),
        "attention_heads": int(model["num_heads"]),
        "attention_head_width": int(model["num_features_head"]),
        "euclidean_degree_channels": len(degrees),
        "radial_basis_features": int(model["num_radial_basis_fn"]),
        "atomic_number_capacity": int(inv_weight.shape[0]) - 1,
        "num_embeddings": 1 + int(bool(model.get("use_charge_embed")))
                            + int(bool(model.get("use_spin_embed"))),
        "short_range_cutoff_angstrom": float(model["cutoff"]),
        "long_range_cutoff_angstrom": float(lr_cutoff),
        "output_heads": heads,
        # --- optional operator flags; absent from a legacy manifest, so the runtime's
        # --- descriptor defaults them off and negotiation names any that are on.
        "use_rms_norm": bool(model.get("use_rms_norm", False)),
        "qk_norm": bool(model.get("qk_norm", False)),
        "use_residual_scalars": bool(model.get("use_residual_scalars", False)),
        "layer_normalization_1": bool(model.get("layer_normalization_1", True)),
        "layer_normalization_2": bool(model.get("layer_normalization_2", True)),
        "residual_mlp_1": bool(model.get("residual_mlp_1", True)),
        "residual_mlp_2": bool(model.get("residual_mlp_2", False)),
        "layer_norm_epsilon": 1.0e-6,
        "qk_nonlinearity": str(model.get("qk_non_linearity", "identity")),
        "repulsion_type": "zbl_learned" if legacy else "nlh",
        "zbl_enabled": 1 if legacy else 0,
        "zbl_cutoff_function": str(model.get("cutoff_fn", "phys")),
        "zbl_switch_off_angstrom": 1.5,
        "zbl_ke": 14.399645351950548,
        "lr_electrostatic_ke": 14.399645351950548,
        "lr_electrostatic_sigma": float(model.get("electrostatic_energy_scale", 4.0)),
        "lr_electrostatic_cuton_angstrom": 0.45 * float(lr_cutoff),
        "lr_fine_structure": 0.0072973525693,
        "lr_bohr_angstrom": 0.5291772105638411,
        "lr_hartree_ev": 27.211386245988,
        "lr_dispersion_scale": float(model.get("dispersion_energy_scale", 1.2)),
        "lr_dispersion_cuton_angstrom": float(lr_cutoff) - float(lr_damping),
        "lr_pair_scale": 0.5,
        "embedding_scale": float(np.sqrt(
            1 + int(bool(model.get("use_charge_embed")))
              + int(bool(model.get("use_spin_embed"))))),
        "message_normalization": normalization,
        "avg_num_neighbors": float(avg_num_neighbors if avg_num_neighbors is not None else 1.0),
        "runtime_element_mapping": True,
        "native_long_range_terms": ["electrostatics", "dispersion"],
        "native_long_range_observables": ["partial_charges", "hirshfeld_ratios"],
        "native_short_range_physical_terms": ["zbl_repulsion"],
        "physical_reference_table_state_keys": [
            "physical.reference_alphas", "physical.reference_c6"],
    }
    if theory_level is not None:
        arch["energy_theory_level_selected"] = int(theory_level)

    if legacy:
        # Decode the learnable ZBL parameters exactly as the reference does:
        # softplus, then normalise the c coefficients to sum to one.
        def par(name: str) -> float:
            return float(softplus(flat[f"params/observables_0/zbl_repulsion/{name}"])[0])
        cs = [par(f"c{i}") for i in (1, 2, 3, 4)]
        total = sum(cs)
        for i in (1, 2, 3, 4):
            arch[f"zbl_a{i}"] = par(f"a{i}")
            arch[f"zbl_c{i}"] = cs[i - 1] / total
        arch["zbl_p"] = par("p")
        arch["zbl_d"] = par("d")
    return arch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True,
                        help="name of a checkpoint bundled with the so3lr package, or a work directory")
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--theory-level", type=int, default=5,
                        help="energy head channel to collapse to, for multi-head energy "
                             "outputs (default 5)")
    parser.add_argument("--lr-cutoff", type=float, default=12.0,
                        help="long-range cutoff in Angstrom; the reference calculator "
                             "overrides the trained value with this (default 12.0)")
    parser.add_argument("--lr-damping", type=float, default=2.0)
    parser.add_argument("--legacy-float32-constants", action="store_true",
                        help="round derived and physical constants through float32, as the "
                             "PyTorch exporter did. Needed to reproduce a 0.2.0-rc1 file "
                             "exactly; otherwise they are emitted in full float64.")
    parser.add_argument("--verify-against", type=pathlib.Path,
                        help="an existing .so3lr whose tensors must match exactly")
    args = parser.parse_args()

    config, flat = load_checkpoint(args.model)
    mapping = name_mapping(config)
    model = config["model"]
    multi_head = not bool(model.get("legacy_so3lr_bool", False))
    theory_level = args.theory_level if multi_head else None

    claimed: set[str] = set()
    tensors: list[tuple[dict[str, Any], bytes]] = []

    def emit(state_key: str, array: np.ndarray, role: str) -> None:
        # np.ascontiguousarray promotes 0-d to 1-d; scalars must stay rank 0.
        array = array if array.ndim == 0 else np.ascontiguousarray(array)
        dtype = "int64" if array.dtype.kind in "iu" else "float64"
        array = array.astype(np.int64 if dtype == "int64" else np.float64)
        tensors.append((
            {"name": f"wrapper.model.{state_key}", "state_key": state_key,
             "dtype": dtype, "role": role, "shape": list(array.shape)},
            array.tobytes(order="C"),
        ))

    # The per-element energy offset is split across two places: `energy_offset`
    # in the parameter tree (all zeros for every bundled model) and
    # `data.energy_shifts` in the training config. The SO3LR checkpoint
    # carries -163.6181213551537 eV/atom there and zero in the params, which is
    # why the shipped native model's energy_shifts is nonzero while the flax
    # checkpoint's energy_offset is not. Both are additive per-element offsets,
    # so the native tensor is their sum.
    shifts = config.get("data", {}).get("energy_shifts", {})
    shift_vector = np.array(
        [float(shifts.get(str(z), shifts.get(z, 0.0))) for z in range(1, 119)],
        dtype=np.float64,
    )

    for flax_key, state_key in sorted(mapping.items()):
        if flax_key not in flat:
            # `use_final_bias_bool: false` removes the bias from the checkpoint.
            # The native energy head always applies one, and a disabled bias is
            # exactly a zero bias, so synthesising it is numerically exact.
            if (flax_key == "params/observables_0/energy_dense_final/bias"
                    and not model.get("use_final_bias_bool", True)):
                emit(f"model.{state_key}", np.zeros(1, dtype=np.float64), "parameter")
                continue
            fail(f"mapped parameter {flax_key} is absent from the checkpoint")
        claimed.add(flax_key)
        array = reshape(flax_key, state_key, flat[flax_key], theory_level)
        if state_key == "atomic_energy_output_block.energy_shifts":
            if array.shape != shift_vector.shape:
                fail(f"energy_shifts shape {array.shape} != {shift_vector.shape}")
            array = array + shift_vector
        emit(f"model.{state_key}", array, "parameter")
        # The PyTorch state dict exposes each FilterNet under two attribute
        # paths, and different native kernels read different ones, so both must
        # be present.
        if ".filter_net_" in state_key:
            head, sep, tail = state_key.partition(".filter_net_")
            emit(f"model.{head}.euclidean_attention_block{sep}{tail}", array, "parameter")

    # Parameters the shared name table does not cover.
    for flax_key in sorted(flat):
        if flax_key in claimed:
            continue
        if flax_key in ("params/resid_lambdas", "params/x0_lambdas"):
            claimed.add(flax_key)
            emit(f"model.{flax_key.split('/')[-1]}", flat[flax_key], "parameter")
        elif flax_key.startswith("params/observables_2/"):
            claimed.add(flax_key)
            leaf = flax_key[len("params/observables_2/"):].replace("/", ".")
            state = {"Embed_0.embedding": "hirshfeld_output_block.v_shift_embedding.weight",
                     "a0_ratios_dense_regression.kernel": "hirshfeld_output_block.transform_features.0.weight",
                     "a0_ratios_dense_regression.bias": "hirshfeld_output_block.transform_features.0.bias",
                     "a0_ratios_dense_final.kernel": "hirshfeld_output_block.transform_features.2.weight",
                     "a0_ratios_dense_final.bias": "hirshfeld_output_block.transform_features.2.bias",
                     "hirshfeld_ratios_dense_regression.kernel": "hirshfeld_output_block.transform_features.0.weight",
                     "hirshfeld_ratios_dense_regression.bias": "hirshfeld_output_block.transform_features.0.bias",
                     "hirshfeld_ratios_dense_final.kernel": "hirshfeld_output_block.transform_features.2.weight",
                     "hirshfeld_ratios_dense_final.bias": "hirshfeld_output_block.transform_features.2.bias",
                     }.get(leaf)
            if state is None:
                fail(f"unhandled Hirshfeld/a0-head parameter {flax_key}")
            emit(f"model.{state}", reshape(flax_key, state, flat[flax_key], None), "parameter")
        elif flax_key.startswith("params/observables_3/"):
            claimed.add(flax_key)
            leaf = flax_key[len("params/observables_3/"):].replace("/", ".")
            state = {"Embed_0.embedding": "c6_ratios_output_block.v_shift_embedding.weight",
                     "c6_ratios_dense_regression.kernel": "c6_ratios_output_block.transform_features.0.weight",
                     "c6_ratios_dense_regression.bias": "c6_ratios_output_block.transform_features.0.bias",
                     "c6_ratios_dense_final.kernel": "c6_ratios_output_block.transform_features.2.weight",
                     "c6_ratios_dense_final.bias": "c6_ratios_output_block.transform_features.2.bias",
                     }.get(leaf)
            if state is None:
                fail(f"unhandled C6-head parameter {flax_key}")
            emit(f"model.{state}", reshape(flax_key, state, flat[flax_key], None), "parameter")

    # The native ratio-head kernel computes |v_shift[Z] + q[Z].key / sqrt(K)|.
    # a0 and C6 ratio heads are |key + q[Z]| with a scalar key, which is that
    # formula with K = 1 and q = ones -- so supply the constant q. (In the legacy
    # Hirshfeld head q is a learned (100, 64) embedding and comes from the
    # checkpoint instead.)
    if not bool(model.get("legacy_so3lr_bool", False)):
        for head in ("hirshfeld_output_block", "c6_ratios_output_block"):
            if head == "c6_ratios_output_block" and \
                    "params/observables_3/Embed_0/embedding" not in flat:
                continue
            emit(f"model.{head}.q_embedding.weight", np.ones((100, 1)), "derived_constant")

    # Gate: refuse to emit a partial model. A tensor nobody claimed means the
    # checkpoint contains something this exporter does not describe, and a
    # silently incomplete file is far worse than a failed conversion.
    unclaimed = sorted(set(flat) - claimed)
    if unclaimed:
        fail("unmapped parameters (refusing to write a partial model):\n  "
             + "\n  ".join(f"{k} {flat[k].shape}" for k in unclaimed))

    def maybe_f32(a: np.ndarray) -> np.ndarray:
        if args.legacy_float32_constants and a.dtype.kind == "f":
            return a.astype(np.float32).astype(np.float64)
        return a

    for name, array in derived_constants(config):
        array = maybe_f32(array)
        if name.startswith("bernstein_"):
            emit(f"model.radial_embedding.radial_basis_fn.{name[len('bernstein_'):]}",
                 array, "derived_constant")
        else:
            for block in range(int(model["num_layers"])):
                for sub in ("euclidean_attention_block", "interaction_block"):
                    key = (f"model.euclidean_transformers.{block}.{sub}."
                           + ("degree_repeats" if name == "degree_repeats"
                              else f"so3_conv_invariants.{name}"))
                    emit(key, array, "derived_constant")
    emit("model.radial_embedding.radial_basis_fn.gamma",
         maybe_f32(np.array(0.9448630629184640)), "derived_constant")
    emit("head", np.array([0], dtype=np.int64), "derived_constant")
    emit("r_max", np.array(float(model["cutoff"])), "derived_constant")
    emit("num_interactions", np.array(int(model["num_layers"]), dtype=np.int64),
         "derived_constant")
    emit("atomic_numbers_map", np.arange(1, 100, dtype=np.int64), "derived_constant")

    for name, table in physical_tables(config):
        emit(name, maybe_f32(table), "physical_constant")

    manifest_base = {
        "schema": SCHEMA,
        "exporter_version": EXPORTER_VERSION,
        "model_family": "SO3LR",
        "source": {
            "framework": "flax",
            "model": str(args.model),
            "hyperparameters_sha256": hashlib.sha256(
                json.dumps(config, sort_keys=True).encode()).hexdigest(),
        },
        "architecture": architecture(config, flat, theory_level,
                                     args.lr_cutoff, args.lr_damping),
        "modules": [],
    }
    manifest_base["architecture"]["module_type_registry"] = {}
    manifest_base["architecture"]["state_tensor_count"] = len(tensors)
    manifest_base["architecture"]["parameter_elements"] = int(
        sum(v.size for v in flat.values()))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = write_native_model(args.output, manifest_base, tensors)
    print(f"wrote {args.output}  ({manifest['tensor_count']} tensors, "
          f"{manifest['tensor_bytes']} payload bytes)")

    if args.verify_against:
        reference_manifest, reference_payload = read_native_model(args.verify_against)
        ref_by_key = {r["state_key"]: r for r in reference_manifest["tensors"]}
        mine, payload = read_native_model(args.output)
        mine_by_key = {r["state_key"]: r for r in mine["tensors"]}
        missing = sorted(set(ref_by_key) - set(mine_by_key))
        extra = sorted(set(mine_by_key) - set(ref_by_key))
        exact, f32_only, genuine = 0, [], []
        for key in sorted(set(ref_by_key) & set(mine_by_key)):
            a, b = ref_by_key[key], mine_by_key[key]
            va = np.frombuffer(reference_payload[a["offset"]:a["offset"] + a["nbytes"]],
                               dtype=np.int64 if a["dtype"] == "int64" else np.float64)
            vb = np.frombuffer(payload[b["offset"]:b["offset"] + b["nbytes"]],
                               dtype=np.int64 if b["dtype"] == "int64" else np.float64)
            if a["shape"] == b["shape"] and np.array_equal(va, vb):
                exact += 1
                continue
            if va.size != vb.size:
                genuine.append(f"{key}: shape {a['shape']} vs {b['shape']}")
                continue
            fa, fb = va.astype(float), vb.astype(float)
            worst = float(np.max(np.abs(fa - fb)))
            # Equal once both sides are rounded through float32 -> the
            # difference is the PyTorch export path's single-precision
            # constants, not a disagreement about the model.
            if np.array_equal(fa.astype(np.float32), fb.astype(np.float32)):
                f32_only.append(f"{key}: max|delta|={worst:.3e}")
            else:
                n = int((np.abs(fa - fb) > 0).sum())
                genuine.append(f"{key}: max|delta|={worst:.3e} over {n}/{va.size} entries")
        print(f"verify: {exact} exact, {len(f32_only)} float32-equivalent, "
              f"{len(genuine)} genuine, {len(missing)} missing, {len(extra)} extra")
        for line in missing[:10]:
            print(f"  missing: {line}")
        for line in extra[:10]:
            print(f"  extra:   {line}")
        for line in f32_only[:10]:
            print(f"  float32: {line}")
        for line in genuine[:15]:
            print(f"  GENUINE: {line}")
        ok = not genuine and not missing and not extra
        print("EXPORT_VERIFY=" + ("PASS" if ok else "FAIL"))
        if not ok:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
