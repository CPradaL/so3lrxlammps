#!/usr/bin/env python3
"""Float64 JAX reference for the native SO3LR runtime.

This script evaluates a LAMMPS data file with the reference implementation
(the JAX `so3lr` package) and writes the energy, per-atom forces and the
per-term decomposition as canonical JSON.

    so3lr_reference.py --model so3lr --data alanine.data \
        --types C H O N --output ref.json [--periodic] \
        [--charge Q] [--multiplicity M] [--float64-params]

Two things this script does deliberately:

* **Non-periodic systems get a zero cell.** glp's ``atoms_to_system`` keeps any
  nonzero cell and ignores ``atoms.pbc``, and ``system_to_graph`` then computes
  every edge vector with the minimum-image periodic displacement. A molecule
  read from a LAMMPS data file carries its box as a cell, so if that box is
  shorter than twice the long-range cutoff in any direction the long-range
  distances are silently wrong (0.31 eV on alanine with a shrink-wrapped cell).
  Zeroing the cell makes glp pass ``cell=None``.

* **The energy is reported both raw and with the config's per-element shift.**
  The JAX calculator does not apply ``data.energy_shifts``; the native model
  folds it into its energy head (it is -163.618 eV/atom for SO3LR).
  ``energy_native_convention`` is directly comparable to LAMMPS.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402

SYMBOL_TO_Z = {"H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
               "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
               "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Br": 35, "I": 53}

# Per-atom quantities requested from the model, when it produces them.
TERMS_HIRSHFELD = ["nn_energy", "zbl_repulsion", "electrostatic_energy", "dispersion_energy",
            "partial_charges", "hirshfeld_ratios"]
TERMS_A0_C6 = ["nn_energy", "zbl_repulsion", "electrostatic_energy", "dispersion_energy",
            "partial_charges", "a0_ratios", "c6_ratios"]


def load_atoms(data: pathlib.Path, types: list[str], periodic: bool):
    from ase.io import read
    z_of_type = {i + 1: SYMBOL_TO_Z[s.capitalize()] for i, s in enumerate(types)}
    atoms = read(str(data), format="lammps-data", Z_of_type=z_of_type, atom_style="atomic")
    if periodic:
        atoms.set_pbc(True)
        lengths = atoms.cell.lengths()
        if np.any(lengths < 24.0):
            raise SystemExit(
                f"periodic cell {lengths.round(2)} is shorter than 2 x 12 A in some "
                "direction; the minimum-image convention would be violated")
    else:
        atoms.set_pbc(False)
        atoms.set_cell(np.zeros((3, 3)))  # see module docstring
    return atoms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True, type=pathlib.Path)
    parser.add_argument("--types", required=True, nargs="+",
                        help="element symbol of each LAMMPS atom type, in type order")
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--periodic", action="store_true")
    parser.add_argument("--charge", type=int, default=0, help="total charge")
    parser.add_argument("--multiplicity", type=int, default=1, help="spin multiplicity")
    parser.add_argument("--float64-params", action="store_true",
                        help="cast the float32 checkpoint to float64 before use (see below)")
    parser.add_argument("--lr-cutoff", type=float, default=12.0)
    parser.add_argument("--lr-damping", type=float, default=2.0)
    args = parser.parse_args()

    if args.float64_params:
        # The checkpoint is float32. Dense layers promote it against float64
        # activations, but embedding lookups do not: ChargeSpinEmbedSparse then
        # forms q.k and its softplus in float32 (relative error ~1e-7), the
        # reference's own floor for charged/open-shell systems. Casting the
        # loaded parameters removes it. The values are unchanged (float32 is
        # exact in float64), so neutral-singlet results are identical.
        import pickle
        _load = pickle.load
        def _load_float64(*a, **k):
            obj = _load(*a, **k)
            return jax.tree_util.tree_map(
                lambda x: np.asarray(x, np.float64)
                if hasattr(x, "dtype") and np.issubdtype(x.dtype, np.floating) else x, obj)
        pickle.load = _load_float64

    from so3lr.ase_utils import make_ase_calculator
    from so3lr.model_registry import resolve_model

    atoms = load_atoms(args.data, args.types, args.periodic)
    # glp's atoms_to_system reads these two keys (num_unpaired = M - 1).
    atoms.info["charge"] = args.charge
    atoms.info["multiplicity"] = args.multiplicity
    workdir = pathlib.Path(resolve_model(args.model)[0])
    config = json.loads((workdir / "hyperparameters.json").read_text())
    legacy = bool(config["model"].get("legacy_so3lr_bool", False))
    terms = TERMS_HIRSHFELD if legacy else TERMS_A0_C6

    calc = make_ase_calculator(model=args.model, dtype=np.float64,
                               lr_cutoff=args.lr_cutoff,
                               dispersion_energy_cutoff_lr_damping=args.lr_damping,
                               observables=terms, has_aux=True)
    atoms.calc = calc
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=np.float64)

    shifts = config.get("data", {}).get("energy_shifts", {})
    z = atoms.get_atomic_numbers()
    shift_per_atom = [float(shifts.get(str(int(zi)), 0.0)) for zi in z]
    shift_total = float(sum(shift_per_atom))

    per_atom = {}
    for key in terms:
        value = calc.results.get(key)
        if value is None and isinstance(calc.results.get("aux"), dict):
            value = calc.results["aux"].get(key)
        if value is not None:
            per_atom[key] = np.asarray(value, dtype=np.float64).reshape(-1).tolist()

    result = {
        "schema": "so3lr-jaxref-v1",
        "model": args.model,
        "data": str(args.data),
        "types": args.types,
        "periodic": bool(args.periodic),
        "charge": args.charge,
        "multiplicity": args.multiplicity,
        "float64_params": bool(args.float64_params),
        "natoms": int(len(atoms)),
        "atomic_numbers": [int(v) for v in z],
        "lr_cutoff": args.lr_cutoff,
        "lr_damping": args.lr_damping,
        "energy": energy,
        "energy_shift_total": shift_total,
        "energy_shift_per_atom": shift_per_atom,
        "energy_native_convention": energy + shift_total,
        "forces": forces.tolist(),
        "per_atom": per_atom,
        "term_totals": {k: float(np.sum(v)) for k, v in per_atom.items()
                        if k.endswith("energy") or k == "zbl_repulsion"},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")))
    print(f"{args.model}: E={energy:.10f} eV (native convention {energy + shift_total:.10f}), "
          f"|F|max={np.abs(forces).max():.6f}, terms={sorted(result['term_totals'])}")


if __name__ == "__main__":
    main()
