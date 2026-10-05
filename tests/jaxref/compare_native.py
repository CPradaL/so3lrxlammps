#!/usr/bin/env python3
"""Compare a native LAMMPS single point against the JAX reference.

    compare_native.py ref.json forces.dump lammps.log [--tol-force 1e-9]

Energy is compared in the native convention (JAX energy plus the config's
per-element shift, which the native model folds into its energy head).
"""
import argparse
import json
import re
import sys

import numpy as np


def read_dump(path):
    lines = open(path).read().split("\n")
    start = next(i for i, l in enumerate(lines) if l.startswith("ITEM: ATOMS"))
    cols = lines[start].split()[2:]
    rows = [l.split() for l in lines[start + 1:] if l.strip()]
    table = {c: np.array([float(r[k]) for r in rows]) for k, c in enumerate(cols)}
    order = np.argsort(table["id"])
    return np.stack([table["fx"][order], table["fy"][order], table["fz"][order]], axis=1)


def read_pe(path):
    m = re.findall(r"SO3LR_SINGLE_POINT_PE\s+(\S+)", open(path).read())
    if not m:
        sys.exit(f"no SO3LR_SINGLE_POINT_PE line in {path}")
    return float(m[-1])


def compare_terms(ref, dbg):
    """Per-atom decomposition: localises a failure to one component."""
    order = np.argsort(np.asarray(dbg["tag"]))
    nat = {k: np.asarray(v)[order] for k, v in dbg.items() if isinstance(v, list)}
    pa = {k: np.asarray(v) for k, v in ref["per_atom"].items()}
    shift = np.asarray(ref.get("energy_shift_per_atom", np.zeros(ref["natoms"])))

    def row(label, native, reference):
        d = native - reference
        scale = max(np.abs(reference).max(), 1e-300)
        print(f"    {label:28s} max|d| = {np.abs(d).max():.3e}   rel = {np.abs(d).max()/scale:.3e}")

    print("  per-atom terms (native vs reference):")
    row("network energy", nat["learned_energy"] - shift, pa["nn_energy"])
    row("repulsion", nat["zbl_energy"], pa["zbl_repulsion"])
    row("partial charges", nat["partial_charges"], pa["partial_charges"])
    ratio_key = "hirshfeld_ratios" if "hirshfeld_ratios" in pa else "a0_ratios"
    row(f"{ratio_key}", nat["hirshfeld_ratios"], pa[ratio_key])
    if "c6_ratios" in pa and "c6_ratios" in nat:
        row("c6_ratios", nat["c6_ratios"], pa["c6_ratios"])
    lr_ref = pa["electrostatic_energy"] + pa["dispersion_energy"]
    row("long-range (per atom)", nat["physical_energy"], lr_ref)
    print(f"    {'electrostatics total':28s} native {dbg['electrostatic_total']:.10f}  "
          f"reference {pa['electrostatic_energy'].sum():.10f}")
    print(f"    {'dispersion total':28s} native {dbg['dispersion_total']:.10f}  "
          f"reference {pa['dispersion_energy'].sum():.10f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("reference")
    p.add_argument("dump")
    p.add_argument("log")
    p.add_argument("--tol-force", type=float, default=1e-9, help="max |dF| in eV/A")
    p.add_argument("--tol-energy-per-atom", type=float, default=1e-9)
    p.add_argument("--debug", help="native per-atom dump from SO3LR_NATIVE_DEBUG_DUMP")
    a = p.parse_args()

    ref = json.load(open(a.reference))
    f_ref = np.asarray(ref["forces"])
    f_nat = read_dump(a.dump)
    if f_nat.shape != f_ref.shape:
        sys.exit(f"shape mismatch: native {f_nat.shape} vs reference {f_ref.shape}")
    e_nat = read_pe(a.log)
    e_ref = ref["energy_native_convention"]
    n = ref["natoms"]

    df = f_nat - f_ref
    de = e_nat - e_ref
    print(f"model {ref['model']}  natoms {n}  periodic {ref['periodic']}")
    print(f"  energy   native {e_nat:.10f}  reference {e_ref:.10f}")
    print(f"           |dE| = {abs(de):.3e} eV   ({abs(de)/n:.3e} eV/atom, rel {abs(de)/abs(e_ref):.3e})")
    print(f"  forces   max|dF| = {np.abs(df).max():.3e} eV/A   "
          f"MAE = {np.abs(df).mean():.3e}   RMS = {np.sqrt((df**2).mean()):.3e}")
    print(f"           max|F|  = {np.abs(f_ref).max():.3e} eV/A   "
          f"max rel = {np.abs(df).max()/np.abs(f_ref).max():.3e}")
    if a.debug:
        compare_terms(ref, json.load(open(a.debug)))
    ok = np.abs(df).max() <= a.tol_force and abs(de) / n <= a.tol_energy_per_atom
    print("COMPARE=" + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
