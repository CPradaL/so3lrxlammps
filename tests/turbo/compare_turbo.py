#!/usr/bin/env python3
"""Compare so3lr/turbo against so3lr/native/mpi, image by image.

    python compare_turbo.py OUT_DIR [--json summary.json]

OUT_DIR holds <model>/{ordinary,turbo}/<tag>.{log,forces.dump,final.dump}
as written by job_turbo_validate.sh. Tags ending in "_dup" are compared with
the tag they duplicate as well (same molecule twice in one pack).
"""
import argparse
import json
import pathlib
import re
import sys

import numpy as np


def dump_frame(path, cols):
    lines = path.read_text().split("\n")
    starts = [k for k, l in enumerate(lines) if l.startswith("ITEM: ATOMS")]
    i = starts[-1]
    names = lines[i].split()[2:]
    rows = []
    for l in lines[i + 1:]:
        if not l.strip() or l.startswith("ITEM"):
            break
        rows.append([float(x) for x in l.split()])
    rows = np.array(rows)
    rows = rows[np.argsort(rows[:, names.index("id")])]
    return rows[:, [names.index(c) for c in cols]]


def marker(path, kind, key):
    m = re.findall(rf"{kind} tag=\S+ .*?{key}=(\S+)", path.read_text())
    return float(m[-1]) if m else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--json")
    ap.add_argument("--tol-force", type=float, default=1e-9)
    ap.add_argument("--tol-energy", type=float, default=1e-9)
    a = ap.parse_args()
    root = pathlib.Path(a.out)
    summary, ok = {}, True
    for mdir in sorted(p for p in root.iterdir() if (p / "turbo").is_dir()):
        model = mdir.name
        print(f"=== {model} ===")
        summary[model] = {}
        for log in sorted((mdir / "turbo").glob("*.log")):
            tag = log.stem
            base = tag[:-4] if tag.endswith("_dup") else tag
            ref = mdir / "ordinary"
            r = {}
            try:
                ft = dump_frame(mdir / "turbo" / f"{tag}.forces.dump", ["fx", "fy", "fz"])
                fo = dump_frame(ref / f"{base}.forces.dump", ["fx", "fy", "fz"])
                r["sp_dE"] = abs(marker(log, "SO3LR_SP", "pe") - marker(ref / f"{base}.log", "SO3LR_SP", "pe"))
                r["sp_max_dF"] = float(np.abs(ft - fo).max())
                r["max_F"] = float(np.abs(fo).max())
                r["atoms"] = len(fo)
                fin_t = mdir / "turbo" / f"{tag}.final.dump"
                if fin_t.exists():
                    xt = dump_frame(fin_t, ["x", "y", "z"])
                    xo = dump_frame(ref / f"{base}.final.dump", ["x", "y", "z"])
                    r["md_max_dx"] = float(np.abs(xt - xo).max())
                    r["md_dE_total"] = abs(marker(log, "SO3LR_MD", "etotal")
                                           - marker(ref / f"{base}.log", "SO3LR_MD", "etotal"))
                passed = r["sp_dE"] <= a.tol_energy * max(1, r["atoms"]) and r["sp_max_dF"] <= a.tol_force
            except Exception as exc:  # missing output = failure
                r["error"] = str(exc)
                passed = False
            r["pass"] = bool(passed)
            ok &= passed
            summary[model][tag] = r
            md = (f"  NVE: max|dx| {r['md_max_dx']:.2e} A, |dEtot| {r['md_dE_total']:.2e} eV"
                  if "md_max_dx" in r else "")
            if "error" in r:
                print(f"  {tag:14s} FAIL  {r['error']}")
            else:
                print(f"  {tag:14s} {'ok  ' if passed else 'FAIL'}  {r['atoms']:4d} atoms  "
                      f"|dE| {r['sp_dE']:.2e} eV  max|dF| {r['sp_max_dF']:.2e} eV/A "
                      f"(max|F| {r['max_F']:.2f}){md}")
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(summary, indent=1))
    print("SO3LR_TURBO_COMPARE", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
