#!/usr/bin/env python3
"""Parse LAMMPS scaling benchmark logs and generate scaling plots.

Dynamically discovers test directories under results/, handles repetitions
(rep_N/ subdirectories), computes mean +/- std, and generates plots with
error bars.

Usage:
    python analyze_scaling.py [--results-dir results/] [--plots-dir plots/]
"""

import argparse
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_log(log_path):
    """Parse a LAMMPS log file for timing data.

    Returns dict with keys: loop_time, nprocs, nsteps, natoms, pair_pct,
    neigh_pct, comm_pct, or None if parsing fails.
    """
    text = Path(log_path).read_text()

    # Find the SECOND "Loop time" line (benchmark run, after warmup)
    loop_matches = re.findall(
        r"Loop time of ([\d.]+) on (\d+) procs for (\d+) steps with (\d+) atoms",
        text,
    )
    if len(loop_matches) < 2:
        if len(loop_matches) == 1:
            match = loop_matches[0]
        else:
            print(f"  WARNING: No Loop time found in {log_path}")
            return None
    else:
        match = loop_matches[1]  # Second run = benchmark

    result = {
        "loop_time": float(match[0]),
        "nprocs": int(match[1]),
        "nsteps": int(match[2]),
        "natoms": int(match[3]),
    }

    # Parse the MPI task timing breakdown for Pair, Neigh, Comm percentages.
    #
    # Two fixes over the ML-IAP version: (1) the row has six columns
    # (min | avg | max | %varavg | %CPU | %total) and the %total is the LAST
    # one -- the old pattern required the line to end after `max`, so it never
    # matched and every percentage silently came out as 0; (2) the breakdown
    # must be read from the TIMED run, i.e. after the second "Loop time" line,
    # not from the warmup run that precedes it.
    loop_positions = [m.start() for m in re.finditer(r"^Loop time of", text, re.MULTILINE)]
    timed_text = text[loop_positions[1]:] if len(loop_positions) >= 2 else text
    for key in ["Pair", "Neigh", "Comm"]:
        pct_match = re.search(
            rf"^{key}\s+\|.*\|\s*([\d.]+)\s*$",
            timed_text,
            re.MULTILINE,
        )
        if pct_match:
            result[f"{key.lower()}_pct"] = float(pct_match.group(1))

    return result


def collect_results(results_dir):
    """Walk results directory and collect all parsed log data.

    Dynamically discovers test directories (strong_*, weak_*).
    Handles repetitions: if rep_N/ subdirs exist, collects all reps as a list.
    For backward compat, if no rep_N/ dirs, reads log.lammps directly.

    Returns dict: {test_name: {ngpus: [result, result, ...]}}
    """
    data = {}

    if not os.path.isdir(results_dir):
        return data

    for test_name in sorted(os.listdir(results_dir)):
        type_dir = os.path.join(results_dir, test_name)
        if not os.path.isdir(type_dir):
            continue

        data[test_name] = {}

        for gpu_dir in sorted(os.listdir(type_dir)):
            if not gpu_dir.startswith("gpu_"):
                continue
            ngpus = int(gpu_dir.split("_")[1])
            gpu_path = os.path.join(type_dir, gpu_dir)

            # Check for rep_N/ subdirectories
            rep_dirs = sorted([
                d for d in os.listdir(gpu_path)
                if d.startswith("rep_") and os.path.isdir(os.path.join(gpu_path, d))
            ])

            results_list = []
            if rep_dirs:
                for rep_dir in rep_dirs:
                    log_path = os.path.join(gpu_path, rep_dir, "log.lammps")
                    if os.path.exists(log_path):
                        result = parse_log(log_path)
                        if result:
                            results_list.append(result)
                            print(f"  {test_name}/gpu_{ngpus:03d}/{rep_dir}: "
                                  f"{result['loop_time']:.2f}s, {result['natoms']} atoms")
            else:
                # Backward compat: no rep dirs, read log.lammps directly
                log_path = os.path.join(gpu_path, "log.lammps")
                if os.path.exists(log_path):
                    result = parse_log(log_path)
                    if result:
                        results_list.append(result)
                        print(f"  {test_name}/gpu_{ngpus:03d}: "
                              f"{result['loop_time']:.2f}s, {result['natoms']} atoms")
                else:
                    print(f"  No log found: {log_path}")

            if results_list:
                data[test_name][ngpus] = results_list

    return data


def _mean_std(values):
    """Return (mean, std) for a list of values."""
    arr = np.array(values)
    return arr.mean(), arr.std() if len(arr) > 1 else 0.0


def compute_metrics(data):
    """Compute scaling metrics from collected data with mean/std across reps."""
    metrics = {}

    for label, results_by_gpu in data.items():
        if not results_by_gpu:
            continue

        gpu_counts = sorted(results_by_gpu.keys())
        if not gpu_counts:
            continue

        is_strong = label.startswith("strong")

        # Compute mean/std of loop_time for each GPU count
        times_mean = []
        times_std = []
        natoms_list = []
        nsteps_list = []
        comm_mean = []
        comm_std = []

        for g in gpu_counts:
            reps = results_by_gpu[g]
            t_vals = [r["loop_time"] for r in reps]
            c_vals = [r.get("comm_pct", 0) for r in reps]
            tm, ts = _mean_std(t_vals)
            cm, cs = _mean_std(c_vals)
            times_mean.append(tm)
            times_std.append(ts)
            comm_mean.append(cm)
            comm_std.append(cs)
            natoms_list.append(reps[0]["natoms"])
            nsteps_list.append(reps[0]["nsteps"])

        # Reference point (smallest GPU count)
        t1 = times_mean[0]
        s1 = times_std[0]
        n1 = gpu_counts[0]

        m = {
            "gpus": gpu_counts,
            "times": times_mean,
            "times_std": times_std,
            "comm_pct": comm_mean,
            "comm_std": comm_std,
            "throughput": [
                natoms_list[i] * nsteps_list[i] / times_mean[i]
                for i in range(len(gpu_counts))
            ],
            "throughput_std": [
                natoms_list[i] * nsteps_list[i] * times_std[i] / times_mean[i]**2
                for i in range(len(gpu_counts))
            ],
        }

        if is_strong:
            # Speedup = T_ref / T_n
            speedup = [t1 / times_mean[i] for i in range(len(gpu_counts))]
            # Error propagation: speedup * sqrt((s1/t1)^2 + (sn/tn)^2)
            speedup_std = [
                speedup[i] * np.sqrt((s1/t1)**2 + (times_std[i]/times_mean[i])**2)
                if times_mean[i] > 0 else 0.0
                for i in range(len(gpu_counts))
            ]
            efficiency = [speedup[i] / (gpu_counts[i] / n1)
                          for i in range(len(gpu_counts))]
            efficiency_std = [speedup_std[i] / (gpu_counts[i] / n1)
                              for i in range(len(gpu_counts))]
            m["speedup"] = speedup
            m["speedup_std"] = speedup_std
            m["efficiency"] = efficiency
            m["efficiency_std"] = efficiency_std
        else:
            # Weak scaling
            m["time_per_step"] = [
                times_mean[i] / nsteps_list[i]
                for i in range(len(gpu_counts))
            ]
            m["time_per_step_std"] = [
                times_std[i] / nsteps_list[i]
                for i in range(len(gpu_counts))
            ]
            # Weak efficiency = T1 / Tn
            efficiency = [t1 / times_mean[i] for i in range(len(gpu_counts))]
            efficiency_std = [
                efficiency[i] * np.sqrt((s1/t1)**2 + (times_std[i]/times_mean[i])**2)
                if times_mean[i] > 0 else 0.0
                for i in range(len(gpu_counts))
            ]
            m["efficiency"] = efficiency
            m["efficiency_std"] = efficiency_std

        metrics[label] = m

    return metrics


def _label_from_dirname(dirname):
    """Derive a human-readable label from a directory name.

    E.g. 'strong_192k' -> '192K', 'weak_30k' -> 'Weak 30K', 'weak' -> 'Weak'.
    """
    if dirname.startswith("strong_"):
        return dirname.replace("strong_", "").upper()
    elif dirname.startswith("weak_"):
        return "Weak " + dirname.replace("weak_", "").upper()
    elif dirname == "weak":
        return "Weak"
    return dirname


# Color/marker cycle for dynamic test discovery
_COLORS = ["tab:blue", "tab:red", "tab:green", "tab:orange", "tab:purple",
           "tab:brown", "tab:pink", "tab:cyan"]
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h"]


def _style(idx):
    return _COLORS[idx % len(_COLORS)], _MARKERS[idx % len(_MARKERS)]


def plot_strong_speedup(metrics, plots_dir):
    """Plot 1: Strong scaling speedup vs GPUs."""
    fig, ax = plt.subplots(figsize=(8, 6))

    strong_labels = sorted(k for k in metrics if k.startswith("strong"))
    if not strong_labels:
        plt.close(fig)
        return

    for idx, label in enumerate(strong_labels):
        m = metrics[label]
        color, marker = _style(idx)
        ax.errorbar(m["gpus"], m["speedup"], yerr=m["speedup_std"],
                     fmt=f"-{marker}", color=color, capsize=4,
                     label=_label_from_dirname(label), linewidth=2, markersize=8)

    # Ideal line
    all_gpus = set()
    for label in strong_labels:
        all_gpus.update(metrics[label]["gpus"])
    gpus = sorted(all_gpus)
    g0 = gpus[0]
    ax.plot(gpus, [g / g0 for g in gpus], "--k", alpha=0.5, label="Ideal")

    ax.set_xlabel("Number of GPUs")
    ax.set_ylabel("Speedup")
    ax.set_title("Strong Scaling: Speedup")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.legend()
    ax.grid(True, alpha=0.3)
    for fmt in ["png", "pdf"]:
        fig.savefig(os.path.join(plots_dir, f"strong_speedup.{fmt}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_strong_efficiency(metrics, plots_dir):
    """Plot 2: Strong scaling parallel efficiency vs GPUs."""
    fig, ax = plt.subplots(figsize=(8, 6))

    strong_labels = sorted(k for k in metrics if k.startswith("strong"))
    if not strong_labels:
        plt.close(fig)
        return

    for idx, label in enumerate(strong_labels):
        m = metrics[label]
        color, marker = _style(idx)
        ax.errorbar(m["gpus"], [e * 100 for e in m["efficiency"]],
                     yerr=[e * 100 for e in m["efficiency_std"]],
                     fmt=f"-{marker}", color=color, capsize=4,
                     label=_label_from_dirname(label), linewidth=2, markersize=8)

    ax.axhline(100, color="k", linestyle="--", alpha=0.5, label="Ideal")
    ax.set_xlabel("Number of GPUs")
    ax.set_ylabel("Parallel Efficiency (%)")
    ax.set_title("Strong Scaling: Efficiency")
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(True, alpha=0.3)
    for fmt in ["png", "pdf"]:
        fig.savefig(os.path.join(plots_dir, f"strong_efficiency.{fmt}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_weak_time(metrics, plots_dir):
    """Plot 3: Weak scaling time per step vs GPUs."""
    weak_labels = sorted(k for k in metrics if k.startswith("weak"))
    if not weak_labels:
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    for idx, label in enumerate(weak_labels):
        m = metrics[label]
        color, marker = _style(idx)
        ax.errorbar(m["gpus"], m["time_per_step"], yerr=m["time_per_step_std"],
                     fmt=f"-{marker}", color=color, capsize=4,
                     label=_label_from_dirname(label), linewidth=2, markersize=8)
        ax.axhline(m["time_per_step"][0], color=color, linestyle="--", alpha=0.3)

    ax.set_xlabel("Number of GPUs")
    ax.set_ylabel("Time per Step (s)")
    ax.set_title("Weak Scaling: Time per Step")
    ax.set_xscale("log", base=2)
    ax.legend()
    ax.grid(True, alpha=0.3)
    for fmt in ["png", "pdf"]:
        fig.savefig(os.path.join(plots_dir, f"weak_time.{fmt}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_weak_efficiency(metrics, plots_dir):
    """Plot 4: Weak scaling efficiency vs GPUs."""
    weak_labels = sorted(k for k in metrics if k.startswith("weak"))
    if not weak_labels:
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    for idx, label in enumerate(weak_labels):
        m = metrics[label]
        color, marker = _style(idx)
        ax.errorbar(m["gpus"], [e * 100 for e in m["efficiency"]],
                     yerr=[e * 100 for e in m["efficiency_std"]],
                     fmt=f"-{marker}", color=color, capsize=4,
                     label=_label_from_dirname(label), linewidth=2, markersize=8)

    ax.axhline(100, color="k", linestyle="--", alpha=0.5, label="Ideal")
    ax.set_xlabel("Number of GPUs")
    ax.set_ylabel("Weak Scaling Efficiency (%)")
    ax.set_title("Weak Scaling: Efficiency")
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(True, alpha=0.3)
    for fmt in ["png", "pdf"]:
        fig.savefig(os.path.join(plots_dir, f"weak_efficiency.{fmt}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_comm_overhead(metrics, plots_dir):
    """Plot 5: Communication overhead vs GPUs."""
    fig, ax = plt.subplots(figsize=(8, 6))

    all_labels = sorted(metrics.keys())
    has_data = False
    for idx, label in enumerate(all_labels):
        m = metrics[label]
        if any(c > 0 for c in m["comm_pct"]):
            color, marker = _style(idx)
            ax.errorbar(m["gpus"], m["comm_pct"], yerr=m["comm_std"],
                         fmt=f"-{marker}", color=color, capsize=4,
                         label=_label_from_dirname(label), linewidth=2, markersize=8)
            has_data = True

    if not has_data:
        plt.close(fig)
        return

    ax.set_xlabel("Number of GPUs")
    ax.set_ylabel("Communication (%)")
    ax.set_title("Communication Overhead vs GPU Count")
    ax.set_xscale("log", base=2)
    ax.legend()
    ax.grid(True, alpha=0.3)
    for fmt in ["png", "pdf"]:
        fig.savefig(os.path.join(plots_dir, f"comm_overhead.{fmt}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_throughput(metrics, plots_dir):
    """Plot 6: Throughput (atom*step/s) vs GPUs."""
    fig, ax = plt.subplots(figsize=(8, 6))

    all_labels = sorted(metrics.keys())
    for idx, label in enumerate(all_labels):
        m = metrics[label]
        color, marker = _style(idx)
        ax.errorbar(m["gpus"], m["throughput"], yerr=m["throughput_std"],
                     fmt=f"-{marker}", color=color, capsize=4,
                     label=_label_from_dirname(label), linewidth=2, markersize=8)

    ax.set_xlabel("Number of GPUs")
    ax.set_ylabel("Throughput (atom·step/s)")
    ax.set_title("Throughput vs GPU Count")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.legend()
    ax.grid(True, alpha=0.3)
    for fmt in ["png", "pdf"]:
        fig.savefig(os.path.join(plots_dir, f"throughput.{fmt}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_summary(metrics):
    """Print a summary table of results."""
    for label in sorted(metrics.keys()):
        m = metrics[label]
        is_strong = label.startswith("strong")
        print(f"\n{'=' * 80}")
        print(f"  {label.upper()}")
        print(f"{'=' * 80}")

        if is_strong:
            print(f"{'GPUs':>6} {'Time(s)':>12} {'Speedup':>12} {'Eff(%)':>12} "
                  f"{'Throughput':>15} {'Comm(%)':>12}")
            print("-" * 80)
            for i, g in enumerate(m["gpus"]):
                t = m["times"][i]
                ts = m["times_std"][i]
                sp = m["speedup"][i]
                ss = m["speedup_std"][i]
                eff = m["efficiency"][i] * 100
                tp = m["throughput"][i]
                comm = m["comm_pct"][i]
                print(f"{g:>6} {t:>8.2f}±{ts:<4.2f} {sp:>7.2f}±{ss:<4.2f}x "
                      f"{eff:>9.1f}% {tp:>15.0f} {comm:>9.1f}%")
        else:
            print(f"{'GPUs':>6} {'Time(s)':>12} {'T/step(s)':>12} {'Eff(%)':>12} "
                  f"{'Throughput':>15} {'Comm(%)':>12}")
            print("-" * 80)
            for i, g in enumerate(m["gpus"]):
                t = m["times"][i]
                ts = m["times_std"][i]
                tps = m["time_per_step"][i]
                tpss = m["time_per_step_std"][i]
                eff = m["efficiency"][i] * 100
                tp = m["throughput"][i]
                comm = m["comm_pct"][i]
                print(f"{g:>6} {t:>8.2f}±{ts:<4.2f} {tps:>8.4f}±{tpss:<6.4f} "
                      f"{eff:>9.1f}% {tp:>15.0f} {comm:>9.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Analyze LAMMPS scaling benchmarks")
    parser.add_argument("--results-dir", default="results/",
                        help="Results directory")
    parser.add_argument("--plots-dir", default="plots/",
                        help="Output directory for plots")
    args = parser.parse_args()

    os.makedirs(args.plots_dir, exist_ok=True)

    print("Collecting results...")
    data = collect_results(args.results_dir)

    total = sum(len(v) for v in data.values())
    if total == 0:
        print("No results found. Run the benchmarks first.")
        return

    print(f"\nFound {total} GPU configurations across {len(data)} test(s).")

    metrics = compute_metrics(data)
    print_summary(metrics)

    print("\nGenerating plots...")
    plot_strong_speedup(metrics, args.plots_dir)
    plot_strong_efficiency(metrics, args.plots_dir)
    plot_weak_time(metrics, args.plots_dir)
    plot_weak_efficiency(metrics, args.plots_dir)
    plot_comm_overhead(metrics, args.plots_dir)
    plot_throughput(metrics, args.plots_dir)
    print(f"Plots saved to {args.plots_dir}/")


if __name__ == "__main__":
    main()
