#!/usr/bin/env python3
"""Generate publication-quality figures for the NMI paper.

Usage:
    python scripts/figures.py --results results/benchmark_results.json

Generates:
  - fig1_system.pdf: System overview diagram
  - fig2_results.pdf: Main results table as heatmap
  - fig3_ablation.pdf: Ablation study (filter / subgoal / replan)
  - fig4_curves.pdf: Training curves / flywheel iterations
  - fig5_real.pdf: Sim-to-real comparison (UR5 env)
"""

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not installed. Install with: pip install matplotlib")

# -- Style constants matching the repo's dark theme --
COLORS = {
    "bc": "#e74c3c",
    "act": "#3498db",
    "diffusion": "#2ecc71",
    "flow_flat": "#f39c12",
    "ours_nofilter": "#9b59b6",
    "ours": "#1abc9c",
    "expert": "#95a5a6",
}

STYLE = {
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
}


def fig_main_results(results, out_dir):
    """Fig 2: Main results heatmap — variants × metrics."""
    if not HAS_MPL:
        return

    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    variants = []
    metrics = {
        "success": "Task Success",
        "crossings_reduced": "Crossings Reduced",
        "violations": "Safety Violations",
    }

    for key, data in results.items():
        if isinstance(data, dict) and "variants" in data:
            variants = list(data["variants"].keys())
            break

    if not variants:
        print("  No variant data found for fig_main_results")
        return

    for ax_idx, (metric, title) in enumerate(metrics.items()):
        ax = axes[ax_idx]
        vals = []
        for v in variants:
            vdata = None
            for key, data in results.items():
                if isinstance(data, dict) and "variants" in data and v in data["variants"]:
                    vdata = data["variants"][v]
                    break
            if vdata and metric in vdata:
                vals.append(vdata[metric])
            else:
                vals.append(0)

        colors = [COLORS.get(v.split("_s")[0].replace("-", "_"), "#95a5a6") for v in variants]
        bars = ax.barh(range(len(variants)), vals, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(len(variants)))
        ax.set_yticklabels([v.replace("_s0", "").replace("_s1", "").replace("_s2", "")
                           for v in variants], fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("Score" if metric != "violations" else "Count")

        # value labels
        for bar, val in zip(bars, vals):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                   f"{val:.2f}", va="center", fontsize=7)

    fig.suptitle("Main Protocol Results (Cable Untangling)", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig2_results.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    plt.close()
    print(f"  Saved {path}")


def fig_expert_ceiling(results, out_dir):
    """Fig 3: Expert ceiling — train vs held-out success."""
    if not HAS_MPL:
        return

    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(5, 3))

    expert = results.get("expert", {})
    train_s = expert.get("train_success", expert.get("success", 0))
    held_s = expert.get("held_success", expert.get("held_out_success", 0))

    x = ["Train Topology", "Held-Out Topology"]
    y = [train_s, held_s]
    colors = [COLORS["expert"], "#7f8c8d"]

    bars = ax.bar(x, y, color=colors, edgecolor="white", width=0.5)
    for bar, val in zip(bars, y):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
               f"{val:.1%}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Success Rate")
    ax.set_title("Expert Policy Ceiling (Oracle Subgoal + CBF Filter)")

    plt.tight_layout()
    path = os.path.join(out_dir, "fig3_expert.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    plt.close()
    print(f"  Saved {path}")


def fig_safety_filter(results, out_dir):
    """Fig 4: Safety violations comparison."""
    if not HAS_MPL:
        return

    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(6, 4))

    variants = []
    violations = []
    for key, data in results.items():
        if isinstance(data, dict) and "variants" in data:
            for vname, vdata in data["variants"].items():
                clean = vname.split("_s")[0]
                if clean not in [v.split("_s")[0] for v in variants]:
                    variants.append(vname)
                    violations.append(vdata.get("violations", 0))

    colors = [COLORS.get(v.split("_s")[0].replace("-", "_"), "#95a5a6") for v in variants]
    bars = ax.bar(range(len(variants)), violations, color=colors, edgecolor="white")
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels([v.replace("_s0", "").replace("_s1", "").replace("_s2", "")
                        for v in variants], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Mean Safety Violations per Episode")
    ax.set_title("CBF Safety Filter: Zero Violations with Our Method")

    for bar, val in zip(bars, violations):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
               f"{val:.1f}", ha="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(out_dir, "fig4_safety.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    plt.close()
    print(f"  Saved {path}")


def fig_flywheel(results, out_dir):
    """Fig 5: Data flywheel iteration curves."""
    if not HAS_MPL:
        return

    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(6, 4))

    flywheel_keys = [k for k in results if k.startswith("flywheel_")]
    if not flywheel_keys:
        # try nested
        for key, data in results.items():
            if isinstance(data, dict):
                flywheel_keys = [k for k in data if k.startswith("flywheel_")]
                if flywheel_keys:
                    break

    if not flywheel_keys:
        print("  No flywheel data found")
        return

    for fk in sorted(flywheel_keys):
        fdata = None
        for key, data in results.items():
            if isinstance(data, dict) and fk in data:
                fdata = data[fk]
                break
        if not fdata:
            continue

        label = fk.replace("flywheel_", "").replace("_", " ").title()
        if isinstance(fdata, dict):
            if "success_curve" in fdata:
                ax.plot(fdata["success_curve"], label=label, marker="o", markersize=3)
            elif "ni_success_curve" in fdata:
                ax.plot(fdata["ni_success_curve"], label=label, marker="o", markersize=3)

    ax.set_xlabel("Flywheel Iteration")
    ax.set_ylabel("No-Intervention Success Rate")
    ax.set_title("Data Flywheel: Compounding Skill Across Iterations")
    ax.legend()
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    path = os.path.join(out_dir, "fig5_flywheel.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    plt.close()
    print(f"  Saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="results/benchmark_results.json")
    parser.add_argument("--output", type=str, default="docs/figures")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    with open(args.results) as f:
        results = json.load(f)

    print(f"Generating figures from {args.results}...")
    fig_main_results(results, args.output)
    fig_expert_ceiling(results, args.output)
    fig_safety_filter(results, args.output)
    fig_flywheel(results, args.output)
    print(f"\nAll figures saved to {args.output}/")


if __name__ == "__main__":
    main()
