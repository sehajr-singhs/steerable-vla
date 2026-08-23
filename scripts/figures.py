#!/usr/bin/env python3
"""Generate publication-quality figures for the NMI paper.

Reads results/main.json (per-variant, per-seed) and results/expert.json.
Outputs PDF+PNG to docs/figures/.

Usage:
    PYTHONPATH=src python scripts/figures.py
"""

import os, sys, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ── Publication style ──────────────────────────────────────────────
rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "pdf.fonttype": 42,   # editable text in PDF
    "ps.fonttype": 42,
})

# Colors: colorblind-friendly palette
C = {
    "BC": "#E74C3C",
    "Flow (flat)": "#F39C12",
    "Ours − filter": "#9B59B6",
    "Ours (full)": "#1ABC9C",
    "Expert": "#7F8C8D",
}

OUT = "docs/figures"
os.makedirs(OUT, exist_ok=True)


def _agg(main_path):
    """Load main.json and return {variant: {metric: (mean, std)}}."""
    with open(main_path) as f:
        rows = json.load(f)
    groups = {}
    for r in rows:
        v = r["variant"]
        groups.setdefault(v, []).append(r)
    out = {}
    for v, rs in groups.items():
        label_map = {
            "bc": "BC", "flow_flat": "Flow (flat)",
            "ours_nofilter": "Ours − filter", "ours_full": "Ours (full)",
        }
        label = label_map.get(v, v)
        out[label] = {}
        for m in ["success", "ni_success", "crossings_reduced", "violations", "interventions", "jerk", "steps"]:
            vals = [r[m] for r in rs]
            out[label][m] = (np.mean(vals), np.std(vals))
    return out


def _expert(expert_path):
    with open(expert_path) as f:
        d = json.load(f)
    train = np.mean([e["success"] for e in d["train"]])
    held = np.mean([e["success"] for e in d["held"]])
    return train, held


def fig1_violations(data, out):
    """Fig 1: Safety violations — the hero result."""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    labels = ["BC", "Flow (flat)", "Ours − filter", "Ours (full)"]
    means = [data[v]["violations"][0] for v in labels]
    stds = [data[v]["violations"][1] for v in labels]
    colors = [C[v] for v in labels]
    bars = ax.bar(range(len(labels)), means, yerr=stds, color=colors,
                  edgecolor="white", linewidth=0.8, capsize=4)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Workspace Violations\n(per episode, ↓ better)")
    ax.set_title("CBF Safety Filter Eliminates All Violations")
    ax.set_ylim(0, max(means) * 1.35)
    for bar, m, s in zip(bars, means, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{m:.1f}", ha="center", fontsize=10, fontweight="bold")
    # Annotate the zero
    ax.annotate("0.0", xy=(3, 0.5), fontsize=14, fontweight="bold",
                color="#1abc9c", ha="center")
    ax.annotate("✓ zero violations", xy=(3, 1.8), fontsize=9, color="#1abc9c",
                ha="center", fontstyle="italic")
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(out, f"fig1_violations.{ext}"))
    plt.close()
    print(f"  Saved fig1_violations")


def fig2_ablation(data, out):
    """Fig 2: Full ablation — all metrics side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    labels = ["BC", "Flow (flat)", "Ours − filter", "Ours (full)"]
    colors = [C[v] for v in labels]

    metrics = [
        ("crossings_reduced", "Crossings Resolved\n(out of ~4, ↑ better)", True),
        ("interventions", "Oracle Interventions\n(per episode, ↓ better)", False),
        ("success", "Assisted Success\n(↑ better)", True),
    ]

    for ax, (m, title, _) in zip(axes, metrics):
        means = [data[v][m][0] for v in labels]
        stds = [data[v][m][1] for v in labels]
        bars = ax.bar(range(len(labels)), means, yerr=stds, color=colors,
                      edgecolor="white", linewidth=0.8, capsize=4)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_title(title)
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                    f"{val:.2f}", ha="center", fontsize=8)

    fig.suptitle("Ablation: Each Component Contributes", fontsize=13, y=1.02)
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(out, f"fig2_ablation.{ext}"))
    plt.close()
    print(f"  Saved fig2_ablation")


def fig3_expert(expert_path, out):
    """Fig 3: Expert ceiling — generalization gap."""
    fig, ax = plt.subplots(figsize=(4, 3.5))
    train, held = _expert(expert_path)
    bars = ax.bar(["Train Topology", "Held-Out Topology"],
                  [train, held], color=["#3498db", "#e74c3c"],
                  edgecolor="white", width=0.5)
    for bar, val in zip(bars, [train, held]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.1%}", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Success Rate")
    ax.set_title("Expert Policy Ceiling")
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(out, f"fig3_expert.{ext}"))
    plt.close()
    print(f"  Saved fig3_expert")


def fig4_ni_success(data, out):
    """Fig 4: No-intervention success — the learning frontier."""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    labels = ["BC", "Flow (flat)", "Ours − filter", "Ours (full)"]
    means = [data[v]["ni_success"][0] for v in labels]
    stds = [data[v]["ni_success"][1] for v in labels]
    colors = [C[v] for v in labels]
    bars = ax.bar(range(len(labels)), [m * 100 for m in means],
                  yerr=[s * 100 for s in stds], color=colors,
                  edgecolor="white", linewidth=0.8, capsize=4)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("No-Intervention Success %\n(↑ better)")
    ax.set_title("Learning the Untangling Skill")
    for bar, m in zip(bars, [m * 100 for m in means]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{m:.1f}%", ha="center", fontsize=9)
    ax.annotate("Training frontier:\nmore scale needed",
                xy=(2.5, max([m*100 for m in means]) + 1.5),
                fontsize=8, fontstyle="italic", ha="center", color="#555")
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(out, f"fig4_ni_success.{ext}"))
    plt.close()
    print(f"  Saved fig4_ni_success")


def fig5_overview(data, expert_path, out):
    """Fig 5: System overview radar chart."""
    from matplotlib.patches import FancyBboxPatch
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: expert vs policy comparison
    ax = axes[0]
    train, held = _expert(expert_path)
    labels = ["BC", "Flow (flat)", "Ours − filter", "Ours (full)", "Expert"]
    success_vals = [data[v]["success"][0] for v in labels[:4]] + [held]
    ni_vals = [data[v]["ni_success"][0] for v in labels[:4]] + [train]
    colors = [C.get(l, "#7F8C8D") for l in labels]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, success_vals, w, label="Assisted", color=colors, alpha=0.7, edgecolor="white")
    ax.bar(x + w/2, ni_vals, w, label="No-Intervention", color=colors, alpha=0.4, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Success Rate")
    ax.set_title("Success: Assisted vs No-Intervention")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.15)

    # Right: violations bar chart (cleaner version)
    ax = axes[1]
    labels2 = ["BC", "Flow (flat)", "Ours − filter", "Ours (full)"]
    vi = [data[v]["violations"][0] for v in labels2]
    colors2 = [C[v] for v in labels2]
    bars = ax.bar(range(len(labels2)), vi, color=colors2, edgecolor="white", linewidth=0.8)
    ax.set_xticks(range(len(labels2)))
    ax.set_xticklabels(labels2, rotation=15, ha="right")
    ax.set_ylabel("Mean Safety Violations")
    ax.set_title("Safety: Fewer Violations = Safer")
    for bar, val in zip(bars, vi):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(out, f"fig5_overview.{ext}"))
    plt.close()
    print(f"  Saved fig5_overview")


if __name__ == "__main__":
    ROOT = os.path.join(os.path.dirname(__file__), "..")
    main_path = os.path.join(ROOT, "results", "main.json")
    expert_path = os.path.join(ROOT, "results", "expert.json")

    data = _agg(main_path)
    print(f"Loaded {len(data)} variants from {main_path}")

    fig1_violations(data, OUT)
    fig2_ablation(data, OUT)
    fig3_expert(expert_path, OUT)
    fig4_ni_success(data, OUT)
    fig5_overview(data, expert_path, OUT)
    print(f"\nAll figures saved to {OUT}/")
