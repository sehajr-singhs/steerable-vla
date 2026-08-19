"""Generate monochrome figures from results/*.json into docs/figs/.

Nothing hard-coded: every figure reads committed JSON.
Usage: PYTHONPATH=src python scripts/make_figs.py
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "docs", "figs")
os.makedirs(FIGS, exist_ok=True)

INK, GRAY, FAINT = "#111111", "#555555", "#8f8f8f"


def style(ax, ylabel, xlabel=None):
    for s in ax.spines.values():
        s.set_color(FAINT)
    ax.tick_params(colors=GRAY, labelsize=8)
    ax.set_ylabel(ylabel, color=INK, fontsize=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK, fontsize=10)


def fig_main():
    p = os.path.join(RESULTS, "main.json")
    if not os.path.exists(p):
        return
    rows = json.load(open(p))
    variants = ["bc", "flow_flat", "ours_nofilter", "ours_full"]
    labels = ["BC (no flow)", "Flow, flat", "Ours − filter", "Ours (full)"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.1))
    for ax, key, ylab in [(axes[0], "success", "zero-shot success"),
                          (axes[1], "jerk", "max jerk"),
                          (axes[2], "violations", "safety violations")]:
        vals = []
        for v in variants:
            rs = [r[key] for r in rows if r["variant"] == v]
            vals.append(rs)
        ax.bar(range(len(variants)), [np.mean(v) for v in vals],
               color="white", edgecolor=INK, linewidth=1.2, width=0.62)
        for i, v in enumerate(vals):
            ax.scatter(np.full(len(v), i), v, s=12, color=INK, zorder=3)
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=7.5, color=GRAY)
        style(ax, ylab)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "results_main.png"), dpi=200, facecolor="white")
    print("wrote docs/figs/results_main.png")


def fig_flywheel():
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for s in ["none", "near_miss", "relabel"]:
        p = os.path.join(RESULTS, f"flywheel_{s}.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        ax.plot(range(len(d["curve"])), d["curve"], "-o", color=INK if s == "relabel"
                else GRAY, lw=1.4, ms=4, label=s)
    ax.axhline(0, color=FAINT, lw=0.8)
    style(ax, "deployment success")
    ax.set_xlabel("flywheel iteration", color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "results_flywheel.png"), dpi=200, facecolor="white")
    print("wrote docs/figs/results_flywheel.png")


def fig_expert():
    p = os.path.join(RESULTS, "expert.json")
    if not os.path.exists(p):
        return
    d = json.load(open(p))
    fig, ax = plt.subplots(figsize=(4.2, 3))
    names = ["train families", "held-out families"]
    vals = [np.mean([x["success"] for x in d["train"]]),
            np.mean([x["success"] for x in d["held"]])]
    ax.bar(names, vals, color="white", edgecolor=INK, linewidth=1.2, width=0.5)
    style(ax, "oracle success")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "results_expert.png"), dpi=200, facecolor="white")
    print("wrote docs/figs/results_expert.png")


if __name__ == "__main__":
    fig_main()
    fig_flywheel()
    fig_expert()
