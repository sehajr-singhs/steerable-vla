"""Monochrome visualization: cable episodes and result figures.

All figures are strictly black / gray / white to match the study's aesthetic.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .envs.geometry import count_crossings

INK = "#111111"
GRAY = "#555555"
FAINT = "#8f8f8f"


def plot_cable(env, ax=None, title=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3))
    x = env.x
    ax.plot(x[:, 0], x[:, 1], "-o", color=INK, lw=1.6, ms=4, zorder=2)
    ax.plot(x[0, 0], x[0, 1], "s", color=INK, ms=8, zorder=3)
    ax.plot(x[-1, 0], x[-1, 1], "s", color=INK, ms=8, zorder=3)
    ax.plot([env.gripper[0]], [env.gripper[1]], "^", color=GRAY, ms=10, zorder=3)
    hits, pts = count_crossings(x)
    for p in pts:
        ax.plot(p[0], p[1], "x", color=GRAY, ms=9, zorder=4)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-0.5, 1.1)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(FAINT)
    if title:
        ax.set_title(title, color=INK, fontsize=10)
    return ax


def style_ax(ax, ylabel):
    for s in ax.spines.values():
        s.set_color(FAINT)
    ax.tick_params(colors=GRAY, labelsize=8)
    ax.set_ylabel(ylabel, color=INK, fontsize=10)
    ax.yaxis.label.set_color(INK)
