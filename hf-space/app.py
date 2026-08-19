"""Steerable VLA — results dashboard (Gradio Space).

Renders the committed study results: variant comparison, safety envelope,
and the data-flywheel curves. Includes a live miniature demo: run the
scripted oracle on a fresh tangled cable and watch it untangle.

Everything reads from `results/` bundled with the Space (synced from the
repo by scripts/sync_hf_space.py). No GPU needed.
"""

import json
import os

import gradio as gr
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

VARIANT_LABELS = {
    "bc": "BC (flat MLP)",
    "flow_flat": "Flow, no subgoal",
    "ours_nofilter": "Flow + SMC (no filter)",
    "ours_full": "Flow + SMC + CBF (ours)",
}


def _load(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)


def variant_chart():
    """Grouped bar chart: ni_success + interventions by variant."""
    rows = _load("main.json")
    labels, ni, iv, viol = [], [], [], []
    for r in sorted(rows, key=lambda x: x["variant"]):
        labels.append(VARIANT_LABELS.get(r["variant"], r["variant"]))
        ni.append(r["ni_success"])
        iv.append(r["interventions"])
        viol.append(r["violations"])
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(labels))
    w = 0.32
    ax.bar(x - w, ni, w, label="zero-shot success (no interventions)", color="#111")
    ax.bar(x, iv, w, label="oracle interventions / ep", color="#777")
    ax.bar(x + w, viol, w, label="safety violations / ep", color="#ccc")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=14, ha="right", fontsize=8)
    ax.set_ylabel("mean per episode")
    ax.legend(fontsize=8, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def flywheel_chart():
    """Flywheel success curves: the strategy decides whether it compounds."""
    fig, ax = plt.subplots(figsize=(9, 4.2))
    styles = {"none": ("o", "#999"), "near_miss": ("s", "#555"),
              "relabel": ("^", "#111")}
    for strat, (mk, col) in styles.items():
        try:
            d = _load(f"flywheel_{strat}.json")
        except FileNotFoundError:
            continue
        curve = d["curve"]
        ax.plot(range(1, len(curve) + 1), curve, marker=mk, color=col,
                label=strat, linewidth=1.6)
    ax.set_xlabel("flywheel iteration")
    ax.set_ylabel("deployment success")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def summary_table():
    rows = sorted(_load("main.json"), key=lambda x: x["variant"])
    head = ["variant", "ni_success", "success", "interventions",
            "violations", "jerk", "steps", "n"]
    data = [[VARIANT_LABELS.get(r["variant"], r["variant"]),
             f"{r['ni_success']:.3f}", f"{r['success']:.3f}",
             f"{r['interventions']:.2f}", f"{r['violations']:.3f}",
             f"{r['jerk']:.3f}", f"{r['steps']:.0f}", r["n"]] for r in rows]
    return head, data


def run_demo(crossings, stiffness):
    """Run the oracle on a fresh cable; return before/after figures."""
    import sys
    sys.path.insert(0, HERE)
    from steerable.config import EnvConfig
    from steerable.envs.cable import CableEnv
    from steerable.policies.expert import run_expert

    cfg = EnvConfig()
    env = CableEnv(cfg, seed=7, stiffness_mult=float(stiffness),
                   crossing_target=int(crossings))

    def snap(env):
        fig, ax = plt.subplots(figsize=(7, 3.2))
        x = env.x
        ax.plot(x[:, 0], x[:, 1], "-o", ms=3, lw=1.6, color="#111")
        ax.plot(env.gripper[0], env.gripper[1], "s", ms=7, color="#555")
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-0.4, 1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        fig.tight_layout()
        return fig

    before = snap(env)
    r = run_expert(env)
    after = snap(env)
    msg = (f"crossings {env.crossings0} -> {env.crossings()} | "
           f"success={r['success']} | steps={r['steps']} | "
           f"violations={env.violations}")
    return before, after, msg


with gr.Blocks(theme=gr.themes.Base(), css="footer{display:none}") as demo:
    gr.Markdown(
        "# π — Steerable VLA (miniature study)\n"
        "Zero-shot generalization in a cable-untangling toy: a conditional "
        "flow-matching action expert, steerable multimodal conditioning, and a "
        "runtime CBF–QP safety envelope. Results are committed and reproduced "
        "by the harness — no numbers are fabricated."
    )
    with gr.Row():
        with gr.Column():
            gr.Markdown("## Variants (held-out families, zero-shot)")
            gr.Plot(variant_chart())
        with gr.Column():
            gr.Markdown("## Data flywheel")
            gr.Plot(flywheel_chart())
    gr.Markdown("## Numbers")
    gr.Dataframe(value=summary_table(), headers=["variant", "ni_success",
                                                 "success", "interventions",
                                                 "violations", "jerk",
                                                 "steps", "n"])
    gr.Markdown("## Live oracle demo")
    with gr.Row():
        crossings = gr.Slider(1, 4, value=2, step=1, label="crossing target")
        stiffness = gr.Slider(0.6, 1.5, value=1.0, step=0.1,
                              label="stiffness multiplier")
        btn = gr.Button("Untangle a fresh cable")
    with gr.Row():
        before = gr.Plot(label="before")
        after = gr.Plot(label="after")
    out = gr.Markdown()
    btn.click(run_demo, [crossings, stiffness], [before, after, out])

if __name__ == "__main__":
    demo.launch()
