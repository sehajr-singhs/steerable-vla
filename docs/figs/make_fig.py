"""Generate fig1_architecture.png — the Steerable VLA hierarchy, monochrome.

Run:  python make_fig.py
Output: fig1_architecture.png (white background, black/gray ink, no color)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK = "#111111"
GRAY = "#555555"
FAINT = "#e8e8e8"
FACE = "#ffffff"

fig, ax = plt.subplots(figsize=(9.2, 6.4), dpi=200)
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis("off")


def box(x, y, w, h, title, lines, fc=FACE, ec=INK, tfs=10.5, lfs=8.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
                                fc=fc, ec=ec, lw=1.3))
    ax.text(x + w / 2, y + h - 0.34, title, ha="center", va="top",
            fontsize=tfs, fontweight="bold", color=INK)
    if lines:
        ax.text(x + w / 2, y + 0.34, lines, ha="center", va="bottom",
                fontsize=lfs, color=GRAY, linespacing=1.5)


def arrow(x1, y1, x2, y2, lw=1.4, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=14, lw=lw, color=INK))


def label(x, y, s, ha="right", fs=8.2, color=GRAY):
    ax.text(x, y, s, ha=ha, va="center", fontsize=fs, color=color, style="italic")


# --- Top row: VLM planner -------------------------------------------------
box(0.6, 9.6, 5.4, 2.2,
    "Vision–Language Planner Λ_φ  (3B+ VLM)",
    "instruction ℓ  +  observation o_t\n→  dense visual subgoals  g_1..g_K\n(keyframe κ_k  +  language step ℓ_k)")

# planner inputs
label(0.4, 11.6, "instruction ℓ", ha="left")
arrow(0.9, 11.6, 0.9, 11.25)
label(0.4, 11.0, "observation o_t", ha="left")
arrow(0.9, 11.0, 0.9, 10.4)

# --- Steering column (right) ----------------------------------------------
box(6.9, 9.6, 2.7, 2.2,
    "Steering",
    "human corrections\nworld-model replans\nconstraint setpoints\nu_1..u_J  (neutral u⁰)")
arrow(9.6, 10.7, 8.5, 10.7)

# --- Middle: SMC + flow expert --------------------------------------------
box(0.6, 6.4, 5.4, 2.2,
    "Steerable Multimodal Conditioning (SMC)",
    "v_θ^S = v_θ^N + Σ_j λ_j(x_s,s) · v_θ^j(x_s,s,u_j)\nλ_j = σ(w_j^T φ)   ·   v_θ^j(·, u⁰) ≡ 0\nGrönwall bound:  ‖x_S − x_N‖ ≤ (M/L_v)(e^{L_v Δs} − 1)")
arrow(3.3, 9.6, 3.3, 8.6)          # subgoals in
arrow(3.3, 6.4, 3.3, 5.3)          # conditioned flow down

box(0.6, 3.1, 5.4, 2.2,
    "Flow-Matching Action Expert v_θ",
    "conditional flow matching on action chunks\nL_CFM = E ‖v_θ(x_s,s,c) − (x_1 − x_0)‖²\nx_1 = x_0 + ∫₀¹ v_θ(x_s,s,c) ds   (2–10 steps)")
arrow(3.3, 3.1, 3.3, 2.0)

# --- Safety + robot -------------------------------------------------------
box(0.6, 0.4, 5.4, 1.6,
    "CBF–QP Safety Filter  (runtime verification)",
    "u* = argmin ½‖u − u_cmd‖²_W   s.t.  ḣ + αh ≥ 0,  u ∈ U_act\nforward invariance:  h(x(0)) ≥ 0 ⇒ h(x(t)) ≥ 0")
arrow(3.3, 2.0, 3.3, 1.3)

box(6.9, 0.4, 2.7, 1.6,
    "Robot",
    "50 Hz receding horizon\ncanonical action frame")
arrow(6.0, 1.2, 6.9, 1.2)

# --- Observation feedback loop --------------------------------------------
arrow(8.25, 0.4, 8.25, 7.6, lw=1.1, style="<|-")
label(9.05, 4.0, "observation o_t feedback\n(RGB-D, proprio, tactile)", ha="left")

# small legend
ax.text(0.6, 11.9, "Steerable VLA Flow-Matching Hierarchy", fontsize=12.5,
        fontweight="bold", color=INK, va="top")

plt.tight_layout(pad=0.4)
plt.savefig("fig1_architecture.png", bbox_inches="tight", facecolor="white")
print("wrote fig1_architecture.png")
